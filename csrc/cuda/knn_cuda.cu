#include "radius_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 256
#define PTS_BUFFER 2048

template <typename scalar_t> struct Cosine {
  static inline __device__ scalar_t dot(const scalar_t *a, const scalar_t *b,
                                        int64_t n_a, int64_t n_b,
                                        int64_t size) {
    scalar_t result = 0;
    for (int64_t i = 0; i < size; i++) {
      result += a[n_a * size + i] * b[n_b * size + i];
    }
    return result;
  }

  static inline __device__ scalar_t norm(const scalar_t *a, int64_t n_a,
                                         int64_t size) {
    scalar_t result = 0;
    for (int64_t i = 0; i < size; i++) {
      result += a[n_a * size + i] * a[n_a * size + i];
    }
    return sqrt(result);
  }
};

template <typename scalar_t>
__global__ void
knn_kernel(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
           const int64_t *__restrict__ ptr_x, const int64_t *__restrict__ ptr_y,
           const int64_t m, const int64_t dim, int64_t *__restrict__ best_idx, scalar_t *__restrict__ best_dist,
           const int64_t num_examples, const bool cosine) {

  const int64_t n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);
  
  for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
    scalar_t tmp_dist = 0;
    
    if (cosine) {
      tmp_dist = Cosine<scalar_t>::dot(x, y, n_x, n_y, dim) /
                 (Cosine<scalar_t>::norm(x, n_x, dim) *
                  Cosine<scalar_t>::norm(y, n_y, dim));
      tmp_dist = 1. - tmp_dist;
    } else {
      for (int64_t d = 0; d < dim; d++) {
        tmp_dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
                    (x[n_x * dim + d] - y[n_y * dim + d]);
      }
    }

    for (int64_t e1 = 0; e1 < PTS_BUFFER; e1++) {
      if (best_dist[e1 + PTS_BUFFER*n_y] > tmp_dist) {
        for (int64_t e2 = PTS_BUFFER - 1; e2 > e1; e2--) {
          best_dist[e2 + PTS_BUFFER*n_y] = best_dist[e2 - 1 + PTS_BUFFER*n_y];
          best_idx[e2 + PTS_BUFFER*n_y] = best_idx[e2 - 1 + PTS_BUFFER*n_y];
        }
        best_dist[e1 + PTS_BUFFER*n_y] = tmp_dist;
        best_idx[e1 + PTS_BUFFER*n_y] = n_x;
        break;
      }
    }
  }
}

template <typename scalar_t>
__global__ void
finalize_kernel(int64_t *__restrict__ row, int64_t *__restrict__ col, int64_t *__restrict__ last_n_y, int64_t *__restrict__ last_e, const int64_t k, int64_t *__restrict__ numk,
                const int64_t *__restrict__ ptr_y, const int64_t *__restrict__ best_idx, const scalar_t *__restrict__ best_dist, const long num_examples) {

  const int64_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= num_examples)
    return;

  const int64_t y_start = ptr_y[batch_idx];
  const int64_t y_end = ptr_y[batch_idx+1];

  for (int64_t e = 0; e < PTS_BUFFER; e++) {
    for (int64_t n_y = y_start; n_y < y_end; n_y++) {
      if ( numk[n_y] == k ) {
        continue;
      }
      
      const int64_t this_best_idx = best_idx[e + PTS_BUFFER*n_y];

      if ( last_n_y[this_best_idx] == -1 ) { // this_best_idx was never used for other cluster
        last_n_y[this_best_idx] = n_y; // this_best_idx is associated with n_y cluster
        last_e[this_best_idx] = e; // this_best_idx is found at e_{th} iteration in range(PTS_BUFFER)

        row[n_y * k + numk[n_y]] = n_y;
        col[n_y * k + numk[n_y]] = this_best_idx;
        numk[n_y] = numk[n_y] + 1;

      } else if ( last_e[this_best_idx] == e ) { // this_best_idx was already used for other cluster, but exactly at this e_{th} iteration in range(PTS_BUFFER)
          if ( best_dist[e + PTS_BUFFER * last_n_y[best_idx[e + PTS_BUFFER * n_y]]] > best_dist[e + PTS_BUFFER * n_y] ) { // if the distance is shorter for this cluster
            numk[last_n_y[this_best_idx]] = numk[last_n_y[this_best_idx]] - 1; // cancel previous allocation of this_best_idx to last_n_y

            last_n_y[this_best_idx] = n_y; // this_best_idx is newly associated with n_y cluster
            last_e[this_best_idx] = e; // this_best_idx is found at e_{th} iteration in range(PTS_BUFFER)

            row[n_y * k + numk[n_y]] = n_y; // update allocation of this_best_idx to this_n_y
            col[n_y * k + numk[n_y]] = this_best_idx;
            numk[n_y] = numk[n_y] + 1;
          }
      } else {
        ;
      }
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
knn_cuda(const torch::Tensor x, const torch::Tensor y,
        torch::optional<torch::Tensor> ptr_x,
        torch::optional<torch::Tensor> ptr_y, const int64_t k,
        const bool cosine) {

  CHECK_CUDA(x);
  CHECK_CONTIGUOUS(x);
  CHECK_INPUT(x.dim() == 2);
  CHECK_CUDA(y);
  CHECK_CONTIGUOUS(y);
  CHECK_INPUT(y.dim() == 2);
  CHECK_INPUT(x.size(1) == y.size(1));
  AT_ASSERTM(k <= 200, "`k` needs to smaller than or equal to 200");

  if (ptr_x.has_value()) {
    CHECK_CUDA(ptr_x.value());
    CHECK_INPUT(ptr_x.value().dim() == 1);
  } else
    ptr_x = torch::arange(0, x.size(0) + 1, x.size(0),
                          x.options().dtype(torch::kLong));

  if (ptr_y.has_value()) {
    CHECK_CUDA(ptr_y.value());
    CHECK_INPUT(ptr_y.value().dim() == 1);
  } else
    ptr_y = torch::arange(0, y.size(0) + 1, y.size(0),
                          y.options().dtype(torch::kLong));

  CHECK_INPUT(ptr_x.value().numel() == ptr_y.value().numel());

  cudaSetDevice(x.get_device());
  auto best_idx = torch::empty(PTS_BUFFER * y.size(0), ptr_y.value().options());
  auto best_dist = torch::full(PTS_BUFFER * y.size(0), 5e4, ptr_y.value().options().dtype(torch::kFloat32));

  auto last_e = torch::full(x.size(0), -1, ptr_y.value().options());
  auto last_n_y = torch::full(x.size(0), -1, ptr_y.value().options());

  auto row = torch::empty(y.size(0) * k, ptr_y.value().options());
  auto col = torch::full(y.size(0) * k, -1, ptr_y.value().options());

  dim3 BLOCKS((y.size(0) + THREADS - 1) / THREADS);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_type = x.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
    knn_kernel<scalar_t><<<BLOCKS, THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        ptr_x.value().data_ptr<int64_t>(), ptr_y.value().data_ptr<int64_t>(),
        /*m=*/y.size(0), /*dim=*/x.size(1), best_idx.data_ptr<int64_t>(), best_dist.data_ptr<scalar_t>(),
        /*num_examples=*/ptr_x.value().numel() - 1, cosine);
  });

  auto numk_tensor = torch::full(y.size(0), 0, ptr_y.value().options());

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
    finalize_kernel<scalar_t><<<BLOCKS, THREADS, 0, stream>>>(
        row.data_ptr<int64_t>(), col.data_ptr<int64_t>(), last_n_y.data_ptr<int64_t>(), last_e.data_ptr<int64_t>(), k, numk_tensor.data_ptr<int64_t>(),
        ptr_y.value().data_ptr<int64_t>(), best_idx.data_ptr<int64_t>(), best_dist.data_ptr<scalar_t>(), /*num_examples=*/ptr_x.value().numel() - 1);
  });

  // for (int64_t n_y = 0; n_y < y.size(0); n_y++) {
  //   int64_t numk = 0;
  //   for (int64_t e = 0; e < 1024; e++) {
  //     if ( ptr_x_used[best_idx[e + 1024*n_y]].item().equal(0) ) {
  //       ptr_x_used[best_idx[e + 1024*n_y]] = ptr_x_used[best_idx[e + 1024*n_y]] + 1;
  //       row[n_y * k + numk] = n_y;
  //       col[n_y * k + numk] = best_idx[e + 1024*n_y];
  //       numk = numk + 1;
  //     }

  //     if (numk >= k) {
  //       break;
  //     }
  //   }
  // }

  auto mask = col != -1;

  return std::make_tuple(torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0), last_n_y);
}
