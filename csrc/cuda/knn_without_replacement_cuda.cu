#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 256

template <typename scalar_t>
__global__ void
knn_without_replacement_kernel(const scalar_t *__restrict__ sorted_distances, // array of size (batch_size * x_size * y_size)
                               const int64_t *__restrict__ sorted_inidices, // array of size (batch_size * x_size * y_size)
                               const int64_t k,
                               const int64_t batch_size,
                               const int64_t x_size,
                               const int64_t y_size,
                               scalar_t *__restrict__ topk_dist, // array of size (batch_size * y_size * k)
                               int64_t *__restrict__ topk_idx, // array of size (batch_size * y_size * k)
                               int64_t *__restrict__ current_k, // array of size (batch_size * y_size)
                               int64_t *__restrict__ inidices_record_x, // array of size (batch_size * x_size)
                               int64_t *__restrict__ inidices_record_y // array of size (batch_size * x_size)
                               ) {

  const int64_t batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index >= batch_size)
    return;

  const int64_t total_offset = batch_index * x_size * y_size;
  const int64_t top_k_offset = batch_index * y_size * k;
  const int64_t current_k_offset = batch_index * y_size;

  const int64_t inidices_record_offset = batch_index * x_size;

  for (int64_t n_x = 0; n_x < x_size; n_x++) {
    const int64_t x_offset = n_x*y_size;

    for (int64_t n_y = 0; n_y < y_size; n_y++) {
      const int64_t this_k = current_k[current_k_offset+n_y];
      const int64_t this_top_k_ptr = top_k_offset + n_y * k + this_k;

      if (this_k == k) {
        continue;
      }

      const int64_t this_ptr = total_offset + x_offset + n_y;
  
      const int64_t this_index = sorted_inidices[this_ptr];
      const scalar_t this_dist = sorted_distances[this_ptr];

      const int64_t previous_n_x = inidices_record_x[inidices_record_offset + this_index];

      if (previous_n_x == -1) { // this_index is initially used. :)
        inidices_record_x[inidices_record_offset + this_index] = n_x;
        inidices_record_y[inidices_record_offset + this_index] = n_y;
        current_k[current_k_offset+n_y] = current_k[current_k_offset+n_y] + 1;

        topk_idx[this_top_k_ptr] = this_index;
        topk_dist[this_top_k_ptr] = this_dist;
      }

      else if (previous_n_x == n_x) { // this_index was used, but exactly at this n_x. Let's fight!
        const int64_t previous_n_y = inidices_record_y[inidices_record_offset + this_index];
        const int64_t previous_k = current_k[current_k_offset+previous_n_y];
        const int64_t previous_top_k_index = top_k_offset + previous_n_y * k + previous_k;

        const scalar_t previous_dist = topk_dist[previous_top_k_index];

        if (previous_dist > this_dist) {
          current_k[current_k_offset+previous_n_y] = current_k[current_k_offset+previous_n_y] - 1;

          inidices_record_x[inidices_record_offset + this_index] = n_x;
          inidices_record_y[inidices_record_offset + this_index] = n_y;
          current_k[current_k_offset+n_y] = current_k[current_k_offset+n_y] + 1;
  
          topk_idx[this_top_k_ptr] = this_index;
          topk_dist[this_top_k_ptr] = this_dist;
        }
      }

      else { // this_index was already used by n_x in higher priority. sorry.
        continue;
      }
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
knn_without_replacement_cuda(const torch::Tensor sorted_distances,
                             const torch::Tensor sorted_indices,
                             const int64_t batch_size,
                             const int64_t x_size,
                             const int64_t y_size,
                             const int64_t k) {

  CHECK_CUDA(sorted_distances);
  CHECK_CONTIGUOUS(sorted_distances);
  CHECK_CUDA(sorted_indices);
  CHECK_CONTIGUOUS(sorted_indices);

  cudaSetDevice(sorted_distances.get_device());
  auto topk_dist = torch::empty(batch_size*y_size*k, sorted_indices.options().dtype(torch::kFloat32));
  auto topk_idx = torch::empty(batch_size*y_size*k, sorted_indices.options());
  auto current_k = torch::full(batch_size*y_size, 0, sorted_indices.options());
  auto inidices_record_x = torch::full(batch_size*x_size, -1, sorted_indices.options());
  auto inidices_record_y = torch::full(batch_size*x_size, -1, sorted_indices.options());

  dim3 BLOCKS((batch_size + THREADS - 1) / THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_type = topk_dist.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
    knn_without_replacement_kernel<scalar_t><<<BLOCKS, THREADS, 0, stream>>>(
      sorted_distances.data_ptr<scalar_t>(),
      sorted_indices.data_ptr<int64_t>(),
      k,
      batch_size,
      x_size,
      y_size,
      topk_dist.data_ptr<scalar_t>(),
      topk_idx.data_ptr<int64_t>(),
      current_k.data_ptr<int64_t>(),
      inidices_record_x.data_ptr<int64_t>(),
      inidices_record_y.data_ptr<int64_t>());
  });

  return std::make_tuple(topk_idx, topk_dist);
}
