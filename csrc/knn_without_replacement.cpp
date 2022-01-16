#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/knn_without_replacement_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__knn_without_replacement_cuda(void) { return NULL; }
#else
#endif
#endif

std::tuple<torch::Tensor, torch::Tensor>
knn_without_replacement(torch::Tensor sorted_distances,
                        torch::Tensor sorted_indices,
                        int64_t batch_size,
                        int64_t x_size,
                        int64_t y_size,
                        int64_t k) {
  if (sorted_distances.device().is_cuda()) {
#ifdef WITH_CUDA
    return knn_without_replacement_cuda(sorted_distances, sorted_indices, batch_size, x_size, y_size, k);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    AT_ERROR("Not supported on CPU");
    // if (cosine)
    //   AT_ERROR("`cosine` argument not supported on CPU");
    // return knn_cpu(x, y, ptr_x, ptr_y, k, num_workers);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::knn_without_replacement", &knn_without_replacement);