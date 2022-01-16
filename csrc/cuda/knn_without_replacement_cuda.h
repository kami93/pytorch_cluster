#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor>
knn_without_replacement_cuda(torch::Tensor sorted_distances,
                             torch::Tensor sorted_indices,
                             int64_t batch_size,
                             int64_t x_size,
                             int64_t y_size,
                             int64_t k);