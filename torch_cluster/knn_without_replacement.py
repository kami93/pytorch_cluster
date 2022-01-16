from typing import Optional, Tuple

import torch


@torch.jit.script
def knn_without_replacement(sorted_distances: torch.Tensor,
                            sorted_indices: torch.Tensor,
                            batch_size: int,
                            x_size: int,
                            y_size: int,
                            k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""AA
    """
    topk_idx, topk_dist = torch.ops.torch_cluster.knn_without_replacement(sorted_distances, sorted_indices, batch_size, x_size, y_size, k)

    topk_idx = topk_idx.reshape(batch_size, y_size, k)
    topk_dist = topk_dist.reshape(batch_size, y_size, k)

    return topk_idx, topk_dist