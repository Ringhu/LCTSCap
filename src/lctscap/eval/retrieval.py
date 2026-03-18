"""Retrieval metrics: similarity matrix, R@k, MRR, MedR for cross-modal retrieval."""

from typing import Dict, List

import torch
from torch import Tensor


def compute_similarity_matrix(ts_embeds: Tensor, text_embeds: Tensor) -> Tensor:
    """Compute cosine similarity matrix between time-series and text embeddings.

    Args:
        ts_embeds: time-series embeddings of shape [N, D].
        text_embeds: text embeddings of shape [N, D].

    Returns:
        Cosine similarity matrix of shape [N, N].
    """
    ts_norm = ts_embeds / ts_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    text_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return ts_norm @ text_norm.t()


def _retrieval_metrics_one_direction(
    sim_matrix: Tensor,
    ks: List[int],
    prefix: str,
) -> Dict[str, float]:
    """Compute R@k, MRR, MedR for one retrieval direction.

    The ground-truth assumption is that row i matches column i.

    Args:
        sim_matrix: [N_query, N_gallery] similarity matrix.
        ks: list of k values for Recall@k.
        prefix: metric name prefix, e.g. "t2s" or "s2t".

    Returns:
        Dict of metric_name -> value.
    """
    n = sim_matrix.size(0)
    # Ranks of the diagonal (ground truth) match for each query
    # Higher similarity = better, so rank is number of items with higher similarity + 1
    # Sort each row in descending order and find position of ground truth
    sorted_indices = sim_matrix.argsort(dim=-1, descending=True)

    gt_indices = torch.arange(n, device=sim_matrix.device).unsqueeze(1)
    ranks = (sorted_indices == gt_indices).nonzero(as_tuple=False)[:, 1] + 1  # 1-indexed ranks

    ranks_float = ranks.float()
    results: Dict[str, float] = {}

    for k in ks:
        recall_at_k = (ranks_float <= k).float().mean().item()
        results[f"{prefix}_R@{k}"] = recall_at_k

    results[f"{prefix}_MRR"] = (1.0 / ranks_float).mean().item()
    results[f"{prefix}_MedR"] = ranks_float.median().item()

    return results


def compute_retrieval_metrics(
    sim_matrix: Tensor,
    ks: List[int] | None = None,
) -> Dict[str, float]:
    """Compute retrieval metrics in both directions.

    Assumes the ground-truth pairing is sim_matrix[i, i] (diagonal).

    Directions:
      - t2s: text-to-series retrieval (rows = text queries, cols = series gallery)
      - s2t: series-to-text retrieval (rows = series queries, cols = text gallery)

    Args:
        sim_matrix: [N, N] cosine similarity matrix.
        ks: list of k values for R@k.  Default: [1, 5, 10].

    Returns:
        Dictionary containing t2s_R@k, s2t_R@k, t2s_MRR, s2t_MRR,
        t2s_MedR, s2t_MedR for each k.
    """
    if ks is None:
        ks = [1, 5, 10]

    results: Dict[str, float] = {}

    # text-to-series: each row queries across columns
    results.update(_retrieval_metrics_one_direction(sim_matrix, ks, prefix="t2s"))

    # series-to-text: transpose so series queries across text columns
    results.update(_retrieval_metrics_one_direction(sim_matrix.t(), ks, prefix="s2t"))

    return results
