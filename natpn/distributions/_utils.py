import math
from typing import List
import torch
from ._base import PosteriorUpdate


def mixture_posterior_update(updates: List[PosteriorUpdate]) -> PosteriorUpdate:
    """
    Computes the posterior update from a mixture of updates.

    Args:
        updates: The posterior updates to join into a mixture.

    Returns:
        The joint posterior update.
    """
    stacked_sufficient_statistics = torch.stack([u.sufficient_statistics for u in updates])
    stacked_log_evidences = torch.stack([u.log_evidence for u in updates])

    # Compute the weighted sufficient statistics
    sufficient_statistics_dims = stacked_sufficient_statistics.dim() - stacked_log_evidences.dim()
    weights = stacked_log_evidences.softmax(0)
    weights = weights.view(weights.size() + (1,) * sufficient_statistics_dims)

    sufficient_statistics = (stacked_sufficient_statistics * weights).sum(0)

    # Compute the average log evidence
    num_mixtures = stacked_log_evidences.size(0)
    log_evidence = stacked_log_evidences.logsumexp(0) - math.log(num_mixtures)

    return PosteriorUpdate(sufficient_statistics, log_evidence)
