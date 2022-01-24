from typing import Any
import torch
import torchmetrics


class BrierScore(torchmetrics.Metric):
    """
    Brier score for Categorical predictions.
    """

    norm_sum: torch.Tensor
    norm_count: torch.Tensor

    def __init__(self, compute_on_step: bool = True, dist_sync_fn: Any = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_fn=dist_sync_fn)

        self.add_state("norm_sum", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("norm_count", torch.zeros(1), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        num_items = y_pred.size(0)

        prob = y_pred.clone()
        indices = torch.arange(num_items)
        prob[indices, y_true] -= 1
        norm = prob.norm(dim=-1)

        self.norm_sum.add_(norm.sum())
        self.norm_count.add_(num_items)

    def compute(self) -> torch.Tensor:
        return self.norm_sum / self.norm_count
