from typing import Any, cast
import torch
import torchmetrics


class QuantileCalibrationScore(torchmetrics.Metric):
    """
    Quantile calibration score using the 10-quantiles.
    """

    def __init__(self, compute_on_step: bool = True, dist_sync_fn: Any = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_fn=dist_sync_fn)

        for i in range(10, 100, 10):
            self.add_state(f"level_{i}_sum", torch.zeros(1), dist_reduce_fx="sum")
            self.add_state(f"level_{i}_count", torch.zeros(1), dist_reduce_fx="sum")

    def update(self, confidence_levels: torch.Tensor) -> None:
        for i in range(10, 100, 10):
            contained = ((1 - confidence_levels) <= (i / 100)).float()
            cast(torch.Tensor, getattr(self, f"level_{i}_sum")).add_(contained.sum())
            cast(torch.Tensor, getattr(self, f"level_{i}_count")).add_(contained.numel())

    def compute(self) -> torch.Tensor:
        q_sum = cast(torch.Tensor, getattr(self, "level_10_sum"))
        q_count = cast(torch.Tensor, getattr(self, "level_10_count"))
        squared_sum = (q_sum / q_count - 0.1) ** 2

        for i in range(20, 100, 10):
            q_sum = cast(torch.Tensor, getattr(self, f"level_{i}_sum"))
            q_count = cast(torch.Tensor, getattr(self, f"level_{i}_count"))
            squared_sum += (q_sum / q_count - i / 100) ** 2

        return (squared_sum / 9).sqrt()
