from typing import Any, Tuple
import torch
from torch.utils.data import Dataset, TensorDataset


class OodDataset(Dataset[Tuple[Any, int]]):
    """
    Dataset of ood data.
    """

    def __init__(self, id_data: Dataset[Any], ood_data: Dataset[Any]):
        self.id_data = id_data
        self.ood_data = ood_data
        self.id_len = len(id_data)  # type: ignore
        self.ood_len = len(ood_data)  # type: ignore

    def __len__(self) -> int:
        return self.id_len + self.ood_len

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        if index < self.id_len:
            return self.id_data[index][0], 1
        return self.ood_data[index - self.id_len][0], 0


def tabular_ood_dataset(data_id: torch.Tensor, data_ood: torch.Tensor) -> TensorDataset:
    """
    Constructs a tensor dataset from the in-distribution and out-of-distribution tabular data.
    """
    X = torch.cat([data_id, data_ood])
    y = torch.cat(
        [
            torch.ones(data_id.size(0), dtype=torch.long),
            torch.zeros(data_ood.size(0), dtype=torch.long),
        ]
    )
    return TensorDataset(X, y)


def scale_oodom(x: torch.Tensor) -> torch.Tensor:
    """
    Scales the given input with a constant of 255 such that it can be considered out-of-domain.
    """
    return x * 255
