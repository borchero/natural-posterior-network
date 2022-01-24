from typing import List, Tuple, TypeVar
import torch
from torch.utils.data import Dataset, Subset

T = TypeVar("T")


def tabular_train_test_split(
    *tensors: torch.Tensor,
    train_size: float,
    generator: torch.Generator,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Splits the given tensors randomly into training and test tensors. Each tensor is split with
    the same indices.

    Args:
        tensors: The tensors to split. Must all have the same number of elements in the first
            dimension.
        train_size: The fraction in ``(0, 1)`` to use for the training data.
        generator: The generator to use for generating train/test splits.

    Returns:
        The tensors split into training and test tensors.
    """
    num_items = tensors[0].size(0)
    num_train = round(num_items * train_size)
    permutation = torch.randperm(num_items, generator=generator)
    return [(t[permutation[:num_train]], t[permutation[num_train:]]) for t in tensors]


def dataset_train_test_split(
    dataset: Dataset[T], train_size: float, generator: torch.Generator,
) -> Tuple[Dataset[T], Dataset[T]]:
    """
    Splits the given dataset randomly into training and test items.

    Args:
        dataset: The dataset to split.
        train_size: The fraction in ``(0, 1)`` to use for the training data.
        generator: The generator to use for generating train/test splits.

    Returns:
        The train and test dataset.
    """
    num_items = len(dataset)  # type: ignore
    num_train = round(num_items * train_size)
    permutation = torch.randperm(num_items, generator=generator)
    return (
        Subset(dataset, permutation[:num_train].tolist()),
        Subset(dataset, permutation[num_train:].tolist()),
    )
