# pylint: disable=abstract-method
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import pytorch_lightning as pl
import torch
from lightkit.utils import PathType
from torch.utils.data import DataLoader, Dataset
from ._utils import IdentityScaler

DEFAULT_ROOT = Path.home() / "opt/data/natpn"

OutputType = Literal["categorical", "normal", "poisson"]


class DataModule(pl.LightningDataModule, ABC):
    """
    Simple extension of PyTorch Lightning's data module which provides the input dimension and
    further details.
    """

    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any]
    test_dataset: Dataset[Any]

    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        super().__init__()
        self.root = Path(root or DEFAULT_ROOT)
        self.output_scaler = IdentityScaler()
        self.ood_datasets: Dict[str, Dataset[Any]] = {}
        self.generator = torch.Generator()
        if seed is not None:
            self.generator = self.generator.manual_seed(seed)

    @property
    @abstractmethod
    def output_type(self) -> OutputType:
        """
        Returns the likelihood distribution for the outputs of this data module.
        """

    @property
    @abstractmethod
    def input_size(self) -> torch.Size:
        """
        Returns the size of the data items yielded by the data module.
        """

    @property
    def num_classes(self) -> int:
        """
        Returns the number of classes if the data module yields training data with categorical
        outputs.
        """
        raise NotImplementedError

    @property
    def gradient_accumulation_steps(self) -> int:
        """
        Returns the number of batches from which to accumulate the gradients for training.
        """
        return 1

    def transform_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Transforms a model's output such that the values are in the space of the true targets.
        This is, for example, useful when targets have been transformed for training.

        Args:
            output: The model output to transform.

        Returns:
            The transformed output.
        """
        return self.output_scaler.inverse_transform(output)

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        """
        Returns a set of dataloaders that can be used to measure out-of-distribution detection
        performance.

        Returns:
            A mapping from out-of-distribution dataset names to data loaders.
        """
        return {}
