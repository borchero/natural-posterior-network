import logging
import sys
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torchvision.datasets as tvd  # type: ignore
import torchvision.transforms as T  # type: ignore
from lightkit.data import DataLoader
from lightkit.utils import PathType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from ._base import DataModule, OutputType
from ._registry import register
from ._utils import dataset_train_test_split, OodDataset, scale_oodom, TransformedDataset

logger = logging.getLogger(__name__)


class _CifarDataModule(DataModule, ABC):
    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.did_setup = False
        self.did_setup_ood = False

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 32, 32])

    @property
    @abstractmethod
    def _input_normalizer(self) -> T.Normalize:
        pass

    def prepare_data(self) -> None:
        logger.info("Preparing 'SVHN'...")
        tvd.SVHN(str(self.root / "svhn"), split="test", download=True)
        try:
            logger.info("Preparing 'CelebA'...")
            tvd.CelebA(str(self.root / "celeba"), split="test", download=True)
        except zipfile.BadZipFile:
            logger.error(
                "Downloading 'CelebA' failed due to download restrictions on Google Drive. "
                "Please download manually from https://drive.google.com/drive/folders/"
                "0B7EVK8r0v71pWEZsZE9oNnFzTm8 and put the files into '%s'.",
                self.root / "celeba",
            )
            sys.exit(1)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" and not self.did_setup_ood:
            self.ood_datasets = {
                "svhn": OodDataset(
                    self.test_dataset,
                    tvd.SVHN(
                        str(self.root / "svhn"),
                        split="test",
                        transform=T.Compose([T.ToTensor(), self._input_normalizer]),
                    ),
                ),
                "celeba": OodDataset(
                    self.test_dataset,
                    tvd.CelebA(
                        str(self.root / "celeba"),
                        split="test",
                        transform=T.Compose(
                            [T.Resize([32, 32]), T.ToTensor(), self._input_normalizer]
                        ),
                    ),
                ),
                "svhn_oodom": OodDataset(
                    self.test_dataset,
                    tvd.SVHN(
                        str(self.root / "svhn"),
                        split="test",
                        transform=T.Compose(
                            [T.ToTensor(), T.Lambda(scale_oodom), self._input_normalizer]
                        ),
                    ),
                ),
            }

            # Mark done
            self.did_setup_ood = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=1024,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=4096,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=4096,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=4096, num_workers=2, persistent_workers=True)
            for name, dataset in self.ood_datasets.items()
        }


@register("cifar10")
class Cifar10DataModule(_CifarDataModule):
    """
    Data module for the CIFAR-10 dataset.
    """

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def _input_normalizer(self) -> T.Normalize:
        return T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    def prepare_data(self) -> None:
        logger.info("Preparing 'CIFAR-10 Train'...")
        tvd.CIFAR10(str(self.root / "cifar10"), train=True, download=True)
        logger.info("Preparing 'CIFAR-10 Test'...")
        tvd.CIFAR10(str(self.root / "cifar10"), train=False, download=True)
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            train_data = tvd.CIFAR10(
                str(self.root / "cifar10"),
                train=True,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )
            train_dataset, val_dataset = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )

            self.train_dataset = TransformedDataset(
                train_dataset,
                transform=T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomAffine(15, translate=(0.1, 0.1)),
                    ]
                ),
            )
            self.val_dataset = val_dataset

            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.CIFAR10(
                str(self.root / "cifar10"),
                train=False,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )

        super().setup(stage=stage)


@register("cifar100")
class Cifar100DataModule(_CifarDataModule):
    """
    Data module for the CIFAR-100 dataset.
    """

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def _input_normalizer(self) -> T.Normalize:
        return T.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])

    def prepare_data(self) -> None:
        logger.info("Preparing 'CIFAR-100 Train'...")
        tvd.CIFAR100(str(self.root / "cifar100"), train=True, download=True)
        logger.info("Preparing 'CIFAR-100 Test'...")
        tvd.CIFAR100(str(self.root / "cifar100"), train=False, download=True)
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            train_data = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=True,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )
            train_dataset, val_dataset = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )

            self.train_dataset = TransformedDataset(
                train_dataset,
                transform=T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomRotation(20),
                        T.RandomAffine(15, translate=(0.1, 0.1)),
                    ]
                ),
            )
            self.val_dataset = val_dataset

            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = tvd.CIFAR100(
                str(self.root / "cifar100"),
                train=False,
                transform=T.Compose([T.ToTensor(), self._input_normalizer]),
            )

        super().setup(stage=stage)
