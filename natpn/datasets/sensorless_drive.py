# pylint: disable=abstract-method
import logging
from typing import Any, cast, Dict, Optional
import pandas as pd
import torch
from lightkit.data import DataLoader
from lightkit.utils import PathType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset
from torchvision.datasets.utils import download_url  # type: ignore
from ._base import DataModule, OutputType
from ._registry import register
from ._utils import scale_oodom, StandardScaler, tabular_ood_dataset, tabular_train_test_split

logger = logging.getLogger(__name__)


@register("sensorless-drive")
class SensorlessDriveDataModule(DataModule):
    """
    Data module for the Sensorless Drive dataset.
    """

    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.did_setup = False

    @property
    def output_type(self) -> OutputType:
        return "categorical"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([48])

    @property
    def num_classes(self) -> int:
        return 9

    def prepare_data(self) -> None:
        logger.info("Preparing 'Sensorless Drive'...")
        data_file = self.root / "sensorless-drive" / "data.tsv"
        if not data_file.exists():
            logger.info(
                "'Sensorless Drive' could not be found locally. Downloading to '%s'...", data_file
            )
            data_file.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00325/"
                "Sensorless_drive_diagnosis.txt"
            )
            download_url(url, str(data_file.parent), data_file.name)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            # Read sensorless drive data. This can be done for any stage but only needs to be done
            # once.
            file = self.root / "sensorless-drive" / "data.tsv"
            data = cast(pd.DataFrame, pd.read_csv(file, sep=" ", header=None))  # type: ignore
            X_base = torch.from_numpy(data.to_numpy()[:, :-1]).float()
            y_base = torch.from_numpy(data.to_numpy()[:, -1]).long() - 1

            # Split by classes
            ood_mask = (y_base == 9) + (y_base == 10)
            X, X_ood = X_base[~ood_mask], X_base[ood_mask]
            y, _ = y_base[~ood_mask], y_base[ood_mask]

            # Split training data
            (X_train, X_test), (y_train, y_test) = tabular_train_test_split(
                X, y, train_size=0.8, generator=self.generator
            )
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )

            # Normalize inputs
            input_scaler = StandardScaler().fit(X_train)

            # Initialize datasets
            self.train_dataset = TensorDataset(input_scaler.transform(X_train), y_train)
            self.val_dataset = TensorDataset(input_scaler.transform(X_val), y_val)
            self.test_dataset = TensorDataset(input_scaler.transform(X_test), y_test)
            self.ood_datasets = {
                "sensorless_drive_left_out": tabular_ood_dataset(
                    input_scaler.transform(X_test), input_scaler.transform(X_ood)
                ),
                "sensorless_drive_oodom": tabular_ood_dataset(
                    input_scaler.transform(X_test), input_scaler.transform(scale_oodom(X_ood))
                ),
            }

            # Mark done
            self.did_setup = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=1024, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=4096)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=4096)

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=4096)
            for name, dataset in self.ood_datasets.items()
        }
