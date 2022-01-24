import logging
from typing import Any, Dict, Optional
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


class _UciDataModule(DataModule):
    def __init__(self, root: Optional[PathType] = None, seed: Optional[int] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
            seed: An optional seed which governs how train/test splits are created.
        """
        super().__init__(root, seed)
        self.did_setup = False
        self.did_setup_test = False

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    @property
    def output_type(self) -> OutputType:
        return "normal"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([8])

    def prepare_data(self) -> None:
        # Download energy dataset
        logger.info("Preparing 'Energy'...")
        target = self.root / "energy"
        if not target.exists():
            logger.info("'Energy' could not be found locally. Downloading to '%s'...", target)
            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
            )
            download_url(url, str(target), "data.xlsx")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test" and not self.did_setup_test:
            data = pd.read_excel(str(self.root / "energy" / "data.xlsx"))
            X = torch.from_numpy(data.to_numpy()[:, :8]).float()
            self.ood_datasets = {
                "energy": tabular_ood_dataset(
                    self.test_dataset.tensors[0], self.input_scaler.transform(X)
                ),
                "energy_oodom": tabular_ood_dataset(
                    self.test_dataset.tensors[0], self.input_scaler.transform(scale_oodom(X))
                ),
            }

            # Mark done
            self.did_setup_test = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=512, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=4096)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=4096)

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=4096)
            for name, dataset in self.ood_datasets.items()
        }


@register("concrete")
class ConcreteDataModule(_UciDataModule):
    """
    Data module for the Concrete dataset.
    """

    def prepare_data(self) -> None:
        # Download concrete dataset
        logger.info("Preparing 'Concrete'...")
        target = self.root / "concrete"
        if not target.exists():
            logger.info("'Concrete' could not be found locally. Downloading to '%s'...", target)
            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/"
                "Concrete_Data.xls"
            )
            download_url(url, str(target), "data.xls")

        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            df = pd.read_excel(str(self.root / "concrete" / "data.xls"))
            X = torch.from_numpy(df.to_numpy()[:, :-1]).float()
            y = torch.from_numpy(df.to_numpy()[:, -1]).float()

            # Split data
            (X_train, X_test), (y_train, y_test) = tabular_train_test_split(
                X, y, train_size=0.8, generator=self.generator
            )
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator
            )

            # Fit transforms
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)

            # Create datasets
            self.train_dataset = TensorDataset(
                self.input_scaler.transform(X_train), self.output_scaler.transform(y_train)
            )
            self.val_dataset = TensorDataset(
                self.input_scaler.transform(X_val), self.output_scaler.transform(y_val)
            )
            self.test_dataset = TensorDataset(
                self.input_scaler.transform(X_test), self.output_scaler.transform(y_test)
            )

            # Mark done
            self.did_setup = True

        super().setup(stage)
