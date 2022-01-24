import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zipfile import ZipFile
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


class _BikeSharingDataModule(DataModule):
    """
    Data module for the Bike Sharing dataset.
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
    def input_size(self) -> torch.Size:
        return torch.Size([11])

    def prepare_data(self) -> None:
        logger.info("Preparing 'Bike Sharing'...")
        target = self.root / "bike-sharing"
        if not target.exists():
            logger.info(
                "'Bike Sharing' could not be found locally. Downloading to '%s'...", target
            )
            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
                "Bike-Sharing-Dataset.zip"
            )
            with tempfile.TemporaryDirectory() as tmp:
                download_url(url, tmp, "data.zip")
                with ZipFile(Path(tmp) / "data.zip") as zipfile:
                    zipfile.extractall(target)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.did_setup:
            # Read the hourly data from file
            df = pd.read_csv(self.root / "bike-sharing" / "hour.csv")

            # Read input and outputs and split by season
            season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
            data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for season_idx, df_season in df.groupby("season"):
                data[season_map[int(season_idx)]] = (
                    torch.from_numpy(df_season.iloc[:, 3:-3].to_numpy()).float(),
                    torch.from_numpy(df_season["cnt"].to_numpy()).float(),
                )

            # Split training data
            (X_train, X_test), (y_train, y_test) = tabular_train_test_split(
                *data["summer"], train_size=0.8, generator=self.generator,
            )
            (X_train, X_val), (y_train, y_val) = tabular_train_test_split(
                X_train, y_train, train_size=0.8, generator=self.generator,
            )

            # Normalize data
            input_scaler = StandardScaler().fit(X_train)
            self.output_scaler.fit(y_train)

            # Initialize datasets
            self.train_dataset = TensorDataset(
                input_scaler.transform(X_train), self.output_scaler.transform(y_train)
            )
            self.val_dataset = TensorDataset(
                input_scaler.transform(X_val), self.output_scaler.transform(y_val)
            )
            self.test_dataset = TensorDataset(
                input_scaler.transform(X_test), self.output_scaler.transform(y_test)
            )

            # And initialize OOD datasets
            self.ood_datasets = {
                season: tabular_ood_dataset(
                    input_scaler.transform(X_test), input_scaler.transform(data[season][0])
                )
                for season in ("spring", "fall", "winter")
            }
            self.ood_datasets["winter_oodom"] = tabular_ood_dataset(
                input_scaler.transform(X_test),
                input_scaler.transform(scale_oodom(data["winter"][0])),
            )

            # Mark done
            self.did_setup = True

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


@register("bike-sharing-poisson")
class BikeSharingPoissonDataModule(_BikeSharingDataModule):
    """
    Data module for Bike sharing dataset with Poisson likelihood.
    """

    @property
    def output_type(self) -> OutputType:
        return "poisson"


@register("bike-sharing-normal")
class BikeSharingNormalDataModule(_BikeSharingDataModule):
    """
    Data module for Bike sharing dataset with Normal likelihood.
    """

    def __init__(self, root: Optional[PathType] = None):
        """
        Args:
            root: The directory where the dataset can be found or where it should be downloaded to.
        """
        super().__init__(root)
        self.output_scaler = StandardScaler()

    @property
    def output_type(self) -> OutputType:
        return "normal"
