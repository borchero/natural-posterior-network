import logging
import random
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, cast, Dict, Optional, Tuple
import pandas as pd
import torch
import torchvision.datasets as tvd  # type: ignore
import torchvision.transforms as T  # type: ignore
import torchvision.transforms.functional as F  # type: ignore
from lightkit.data import DataLoader
from lightkit.utils import PathType
from PIL import Image
from PIL.Image import Image as ImageType
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
from torchvision.datasets.utils import (  # type: ignore
    download_file_from_google_drive,
    download_url,
)
from ._base import DataModule, OutputType
from ._registry import register
from ._utils import dataset_train_test_split, OodDataset, scale_oodom, TransformedDataset

logger = logging.getLogger(__name__)


@register("nyu-depth-v2")
class NyuDepthV2DataModule(DataModule):
    """
    Data module for the NYU Depth v2 dataset.
    """

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
        return "normal"

    @property
    def input_size(self) -> torch.Size:
        return torch.Size([3, 640, 480])

    @property
    def gradient_accumulation_steps(self) -> int:
        return 2

    def prepare_data(self) -> None:
        # Download NYU Depth v2
        logger.info("Preparing 'NYU Depth v2'...")
        target = self.root / "nyu-depth-v2"
        if not target.exists():
            logger.info(
                "'NYU Depth v2' could not be found locally. Downloading to '%s'...", target
            )
            with tempfile.TemporaryDirectory() as tmp:
                download_file_from_google_drive(
                    "1fdFu5NGXe4rTLYKD5wOqk9dl-eJOefXo", tmp, "nyu_data.zip"
                )
                with zipfile.ZipFile(Path(tmp) / "nyu_data.zip") as f:
                    f.extractall(target)

        # Download LSUN for OOD
        logger.info("Preparing 'LSUN'...")
        target = self.root / "lsun"
        if not target.exists():
            logger.info("'LSUN' could not be found locally. Downloading to '%s'...", target)
            with tempfile.TemporaryDirectory() as tmp:
                for category in ["classroom", "church_outdoor"]:
                    url = f"http://dl.yf.io/lsun/scenes/{category}_train_lmdb.zip"
                    download_url(url, tmp, f"{category}.zip")
                    with zipfile.ZipFile(Path(tmp) / f"{category}.zip") as f:
                        f.extractall(target)

        # Download Kitti for OOD
        logger.info("Preparing 'Kitti'...")
        tvd.Kitti(str(self.root / "kitti"), train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        normalizer = T.Normalize(mean=[0.4874, 0.4176, 0.4007], std=[0.2833, 0.2909, 0.3031])

        if not self.did_setup:
            train_data = _NyuDepthV2(
                self.root / "nyu-depth-v2",
                train=True,
                transform=T.Compose([T.ToTensor(), normalizer]),
                target_transform=T.Compose(
                    [
                        T.Resize([240, 320]),
                        T.ToTensor(),
                        T.Lambda(_train_depth_transform),
                        T.Lambda(_reciprocal_depth_transform),
                    ]
                ),
            )
            train, val = dataset_train_test_split(
                train_data, train_size=0.8, generator=self.generator
            )
            self.train_dataset = TransformedDataset(
                train,
                transform=_RandomColorChannelPermutation(p=0.25),
                joint_transform=_JointRandomHorizontalFlip(p=0.5),
            )
            self.val_dataset = val

            self.did_setup = True

        if stage == "test" and not self.did_setup_ood:
            self.test_dataset = _NyuDepthV2(
                self.root / "nyu-depth-v2",
                train=False,
                transform=T.Compose([T.ToTensor(), normalizer]),
                target_transform=T.Compose(
                    [
                        T.Resize([240, 320]),
                        T.ToTensor(),
                        T.Lambda(_test_depth_transform),
                        T.Lambda(_reciprocal_depth_transform),
                    ]
                ),
            )
            self.ood_datasets = {
                "lsun_classroom": OodDataset(
                    self.test_dataset,
                    tvd.LSUN(
                        str(self.root / "lsun"),
                        classes=["classroom_train"],
                        transform=T.Compose([T.Resize([480, 640]), T.ToTensor(), normalizer]),
                    ),
                ),
                "lsun_church": OodDataset(
                    self.test_dataset,
                    tvd.LSUN(
                        str(self.root / "lsun"),
                        classes=["church_outdoor_train"],
                        transform=T.Compose([T.Resize([480, 640]), T.ToTensor(), normalizer]),
                    ),
                ),
                "kitti": OodDataset(
                    self.test_dataset,
                    tvd.Kitti(
                        str(self.root / "kitti"),
                        train=False,
                        transform=T.Compose([T.Resize([480, 640]), T.ToTensor(), normalizer]),
                    ),
                ),
                "kitti_oodom": OodDataset(
                    self.test_dataset,
                    tvd.Kitti(
                        str(self.root / "kitti"),
                        train=False,
                        transform=T.Compose(
                            [T.Resize([480, 640]), T.ToTensor(), normalizer, T.Lambda(scale_oodom)]
                        ),
                    ),
                ),
            }
            self.did_setup_ood = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=4, shuffle=True, num_workers=8, persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=16, num_workers=8, persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=16, num_workers=8, persistent_workers=True)

    def ood_dataloaders(self) -> Dict[str, DataLoader[Any]]:
        return {
            name: DataLoader(dataset, batch_size=16, num_workers=8, persistent_workers=True)
            for name, dataset in self.ood_datasets.items()
        }

    def transform_output(self, output: torch.Tensor) -> torch.Tensor:
        return _NyuDepthV2.DEPTH_MAX / (output * _NyuDepthV2.DEPTH_STD + _NyuDepthV2.DEPTH_MEAN)


class _NyuDepthV2(Dataset[Tuple[Any, Any]]):

    DEPTH_MAX = 1000
    DEPTH_MEAN = 4.4106
    DEPTH_STD = 2.2759

    def __init__(
        self,
        root: Path,
        train: bool = True,
        transform: Optional[Callable[[ImageType], Any]] = None,
        target_transform: Optional[Callable[[ImageType], Any]] = None,
    ):
        super().__init__()

        self.transform = transform or _noop
        self.target_transform = target_transform or _noop

        files = cast(
            pd.DataFrame,
            pd.read_csv(
                root / "data" / f"nyu2_{'train' if train else 'test'}.csv",
                header=None,  # type: ignore
            ),
        )

        self.root = root
        self.image_files = files.iloc[:, 0].tolist()
        self.depth_files = files.iloc[:, 1].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.root / self.image_files[index])
        depth = Image.open(self.root / self.depth_files[index])

        return self.transform(image), self.target_transform(depth)


class _RandomColorChannelPermutation:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return item[torch.randperm(item.size(0))]
        return item


class _JointRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, item: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            return F.hflip(item), F.hflip(target)
        return item, target


def _train_depth_transform(depth: torch.Tensor) -> torch.Tensor:
    return (depth * _NyuDepthV2.DEPTH_MAX).clamp(min=0, max=_NyuDepthV2.DEPTH_MAX).squeeze(0)


def _test_depth_transform(depth: torch.Tensor) -> torch.Tensor:
    return (depth / 10).clamp(min=0, max=_NyuDepthV2.DEPTH_MAX).squeeze(0)


def _reciprocal_depth_transform(depth: torch.Tensor) -> torch.Tensor:
    return ((_NyuDepthV2.DEPTH_MAX / depth) - _NyuDepthV2.DEPTH_MEAN) / _NyuDepthV2.DEPTH_STD


def _noop(x: Any) -> Any:
    return x
