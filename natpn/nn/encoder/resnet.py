# pylint: disable=missing-function-docstring
import torch
import torchvision.models as M  # type: ignore
from torch import nn


class ResnetEncoder(nn.Module):
    """
    A ResNet-based encoding layer for images, mapping images into a one-dimensional latent space of
    the specified size.
    """

    def __init__(self, out_dim: int, *, dropout: float = 0.0):
        super().__init__()

        self.model = M.resnet18(pretrained=True)
        self.model.layer1 = nn.Sequential(self.model.layer1, nn.Dropout2d(dropout))
        self.model.layer2 = nn.Sequential(self.model.layer2, nn.Dropout2d(dropout))
        self.model.layer3 = nn.Sequential(self.model.layer3, nn.Dropout2d(dropout))
        self.model.layer4 = nn.Sequential(self.model.layer4, nn.Dropout2d(dropout))
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
