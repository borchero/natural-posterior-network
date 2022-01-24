from __future__ import annotations
import torch


class StandardScaler:
    """
    Standard scaler for tabular data.
    """

    mean_: torch.Tensor
    std_: torch.Tensor

    def fit(self, X: torch.Tensor) -> StandardScaler:
        """
        Fits the mean and std of this scaler via the provided tabular data.
        """
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transforms the provided tabular data with the mean and std that was fitted previously.
        """
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform of tabular data.
        """
        return X * self.std_ + self.mean_


class IdentityScaler:
    """
    A scaler which does nothing.
    """

    def fit(self, _X: torch.Tensor) -> IdentityScaler:
        """
        Noop.
        """
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Identity.
        """
        return X

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Identity.
        """
        return X
