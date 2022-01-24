# pylint: disable=abstract-method
from typing import Any, cast, Dict, List, Tuple, Union
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import optim
from torchmetrics import Accuracy, MeanSquaredError
import natpn.distributions as D
from natpn.datasets import DataModule
from natpn.metrics import BrierScore, QuantileCalibrationScore
from natpn.nn import BayesianLoss, NaturalPosteriorEnsembleModel, NaturalPosteriorNetworkModel
from natpn.nn.output import CategoricalOutput

Batch = Tuple[torch.Tensor, torch.Tensor]


class NaturalPosteriorNetworkLightningModule(pl.LightningModule):
    """
    Lightning module for training and evaluating NatPN.
    """

    def __init__(
        self,
        model: Union[NaturalPosteriorNetworkModel, NaturalPosteriorEnsembleModel],
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = False,
        entropy_weight: float = 1e-5,
    ):
        """
        Args:
            model: The model to train or evaluate. If training, this *must* be a
                :class:`NaturalPosteriorNetworkModel`.
            learning_rate: The learning rate to use for the Adam optimizer.
            learning_rate_decay: Whether to use learning rate decay. If set to ``True``, the
                learning rate schedule is implemented using
                :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`.
            entropy_weight: The weight of the entropy regularizer in the Bayesian loss.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.loss = BayesianLoss(entropy_weight)

        if isinstance(model.output, CategoricalOutput):
            # We have discrete output
            self.output = "discrete"
            self.accuracy = Accuracy(compute_on_step=False, dist_sync_fn=self.all_gather)
            self.brier_score = BrierScore(compute_on_step=False, dist_sync_fn=self.all_gather)
        else:
            # We have continuous output
            self.output = "continuous"
            self.rmse = MeanSquaredError(
                squared=False, compute_on_step=False, dist_sync_fn=self.all_gather
            )
            self.calibration = QuantileCalibrationScore(
                compute_on_step=False, dist_sync_fn=self.all_gather
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        config: Dict[str, Any] = {"optimizer": optimizer}
        if self.learning_rate_decay:
            config["lr_scheduler"] = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.25,
                    patience=self.trainer.max_epochs // 20,
                    threshold=1e-3,
                    min_lr=1e-7,
                ),
                "monitor": "val/loss",
            }
        return config

    def configure_callbacks(self) -> List[pl.Callback]:
        return [
            EarlyStopping("val/loss", min_delta=1e-3, patience=self.trainer.max_epochs // 10),
        ]

    def training_step(self, batch: Batch, _batch_idx: int) -> torch.Tensor:
        X, y_true = batch
        y_pred, log_prob = self.model.forward(X)
        loss = self.loss.forward(y_pred, y_true)
        self.log("train/loss", loss)
        self.log("train/log_prob", log_prob.mean(), prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, _batch_idx: int) -> None:
        X, y_true = batch
        y_pred, log_prob = self.model.forward(X)
        loss = self.loss.forward(y_pred, y_true)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/log_prob", log_prob.mean(), prog_bar=True)
        self._compute_metrics("val", y_pred, y_true)

    def test_step(self, batch: Batch, _batch_idx: int) -> None:
        X, y_true = batch
        y_pred, _ = self.model.forward(X)
        self._compute_metrics("test", y_pred, y_true)

    def _compute_metrics(self, prefix: str, y_pred: D.Posterior, y_true: torch.Tensor) -> None:
        if self.output == "discrete":
            self.accuracy.update(y_pred.maximum_a_posteriori().mean(), y_true)
            self.log(f"{prefix}/accuracy", self.accuracy, prog_bar=True)

            probs = y_pred.maximum_a_posteriori().expected_sufficient_statistics()
            self.brier_score.update(probs, y_true)
            self.log(f"{prefix}/brier_score", self.brier_score, prog_bar=True)
        else:
            dm = cast(DataModule, self.trainer.datamodule)
            predicted = y_pred.maximum_a_posteriori().mean()
            self.rmse.update(dm.transform_output(predicted), dm.transform_output(y_true))
            self.log(f"{prefix}/rmse", self.rmse, prog_bar=True)

            confidence_levels = y_pred.posterior_predictive().symmetric_confidence_level(y_true)
            self.calibration.update(confidence_levels)
            self.log(f"{prefix}/calibration", self.calibration, prog_bar=True)
