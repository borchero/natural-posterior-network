# pylint: disable=abstract-method
from typing import Any, Dict, List
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import optim
from natpn.nn import NaturalPosteriorNetworkModel
from .lightning_module import Batch


class NaturalPosteriorNetworkFlowLightningModule(pl.LightningModule):
    """
    Lightning module for optimizing the normalizing flow of NatPN.
    """

    def __init__(
        self,
        model: NaturalPosteriorNetworkModel,
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = False,
        early_stopping: bool = True,
    ):
        """
        Args:
            model: The model whose flow to optimize.
            learning_rate: The learning rate to use for the Adam optimizer.
            learning_rate_decay: Whether to use a learning rate decay. If set to ``True``, the
                learning rate schedule is implemented using
                :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`.
            early_stopping: Whether to use early stopping for training.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.early_stopping = early_stopping

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.Adam(self.model.flow.parameters(), lr=self.learning_rate)
        config: Dict[str, Any] = {"optimizer": optimizer}
        if self.learning_rate_decay:
            config["lr_scheduler"] = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.25,
                    patience=self.trainer.max_epochs // 20,
                    threshold=1e-3,
                    min_lr=1e-7,
                ),
                "monitor": "val/log_prob",
            }
        return config

    def configure_callbacks(self) -> List[pl.Callback]:
        if not self.early_stopping:
            return []
        return [
            EarlyStopping(
                "val/log_prob",
                min_delta=1e-2,
                mode="max",
                patience=self.trainer.max_epochs // 10,
            ),
        ]

    def training_step(self, batch: Batch, _batch_idx: int) -> torch.Tensor:
        X, _ = batch
        log_prob = self.model.log_prob(X, track_encoder_gradients=False).mean()
        self.log("train/log_prob", log_prob, prog_bar=True)
        return -log_prob

    def validation_step(self, batch: Batch, _batch_idx: int) -> None:
        X, _ = batch
        log_prob = self.model.log_prob(X).mean()
        self.log("val/log_prob", log_prob, prog_bar=True)
