# pylint: disable=abstract-method
from typing import Union
import pytorch_lightning as pl
from torchmetrics import AUROC
from natpn.metrics import AUCPR
from natpn.nn import NaturalPosteriorEnsembleModel, NaturalPosteriorNetworkModel
from .lightning_module import Batch


class NaturalPosteriorNetworkOodTestingLightningModule(pl.LightningModule):
    """
    Lightning module for evaluating the OOD detection performance of NatPN.
    """

    def __init__(
        self,
        model: Union[NaturalPosteriorNetworkModel, NaturalPosteriorEnsembleModel],
        logging_key: str,
    ):
        super().__init__()
        self.model = model
        self.logging_key = logging_key

        self.alea_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.alea_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

        self.epist_conf_pr = AUCPR(compute_on_step=False, dist_sync_fn=self.all_gather)
        self.epist_conf_roc = AUROC(compute_on_step=False, dist_sync_fn=self.all_gather)

    def test_step(self, batch: Batch, _batch_idx: int) -> None:
        X, y = batch
        posterior, log_prob = self.model.forward(X)

        # Aleatoric confidence (from negative uncertainty)
        aleatoric_conf = -posterior.maximum_a_posteriori().uncertainty()
        if aleatoric_conf.dim() > 1:
            aleatoric_conf = aleatoric_conf.mean(tuple(range(1, aleatoric_conf.dim())))

        self.alea_conf_pr.update(aleatoric_conf, y)
        self.log(f"{self.logging_key}/aleatoric_confidence_auc_pr", self.alea_conf_pr)

        self.alea_conf_roc.update(aleatoric_conf, y)
        self.log(f"{self.logging_key}/aleatoric_confidence_auc_roc", self.alea_conf_roc)

        # Epistemic confidence
        epistemic_conf = log_prob
        if epistemic_conf.dim() > 1:
            epistemic_conf = epistemic_conf.mean(tuple(range(1, epistemic_conf.dim())))

        self.epist_conf_pr.update(epistemic_conf, y)
        self.log(f"{self.logging_key}/epistemic_confidence_auc_pr", self.epist_conf_pr)

        self.epist_conf_roc.update(epistemic_conf, y)
        self.log(f"{self.logging_key}/epistemic_confidence_auc_roc", self.epist_conf_roc)
