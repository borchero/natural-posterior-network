from __future__ import annotations
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, cast, Dict, List, Literal, Optional, Union
import torch
from lightkit import BaseEstimator
from pytorch_lightning.callbacks import ModelCheckpoint
from natpn.datasets import DataModule, OutputType
from natpn.nn import CertaintyBudget, NaturalPosteriorEnsembleModel, NaturalPosteriorNetworkModel
from natpn.nn.encoder import (
    DeepImageEncoder,
    DenseDepthEncoder,
    ResnetEncoder,
    ShallowImageEncoder,
    TabularEncoder,
)
from natpn.nn.flow import MaskedAutoregressiveFlow, RadialFlow
from natpn.nn.output import CategoricalOutput, NormalOutput, PoissonOutput
from .lightning_module import NaturalPosteriorNetworkLightningModule
from .lightning_module_flow import NaturalPosteriorNetworkFlowLightningModule
from .lightning_module_ood import NaturalPosteriorNetworkOodTestingLightningModule

logger = logging.getLogger(__name__)

FlowType = Literal["radial", "maf"]
"""
A reference to a flow type that can be used with :class:`NaturalPosteriorNetwork`:

- `radial`: A :class:`~natpn.nn.flow.RadialFlow`.
- `maf`: A :class:`~natpn.nn.flow.MaskedAutoregressiveFlow`.
"""

EncoderType = Literal["tabular", "image-shallow", "image-deep", "resnet", "dense-depth"]
"""
A reference to an encoder class that can be used with :class:`NaturalPosteriorNetwork`:

- `tabular`: A :class:`~natpn.nn.encoder.TabularEncoder`.
- `image-shallow`: A :class:`~natpn.nn.encoder.ShallowImageEncoder`.
- `image-deep`: A :class:`~natpn.nn.encoder.DeepImageEncoder`.
- `resnet`: A :class:`~natpn.nn.encoder.ResnetEncoder`.
- `dense-depth`: A :class:`~natpn.nn.encoder.DenseDepthEncoder`.
"""


class NaturalPosteriorNetwork(BaseEstimator):
    """
    Estimator for the Natural Posterior Network and the Natural Posterior Ensemble.
    """

    #: The fitted model.
    model_: Union[NaturalPosteriorNetworkModel, NaturalPosteriorEnsembleModel]
    #: The input size of the model.
    input_size_: torch.Size
    #: The output type of the model.
    output_type_: OutputType
    #: The number of classes the model predicts if ``output_type_ == "categorical"``.
    num_classes_: Optional[int]

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        encoder: EncoderType = "tabular",
        flow: FlowType = "radial",
        flow_num_layers: int = 8,
        certainty_budget: CertaintyBudget = "normal",
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = False,
        entropy_weight: float = 1e-5,
        warmup_epochs: int = 3,
        finetune: bool = True,
        ensemble_size: Optional[int] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            latent_dim: The dimension of the latent space that the encoder should map to.
            encoder: The type of encoder to use which maps the input to the latent space.
            flow: The type of flow which produces log-probabilities from the latent
                representations.
            flow_num_layers: The number of layers to use for the flow. If ``flow`` is set to
                ``"maf"``, this sets the number of masked autoregressive layers. In between each
                of these layers, another batch normalization layer is added.
            certainty_budget: The certainty budget to use to scale the log-probabilities produced
                by the normalizing flow.
            dropout: The dropout probability to use for dropout layers in the encoder.
            learning_rate: The learning rate to use for training encoder, flow, and linear output
                layer. Applies to warm-up, actual training, and fine-tuning.
            learning_rate_decay: Whether to use a learning rate decay by reducing the learning rate
                when the validation loss plateaus.
            entropy_weight: The strength of the entropy regularizer for the Bayesian loss used for
                the main training procedure.
            warmup_epochs: The number of epochs to run warm-up for. Should be used if the latent
                space is high-dimensional and/or the normalizing flow is complex, i.e. consists of
                many layers.
            finetune: Whether to run fine-tuning after the main training loop. May be set to
                ``False`` to speed up the overall training time if the data is simple. Otherwise,
                it should be kept as ``True`` to improve out-of-distribution detection.
            ensemble_size: The number of NatPN models to ensemble for the final predictions. This
                constructs a Natural Posterior Ensemble which trains multiple NatPN models
                independently and combines their predictions via Bayesian combination. By default,
                this is set to ``None`` which does not create a NatPE.
            trainer_params: Additional parameters which are passed to the PyTorch Ligthning
                trainer. These parameters apply to all fitting runs as well as testing.
        """
        super().__init__(
            user_params=trainer_params,
            overwrite_params=dict(
                log_every_n_steps=1,
                enable_checkpointing=True,
                enable_progress_bar=True,
            ),
        )

        self.latent_dim = latent_dim
        self.encoder = encoder
        self.flow = flow
        self.flow_num_layers = flow_num_layers
        self.certainty_budget: CertaintyBudget = certainty_budget
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.entropy_weight = entropy_weight
        self.warmup_epochs = warmup_epochs
        self.finetune = finetune
        self.ensemble_size = ensemble_size

    # ---------------------------------------------------------------------------------------------
    # RUNNING THE MODEL

    def fit(self, data: DataModule) -> NaturalPosteriorNetwork:
        """
        Fits the Natural Posterior Network with the provided data. Fitting sequentially runs
        warm-up (if ``self.warmup_epochs > 0``), the main training loop, and fine-tuning (if
        ``self.finetune == True``).

        Args:
            data: The data to fit the model with.

        Returns:
            The estimator whose ``model_`` property is set.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            if self.ensemble_size is None:
                model = self._init_model(
                    data.output_type,
                    data.input_size,
                    data.num_classes if data.output_type == "categorical" else 0,
                )
                self.model_ = self._fit_model(model, data, Path(tmp_dir))
            else:
                models = []
                for i in range(self.ensemble_size):
                    logger.info("Fitting model %d/%d...", i + 1, self.ensemble_size)
                    model = self._init_model(
                        data.output_type,
                        data.input_size,
                        data.num_classes if data.output_type == "categorical" else 0,
                    )
                    models.append(self._fit_model(model, data, Path(tmp_dir)))
                self.model_ = NaturalPosteriorEnsembleModel(models)

        # Assign additional fitted attributes
        self.input_size_ = data.input_size
        self.output_type_ = data.output_type
        try:
            self.num_classes_ = data.num_classes
        except NotImplementedError:
            self.num_classes_ = None

        # Return self
        return self

    def score(self, data: DataModule) -> Dict[str, float]:
        """
        Measures the model performance on the given data.

        Args:
            data: The data for which to measure the model performance.

        Returns:
            A dictionary mapping metrics to their values. This dictionary includes a measure of
            accuracy (`"accuracy"` for classification and `"rmse"` for regression) and a
            calibration measure (`"brier_score"` for classification and `"calibration"` for
            regression).
        """
        logger.info("Evaluating on test set...")
        module = NaturalPosteriorNetworkLightningModule(self.model_)
        out = self.trainer().test(module, data, verbose=False)
        return {k.split("/")[1]: v for k, v in out[0].items()}

    def score_ood_detection(self, data: DataModule) -> Dict[str, Dict[str, float]]:
        """
        Measures the model's ability to detect out-of-distribution data.

        Args:
            data: The data module which provides one or more datasets that contain test data along
                with out-of-distribution data.

        Returns:
            A nested dictionary which provides for multiple out-of-distribution datasets (first
            key) multiple metrics for measuring epistemic and aleatoric uncertainty.
        """
        results = {}
        for dataset, loader in data.ood_dataloaders().items():
            logger.info("Evaluating in-distribution vs. %s...", dataset)
            module = NaturalPosteriorNetworkOodTestingLightningModule(
                self.model_, logging_key=f"ood/{dataset}"
            )
            result = self.trainer().test(module, loader, verbose=False)
            results[dataset] = {k.split("/")[2]: v for k, v in result[0].items()}

        return results

    # ---------------------------------------------------------------------------------------------
    # PERSISTENCE

    @property
    def persistent_attributes(self) -> List[str]:
        return [k for k in self.__annotations__ if k != "model_"]

    def save_parameters(self, path: Path) -> None:
        params = {
            k: (
                v
                if k != "trainer_params"
                else {kk: vv for kk, vv in cast(Dict[str, Any], v).items() if kk != "logger"}
            )
            for k, v in self.get_params().items()
        }
        data = json.dumps(params, indent=4)
        with (path / "params.json").open("w+") as f:
            f.write(data)

    def save_attributes(self, path: Path) -> None:
        super().save_attributes(path)
        torch.save(self.model_.state_dict(), path / "parameters.pt")

    def load_attributes(self, path: Path) -> None:
        super().load_attributes(path)
        parameters = torch.load(path / "parameters.pt")
        if self.ensemble_size is None:
            model = self._init_model(self.output_type_, self.input_size_, self.num_classes_ or 0)
            model.load_state_dict(parameters)
            self.model_ = model
        else:
            model = NaturalPosteriorEnsembleModel(
                [
                    self._init_model(self.output_type_, self.input_size_, self.num_classes_ or 0)
                    for _ in range(self.ensemble_size)
                ]
            )
            model.load_state_dict(parameters)
            self.model_ = model

    # ---------------------------------------------------------------------------------------------
    # UTILS

    def _fit_model(
        self, model: NaturalPosteriorNetworkModel, data: DataModule, tmp_dir: Path
    ) -> NaturalPosteriorNetworkModel:
        level = logging.getLogger("pytorch_lightning").getEffectiveLevel()

        # Run warmup
        if self.warmup_epochs > 0:
            warmup_module = NaturalPosteriorNetworkFlowLightningModule(
                model, learning_rate=self.learning_rate, early_stopping=False
            )

            # Get trainer and print information
            logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
            trainer = self.trainer(
                accumulate_grad_batches=data.gradient_accumulation_steps,
                enable_checkpointing=False,
                enable_model_summary=True,
                max_epochs=self.warmup_epochs,
            )
            logging.getLogger("pytorch_lightning").setLevel(level)

            logger.info("Running warmup...")
            trainer.fit(warmup_module, data)

        # Run training
        trainer_checkpoint = ModelCheckpoint(tmp_dir / "training", monitor="val/loss", mode='min')

        logging.getLogger("pytorch_lightning").setLevel(
            logging.INFO if self.warmup_epochs == 0 else level
        )
        trainer = self.trainer(
            accumulate_grad_batches=data.gradient_accumulation_steps,
            callbacks=[trainer_checkpoint],
            enable_model_summary=self.warmup_epochs == 0,
        )
        logging.getLogger("pytorch_lightning").setLevel(level)

        logger.info("Running training...")
        train_module = NaturalPosteriorNetworkLightningModule(
            model,
            learning_rate=self.learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            entropy_weight=self.entropy_weight,
        )
        trainer.fit(train_module, data)

        best_module = NaturalPosteriorNetworkLightningModule.load_from_checkpoint(
            trainer_checkpoint.best_model_path
        )

        # Run fine-tuning
        if self.finetune:
            finetune_checkpoint = ModelCheckpoint(tmp_dir / "finetuning", monitor="val/log_prob", mode='max')
            trainer = self.trainer(
                accumulate_grad_batches=data.gradient_accumulation_steps,
                callbacks=[finetune_checkpoint],
            )

            logger.info("Running fine-tuning...")
            finetune_module = NaturalPosteriorNetworkFlowLightningModule(
                cast(NaturalPosteriorNetworkModel, best_module.model),
                learning_rate=self.learning_rate,
                learning_rate_decay=self.learning_rate_decay,
            )
            trainer.fit(finetune_module, data)

            # Return model
            return NaturalPosteriorNetworkFlowLightningModule.load_from_checkpoint(
                finetune_checkpoint.best_model_path
            ).model
        return cast(NaturalPosteriorNetworkModel, best_module.model)

    def _init_model(
        self, output_type: OutputType, input_size: torch.Size, num_classes: int
    ) -> NaturalPosteriorNetworkModel:
        # Initialize encoder
        if self.encoder == "tabular":
            assert len(input_size) == 1, "Tabular encoder only allows for one-dimensional inputs."
            encoder = TabularEncoder(
                input_size[0], [64] * 3, self.latent_dim, dropout=self.dropout
            )
        elif self.encoder == "image-shallow":
            encoder = ShallowImageEncoder(input_size, self.latent_dim, dropout=self.dropout)
        elif self.encoder == "image-deep":
            encoder = DeepImageEncoder(input_size, self.latent_dim, dropout=self.dropout)
        elif self.encoder == "resnet":
            assert len(input_size) == 3, "Resnet encoder requires three-dimensional inputs."
            encoder = ResnetEncoder(self.latent_dim, dropout=self.dropout)
        elif self.encoder == "dense-depth":
            assert input_size == torch.Size(
                [3, 640, 480]
            ), "DenseDepth encoder requires input of shape [3, 640, 480]."
            encoder = DenseDepthEncoder(self.latent_dim, dropout=self.dropout)
        else:
            raise NotImplementedError

        # Initialize flow
        if self.flow == "radial":
            flow = RadialFlow(self.latent_dim, num_layers=self.flow_num_layers)
        elif self.flow == "maf":
            flow = MaskedAutoregressiveFlow(self.latent_dim, num_layers=self.flow_num_layers)
        else:
            raise NotImplementedError

        # Initialize output
        if output_type == "categorical":
            output = CategoricalOutput(self.latent_dim, num_classes)
        elif output_type == "normal":
            output = NormalOutput(self.latent_dim)
        elif output_type == "poisson":
            output = PoissonOutput(self.latent_dim)
        else:
            raise NotImplementedError

        return NaturalPosteriorNetworkModel(
            self.latent_dim,
            encoder=encoder,
            flow=flow,
            output=output,
            certainty_budget=self.certainty_budget,
        )
