# pylint: disable=missing-function-docstring
import logging
import os
import tempfile
from pathlib import Path
from typing import cast, Dict, Optional
import click
import pytorch_lightning as pl
import torch
from lightkit.utils import PathType
from pytorch_lightning.loggers import WandbLogger
from wandb.wandb_run import Run
from natpn import NaturalPosteriorNetwork, suppress_pytorch_lightning_logs
from natpn.model import EncoderType, FlowType
from natpn.nn import CertaintyBudget
from .datasets import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@click.command()
# -------------------------------------------------------------------------------------------------
# REQUIRED
@click.option(
    "--dataset",
    type=click.Choice(list(DATASET_REGISTRY.keys())),
    required=True,
    help="The dataset to train and evaluate on.",
)
# -------------------------------------------------------------------------------------------------
# GLOBAL CONFIGURATION
@click.option(
    "--seed",
    type=int,
    default=None,
    show_default=True,
    help="A fixed seed to reproduced experiments.",
)
@click.option(
    "--data_path",
    type=click.Path(),
    default=Path.home() / "opt" / "data" / "natpn",
    help="The directory where input data is stored.",
)
@click.option(
    "--output_path",
    type=click.Path(),
    default=None,
    help="The local directory where the final model should be stored. Only uploaded "
    "or discarded if not provided.",
)
@click.option(
    "--experiment",
    type=str,
    default=None,
    help="If provided, tracks the run using Weights & Biases and uploads the trained model.",
)
# -------------------------------------------------------------------------------------------------
# MODEL ARCHITECTURE
@click.option(
    "--latent_dim",
    default=16,
    show_default=True,
    help="The dimension of the model's latent space.",
)
@click.option(
    "--flow_type",
    type=click.Choice(["radial", "maf"]),
    default="radial",
    show_default=True,
    help="The type of normalizing flow to use.",
)
@click.option(
    "--flow_layers",
    default=8,
    show_default=True,
    help="The number of sequential normalizing flow transforms to use.",
)
@click.option(
    "--certainty_budget",
    type=click.Choice(["constant", "exp-half", "exp", "normal"]),
    default="normal",
    show_default=True,
    help="The certainty budget to allocate in the latent space.",
)
@click.option(
    "--ensemble_size",
    type=int,
    default=None,
    help="The number of NatPN models to ensemble for NatPE. Disabled if set to None (default).",
)
# -------------------------------------------------------------------------------------------------
# TRAINING
@click.option(
    "--learning_rate",
    default=1e-3,
    show_default=True,
    help="The learning rate for the Adam optimizer for both training and fine-tuning.",
)
@click.option(
    "--use_learning_rate_decay",
    default=False,
    show_default=True,
    help="Whether to decay the learning rate if the validation loss plateaus for some time.",
)
@click.option(
    "--max_epochs",
    default=100,
    show_default=100,
    help="The maximum number of epochs to run both training and fine-tuning for.",
)
@click.option(
    "--entropy_weight",
    default=1e-5,
    show_default=True,
    help="The weight for the entropy regularizer.",
)
@click.option(
    "--warmup_epochs",
    default=3,
    show_default=True,
    help="The number of warm-up epochs to run prior to training.",
)
@click.option(
    "--run_finetuning",
    default=True,
    show_default=True,
    help="Whether to run fine-tuning after training.",
)
def main(
    dataset: str,
    seed: Optional[int],
    data_path: PathType,
    output_path: Optional[PathType],
    experiment: Optional[str],
    latent_dim: int,
    flow_type: FlowType,
    flow_layers: int,
    certainty_budget: CertaintyBudget,
    ensemble_size: Optional[int],
    learning_rate: float,
    use_learning_rate_decay: bool,
    max_epochs: int,
    entropy_weight: float,
    warmup_epochs: int,
    run_finetuning: bool,
):
    """
    Trains the Natural Posterior Network or an ensemble thereof on a single dataset and evaluates
    its performance.
    """
    logging.getLogger("natpn").setLevel(logging.INFO)
    suppress_pytorch_lightning_logs()

    # Fix randomness
    pl.seed_everything(seed)
    logger.info("Using seed %s.", os.getenv("PL_GLOBAL_SEED"))

    # Initialize logger if needed
    if experiment is not None:
        remote_logger = WandbLogger()
        cast(Run, remote_logger.experiment).config.update(
            {
                "seed": os.getenv("PL_GLOBAL_SEED"),
                "dataset": dataset,
                "latent_dim": latent_dim,
                "flow_type": flow_type,
                "flow_layers": flow_layers,
                "certainty_budget": certainty_budget,
                "ensemble_size": ensemble_size,
                "learning_rate": learning_rate,
                "use_learning_rate_decay": use_learning_rate_decay,
                "max_epochs": max_epochs,
                "entropy_weight": entropy_weight,
                "warmup_epochs": warmup_epochs,
                "run_finetuning": run_finetuning,
            }
        )
    else:
        remote_logger = None

    # Initialize data
    dm = DATASET_REGISTRY[dataset](data_path, seed=int(os.getenv("PL_GLOBAL_SEED") or 0))

    # Initialize estimator
    encoder_map: Dict[str, EncoderType] = {
        "concrete": "tabular",
        "sensorless-drive": "tabular",
        "bike-sharing-normal": "tabular",
        "bike-sharing-poisson": "tabular",
        "mnist": "image-shallow",
        "fashion-mnist": "image-shallow",
        "cifar10": "image-deep",
        "cifar100": "resnet",
        "nyu-depth-v2": "dense-depth",
    }
    cpu_datasets = {
        "concrete",
        "sensorless-drive",
        "bike-sharing-normal",
        "bike-sharing-poisson",
    }

    estimator = NaturalPosteriorNetwork(
        latent_dim=latent_dim,
        encoder=encoder_map[dataset],
        flow=flow_type,
        flow_num_layers=flow_layers,
        certainty_budget=certainty_budget,
        learning_rate=learning_rate,
        learning_rate_decay=use_learning_rate_decay,
        entropy_weight=entropy_weight,
        warmup_epochs=warmup_epochs,
        finetune=run_finetuning,
        ensemble_size=ensemble_size,
        trainer_params=dict(
            max_epochs=max_epochs,
            logger=remote_logger,
            gpus=int(torch.cuda.is_available() and dataset not in cpu_datasets),
        ),
    )

    # Run training
    estimator.fit(dm)

    # Evaluate model
    scores = estimator.score(dm)
    ood_scores = estimator.score_ood_detection(dm)

    # Print scores
    logger.info("Test scores:")
    for key, value in scores.items():
        logger.info("  %s: %.2f", key, value * 100 if key != "rmse" else value)

    logger.info("OOD detection scores:")
    for ood_dataset, metrics in ood_scores.items():
        logger.info("in-distribution vs. '%s'...", ood_dataset)
        for key, value in metrics.items():
            logger.info("  %s: %.2f", key, value * 100)

    # Save model if required
    if output_path is not None:
        estimator.save(output_path)
    if remote_logger is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            estimator.save(tmpdir)
            cast(Run, remote_logger.experiment).log_artifact(tmpdir, name=dataset, type="model")

    logger.info("Done 🎉")
