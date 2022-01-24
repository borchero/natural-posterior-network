import logging
import warnings
from .model import NaturalPosteriorNetwork

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False


def suppress_pytorch_lightning_logs():
    """
    Suppresses annoying PyTorch Lightning logs.
    """
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")
    warnings.filterwarnings("ignore", ".*this may lead to large memory footprint.*")
    warnings.filterwarnings("ignore", ".*DataModule.setup has already been called.*")
    warnings.filterwarnings("ignore", ".*DataModule.teardown has already been called.*")
    warnings.filterwarnings("ignore", ".*Set the gpus flag in your trainer.*")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


__all__ = ["NaturalPosteriorNetwork", "suppress_pytorch_lightning_logs"]
