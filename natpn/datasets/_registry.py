from typing import Callable, Dict, Type, TypeVar
from ._base import DataModule

DATASET_REGISTRY: Dict[str, Type[DataModule]] = {}
T = TypeVar("T", bound=Type[DataModule])


def register(name: str) -> Callable[[T], T]:
    """
    Registers the provided module in the global registry under the specified name.
    """

    def register_module(module: T) -> T:
        DATASET_REGISTRY[name] = module
        return module

    return register_module
