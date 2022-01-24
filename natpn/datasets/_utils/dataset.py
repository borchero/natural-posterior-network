from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset


class TransformedDataset(Dataset[Any]):
    """
    Dataset that applies a transformation to its input and/or outputs.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        joint_transform: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    ):
        self.dataset = dataset
        self.transform = transform or _noop
        self.target_transform = target_transform or _noop
        self.joint_transform = joint_transform or _joint_noop

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int) -> Any:
        X, y = self.dataset[index]
        X_out, y_out = self.transform(X), self.target_transform(y)
        return self.joint_transform(X_out, y_out)


def _noop(x: Any) -> Any:
    return x


def _joint_noop(x: Any, y: Any) -> Tuple[Any, Any]:
    return x, y
