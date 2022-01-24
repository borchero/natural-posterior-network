from typing import Optional
from ._base import NormalizingFlow
from .transforms import BatchNormTransform, MaskedAutoregressiveTransform


class MaskedAutoregressiveFlow(NormalizingFlow):
    """
    Normalizing flow that consists of masked autoregressive transforms with optional batch
    normalizing layers in between.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_hidden_layers: int = 1,
        hidden_layer_size: Optional[int] = None,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            dim: The input dimension of the normalizing flow.
            num_layers: The number of sequential masked autoregressive transforms.
            num_hidden_layers: The number of hidden layers for each autoregressive transform.
            hidden_layer_size_multiplier: The dimension of each hidden layer. Defaults to
                ``3 * dim + 1``.
            use_batch_norm: Whether to insert batch normalizing transforms between transforms.
        """
        transforms = []
        for i in range(num_layers):
            if i > 0 and use_batch_norm:
                transforms.append(BatchNormTransform(dim))
            transform = MaskedAutoregressiveTransform(
                dim,
                [hidden_layer_size or (dim * 3 + 1)] * num_hidden_layers,
            )
            transforms.append(transform)
        super().__init__(transforms)
