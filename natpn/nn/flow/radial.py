from ._base import NormalizingFlow
from .transforms import RadialTransform


class RadialFlow(NormalizingFlow):
    """
    Normalizing flow that consists purely of a series of radial transforms.
    """

    def __init__(self, dim: int, num_layers: int = 8):
        """
        Args:
            dim: The input dimension of the normalizing flow.
            num_layers: The number of sequential radial transforms.
        """
        transforms = [RadialTransform(dim) for _ in range(num_layers)]
        super().__init__(transforms)
