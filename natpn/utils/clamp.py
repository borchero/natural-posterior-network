import torch


def clamp_preserve_gradients(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """
    Clamps the values of the tensor into ``[lower, upper]`` but keeps the gradients.

    Args:
        x: The tensor whose values to constrain.
        lower: The lower limit for the values.
        upper: The upper limit for the values.

    Returns:
        The clamped tensor.
    """
    return x + (x.clamp(min=lower, max=upper) - x).detach()
