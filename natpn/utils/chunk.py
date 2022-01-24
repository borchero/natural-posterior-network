from typing import List
import torch


def chunk_squeeze_last(x: torch.Tensor) -> List[torch.Tensor]:
    """
    Splits the provided tensor into individual elements along the last dimension and returns the
    items with the last dimension squeezed.

    Args:
        x: The tensor to chunk.

    Returns:
        The squeezed chunks.
    """
    chunks = x.chunk(x.size(-1), dim=-1)
    return [c.squeeze(-1) for c in chunks]
