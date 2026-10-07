from __future__ import annotations

from typing import Any
import numpy as np

from ...function.puzzle_tensor import puzzle_tensor
from .hosvd import hosvd


def hosvd_puzzle(
    array: np.ndarray,
    ranks: list[int] | None = None,
    puzzle_max_iter: int = 2,
    puzzle_max_shift: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """Higher-Order Singular Value Decomposition augmented with PuzzleTensor hyperslice alignment.

    First aligns the hyperslices of the tensor to minimize nuclear norm matricization loss,
    then performs multilinear SVD (HOSVD) on the aligned tensor.

    Args:
        array: Input tensor as a NumPy array (ndim >= 2).
        ranks: Optional target ranks for truncated HOSVD along each mode.
        puzzle_max_iter: Number of shift search iterations in PuzzleTensor.
        puzzle_max_shift: Maximum shift radius allowed along any mode.
        **kwargs: Additional keyword arguments passed to HOSVD.

    Returns:
        dict containing HOSVD decomposition outputs (method, core, factors, singular_values, ranks, shape)
        along with the applied PuzzleTensor 'shifts' and 'is_puzzle': True.
    """
    shifted_tensor, shifts = puzzle_tensor(
        array,
        max_iter=puzzle_max_iter,
        max_shift=puzzle_max_shift,
        return_shifts=True,
    )

    result = hosvd(shifted_tensor, ranks=ranks, **kwargs)
    result["method"] = "hosvd_puzzle"
    result["shifts"] = shifts
    result["is_puzzle"] = True
    result["original_shape"] = list(np.asarray(array).shape)

    return result
