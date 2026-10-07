from __future__ import annotations

from typing import Any
import numpy as np

from ...function.puzzle_tensor import puzzle_tensor
from .tucker import tucker


def tucker_puzzle(
    array: np.ndarray,
    ranks: list[int] | None = None,
    max_iter: int = 100,
    tol: float = 1e-7,
    puzzle_max_iter: int = 2,
    puzzle_max_shift: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """Tucker Decomposition augmented with PuzzleTensor hyperslice alignment.

    First aligns the hyperslices of the tensor to reduce effective multilinear rank,
    then performs Tucker decomposition (Higher-Order Orthogonal Iteration) on the aligned tensor.

    Args:
        array: Input tensor as a NumPy array (ndim >= 2).
        ranks: Target ranks along each mode.
        max_iter: Maximum number of HOOI iterations.
        tol: Convergence tolerance for Tucker decomposition.
        puzzle_max_iter: Number of shift search iterations in PuzzleTensor.
        puzzle_max_shift: Maximum shift radius allowed along any mode.
        **kwargs: Additional keyword arguments passed to Tucker decomposition.

    Returns:
        dict containing Tucker decomposition outputs (method, core, factors, shape, ranks)
        along with the applied PuzzleTensor 'shifts' and 'is_puzzle': True.
    """
    shifted_tensor, shifts = puzzle_tensor(
        array,
        max_iter=puzzle_max_iter,
        max_shift=puzzle_max_shift,
        return_shifts=True,
    )

    result = tucker(shifted_tensor, ranks=ranks, max_iter=max_iter, tol=tol, **kwargs)
    result["method"] = "tucker_puzzle"
    result["shifts"] = shifts
    result["is_puzzle"] = True
    result["original_shape"] = list(np.asarray(array).shape)

    return result
