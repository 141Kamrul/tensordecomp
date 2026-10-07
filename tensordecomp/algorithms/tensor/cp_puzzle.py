from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from ...function.puzzle_tensor import puzzle_tensor
from .cp import cp


def cp_puzzle(
    array: np.ndarray,
    rank: int | Sequence[int] | None = None,
    ranks: int | Sequence[int] | None = None,
    max_iter: int = 150,
    tol: float = 1e-7,
    puzzle_max_iter: int = 2,
    puzzle_max_shift: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """CP Decomposition augmented with PuzzleTensor hyperslice alignment.

    First aligns the hyperslices of the tensor to reduce effective multilinear rank,
    then performs CANDECOMP/PARAFAC (CP) decomposition on the aligned tensor.

    Args:
        array: Input tensor as a NumPy array (ndim >= 2).
        rank: Target CP decomposition rank (integer or list of mode ranks).
        ranks: Alternative alias for rank.
        max_iter: Maximum number of CP-ALS alternating least squares iterations.
        tol: Convergence tolerance for CP decomposition.
        puzzle_max_iter: Number of shift search iterations in PuzzleTensor.
        puzzle_max_shift: Maximum shift radius allowed along any mode.
        **kwargs: Additional keyword arguments passed to CP decomposition.

    Returns:
        dict containing CP decomposition outputs (method, weights, factors, shape, rank)
        along with the applied PuzzleTensor 'shifts' and 'is_puzzle': True.
    """
    shifted_tensor, shifts = puzzle_tensor(
        array,
        max_iter=puzzle_max_iter,
        max_shift=puzzle_max_shift,
        return_shifts=True,
    )

    result = cp(shifted_tensor, rank=rank, ranks=ranks, max_iter=max_iter, tol=tol, **kwargs)
    result["method"] = "cp_puzzle"
    result["shifts"] = shifts
    result["is_puzzle"] = True
    result["original_shape"] = list(np.asarray(array).shape)

    return result
