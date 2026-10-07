from __future__ import annotations

from typing import Any
import numpy as np

from ...function.puzzle_tensor import puzzle_tensor
from .tensor_train import tensor_train


def tensor_train_puzzle(
    array: np.ndarray,
    ranks: list[int] | None = None,
    max_rank: int | None = 4,
    tol: float | None = None,
    puzzle_max_iter: int = 2,
    puzzle_max_shift: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """Tensor-Train (TT) Decomposition augmented with PuzzleTensor hyperslice alignment.

    First aligns the hyperslices of the tensor to reduce effective rank,
    then executes sequential TT-SVD on the aligned tensor.

    Args:
        array: Input tensor as a NumPy array (ndim >= 2).
        ranks: Optional target TT-ranks [r_1, ..., r_{d-1}].
        max_rank: Maximum TT-rank bound.
        tol: Relative error threshold epsilon for singular value truncation.
        puzzle_max_iter: Number of shift search iterations in PuzzleTensor.
        puzzle_max_shift: Maximum shift radius allowed along any mode.
        **kwargs: Additional keyword arguments passed to TT-SVD.

    Returns:
        dict containing TT decomposition outputs (method, cores, ranks, singular_values, shape)
        along with the applied PuzzleTensor 'shifts' and 'is_puzzle': True.
    """
    shifted_tensor, shifts = puzzle_tensor(
        array,
        max_iter=puzzle_max_iter,
        max_shift=puzzle_max_shift,
        return_shifts=True,
    )

    result = tensor_train(
        shifted_tensor,
        ranks=ranks,
        max_rank=max_rank,
        tol=tol,
        **kwargs,
    )
    result["method"] = "tensor_train_puzzle"
    result["shifts"] = shifts
    result["is_puzzle"] = True
    result["original_shape"] = list(np.asarray(array).shape)

    return result
