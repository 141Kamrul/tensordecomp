from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...function.tensor_utils import as_float_tensor, matricization, mode_n_product
from ..matrix.svd import svd


def hosvd(
    array: np.ndarray,
    ranks: int | Sequence[int] | None = None,
    rank: int | Sequence[int] | None = None,
) -> dict[str, Any]:
    """Higher-Order Singular Value Decomposition (HOSVD).

    As formulated in Kolda & Bader (2009) [Section 4.2] and De Lathauwer et al. (2000),
    HOSVD is a multilinear generalization of matrix SVD for N-way tensors.

    It computes the leading left singular vectors of each mode-n matrix unfolding X_(n)
    to form orthonormal factor matrices A^(n) and an all-orthogonal core tensor G.

    Args:
        array: Input tensor as a NumPy array (ndim >= 2).
        ranks: Multilinear ranks [R_1, R_2, ..., R_N] for each mode matrix.
            Can also be a single integer to truncate all modes.
            If None, full ranks [I_1, I_2, ..., I_N] are used for exact factorization.
        rank: Alias for ranks.

    Returns:
        dict containing:
            - "method": "hosvd"
            - "core": core tensor G
            - "factors": list of orthonormal factor matrices [A^(1), ..., A^(N)]
            - "singular_values": list of mode-n singular values
            - "ranks": list of multilinear ranks used along each mode
            - "shape": original tensor shape
    """
    tensor = as_float_tensor(array)
    if tensor.ndim < 2:
        raise ValueError("HOSVD requires a tensor with at least 2 dimensions")

    ndim = tensor.ndim
    shape = tensor.shape

    if ranks is None and rank is not None:
        ranks = rank

    target_ranks: list[int] = []
    if ranks is None:
        target_ranks = list(shape)
    elif isinstance(ranks, (int, np.integer)):
        r = max(1, int(ranks))
        target_ranks = [min(r, shape[m]) for m in range(ndim)]
    else:
        parsed = [max(1, int(v)) for v in ranks]
        if len(parsed) == 1:
            target_ranks = [min(parsed[0], shape[m]) for m in range(ndim)]
        elif len(parsed) >= ndim:
            target_ranks = [min(parsed[m], shape[m]) for m in range(ndim)]
        else:
            last = parsed[-1]
            padded = parsed + [last] * (ndim - len(parsed))
            target_ranks = [min(padded[m], shape[m]) for m in range(ndim)]

    factors: list[np.ndarray] = []
    singular_values: list[np.ndarray] = []
    actual_ranks: list[int] = []

    for mode in range(ndim):
        unfolding = matricization(tensor, mode)
        svd_res = svd(unfolding)
        u, s = svd_res["u"], svd_res["singular_values"]

        target_rank = min(target_ranks[mode], u.shape[1])
        u_trunc = u[:, :target_rank]

        factors.append(u_trunc)
        singular_values.append(s[:target_rank])
        actual_ranks.append(target_rank)

    # Compute all-orthogonal core tensor G = tensor ×_1 A^(1)T ×_2 A^(2)T ... ×_N A^(N)T
    core = tensor
    for mode in range(ndim):
        core = mode_n_product(core, factors[mode].T, mode)

    return {
        "method": "hosvd",
        "core": core,
        "factors": factors,
        "singular_values": singular_values,
        "ranks": actual_ranks,
        "min_ranks": [1] * ndim,
        "max_ranks": list(shape),
        "shape": list(tensor.shape),
    }