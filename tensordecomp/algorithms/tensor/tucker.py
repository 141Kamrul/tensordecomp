from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...function.tensor_utils import (
    as_float_tensor,
    matricization,
    mode_n_product,
    norm,
    reconstruct_tucker,
)
from ..matrix.svd import svd


def tucker(
    array: np.ndarray,
    ranks: int | Sequence[int] | None = None,
    rank: int | Sequence[int] | None = None,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> dict[str, Any]:
    """Tucker Decomposition via Higher-Order Orthogonal Iteration (HOOI).

    Decomposes an N-way tensor X into a dense core tensor G multiplied by
    factor matrices along each mode:
        X ≈ G ×_1 A^(1) ×_2 A^(2) ... ×_N A^(N)

    Args:
        array: Input tensor as a NumPy array (ndim >= 2).
        ranks: Multilinear ranks [R_1, R_2, ..., R_N] for each matrix/mode.
               Can also be a single integer to apply to all modes.
               If None, optimal multilinear ranks are determined adaptively.
        max_iter: Maximum HOOI optimization iterations.
        tol: Convergence tolerance on relative core norm change.

    Returns:
        dict containing:
            - "method": "tucker"
            - "core": core tensor G of shape (R_1, R_2, ..., R_N)
            - "factors": list of orthonormal factor matrices [A^(1), ..., A^(N)]
            - "shape": original tensor shape
            - "ranks": multilinear ranks [R_1, ..., R_N]
    """
    tensor = as_float_tensor(array)
    if tensor.ndim < 2:
        raise ValueError("Tucker decomposition requires a tensor with at least 2 dimensions")

    ndim = tensor.ndim
    shape = tensor.shape
    tensor_norm = float(norm(tensor))
    if tensor_norm == 0:
        tensor_norm = 1.0

    # Determine multilinear ranks
    if ranks is None and rank is not None:
        ranks = rank

    target_ranks: list[int] = []
    if ranks is None:
        # Adaptive SVD energy thresholding (retaining >= 90% Frobenius energy per mode)
        for mode in range(ndim):
            unfolding = matricization(tensor, mode)
            s = svd(unfolding)["singular_values"]
            total_energy = float(np.sum(s**2))
            if total_energy == 0:
                target_ranks.append(1)
                continue
            cumsum = np.cumsum(s**2)
            k = int(np.searchsorted(cumsum, 0.90 * total_energy)) + 1
            target_ranks.append(max(1, min(k, shape[mode])))
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

    # Step 1: Initialize factor matrices using HOSVD
    factors: list[np.ndarray] = []
    for mode in range(ndim):
        unfolding = matricization(tensor, mode)
        u = svd(unfolding)["u"]
        target_rank = min(target_ranks[mode], u.shape[1])
        factors.append(u[:, :target_rank])

    # Initial core calculation
    core = tensor
    for m in range(ndim):
        core = mode_n_product(core, factors[m].T, m)

    best_factors = [f.copy() for f in factors]
    best_core = core.copy()
    best_core_norm = float(norm(core))

    # Step 2: Higher-Order Orthogonal Iteration (HOOI) Loop
    prev_norm = best_core_norm
    for _ in range(max_iter):
        for n in range(ndim):
            # Compute Y = tensor ×_1 A^(1)T ... ×_{n-1} A^(n-1)T ×_{n+1} A^(n+1)T ...
            Y = tensor
            for m in range(ndim):
                if m != n:
                    Y = mode_n_product(Y, factors[m].T, m)

            # Unfold Y along mode n and extract leading left singular vectors
            Y_n = matricization(Y, n)
            u = svd(Y_n)["u"]
            target_rank = min(target_ranks[n], u.shape[1])
            factors[n] = u[:, :target_rank]

        # Compute core tensor: G = tensor ×_1 A^(1)T ... ×_N A^(N)T
        current_core = tensor
        for m in range(ndim):
            current_core = mode_n_product(current_core, factors[m].T, m)

        current_norm = float(norm(current_core))

        if current_norm > best_core_norm:
            best_core_norm = current_norm
            best_factors = [f.copy() for f in factors]
            best_core = current_core.copy()

        if abs(current_norm - prev_norm) < tol:
            break
        prev_norm = current_norm

    actual_ranks = [f.shape[1] for f in best_factors]

    return {
        "method": "tucker",
        "core": best_core,
        "factors": best_factors,
        "shape": list(tensor.shape),
        "ranks": actual_ranks,
        "min_ranks": [1] * ndim,
        "max_ranks": list(shape),
    }