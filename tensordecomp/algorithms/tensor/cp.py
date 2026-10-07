from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...function.tensor_utils import (
    as_float_tensor,
    khatri_rao,
    matricization,
    norm,
    pinv,
    reconstruct_cp,
)
from ..matrix.svd import svd


def cp(
    array: np.ndarray,
    rank: int | Sequence[int] | None = None,
    ranks: int | Sequence[int] | None = None,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> dict[str, Any]:
    """CANDECOMP/PARAFAC (CP) Decomposition via Alternating Least Squares (CP-ALS).

    Decomposes an N-way tensor X into a sum of R rank-one component tensors:
        X ≈ sum_{r=1}^R λ_r (a_r^(1) ∘ a_r^(2) ∘ ... ∘ a_r^(N))

    Args:
        array: Input tensor as a NumPy array (ndim >= 2).
        rank: Target CP rank R. Can be an integer or a list/tuple of mode ranks [r_1, ..., r_N].
        ranks: Alternative alias for rank. Can be an integer or mode ranks for each matrix.
        max_iter: Maximum number of ALS optimization iterations.
        tol: Convergence tolerance based on reconstruction relative error.

    Returns:
        dict containing:
            - "method": "cp"
            - "weights": 1D array of component weights λ (length R)
            - "factors": list of factor matrices [A^(1), ..., A^(N)], shape (I_n, R)
            - "shape": original tensor shape
            - "rank": effective CP rank R
            - "mode_ranks": target rank per mode
    """
    tensor = as_float_tensor(array)
    if tensor.ndim < 2:
        raise ValueError("CP decomposition requires a tensor with at least 2 dimensions")

    ndim = tensor.ndim
    tensor_norm = float(norm(tensor))
    if tensor_norm == 0:
        tensor_norm = 1.0

    # Fixed minimum and maximum rank bounds for CP
    min_rank = 1
    max_rank = max(1, int(np.prod(tensor.shape)))

    target_input = rank if rank is not None else ranks
    if target_input is None:
        raw_rank = max(1, min(tensor.shape))
    elif isinstance(target_input, (int, np.integer)):
        raw_rank = int(target_input)
    elif isinstance(target_input, (list, tuple, np.ndarray)) and len(target_input) > 0:
        # CP requires a single rank R across all factor matrices. Disallow mode-specific ranks.
        raw_rank = int(target_input[0])
    else:
        raw_rank = max(1, min(tensor.shape))

    # Clamp strictly within fixed min and max rank bounds
    target_rank = max(min_rank, min(raw_rank, max_rank))

    # Initialize factor matrices using SVD of mode unfoldings
    factors: list[np.ndarray] = []
    for mode in range(ndim):
        unfolding = matricization(tensor, mode)
        u = svd(unfolding)["u"]
        avail = min(u.shape[1], target_rank)
        
        cols: list[np.ndarray] = []
        if avail > 0:
            cols.append(u[:, :avail])
        
        # If target_rank > avail, pad with small random noise
        needed = target_rank - avail
        if needed > 0:
            rng = np.random.default_rng(42 + mode)
            rand_cols = rng.standard_normal((u.shape[0], needed)) * 0.05
            cols.append(rand_cols)
            
        col_mat = np.hstack(cols) if len(cols) > 1 else cols[0]
        # Normalize columns
        c_norms = np.linalg.norm(col_mat, axis=0)
        c_norms[c_norms == 0] = 1.0
        factors.append(col_mat / c_norms)

    weights = np.ones(target_rank, dtype=float)

    # Track the best possible scenario across ALS iterations
    best_factors = [f.copy() for f in factors]
    best_weights = weights.copy()
    initial_recon = reconstruct_cp(best_weights, best_factors)
    best_error = float(norm(tensor - initial_recon) / tensor_norm)

    # CP-ALS Iteration Loop
    for _ in range(max_iter):
        for n in range(ndim):
            # Compute V = *_{m != n} (A^(m)^T A^(m))
            V = np.ones((target_rank, target_rank), dtype=float)
            for m in range(ndim):
                if m != n:
                    V *= (factors[m].T @ factors[m])

            # Khatri-Rao product of factor matrices in ascending order excluding mode n
            mats = [factors[m] for m in range(ndim) if m != n]
            W = khatri_rao(mats)

            # Mode-n unfolding
            X_n = matricization(tensor, n)
            mttkrp = X_n @ W

            # Solve least-squares: A_tilde = MTTKRP @ inv(V)
            try:
                A_tilde = np.linalg.solve(V + 1e-12 * np.eye(target_rank), mttkrp.T).T
            except np.linalg.LinAlgError:
                A_tilde = mttkrp @ pinv(V)

            # Normalize columns and update weights:
            # Modes 0 .. ndim-2 are normalized to unit column norm;
            # the final mode ndim-1 absorbs the column scales into weights.
            if n < ndim - 1:
                col_norms = np.linalg.norm(A_tilde, axis=0)
                clean_norms = np.where(col_norms == 0, 1.0, col_norms)
                factors[n] = A_tilde / clean_norms
            else:
                weights = np.linalg.norm(A_tilde, axis=0)
                clean_norms = np.where(weights == 0, 1.0, weights)
                factors[n] = A_tilde / clean_norms

        # Check current reconstruction accuracy
        current_recon = reconstruct_cp(weights, factors)
        current_error = float(norm(tensor - current_recon) / tensor_norm)

        if current_error < best_error:
            best_error = current_error
            best_factors = [f.copy() for f in factors]
            best_weights = weights.copy()

        if current_error < tol:
            break

    return {
        "method": "cp",
        "weights": best_weights,
        "factors": best_factors,
        "shape": list(tensor.shape),
        "rank": target_rank,
        "min_rank": min_rank,
        "max_rank": max_rank,
    }