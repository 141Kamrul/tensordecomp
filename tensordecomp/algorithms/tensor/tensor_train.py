from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...function.tensor_utils import as_float_tensor, norm
from ..matrix.svd import svd


def tensor_train(
    array: np.ndarray,
    ranks: int | Sequence[int] | None = None,
    rank: int | Sequence[int] | None = None,
    max_rank: int | None = None,
    tol: float | None = 1e-6,
) -> dict[str, Any]:
    """Tensor-Train (TT) Decomposition via TT-SVD algorithm.

    As introduced by I. V. Oseledets (2011) [SIAM J. Sci. Comput., 33(5), 2295-2317],
    TT-SVD decomposes an N-dimensional tensor A into a sequence of 3-mode core tensors
    G_1, G_2, ..., G_d such that:
        A(i_1, i_2, ..., i_d) = G_1(i_1) * G_2(i_2) * ... * G_d(i_d)

    Args:
        array: Input tensor as a NumPy array (ndim >= 2).
        ranks: Target TT-ranks [r_1, r_2, ..., r_{d-1}] for each interior bond.
               Can also be a single integer applied to all interior TT-ranks.
        rank: Alias for ranks.
        max_rank: Maximum TT-rank bound r_k if ranks is None.
        tol: Relative error threshold epsilon for adaptive rank determination.
             Truncates singular values delta = (tol / sqrt(d - 1)) * ||A||_F.

    Returns:
        dict containing:
            - "method": "tensor_train"
            - "cores": list of 3-mode core tensors G_1, ..., G_d (shapes r_{k-1} × I_k × r_k)
            - "ranks": list of TT-ranks [r_0=1, r_1, ..., r_{d-1}, r_d=1]
            - "singular_values": list of singular value arrays at each SVD step
            - "shape": original tensor shape
    """
    tensor = as_float_tensor(array)
    if tensor.ndim < 2:
        raise ValueError("Tensor Train decomposition requires a tensor with at least 2 dimensions")

    ndim = tensor.ndim
    shape = tensor.shape
    tensor_norm = float(norm(tensor))

    if ranks is None and rank is not None:
        ranks = rank

    # Parse target TT-ranks if provided
    explicit_ranks: list[int] | None = None
    if ranks is not None:
        if isinstance(ranks, (int, np.integer)):
            explicit_ranks = [max(1, int(ranks))] * (ndim - 1)
        else:
            parsed = [max(1, int(v)) for v in ranks]
            if len(parsed) == 1:
                explicit_ranks = [parsed[0]] * (ndim - 1)
            elif len(parsed) >= ndim - 1:
                explicit_ranks = parsed[: ndim - 1]
            else:
                last = parsed[-1]
                explicit_ranks = parsed + [last] * ((ndim - 1) - len(parsed))

    delta = 0.0
    if tol is not None and ndim > 1 and tensor_norm > 0 and explicit_ranks is None:
        delta = (tol / np.sqrt(ndim - 1)) * tensor_norm

    cores: list[np.ndarray] = []
    singular_values_list: list[np.ndarray] = []
    tt_ranks: list[int] = [1]

    # Current matrix C to unfold sequentially
    C = tensor.reshape(shape[0], -1)

    for mode in range(ndim - 1):
        n_k = shape[mode]
        r_prev = tt_ranks[-1]

        # Reshape C into matrix of shape (r_prev * n_k, -1)
        C_mat = C.reshape(r_prev * n_k, -1)

        # SVD of unfolding
        svd_res = svd(C_mat)
        u, s, vh = svd_res["u"], svd_res["singular_values"], svd_res["vh"]

        # Determine target rank for this bond
        max_possible = min(C_mat.shape[0], C_mat.shape[1])
        if explicit_ranks is not None:
            rk = min(explicit_ranks[mode], max_possible)
        else:
            # Adaptive rank selection based on SVD decay
            if delta > 0:
                rev_energy = np.cumsum(s[::-1] ** 2)
                cumsum_rev = rev_energy[::-1]
                idx_keep = np.where(cumsum_rev <= delta**2)[0]
                if len(idx_keep) > 0 and idx_keep[0] > 0:
                    rk = idx_keep[0]
                else:
                    rk = len(s)
            else:
                rk = len(s)

            if max_rank is not None:
                rk = min(rk, max_rank)
            rk = max(1, min(rk, max_possible))

        tt_ranks.append(rk)

        # Extract core G_k of shape (r_prev, n_k, rk)
        core_k = u[:, :rk].reshape(r_prev, n_k, rk)
        cores.append(core_k)
        singular_values_list.append(s[:rk])

        # Prepare next matrix C = S * V^H
        C = np.diag(s[:rk]) @ vh[:rk, :]

    # Last core G_d has shape (r_{d-1}, I_d, 1)
    last_r_prev = tt_ranks[-1]
    last_nk = shape[-1]
    last_core = C.reshape(last_r_prev, last_nk, 1)
    cores.append(last_core)
    tt_ranks.append(1)

    # Compute fixed theoretical min and max rank bounds for interior bonds
    min_bond_ranks = [1] * (ndim - 1)
    max_bond_ranks = [
        min(int(np.prod(shape[: k + 1])), int(np.prod(shape[k + 1 :])))
        for k in range(ndim - 1)
    ]

    return {
        "method": "tensor_train",
        "cores": cores,
        "ranks": tt_ranks,
        "singular_values": singular_values_list,
        "shape": list(shape),
        "min_ranks": min_bond_ranks,
        "max_ranks": max_bond_ranks,
    }