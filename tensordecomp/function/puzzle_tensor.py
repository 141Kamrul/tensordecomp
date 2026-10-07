from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from .tensor_utils import as_float_tensor, matricization, norm
from ..algorithms.matrix.svd import svd


def shift_hyperslice(
    tensor: np.ndarray,
    mode: int,
    slice_idx: int,
    target_axis: int,
    shift: int,
    block_bounds: tuple[slice, ...] | None = None,
) -> np.ndarray:
    """Circularly shifts a (D - 1)-dimensional hyperslice of a tensor along a target axis.

    Supports optional sub-block shifting as described in Section 3.4 of Park et al. (KDD 2025).

    Args:
        tensor: Input tensor as a NumPy array (ndim >= 2).
        mode: The axis of the hyperslice (mode index, 0 <= mode < ndim).
        slice_idx: The slice index along the specified mode (0 <= slice_idx < shape[mode]).
        target_axis: The axis along which the hyperslice is shifted (target_axis != mode).
        shift: Integer shift amount (positive or negative).
        block_bounds: Optional tuple of slice objects specifying a sub-block region.

    Returns:
        A copy of the tensor with the specified hyperslice circularly shifted.
    """
    if shift == 0 or mode == target_axis or tensor.ndim < 2:
        return tensor

    transformed = tensor.copy()

    if block_bounds is not None:
        sub_view = transformed[tuple(block_bounds)]
        slice_spec = [slice(None)] * sub_view.ndim
        slice_spec[mode] = slice_idx
        sub_axis = target_axis if target_axis < mode else target_axis - 1
        sub_view[tuple(slice_spec)] = np.roll(
            sub_view[tuple(slice_spec)],
            shift=shift,
            axis=sub_axis,
        )
    else:
        slice_spec = [slice(None)] * tensor.ndim
        slice_spec[mode] = slice_idx
        sub_axis = target_axis if target_axis < mode else target_axis - 1
        transformed[tuple(slice_spec)] = np.roll(
            transformed[tuple(slice_spec)],
            shift=shift,
            axis=sub_axis,
        )

    return transformed


def tensor_nuclear_norm_loss(tensor: np.ndarray) -> float:
    """Computes the multilinear nuclear norm loss proxy from the PuzzleTensor paper:

        L(Z) = sum_{k=1}^D (1 / sqrt(I_k)) * ||Z_(k)||_*

    Minimizing this objective induces sparsity in the HOSVD core tensor,
    lowering effective multilinear rank and enabling more compact factorization.

    Args:
        tensor: Input tensor.

    Returns:
        Aggregated normalized nuclear norm across all mode unfoldings.
    """
    if tensor.ndim < 2:
        return float(norm(tensor))

    total_loss = 0.0
    for mode in range(tensor.ndim):
        unfolding = matricization(tensor, mode)
        # Fast singular value calculation without computing heavy U, V matrices
        singular_values = np.linalg.svd(unfolding, compute_uv=False)
        total_loss += float(np.sum(singular_values) / np.sqrt(tensor.shape[mode]))
    return total_loss


def puzzle_tensor(
    tensor: np.ndarray | Sequence[Any],
    max_iter: int = 2,
    max_shift: int = 2,
    block_size: int | Sequence[int] | None = None,
    return_shifts: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """Applies hyperslice shifting to align tensor patterns and reduce effective rank.

    Based on PuzzleTensor (Park et al., KDD 2025: "PuzzleTensor: A Method-Agnostic
    Data Transformation for Compact Tensor Factorization").

    Uses coordinate-wise greedy alignment to search for hyperslice shifts that minimize
    the multilinear nuclear norm loss objective (Equation 2 in paper). Supports optional
    sub-block decomposition for scaling to large tensors (Section 3.4 in paper).

    Args:
        tensor: Input tensor (2D, 3D, 4D, or N-D array).
        max_iter: Maximum optimization passes over all modes and hyperslices (default: 2).
        max_shift: Maximum search shift radius along each axis (default: 2).
        block_size: Optional sub-block size integer or sequence per mode (Section 3.4).
        return_shifts: If True, returns a tuple of (shifted_tensor, shifts_list).
            If False, returns just the shifted_tensor.

    Returns:
        Shifted tensor with lower effective rank, optionally accompanied by the
        list of applied shift operations for exact reconstruction.
    """
    arr = as_float_tensor(np.asarray(tensor))
    if arr.ndim < 2:
        return (arr, []) if return_shifts else arr

    current = arr.copy()
    shifts_applied: list[dict[str, Any]] = []
    ndim = current.ndim

    # Handle sub-block bounds partitioning (Section 3.4)
    if block_size is not None:
        if isinstance(block_size, int):
            b_sizes = [block_size] * ndim
        else:
            b_sizes = list(block_size)

        blocks_per_mode = []
        for mode_idx in range(ndim):
            dim_len = current.shape[mode_idx]
            bsize = b_sizes[mode_idx] if mode_idx < len(b_sizes) else b_sizes[0]
            mode_slices = []
            start = 0
            while start < dim_len:
                end = min(start + bsize, dim_len)
                mode_slices.append(slice(start, end))
                start = end
            blocks_per_mode.append(mode_slices)

        import itertools
        all_block_bounds = list(itertools.product(*blocks_per_mode))
    else:
        all_block_bounds = [None]

    for block_bounds in all_block_bounds:
        if block_bounds is not None:
            sub_tensor = current[block_bounds]
            if sub_tensor.ndim < 2 or min(sub_tensor.shape) < 2:
                continue
        else:
            sub_tensor = current

        current_loss = tensor_nuclear_norm_loss(sub_tensor)

        for _ in range(max_iter):
            improved = False
            for mode in range(sub_tensor.ndim):
                dim_size = sub_tensor.shape[mode]
                for slice_idx in range(dim_size):
                    for target_axis in range(sub_tensor.ndim):
                        if target_axis == mode:
                            continue

                        target_dim_size = sub_tensor.shape[target_axis]
                        shift_bound = min(max_shift, target_dim_size // 2)
                        if shift_bound < 1:
                            continue

                        best_shift = 0
                        best_loss = current_loss

                        slice_spec = [slice(None)] * sub_tensor.ndim
                        slice_spec[mode] = slice_idx
                        sub_axis = target_axis if target_axis < mode else target_axis - 1

                        orig_slice = sub_tensor[tuple(slice_spec)].copy()

                        for candidate_shift in range(-shift_bound, shift_bound + 1):
                            if candidate_shift == 0:
                                continue

                            # Temporary in-place slice shift evaluation
                            sub_tensor[tuple(slice_spec)] = np.roll(
                                orig_slice, shift=candidate_shift, axis=sub_axis
                            )
                            candidate_loss = tensor_nuclear_norm_loss(sub_tensor)

                            if candidate_loss < best_loss - 1e-5:
                                best_loss = candidate_loss
                                best_shift = candidate_shift

                        # Restore or apply best shift
                        if best_shift != 0:
                            sub_tensor[tuple(slice_spec)] = np.roll(
                                orig_slice, shift=best_shift, axis=sub_axis
                            )
                            current_loss = best_loss
                            op_dict: dict[str, Any] = {
                                "mode": mode,
                                "slice_idx": slice_idx,
                                "target_axis": target_axis,
                                "shift": best_shift,
                            }
                            if block_bounds is not None:
                                op_dict["block_bounds"] = block_bounds
                            shifts_applied.append(op_dict)
                            improved = True
                        else:
                            sub_tensor[tuple(slice_spec)] = orig_slice

            if not improved:
                break

    if return_shifts:
        return current, shifts_applied
    return current


def invert_puzzle_tensor(
    shifted_tensor: np.ndarray,
    shifts: list[dict[str, Any]],
) -> np.ndarray:
    """Exact inverse transformation of puzzle_tensor.

    Reverses applied hyperslice shifts in reverse chronological order to
    restore the original tensor losslessly.

    Args:
        shifted_tensor: The tensor produced by puzzle_tensor.
        shifts: The list of shift operations returned when return_shifts=True.

    Returns:
        The exact reconstructed original tensor.
    """
    recovered = shifted_tensor.copy()
    for op in reversed(shifts):
        bounds = op.get("block_bounds")
        recovered = shift_hyperslice(
            recovered,
            mode=op["mode"],
            slice_idx=op["slice_idx"],
            target_axis=op["target_axis"],
            shift=-op["shift"],
            block_bounds=bounds,
        )
    return recovered

