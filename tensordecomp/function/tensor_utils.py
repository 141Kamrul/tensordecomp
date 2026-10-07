from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def as_float_tensor(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=float)


def matricization(tensor: np.ndarray, mode: int) -> np.ndarray:
    moved = np.moveaxis(tensor, mode, 0)
    return moved.reshape(tensor.shape[mode], -1)


def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    product = np.tensordot(matrix, tensor, axes=(1, mode))
    return np.moveaxis(product, 0, mode)


def multi_mode_product(tensor: np.ndarray, matrices: Iterable[np.ndarray]) -> np.ndarray:
    result = tensor
    for mode, matrix in enumerate(matrices):
        result = mode_n_product(result, matrix, mode)
    return result


def khatri_rao(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("Khatri-Rao product requires at least one matrix")
    result = matrices[0]
    for mat in matrices[1:]:
        if result.shape[1] != mat.shape[1]:
            raise ValueError(f"Column count mismatch for Khatri-Rao product: {result.shape[1]} vs {mat.shape[1]}")
        result = (result[:, None, :] * mat[None, :, :]).reshape(-1, result.shape[1])
    return result


def pinv(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(matrix)


def norm(array: np.ndarray, axis: int | tuple[int, ...] | None = None) -> Any:
    return np.linalg.norm(array, axis=axis)



def reconstruct_cp(weights: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    rank = int(weights.shape[0])
    tensor: np.ndarray | None = None

    for component_index in range(rank):
        outer: np.ndarray | None = None
        for factor in factors:
            vector = factor[:, component_index]
            outer = vector if outer is None else np.multiply.outer(outer, vector)

        if outer is None:
            continue

        component = weights[component_index] * outer
        tensor = component if tensor is None else tensor + component

    if tensor is None:
        raise ValueError("CP reconstruction requires at least one factor")

    return tensor


def reconstruct_tucker(core: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    return multi_mode_product(core, factors)


def reconstruct_tt(cores: list[np.ndarray]) -> np.ndarray:
    tensor = cores[0]
    for core in cores[1:]:
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))

    return np.squeeze(tensor, axis=(0, -1))


def count_parameters(values: Iterable[np.ndarray]) -> int:
    return sum(int(np.prod(value.shape)) for value in values)