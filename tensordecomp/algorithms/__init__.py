from __future__ import annotations

from typing import Any

import numpy as np

from .matrix.lu import lu
from .tensor.cp import cp
from .tensor.cp_puzzle import cp_puzzle
from .matrix.eigendecomposition import eigendecomposition
from .tensor.hosvd import hosvd
from .tensor.hosvd_puzzle import hosvd_puzzle
from .matrix.qr import qr
from .tensor.tensor_train import tensor_train
from .tensor.tensor_train_puzzle import tensor_train_puzzle
from .tensor.tucker import tucker
from .tensor.tucker_puzzle import tucker_puzzle
from .matrix.svd import svd

SUPPORTED_ALGORITHMS: tuple[str, ...] = (
    "cp",
    "cp_puzzle",
    "tucker",
    "tucker_puzzle",
    "hosvd",
    "hosvd_puzzle",
    "tensor_train",
    "tensor_train_puzzle",
    "svd",
    "eigendecomposition",
    "qr",
    "lu",
)


def run_algorithm(array: np.ndarray, algorithm: str, **kwargs: Any) -> dict[str, Any]:
    norm_algo = algorithm.lower().replace("+", "_")

    if norm_algo == "cp":
        return cp(array, **kwargs)

    if norm_algo == "cp_puzzle":
        return cp_puzzle(array, **kwargs)

    if norm_algo == "tucker":
        return tucker(array, **kwargs)

    if norm_algo == "tucker_puzzle":
        return tucker_puzzle(array, **kwargs)

    if norm_algo == "hosvd":
        return hosvd(array, **kwargs)

    if norm_algo == "hosvd_puzzle":
        return hosvd_puzzle(array, **kwargs)

    if norm_algo == "tensor_train":
        return tensor_train(array, **kwargs)

    if norm_algo == "tensor_train_puzzle":
        return tensor_train_puzzle(array, **kwargs)

    if norm_algo == "svd":
        return svd(array, **kwargs)

    if norm_algo == "eigendecomposition":
        return eigendecomposition(array, **kwargs)

    if norm_algo == "qr":
        return qr(array, **kwargs)

    if norm_algo == "lu":
        return lu(array, **kwargs)

    raise ValueError(f"Unsupported algorithm while running: {algorithm}")