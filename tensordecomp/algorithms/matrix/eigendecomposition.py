from __future__ import annotations

from typing import Any

import numpy as np


def eigendecomposition(array: np.ndarray) -> dict[str, Any]:
    if array.ndim != 2:
        raise ValueError(
            f"Input array must be 2D for eigendecomposition, got {array.ndim}D."
        )
    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"Input matrix must be square (rows == cols), got shape {array.shape}."
        )

    eigenvalues, eigenvectors = np.linalg.eig(array)

    # Cast to real if imaginary parts are negligible to simplify representation and rendering
    if np.iscomplexobj(eigenvalues) and np.allclose(np.imag(eigenvalues), 0, atol=1e-12):
        eigenvalues = np.real(eigenvalues)
    if np.iscomplexobj(eigenvectors) and np.allclose(np.imag(eigenvectors), 0, atol=1e-12):
        eigenvectors = np.real(eigenvectors)

    return {
        "eigenvalues": eigenvalues,
        "q": eigenvectors,
        "eigenvectors": eigenvectors,
    }