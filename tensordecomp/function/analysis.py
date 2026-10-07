from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..algorithms import run_algorithm
from .tensor_utils import (
    count_parameters,
    norm,
    reconstruct_cp,
    reconstruct_tt,
    reconstruct_tucker,
)


def analyze_decomposition(array: np.ndarray, algorithm: str, result: dict[str, Any]) -> dict[str, Any]:
    tensor = np.asarray(array, dtype=float)
    reconstructed = reconstruct_tensor(algorithm, result)
    absolute_error = float(norm(tensor - reconstructed))
    relative_error = float(absolute_error / (norm(tensor) + 1e-12))
    
    # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
    mean_absolute_error = float(np.mean(np.abs(tensor - reconstructed)))
    root_mean_squared_error = float(np.sqrt(np.mean((tensor - reconstructed) ** 2)))
    
    # Calculate reconstructed matrix/tensor element head: min(10, len(tensor))
    head_len = min(10, len(tensor)) if tensor.ndim > 0 else 0
    reconstructed_head = reconstructed[:head_len] if tensor.ndim > 0 else reconstructed

    compressed_parameters = count_compressed_parameters(algorithm, result)
    original_parameters = int(tensor.size)

    return {
        "algorithm": algorithm,
        "original_parameters": original_parameters,
        "compressed_parameters": compressed_parameters,
        "compression_ratio": round(float(original_parameters / max(1, compressed_parameters)), 3),
        "absolute_error": round(absolute_error, 6),
        "mean_absolute_error": round(mean_absolute_error, 6),
        "root_mean_squared_error": round(root_mean_squared_error, 6),
        "relative_error": round(relative_error, 6),
        "reconstructed_head": reconstructed_head,
        "reconstructed_tensor": reconstructed.tolist() if hasattr(reconstructed, "tolist") else reconstructed,
    }


def compare_methods(array: np.ndarray, algorithms: Iterable[str]) -> list[dict[str, Any]]:
    from time import perf_counter
    from .benchmark import estimate_flops, get_complexity_formula

    comparison: list[dict[str, Any]] = []
    for algorithm in algorithms:
        # Time the algorithm (average of 3 runs for stability)
        durations = []
        result = None
        for _ in range(3):
            t0 = perf_counter()
            result = run_algorithm(array, algorithm)
            durations.append((perf_counter() - t0) * 1000)

        execution_time_ms = round(float(sum(durations) / len(durations)), 3)
        analysis = analyze_decomposition(array, algorithm, result)

        # Estimate FLOPS and complexity formula
        flops = estimate_flops(array.shape, algorithm, result)
        if flops >= 1e9:
            flops_str = f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            flops_str = f"{flops / 1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            flops_str = f"{flops / 1e3:.2f} KFLOPs"
        else:
            flops_str = f"{flops} FLOPs"
        complexity = get_complexity_formula(array.shape, algorithm)

        comparison.append(
            {
                "algorithm": algorithm,
                "compression_ratio": analysis["compression_ratio"],
                "relative_error": analysis["relative_error"],
                "absolute_error": analysis["absolute_error"],
                "mean_absolute_error": analysis["mean_absolute_error"],
                "root_mean_squared_error": analysis["root_mean_squared_error"],
                "original_parameters": analysis["original_parameters"],
                "compressed_parameters": analysis["compressed_parameters"],
                "execution_time_ms": execution_time_ms,
                "flops": flops,
                "flops_str": flops_str,
                "complexity": complexity,
            }
        )

    return comparison



from .puzzle_tensor import invert_puzzle_tensor


def reconstruct_tensor(algorithm: str, result: dict[str, Any]) -> np.ndarray:
    norm_algo = algorithm.lower().replace("+", "_")
    base_algo = norm_algo.replace("_puzzle", "")

    if base_algo == "cp":
        reconstructed = reconstruct_cp(result["weights"], result["factors"])
    elif base_algo in ("tucker", "hosvd"):
        reconstructed = reconstruct_tucker(result["core"], result["factors"])
    elif base_algo == "tensor_train":
        reconstructed = reconstruct_tt(result["cores"])
    elif base_algo == "svd":
        reconstructed = result["u"] @ np.diag(result["singular_values"]) @ result["vh"]
    elif base_algo == "eigendecomposition":
        q = result.get("q", result.get("eigenvectors"))
        reconstructed = q @ np.diag(result["eigenvalues"]) @ np.linalg.inv(q)
    elif base_algo == "qr":
        reconstructed = result["q"] @ result["r"]
    elif base_algo == "lu":
        reconstructed = result["l"] @ result["u"]
    else:
        raise ValueError(f"Unsupported algorithm while reconstructing tensor: {algorithm}")

    if result.get("is_puzzle") or "shifts" in result or norm_algo.endswith("_puzzle"):
        shifts = result.get("shifts", [])
        if shifts:
            reconstructed = invert_puzzle_tensor(reconstructed, shifts)

    return reconstructed


def count_compressed_parameters(algorithm: str, result: dict[str, Any]) -> int:
    norm_algo = algorithm.lower().replace("+", "_")
    base_algo = norm_algo.replace("_puzzle", "")

    if base_algo == "cp":
        count = int(np.prod(result["weights"].shape)) + count_parameters(result["factors"])
    elif base_algo in {"tucker", "hosvd"}:
        count = int(np.prod(result["core"].shape)) + count_parameters(result["factors"])
    elif base_algo == "tensor_train":
        count = count_parameters(result["cores"])
    elif base_algo == "svd":
        count = int(np.prod(result["u"].shape)) + int(np.prod(result["singular_values"].shape)) + int(np.prod(result["vh"].shape))
    elif base_algo == "eigendecomposition":
        q = result.get("q", result.get("eigenvectors"))
        count = int(np.prod(q.shape)) + int(np.prod(result["eigenvalues"].shape))
    elif base_algo == "qr":
        count = int(np.prod(result["q"].shape)) + int(np.prod(result["r"].shape))
    elif base_algo == "lu":
        count = int(np.prod(result["l"].shape)) + int(np.prod(result["u"].shape))
    else:
        raise ValueError(f"Unsupported algorithm while counting compressed parameters: {algorithm}")

    if result.get("is_puzzle") or "shifts" in result or norm_algo.endswith("_puzzle"):
        shifts = result.get("shifts", [])
        count += len(shifts)

    return count