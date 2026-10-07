from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ..algorithms import run_algorithm
from .analysis import analyze_decomposition


def estimate_flops(shape: tuple[int, ...], algorithm: str, last_result: dict[str, Any] | None) -> int:
    def svd_flops(m: int, n: int) -> int:
        M, N = max(m, n), min(m, n)
        return int(4 * (M**2) * N + 8 * M * (N**2) + 9 * (N**3))

    base_algo = algorithm.lower().replace("+", "_").replace("_puzzle", "")

    if len(shape) >= 3:
        ndim = len(shape)
        prod_N = int(np.prod(shape))
        if base_algo == "tensor_train":
            total_flops = 0
            r_prev = 1
            for k in range(ndim - 1):
                m_k = r_prev * shape[k]
                n_k = int(np.prod(shape[k + 1 :]))
                total_flops += svd_flops(m_k, n_k)
                rk = 2
                if last_result and "cores" in last_result:
                    try:
                        rk = np.array(last_result["cores"][k]).shape[2]
                    except Exception:
                        pass
                r_prev = rk
            return total_flops
        elif base_algo in ("tucker", "hosvd"):
            total_svd_flops = sum(
                svd_flops(shape[m], prod_N // shape[m])
                for m in range(ndim)
            )
            core_ranks = list(shape)
            if last_result and "core" in last_result:
                try:
                    core_ranks = list(np.array(last_result["core"]).shape)
                except Exception:
                    pass
            elif last_result and "ranks" in last_result:
                try:
                    core_ranks = list(last_result["ranks"])
                except Exception:
                    pass
            projection_flops = 2 * prod_N * sum(core_ranks)
            return total_svd_flops + projection_flops
        elif base_algo == "cp":
            R = 3
            if last_result and "factors" in last_result:
                try:
                    R = int(np.array(last_result["factors"][0]).shape[1])
                except Exception:
                    pass
            elif last_result and "rank" in last_result:
                try:
                    R = int(last_result["rank"])
                except Exception:
                    pass
            return 50 * (2 * ndim * prod_N * R + sum(shape) * (R**2) + ndim * (R**3))
    else:
        m = shape[0]
        n = shape[1] if len(shape) > 1 else 1
        
        if base_algo == "svd":
            return svd_flops(m, n)
        elif base_algo == "qr":
            M, N = max(m, n), min(m, n)
            return int(2 * (N**2) * (M - N/3.0))
        elif base_algo == "lu":
            M, N = max(m, n), min(m, n)
            return int(2/3.0 * (N**3) + (N**2) * (M - N))
        elif base_algo == "eigendecomposition":
            return int(9 * (m**3))
            
    return int(m * n)


def get_complexity_formula(shape: tuple[int, ...], algorithm: str) -> str:
    base_algo = algorithm.lower().replace("+", "_").replace("_puzzle", "")
    suffix = " + Puzzle" if "_puzzle" in algorithm.lower() or "+puzzle" in algorithm.lower() else ""

    if len(shape) >= 3:
        if base_algo == "tensor_train":
            return f"O((∏ N_i) • R){suffix}"
        elif base_algo in ("tucker", "hosvd"):
            return f"O((∏ N_i) • (∑ R_i) + ∑ SVD_i){suffix}"
        elif base_algo == "cp":
            return f"O((∏ N_i) • R • Iterations){suffix}"
    else:
        if base_algo == "svd":
            return "O(4M²N + 8MN² + 9N³)"
        elif base_algo == "qr":
            return "O(2N²(M - N/3))"
        elif base_algo == "lu":
            return "O(2/3 N³ + N²(M-N))"
        elif base_algo == "eigendecomposition":
            return "O(9 N³)"
    return f"O((∏ N_i)){suffix}"


def benchmark_algorithm(array: np.ndarray, algorithm: str, repeats: int = 1) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("Benchmark repeats must be at least 1")

    durations_ms: list[float] = []
    last_result: dict[str, Any] | None = None

    for _ in range(repeats):
        start = perf_counter()
        last_result = run_algorithm(array, algorithm)
        durations_ms.append((perf_counter() - start) * 1000)

    flops = estimate_flops(array.shape, algorithm, last_result)
    if flops >= 1e9:
        flops_str = f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        flops_str = f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        flops_str = f"{flops / 1e3:.2f} KFLOPs"
    else:
        flops_str = f"{flops} FLOPs"

    complexity = get_complexity_formula(array.shape, algorithm)
    
    # Calculate compression analysis details
    original_parameters = int(array.size)
    compressed_parameters = original_parameters
    compression_ratio = 1.0
    if last_result:
        try:
            analysis = analyze_decomposition(array, algorithm, last_result)
            original_parameters = analysis.get("original_parameters", original_parameters)
            compressed_parameters = analysis.get("compressed_parameters", compressed_parameters)
            compression_ratio = analysis.get("compression_ratio", compression_ratio)
        except Exception:
            pass

    return {
        "algorithm": algorithm,
        "repeats": repeats,
        "execution_time_ms": round(float(sum(durations_ms) / len(durations_ms)), 3),
        "flops": flops,
        "flops_str": flops_str,
        "complexity": complexity,
        "original_parameters": original_parameters,
        "compressed_parameters": compressed_parameters,
        "compression_ratio": compression_ratio,
        "result_keys": list((last_result or {}).keys()),
    }