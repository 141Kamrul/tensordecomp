from __future__ import annotations

"""tensordecomp: High-Dimensional Tensor Decompositions and PuzzleTensor Optimization Library."""

from .algorithms import (
    SUPPORTED_ALGORITHMS,
    cp,
    cp_puzzle,
    eigendecomposition,
    hosvd,
    hosvd_puzzle,
    lu,
    qr,
    run_algorithm,
    svd,
    tensor_train,
    tensor_train_puzzle,
    tucker,
    tucker_puzzle,
)
from .function.analysis import (
    analyze_decomposition,
    benchmark_methods_graph,
    compression_methods_graph,
    compare_methods,
    compare_methods_graph,
    error_methods_graph,
    reconstruct_tensor,
    time_methods_graph,
)
from .function.benchmark import (
    benchmark_algorithm,
    estimate_flops,
    get_complexity_formula,
)
from .function.puzzle_tensor import (
    invert_puzzle_tensor,
    puzzle_tensor,
    shift_hyperslice,
    tensor_nuclear_norm_loss,
)
from .function.tensor_utils import (
    as_float_tensor,
    count_parameters,
    khatri_rao,
    matricization,
    mode_n_product,
    multi_mode_product,
    norm,
    pinv,
    reconstruct_cp,
    reconstruct_tt,
    reconstruct_tucker,
)
from .io import parse_tensor_input

__version__ = "0.1.0"

__all__ = [
    "SUPPORTED_ALGORITHMS",
    "analyze_decomposition",
    "as_float_tensor",
    "benchmark_algorithm",
    "benchmark_methods_graph",
    "compression_methods_graph",
    "compare_methods",
    "compare_methods_graph",
    "count_parameters",
    "cp",
    "cp_puzzle",
    "eigendecomposition",
    "error_methods_graph",
    "estimate_flops",
    "get_complexity_formula",
    "hosvd",
    "hosvd_puzzle",
    "invert_puzzle_tensor",
    "khatri_rao",
    "lu",
    "matricization",
    "mode_n_product",
    "multi_mode_product",
    "norm",
    "parse_tensor_input",
    "pinv",
    "puzzle_tensor",
    "qr",
    "reconstruct_cp",
    "reconstruct_tensor",
    "reconstruct_tt",
    "reconstruct_tucker",
    "run_algorithm",
    "shift_hyperslice",
    "svd",
    "tensor_nuclear_norm_loss",
    "tensor_train",
    "tensor_train_puzzle",
    "time_methods_graph",
    "tucker",
    "tucker_puzzle",
]
