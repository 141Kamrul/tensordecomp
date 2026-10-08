# tensordecomp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**tensordecomp** is a lightweight, high-performance Python library for multi-dimensional tensor decompositions and PuzzleTensor sparsity-inducing permutation algorithms.

---

## Features

- **Core Tensor Decompositions**:
  - CANDECOMP/PARAFAC (`CP`) via Alternating Least Squares (ALS)
  - Tucker Decomposition (Higher-Order Orthogonal Iteration / HOSVD core)
  - Higher-Order Singular Value Decomposition (`HOSVD`)
  - Tensor Train (`TT` / TT-SVD)
- **PuzzleTensor Optimization** (KDD 2025 Paper):
  - Greedy cyclic slice shifting to minimize matricization SVD decay cost $F(A)$
  - SVD energy concentration for enhanced low-rank compressibility
  - Augmented pipelines: `cp_puzzle`, `tucker_puzzle`, `hosvd_puzzle`, `tensor_train_puzzle`
- **Reconstruction, Accuracy & Benchmarking**:
  - Exact/approximate tensor reconstruction from decomposed factors
  - Numerical metrics: Relative Error, RMSE, MAE, Compression Ratio
  - Execution speed benchmarks and FLOP estimation
- **Zero Heavy Dependencies**:
  - Requires only `numpy>=1.26`

---

## Installation

### Local Editable Install (Development Mode)
```bash
pip install -e ./tensordecomp
# or with uv
uv pip install -e ./tensordecomp
```

To use the comparison plots, install the optional plotting dependency:

```bash
pip install -e "./tensordecomp[plotting]"
```

---

## Quickstart

```python
import numpy as np
import tensordecomp as td

# 1. Create a 3D tensor
tensor = np.random.rand(4, 5, 6)

# 2. Run CP Decomposition
cp_res = td.cp(tensor, rank=3)
print("CP Factors:", [f.shape for f in cp_res["factors"]])

# 3. Run PuzzleTensor Shifting
shifted_tensor, shifts = td.puzzle_tensor(tensor, max_shift=2)

# 4. Run Puzzle-Augmented Tucker Decomposition
tucker_res = td.tucker_puzzle(tensor, ranks=[3, 3, 3])
print("Tucker Core Shape:", tucker_res["core"].shape)

# 5. Numerical Accuracy & Metrics
analysis = td.analyze_decomposition(tensor, "tucker_puzzle", tucker_res)
print(f"RMSE: {analysis['root_mean_squared_error']:.6f}")
print(f"Compression: {analysis['compression_ratio']:.2f}x")

# 6. Compare All Methods Side-by-Side
comparison = td.compare_methods(tensor, ["cp", "tucker", "hosvd", "tensor_train"])
for row in comparison:
    print(f"{row['algorithm']}: {row['execution_time_ms']:.2f}ms, error={row['relative_error']:.4e}")

# 7. Plot execution time and relative error for all methods
figure = td.compare_methods_graph(
    tensor,
    ["cp", "tucker", "hosvd", "tensor_train"],
    show=False,
)
figure.savefig("method-comparison.png")

# Separate figures for all eight tensor methods, including PuzzleTensor methods.
td.benchmark_methods_graph(tensor, show=True)
td.error_methods_graph(tensor, show=True)
td.time_methods_graph(tensor, show=True)
td.compression_methods_graph(tensor, show=True)
```

---

## License

MIT License.
