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
pip install -e .
# or with uv
uv pip install -e .
```

To use the comparison plots in an editable installation, install the optional
plotting dependency:

```bash
pip install -e ".[plotting]"
```

For a regular installation from PyPI:

```bash
pip install "tensordecomp[plotting]"
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
shifted_tensor, shifts = td.puzzle_tensor(tensor, max_shift=2, return_shifts=False)

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

# 7. Compare methods numerically
figure = td.compare_methods_graph(
    tensor,
    [
        "cp", "cp_puzzle", "tucker", "tucker_puzzle",
        "hosvd", "hosvd_puzzle", "tensor_train", "tensor_train_puzzle",
    ],
    show=False,
)
# In a notebook, evaluate `figure` to render it.

# 8. Create separate benchmark figures for all eight default methods.
# Each function returns a matplotlib.figure.Figure. Use show=False in scripts
# or notebooks where you want to display or customize the figure yourself.
td.benchmark_methods_graph(tensor, show=True)
td.error_methods_graph(tensor, show=True)
td.time_methods_graph(tensor, show=True)
td.compression_methods_graph(tensor, show=True)
```

The graph functions compare these methods by default:

- `cp` and `cp_puzzle`
- `tucker` and `tucker_puzzle`
- `hosvd` and `hosvd_puzzle`
- `tensor_train` and `tensor_train_puzzle`

Use `show=False` when running in a headless environment, then display the
returned figure in a notebook or save it with Matplotlib. The plotting
dependency is optional; numerical decomposition and benchmarking functions
only require NumPy.

For an interactive example, see
[`compare_methods_graph_demo.ipynb`](./compare_methods_graph_demo.ipynb).

---

## License

MIT License.
