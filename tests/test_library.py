import numpy as np
import pytest

import tensordecomp as td


def test_library_exports():
    expected_exports = [
        "cp",
        "tucker",
        "hosvd",
        "tensor_train",
        "cp_puzzle",
        "tucker_puzzle",
        "hosvd_puzzle",
        "tensor_train_puzzle",
        "puzzle_tensor",
        "invert_puzzle_tensor",
        "analyze_decomposition",
        "compare_methods",
        "compare_methods_graph",
        "benchmark_methods_graph",
        "compression_methods_graph",
        "error_methods_graph",
        "time_methods_graph",
        "benchmark_algorithm",
        "reconstruct_tensor",
        "parse_tensor_input",
    ]
    for export_name in expected_exports:
        assert hasattr(td, export_name), f"Missing export: {export_name}"


def test_parse_tensor_input():
    arr = td.parse_tensor_input("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]")
    assert arr.shape == (2, 2, 2)
    assert arr[0, 1, 0] == 3.0


def test_cp_decomposition():
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    result = td.cp(tensor, rank=2)
    assert result["method"] == "cp"
    assert len(result["factors"]) == 3
    assert result["factors"][0].shape == (2, 2)
    assert result["factors"][1].shape == (3, 2)
    assert result["factors"][2].shape == (4, 2)


def test_tucker_decomposition():
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    result = td.tucker(tensor, ranks=[2, 2, 2])
    assert result["method"] == "tucker"
    assert result["core"].shape == (2, 2, 2)
    assert len(result["factors"]) == 3


def test_hosvd_and_reconstruction():
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    result = td.hosvd(tensor)
    assert result["method"] == "hosvd"
    reconstructed = td.reconstruct_tensor("hosvd", result)
    assert reconstructed.shape == tensor.shape
    np.testing.assert_allclose(reconstructed, tensor, rtol=1e-4, atol=1e-4)


def test_tensor_train():
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    result = td.tensor_train(tensor, max_rank=3)
    assert result["method"] == "tensor_train"
    assert len(result["cores"]) == 3
    reconstructed = td.reconstruct_tensor("tensor_train", result)
    assert reconstructed.shape == tensor.shape


def test_puzzle_tensor_roundtrip():
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    shifted, shifts = td.puzzle_tensor(tensor, max_shift=2, max_iter=3, return_shifts=True)
    assert shifted.shape == tensor.shape
    restored = td.invert_puzzle_tensor(shifted, shifts)
    np.testing.assert_allclose(restored, tensor, rtol=1e-5, atol=1e-5)


def test_puzzle_tensor_subblock_roundtrip():
    tensor = np.arange(64, dtype=float).reshape(4, 4, 4)
    shifted, shifts = td.puzzle_tensor(tensor, max_shift=1, block_size=2, return_shifts=True)
    assert shifted.shape == tensor.shape
    restored = td.invert_puzzle_tensor(shifted, shifts)
    np.testing.assert_allclose(restored, tensor, rtol=1e-5, atol=1e-5)



def test_puzzle_augmented_methods():
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    algos = ["cp_puzzle", "tucker_puzzle", "hosvd_puzzle", "tensor_train_puzzle"]
    for algo in algos:
        result = td.run_algorithm(tensor, algo)
        assert result["method"] == algo
        assert "shifts" in result
        assert result.get("is_puzzle") is True
        analysis = td.analyze_decomposition(tensor, algo, result)
        assert analysis["compression_ratio"] > 0
        assert "root_mean_squared_error" in analysis


def test_compare_methods():
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    comp = td.compare_methods(tensor, ["cp", "tucker", "hosvd", "tensor_train"])
    assert len(comp) == 4
    for row in comp:
        assert "compression_ratio" in row
        assert "relative_error" in row
        assert "execution_time_ms" in row


def test_compare_methods_graph():
    pytest.importorskip("matplotlib")
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    figure = td.compare_methods_graph(tensor, ["hosvd"], show=False)
    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Execution time"
    assert figure.axes[1].get_title() == "Relative reconstruction error"
    figure.clf()


def test_separate_method_graphs():
    pytest.importorskip("matplotlib")
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
    for graph_function, expected_axes in (
        (td.benchmark_methods_graph, 3),
        (td.compression_methods_graph, 1),
        (td.error_methods_graph, 1),
        (td.time_methods_graph, 1),
    ):
        figure = graph_function(tensor, ["hosvd"], show=False)
        assert len(figure.axes) == expected_axes
        figure.clf()


def test_mode_and_multilinear_ranks():
    tensor = np.arange(24, dtype=float).reshape(2, 3, 4)

    # 1. CP with single scalar rank and bounds
    cp_res1 = td.cp(tensor, rank=3)
    assert cp_res1["rank"] == 3
    assert cp_res1["min_rank"] == 1
    assert cp_res1["max_rank"] == 24
    for f in cp_res1["factors"]:
        assert f.shape[1] == 3

    # CP disallows multiple mode ranks and enforces single rank R
    cp_res2 = td.cp(tensor, ranks=[2, 3, 2])
    assert cp_res2["rank"] == 2
    assert len(cp_res2["factors"]) == 3
    for f in cp_res2["factors"]:
        assert f.shape[1] == 2

    # 2. Tucker with multilinear ranks and fixed mode bounds
    tucker_res1 = td.tucker(tensor, ranks=[2, 2, 3])
    assert tucker_res1["ranks"] == [2, 2, 3]
    assert tucker_res1["min_ranks"] == [1, 1, 1]
    assert tucker_res1["max_ranks"] == [2, 3, 4]
    assert tucker_res1["core"].shape == (2, 2, 3)
    tucker_res2 = td.tucker(tensor, ranks=2)
    assert tucker_res2["ranks"] == [2, 2, 2]

    # 3. HOSVD with multilinear ranks and fixed mode bounds
    hosvd_res = td.hosvd(tensor, ranks=[2, 2, 2])
    assert hosvd_res["ranks"] == [2, 2, 2]
    assert hosvd_res["min_ranks"] == [1, 1, 1]
    assert hosvd_res["max_ranks"] == [2, 3, 4]
    assert hosvd_res["core"].shape == (2, 2, 2)

    # 4. TT with TT-ranks and fixed bond bounds
    tt_res = td.tensor_train(tensor, ranks=[2, 2])
    assert tt_res["ranks"] == [1, 2, 2, 1]
    assert tt_res["min_ranks"] == [1, 1]
    assert tt_res["max_ranks"] == [2, 4]  # min(2, 12) = 2, min(6, 4) = 4
    assert len(tt_res["cores"]) == 3
    for core in tt_res["cores"]:
        assert core.ndim == 3


def test_complexity_formula_product_of_n():
    shape = (2, 3, 4)
    cp_c = td.get_complexity_formula(shape, "cp")
    assert "∏ N_i" in cp_c
    assert "N₁N₂N₃" not in cp_c

    tt_c = td.get_complexity_formula(shape, "tensor_train")
    assert "∏ N_i" in tt_c
    assert "N₁N₂N₃" not in tt_c

    tucker_c = td.get_complexity_formula(shape, "tucker")
    assert "∏ N_i" in tucker_c
    assert "N₁N₂N₃" not in tucker_c
