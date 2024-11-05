# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the Python weight loading interface."""


import platform
import pytest
import numpy as np
import torch
from max.dtype import DType
from max.graph import Graph, Weight
from max.graph.weights import GGUFWeights, PytorchWeights, SafetensorWeights


def test_weight(session) -> None:
    """Tests adding an external weight to a graph."""
    with Graph("graph_with_weights") as graph:
        weight_shape = [5, 10]
        weight = np.random.uniform(1, 100, size=weight_shape).astype(np.int64)
        w = Weight(
            "random_weight",
            dtype=DType.int64,
            shape=weight_shape,
        )
        graph.output(graph.add_weight(w))
        compiled = session.load(
            graph, weights_registry={"random_weight": weight}
        )
        output = compiled.execute_legacy()

        np.testing.assert_array_equal(weight, output["output0"])


def test_weight_offset(session) -> None:
    """Tests adding an external weight to a graph."""
    with Graph("graph_with_offset_weights") as graph:
        weight_shape = [5, 10]
        weight = np.random.uniform(1, 100, size=weight_shape).astype(np.int64)
        w = Weight(
            "random_weight",
            dtype=DType.int64,
            shape=weight_shape,
        )
        graph.output(graph.add_weight(w))
        compiled = session.load(
            graph, weights_registry={"random_weight": weight}
        )
        output = compiled.execute_legacy()

        np.testing.assert_array_equal(weight, output["output0"])


def _test_data():
    return {
        "a": np.arange(10, dtype=np.int32).reshape(5, 2),
        "b": np.full((1, 2, 3), 3.5, dtype=np.float64),
        "c": np.array(5432.1, dtype=np.float32),
        "fancy/name": np.array([1, 2, 3], dtype=np.int64),
        # This is actually saved as bf16 in gen_external_checkpoints.
        "bf16": torch.tensor([123, 45], dtype=torch.bfloat16),
    }


@pytest.mark.skipif(
    platform.machine() in ["arm64", "aarch64"],
    reason="BF16 is not supported on ARM CPU architecture",
)
def test_load_pytorch(session, graph_testdata) -> None:
    """Tests adding an external weight to a graph."""
    expected_dict = _test_data()
    flat_keys = list(expected_dict.keys())
    expected = [expected_dict[k] for k in flat_keys]

    weights = PytorchWeights(graph_testdata / "example_data.pt")
    with Graph("graph_with_pt_weights") as graph:
        loaded = {k: graph.add_weight(w.allocate()) for k, w in weights.items()}
        graph.output(*[loaded[k] for k in flat_keys])
        compiled = session.load(
            graph, weights_registry=weights.allocated_weights
        )
        output = compiled.execute()

        assert len(expected) == len(output)
        for n, expected in enumerate(expected):
            if flat_keys[n] == "bf16":
                assert torch.equal(expected, torch.from_dlpack(output[n]))
            else:
                np.testing.assert_array_equal(expected, output[n].to_numpy())


@pytest.mark.skipif(
    platform.machine() in ["arm64", "aarch64"],
    reason="BF16 is not supported on ARM CPU architecture",
)
def test_load_gguf(session, graph_testdata) -> None:
    """Tests adding an external weight to a graph."""
    expected_dict = _test_data()
    expected_dict["quantized"] = np.arange(0, 288, dtype=np.uint8).reshape(
        2, 144
    )

    flat_keys = list(expected_dict.keys())
    expected = [expected_dict[k] for k in flat_keys]

    weights = GGUFWeights(graph_testdata / "example_data.gguf")
    with Graph("graph_with_gguf_weights") as graph:
        loaded = {k: graph.add_weight(w.allocate()) for k, w in weights.items()}
        graph.output(*[loaded[k] for k in flat_keys])
        compiled = session.load(
            graph, weights_registry=weights.allocated_weights
        )
        output = compiled.execute()

        assert len(expected) == len(output)
        for n, expected in enumerate(expected):
            if flat_keys[n] == "bf16":
                assert torch.equal(expected, torch.from_dlpack(output[n]))
            else:
                np.testing.assert_array_equal(expected, output[n].to_numpy())


@pytest.mark.skipif(
    platform.machine() in ["arm64", "aarch64"],
    reason="BF16 is not supported on ARM CPU architecture",
)
def test_load_safetensors(session, graph_testdata) -> None:
    """Tests adding an external weight to a graph."""
    expected_base_dict = _test_data()
    expected_dict = {
        f"{i}.{k}": v
        for k, v in expected_base_dict.items()
        for i in range(1, 3)
    }
    flat_keys = list(expected_dict.keys())
    expected = [expected_dict[k] for k in flat_keys]

    weights = SafetensorWeights(
        [graph_testdata / f"example_data_{i}.safetensors" for i in range(1, 3)]
    )
    with Graph("graph_with_pt_weights") as graph:
        loaded = {k: graph.add_weight(w.allocate()) for k, w in weights.items()}
        graph.output(*[loaded[k] for k in flat_keys])
        compiled = session.load(
            graph, weights_registry=weights.allocated_weights
        )
        output = compiled.execute()
        assert len(expected) == len(output)
        for n, expected in enumerate(expected):
            if flat_keys[n].endswith("bf16"):
                assert torch.equal(expected, torch.from_dlpack(output[n]))
            else:
                np.testing.assert_array_equal(expected, output[n].to_numpy())
