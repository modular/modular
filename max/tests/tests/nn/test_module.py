# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for max.nn.Module."""

import numpy as np
import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, Weight
from max.nn import Module


class TestLayer(Module):
    def __init__(self):
        super().__init__()
        self.weight = Weight("weight", DType.float32, [10])

    def __call__(self):
        return self.weight


class TestModel(Module):
    def __init__(self):
        super().__init__()
        self.layer = TestLayer()

    def __call__(self):
        return self.layer() * 2


def test_state_dict(session: InferenceSession) -> None:
    module = TestModel()
    state_dict = module.state_dict()
    assert "layer.weight" in state_dict

    # The weight should be initialized to zeros.
    expected_weight = np.zeros([10], dtype=np.float32)
    np.testing.assert_array_equal(
        expected_weight,
        state_dict["layer.weight"].to_numpy(),  # type: ignore
    )

    graph = Graph("initialize_state_dict", module)
    model = session.load(graph, weights_registry=state_dict)
    outputs = model()[0]
    np.testing.assert_array_equal(outputs.to_numpy(), expected_weight * 2)  # type: ignore


def test_load_state_dict(session: InferenceSession) -> None:
    module = TestModel()
    weight = np.random.uniform(size=[10]).astype(np.float32)
    assert weight.flags.aligned
    module.load_state_dict({"layer.weight": weight})
    state_dict = module.state_dict()
    np.testing.assert_array_equal(weight, state_dict["layer.weight"])  # type: ignore

    graph = Graph("load_state_dict", module)
    model = session.load(graph, weights_registry=state_dict)
    outputs = model()[0]
    np.testing.assert_array_equal(outputs.to_numpy(), weight * 2)  # type: ignore


def test_load_state_dict_with_unaligned_weights(
    session: InferenceSession,
) -> None:
    # Create an unaligned numpy array.
    weight = np.arange(10, dtype=np.float32)
    unaligned_weight = np.array(
        [15] + weight.view(np.uint8).tolist(),
        np.uint8,
    )[1:].view(np.float32)
    assert not unaligned_weight.flags.aligned

    module = TestModel()
    with pytest.raises(ValueError, match="Found unaligned weight"):
        module.load_state_dict({"layer.weight": unaligned_weight})

    # Module should be able to load weights with `weight_alignment=1`.
    module.load_state_dict(
        {"layer.weight": unaligned_weight}, weight_alignment=1
    )
    graph = Graph("load_state_dict_unaligned", module)
    model = session.load(graph, weights_registry=module.state_dict())
    outputs = model()[0]
    np.testing.assert_array_equal(outputs.to_numpy(), weight * 2)  # type: ignore
