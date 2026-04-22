# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for max.nn.Module."""

import numpy as np
import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, SymbolicDim, Weight
from max.nn import Module


class TestLayer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Weight("weight", DType.float32, [10], DeviceRef.CPU())

    def __call__(self):
        return self.weight


class TestModel(Module):
    def __init__(self) -> None:
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
    np.testing.assert_array_equal(outputs.to_numpy(), expected_weight * 2)


def test_load_state_dict(session: InferenceSession) -> None:
    module = TestModel()
    weight = np.random.uniform(size=[10]).astype(np.float32)
    assert weight.flags.aligned
    module.load_state_dict({"layer.weight": weight})
    state_dict = module.state_dict()
    np.testing.assert_array_equal(weight, state_dict["layer.weight"])

    graph = Graph("load_state_dict", module)
    model = session.load(graph, weights_registry=state_dict)
    outputs = model()[0]
    np.testing.assert_array_equal(outputs.to_numpy(), weight * 2)


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
    np.testing.assert_array_equal(outputs.to_numpy(), weight * 2)


def test_load_state_dict_with_symbolic_dimensions() -> None:
    """Test that load_state_dict works with weights that have symbolic dimensions."""

    class SymbolicModule(Module):
        def __init__(self) -> None:
            super().__init__()
            batch = SymbolicDim("batch")
            self.weight = Weight(
                "weight", DType.float32, (batch, 10), DeviceRef.CPU()
            )

        def __call__(self):
            return self.weight

    module = SymbolicModule()

    # Test loading with concrete batch size - should work
    weight_array = np.random.uniform(size=[32, 10]).astype(np.float32)
    module.load_state_dict({"weight": weight_array})

    # Verify the weight was loaded correctly
    state_dict = module.state_dict()
    np.testing.assert_array_equal(weight_array, state_dict["weight"])


def test_load_state_dict_symbolic_validation_errors() -> None:
    """Test that validation errors work correctly with symbolic dimensions."""

    class SymbolicModule(Module):
        def __init__(self) -> None:
            super().__init__()
            batch = SymbolicDim("batch")
            # Static, symbolic, static pattern
            self.weight = Weight(
                "weight", DType.float32, (1024, batch, 256), DeviceRef.CPU()
            )

        def __call__(self):
            return self.weight

    module = SymbolicModule()

    # Wrong static dimension should fail
    wrong_static = np.random.uniform(size=[512, 32, 256]).astype(np.float32)
    with pytest.raises(ValueError, match="expected 1024, got 512"):
        module.load_state_dict({"weight": wrong_static})

    # Wrong rank should fail
    wrong_rank = np.random.uniform(size=[1024, 32]).astype(np.float32)
    with pytest.raises(ValueError, match="shape rank"):
        module.load_state_dict({"weight": wrong_rank})

    # Correct shape should work
    correct_array = np.random.uniform(size=[1024, 32, 256]).astype(np.float32)
    module.load_state_dict({"weight": correct_array})
