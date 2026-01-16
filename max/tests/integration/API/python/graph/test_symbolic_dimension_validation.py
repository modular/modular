# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Tests for symbolic dimension validation at runtime."""

from typing import Any

import numpy as np
import pytest
import torch.utils.dlpack
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, ops


def to_numpy(x: Any) -> np.ndarray:
    """Convert model output to numpy array."""
    if isinstance(x, Tensor):
        return x.to_numpy()
    return torch.utils.dlpack.from_dlpack(x).numpy()


def test_same_symbolic_dimension_mismatch() -> None:
    """Test that two inputs sharing the same symbolic dimension must have
    matching concrete values.

    This graph has two inputs with shape ["seq_len", 768]. When executed with
    different values for seq_len (e.g., input0 has seq_len=10, input1 has
    seq_len=12), the engine should return an error.
    """
    input_type = TensorType(
        dtype=DType.float32, shape=("seq_len", 768), device=DeviceRef.CPU()
    )

    with Graph("same_symbolic", input_types=(input_type, input_type)) as graph:
        result = graph.inputs[0].tensor + graph.inputs[1].tensor
        graph.output(result)

    cpu = CPU()
    session = InferenceSession(devices=[cpu])
    model = session.load(graph)

    # Create inputs with different seq_len values (10 vs 12)
    input0 = Tensor.from_numpy(np.zeros((10, 768), dtype=np.float32)).to(cpu)
    input1 = Tensor.from_numpy(np.zeros((12, 768), dtype=np.float32)).to(cpu)

    with pytest.raises(Exception, match="symbolic dimension"):
        model(input0, input1)


def test_algebraic_dimension_fails_constraints() -> None:
    """Test that algebraic dimension constraints are validated at runtime.

    This graph reshapes input [batch, dim] -> [batch, 4, dim // 4].
    When executed with dim=63 (not divisible by 4), the engine should return
    an error.
    """
    input_type = TensorType(
        dtype=DType.float32, shape=("batch", "dim"), device=DeviceRef.CPU()
    )

    with Graph("algebraic", input_types=(input_type,)) as graph:
        x = graph.inputs[0].tensor
        x = x.rebind(["batch", 4 * (Dim("dim") // 4)])
        reshaped = ops.reshape(x, (Dim("batch"), 4, Dim("dim") // 4))
        graph.output(reshaped)

    cpu = CPU()
    session = InferenceSession(devices=[cpu])
    model = session.load(graph)

    # Create input with dim=63 (not divisible by 4)
    input_data = Tensor.from_numpy(np.zeros((8, 63), dtype=np.float32)).to(cpu)

    with pytest.raises(Exception, match="rebind shape check failed"):
        model(input_data)


def test_algebraic_dimension_meets_constraints() -> None:
    """Test that algebraic dimension constraints pass when met.

    This is a sanity check that the graph works correctly when the constraints
    are satisfied (dim=64 is divisible by 4).
    """
    input_type = TensorType(
        dtype=DType.float32, shape=("batch", "dim"), device=DeviceRef.CPU()
    )

    with Graph("algebraic", input_types=(input_type,)) as graph:
        x = graph.inputs[0].tensor
        x = x.rebind(["batch", 4 * (Dim("dim") // 4)])
        reshaped = ops.reshape(x, (Dim("batch"), 4, Dim("dim") // 4))
        graph.output(reshaped)

    cpu = CPU()
    session = InferenceSession(devices=[cpu])
    model = session.load(graph)

    # Create input with dim=64 (divisible by 4)
    input_data = Tensor.from_numpy(np.zeros((8, 64), dtype=np.float32)).to(cpu)

    results = model(input_data)
    result_np = to_numpy(results[0])

    assert result_np.shape == (8, 4, 16)


def test_same_symbolic_dimension_matches() -> None:
    """Test that two inputs sharing the same symbolic dimension work when
    the concrete values match.

    This is a sanity check that the graph works correctly when both inputs
    have the same seq_len value.
    """
    input_type = TensorType(
        dtype=DType.float32, shape=("seq_len", 768), device=DeviceRef.CPU()
    )

    with Graph("same_symbolic", input_types=(input_type, input_type)) as graph:
        result = graph.inputs[0].tensor + graph.inputs[1].tensor
        graph.output(result)

    cpu = CPU()
    session = InferenceSession(devices=[cpu])
    model = session.load(graph)

    # Create inputs with the same seq_len value (10)
    input0 = Tensor.from_numpy(np.ones((10, 768), dtype=np.float32)).to(cpu)
    input1 = Tensor.from_numpy(np.ones((10, 768), dtype=np.float32) * 2).to(cpu)

    results = model(input0, input1)
    result_np = to_numpy(results[0])

    assert result_np.shape == (10, 768)
    np.testing.assert_allclose(result_np, np.ones((10, 768)) * 3)
