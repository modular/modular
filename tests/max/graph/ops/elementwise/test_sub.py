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
"""Test the max.graph Python bindings."""

from conftest import broadcast_shapes, broadcastable_tensor_types
from hypothesis import assume, event, given
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, TensorType
from max.graph.ops import sub


@given(tensor_type=...)
def test_sub__same_type(tensor_type: TensorType) -> None:
    with Graph("sub", input_types=[tensor_type, tensor_type]) as graph:
        op = sub(graph.inputs[0].tensor, graph.inputs[1].tensor)
        assert op.type == tensor_type


@given(tensor_type=...)
def test_sub__same_type__operator(tensor_type: TensorType) -> None:
    with Graph("sub", input_types=[tensor_type, tensor_type]) as graph:
        op = graph.inputs[0].tensor - graph.inputs[1].tensor
        assert op.type == tensor_type


@given(d1=..., d2=..., shape=...)
def test_sub__promoted_dtype__operator(
    d1: DType, d2: DType, shape: list[Dim]
) -> None:
    assume(d1 != d2)
    t1 = TensorType(d1, shape, device=DeviceRef.CPU())
    t2 = TensorType(d2, shape, device=DeviceRef.CPU())
    with Graph("sub", input_types=[t1, t2]) as graph:
        i0, i1 = graph.inputs[0].tensor, graph.inputs[1].tensor
        try:
            assert (i0 - i1).dtype in (d1, d2)
            assert (i1 - i0).dtype in (d1, d2)
            assert (i0 - i1).type == (i1 - i0).type
            event("types promote")
        except ValueError as e:
            assert "Unsafe cast" in str(e)
            event("types don't promote")


@given(types=broadcastable_tensor_types(2))
def test_sub__broadcast__operator(types: list[TensorType]) -> None:
    t1, t2 = types
    broadcast_shape = broadcast_shapes(t1.shape, t2.shape)
    with Graph("sub", input_types=[t1, t2]) as graph:
        i0, i1 = graph.inputs[0].tensor, graph.inputs[1].tensor
        assert (i0 - i1).shape == broadcast_shape
        assert (i1 - i0).shape == broadcast_shape


@given(tensor_type=...)
def test_sub__python_int__operator(tensor_type: TensorType) -> None:
    with Graph("sub", input_types=[tensor_type, tensor_type]) as graph:
        op = graph.inputs[0].tensor - 1
        assert op.type == tensor_type
