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
"""test the max.graph python bindings."""

from collections.abc import Callable

import pytest
from conftest import GraphBuilder, dtypes, float_dtypes, tensor_types
from hypothesis import given
from max.graph import TensorType, TensorValue
from max.graph.ops import ceil, cos, floor, round, rsqrt, sin, sqrt, tanh, trunc

# atanh is covered in test_atanh.py.
FLOAT_ONLY_UNARY_OPS = [ceil, cos, floor, round, rsqrt, sin, sqrt, tanh, trunc]


@given(tensor_type=tensor_types(dtypes=float_dtypes()))
def test_ceil_same_type(
    graph_builder: GraphBuilder, tensor_type: TensorType
) -> None:
    """ceil preserves the input shape, dtype, and device."""
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        op = ceil(x)
        assert op.type == tensor_type
        graph.output(op)


@pytest.mark.parametrize("op", FLOAT_ONLY_UNARY_OPS)
@given(tensor_type=tensor_types(dtypes=float_dtypes()))
def test_float_only_unary_same_type(
    graph_builder: GraphBuilder,
    op: Callable[[TensorValue], TensorValue],
    tensor_type: TensorType,
) -> None:
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        assert op(x).type == tensor_type


@pytest.mark.parametrize("op", FLOAT_ONLY_UNARY_OPS)
@given(
    tensor_type=tensor_types(dtypes=dtypes.filter(lambda d: not d.is_float()))
)
def test_float_only_unary_rejects_non_float(
    graph_builder: GraphBuilder,
    op: Callable[[TensorValue], TensorValue],
    tensor_type: TensorType,
) -> None:
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        with pytest.raises(ValueError, match="requires a floating-point input"):
            op(x)
