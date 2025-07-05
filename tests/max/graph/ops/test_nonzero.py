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

import pytest
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

shared_dtypes = st.shared(st.from_type(DType))


@given(input_type=...)
def test_nonzero(input_type: TensorType) -> None:
    with Graph("nonzero ", input_types=[input_type]) as graph:
        out = ops.nonzero(graph.inputs[0], "nonzero")
        assert out.dtype == DType.int64
        assert out.shape == ["nonzero", input_type.rank]
        graph.output(out)


@given(dtype=shared_dtypes)
def test_nonzero_scalar_error(dtype: DType) -> None:
    """Test that nonzero raises an error with a scalar input for any dtype."""
    scalar_type = TensorType(dtype, [], device=DeviceRef.CPU())
    with Graph("nonzero_scalar", input_types=[scalar_type]) as graph:
        with pytest.raises(ValueError, match="Scalar inputs not supported"):
            ops.nonzero(graph.inputs[0], "nonzero")
