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
"""Test the max.graph Python bindings."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from max._core.dialects import kgen
from max._core.dialects import mo as _mo
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

shared_dtypes = st.shared(st.from_type(DType))


@given(input_type=...)
def test_nonzero(input_type: TensorType) -> None:
    with Graph("nonzero ", input_types=[input_type]) as graph:
        out = ops.nonzero(graph.inputs[0].tensor, "nonzero")
        assert out.dtype == DType.int64
        assert out.shape == ["nonzero", input_type.rank]
        graph.output(out)


@given(dtype=shared_dtypes)
def test_nonzero_scalar_error(dtype: DType) -> None:
    """Test that nonzero raises an error with a scalar input for any dtype."""
    scalar_type = TensorType(dtype, [], device=DeviceRef.CPU())
    with Graph("nonzero_scalar", input_types=[scalar_type]) as graph:
        with pytest.raises(ValueError, match="Scalar inputs not supported"):
            ops.nonzero(graph.inputs[0].tensor, "nonzero")


def test_nonzero_rank3_result_error() -> None:
    """Constructing mo.arg_nonzero with a non-rank-2 result must raise
    ValueError, not abort: the verifier's error surfaces as a Python
    exception."""
    input_type = TensorType(DType.float32, [4, 4], device=DeviceRef.CPU())
    with Graph("nonzero_bad_result", input_types=[input_type]) as graph:
        bad_result_type = TensorType(
            DType.int64, ["count", 2, 1], device=DeviceRef.CPU()
        )
        with pytest.raises(ValueError, match="rank must be 2"):
            graph._add_op_generated(
                _mo.ArgNonzeroOp,
                bad_result_type,
                graph.inputs[0].tensor,
                kgen.ParamDeclArrayAttr([]),
            )


def test_nonzero_non_si64_result_error() -> None:
    """Constructing mo.arg_nonzero with a non-si64 result must raise
    ValueError."""
    input_type = TensorType(DType.float32, [4, 4], device=DeviceRef.CPU())
    with Graph("nonzero_bad_dtype", input_types=[input_type]) as graph:
        bad_result_type = TensorType(
            DType.float32, ["count", 2], device=DeviceRef.CPU()
        )
        with pytest.raises(ValueError, match="signed 64 integer element type"):
            graph._add_op_generated(
                _mo.ArgNonzeroOp,
                bad_result_type,
                graph.inputs[0].tensor,
                kgen.ParamDeclArrayAttr([]),
            )
