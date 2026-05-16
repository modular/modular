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
"""Verifier-level error tests for the underlying rmo.mo.*_pool ops."""

import numpy as np
import pytest
from max._core.dialects import builtin, kgen
from max._core.dialects import rmo as _rmo
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def _i64_const(values: list[int]) -> object:
    return ops.constant(
        np.array(values, dtype=np.int64), DType.int64, DeviceRef.CPU()
    )


def test_mo_max_pool_non_rank1_filter_shape_error() -> None:
    """Constructing rmo.mo.max_pool with a rank-2 filter_shape must raise
    TypeError, not abort. The op's MO_Rank1IndexTensor operand constraint is
    enforced by the verifier and surfaces as a Python exception."""
    x_type = TensorType(DType.float32, [1, 8, 8, 4], device=DeviceRef.CPU())
    result_type = TensorType(
        DType.float32, [1, 4, 4, 4], device=DeviceRef.CPU()
    )
    with Graph("max_pool_bad_filter_shape", input_types=[x_type]) as graph:
        bad_filter_shape = ops.constant(
            np.array([[2, 2], [2, 2]], dtype=np.int64),
            DType.int64,
            DeviceRef.CPU(),
        )
        with pytest.raises(TypeError, match="rank-1 tensor with indices"):
            graph._add_op_generated(
                _rmo.MoMaxPoolOp,
                result_type,
                graph.inputs[0].tensor,
                bad_filter_shape,
                _i64_const([1, 1]),
                _i64_const([1, 1]),
                _i64_const([0, 0, 0, 0]),
                kgen.ParamDeclArrayAttr([]),
            )


def test_mo_avg_pool_float_strides_error() -> None:
    """Constructing rmo.mo.avg_pool with float strides must raise TypeError;
    the MO_Rank1IndexTensor constraint requires integer/index elements."""
    x_type = TensorType(DType.float32, [1, 8, 8, 4], device=DeviceRef.CPU())
    result_type = TensorType(
        DType.float32, [1, 4, 4, 4], device=DeviceRef.CPU()
    )
    with Graph("avg_pool_bad_strides", input_types=[x_type]) as graph:
        bad_strides = ops.constant(
            np.array([1.0, 1.0], dtype=np.float32),
            DType.float32,
            DeviceRef.CPU(),
        )
        with pytest.raises(TypeError, match="rank-1 tensor with indices"):
            graph._add_op_generated(
                _rmo.MoAvgPoolOp,
                result_type,
                graph.inputs[0].tensor,
                _i64_const([2, 2]),
                bad_strides,
                _i64_const([1, 1]),
                _i64_const([0, 0, 0, 0]),
                builtin.BoolAttr(True),  # count_boundary
                kgen.ParamDeclArrayAttr([]),
            )
