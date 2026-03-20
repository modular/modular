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
"""Tests for the block_scaled_matmul_with_epilogue public API."""

from std.math import ceildiv

from std.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from linalg.fp4_quantization import block_scaled_matmul_with_epilogue
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from std.utils.index import IndexList


def test_block_scaled_matmul_zero_rows(ctx: DeviceContext) raises:
    """Test that block_scaled_matmul_with_epilogue handles zero-row inputs."""
    var m = 0
    var n = 128
    var k = 128
    var k_packed = k // 2

    var a_shape = IndexList[2](m, k_packed)
    var b_shape = IndexList[2](n, k_packed)
    var c_shape = IndexList[2](m, n)

    comptime a_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime b_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime c_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var a_runtime_layout = RuntimeLayout[a_layout].row_major(a_shape)
    var b_runtime_layout = RuntimeLayout[b_layout].row_major(b_shape)
    var c_runtime_layout = RuntimeLayout[c_layout].row_major(c_shape)

    var a_device = ctx.enqueue_create_buffer[DType.uint8](
        a_shape.flattened_length()
    )
    var b_device = ctx.enqueue_create_buffer[DType.uint8](
        b_shape.flattened_length()
    )
    var c_device = ctx.enqueue_create_buffer[DType.bfloat16](
        c_shape.flattened_length()
    )

    var a_tensor = LayoutTensor[DType.uint8, a_layout](
        a_device, a_runtime_layout
    )
    var b_tensor = LayoutTensor[DType.uint8, b_layout](
        b_device, b_runtime_layout
    )
    var c_tensor = LayoutTensor[DType.bfloat16, c_layout](
        c_device, c_runtime_layout
    )

    var a_scales_dim0 = ceildiv(m, SF_MN_GROUP_SIZE)
    var a_scales_dim1 = ceildiv(k, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    var b_scales_dim0 = ceildiv(n, SF_MN_GROUP_SIZE)
    var b_scales_dim1 = ceildiv(k, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)

    var a_scales_total = (
        a_scales_dim0 * a_scales_dim1 * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    var b_scales_total = (
        b_scales_dim0 * b_scales_dim1 * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )

    var a_scales_device = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](
        max(a_scales_total, 1)
    )
    var b_scales_device = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](
        max(b_scales_total, 1)
    )

    var a_scales_tensor = TileTensor(
        a_scales_device,
        row_major(
            (
                Idx(a_scales_dim0),
                Idx(a_scales_dim1),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1]](),
                Idx[SF_ATOM_K](),
            )
        ),
    )
    var b_scales_tensor = TileTensor(
        b_scales_device,
        row_major(
            (
                Idx(b_scales_dim0),
                Idx(b_scales_dim1),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1]](),
                Idx[SF_ATOM_K](),
            )
        ),
    )

    block_scaled_matmul_with_epilogue[
        SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        transpose_b=True,
    ](
        c_tensor.as_any_origin(),
        a_tensor.as_any_origin(),
        b_tensor.as_any_origin(),
        a_scales_tensor,
        b_scales_tensor,
        1.0,
        ctx,
    )
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        test_block_scaled_matmul_zero_rows(ctx)
    print("\n=== ALL TESTS PASSED ===\n")
