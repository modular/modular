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

from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random, zero
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.fp4_quantization import (
    block_scales_interleave_fp4,
)
from testing import assert_equal
from buffer import Dim
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    _get_scale_factor,
)
from math import ceildiv, align_up


fn test_block_scales_interleave_fp4[
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim) raises:
    alias input_scales_static_shape = DimList(m.dim, n.dim)
    var input_scales_dynamic_shape = DimList(m.value, n.value)

    var input_scales_host = HostNDBuffer[
        scales_dtype, 2, input_scales_static_shape
    ](input_scales_dynamic_shape)
    var input_scales_device = DeviceNDBuffer[
        scales_dtype, 2, input_scales_static_shape
    ](input_scales_dynamic_shape, ctx=ctx)

    random(input_scales_host.tensor)

    comptime output_scales_shape_static = DimList(
        ceildiv(m.dim, SF_MN_GROUP_SIZE),
        ceildiv(n.dim, SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var output_scales_shape_dynamic = DimList(
        ceildiv(m.value, SF_MN_GROUP_SIZE),
        ceildiv(n.value, SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var output_scales_host = HostNDBuffer[
        scales_dtype, 5, output_scales_shape_static
    ](output_scales_shape_dynamic)
    var output_scales_device = DeviceNDBuffer[
        scales_dtype, 5, output_scales_shape_static
    ](output_scales_shape_dynamic, ctx=ctx)

    ctx.enqueue_copy(input_scales_device.buffer, input_scales_host.tensor.data)

    var output_scales_tensor = from_ndbuffer_row_major(
        output_scales_device.tensor
    )
    var input_scales_tensor = from_ndbuffer_row_major(
        input_scales_device.tensor
    )

    block_scales_interleave_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        input_scales_tensor,
        output_scales_tensor,
    )

    ctx.enqueue_copy(
        output_scales_host.tensor.data, output_scales_device.buffer
    )
    ctx.synchronize()

    var output_scales_tensor_host = from_ndbuffer_row_major(
        output_scales_host.tensor
    )
    var input_scales_tensor_host = from_ndbuffer_row_major(
        input_scales_host.tensor
    )

    for row_idx in range(0, align_up(m.value, SF_MN_GROUP_SIZE)):
        for col_idx in range(0, align_up(n.value, SF_ATOM_K)):
            var swizzled_sf = _get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                output_scales_tensor_host, row_idx, col_idx * SF_VECTOR_SIZE
            )
            if row_idx < m.value and col_idx < n.value:
                var ref_sf = rebind[Scalar[scales_dtype]](
                    input_scales_tensor_host[row_idx, col_idx]
                )
                assert_equal(
                    ref_sf.cast[DType.float64](),
                    swizzled_sf.cast[DType.float64](),
                )
            else:
                assert_equal(Float64(0.0), swizzled_sf.cast[DType.float64]())


def main():
    with DeviceContext() as ctx:
        test_block_scales_interleave_fp4[NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE](
            ctx, dynamic(128), static[4]()
        )
        test_block_scales_interleave_fp4[NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE](
            ctx, dynamic(129), static[4]()
        )
        test_block_scales_interleave_fp4[NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE](
            ctx, dynamic(129), static[4 + 1]()
        )
        test_block_scales_interleave_fp4[NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE](
            ctx, dynamic(1024), static[1024]()
        )
        test_block_scales_interleave_fp4[NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE](
            ctx, dynamic(16384), static[3328]()
        )
        test_block_scales_interleave_fp4[NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE](
            ctx, dynamic(53248), static[1024]()
        )
