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

from max.gpu.host import DeviceContext
from linalg.block_scaled_quantization import (
    block_scales_interleave_fp4,
)
from std.testing import assert_equal
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP4_SF_VECTOR_SIZE,
    MXFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    get_scale_factor,
)
from std.math import ceildiv, align_up


def test_block_scales_interleave_fp4[
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    N: Int,
](ctx: DeviceContext, m: Int, n: Int) raises:
    # Input scales [m, n]; output scales in the interleaved 5D TCGEN layout.
    var input_layout = row_major(m, Idx[N])
    var output_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(n, SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var input_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        input_layout.size()
    )
    var output_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        output_layout.size()
    )

    with input_scales_device.map_to_host() as input_host:
        var input_host_tensor = TileTensor(input_host, input_layout)
        random(input_host_tensor)

    with output_scales_device.map_to_host() as output_host:
        for i in range(len(output_host)):
            output_host[i] = 0

    var input_scales_tensor = TileTensor(input_scales_device, input_layout)
    var output_scales_tensor = TileTensor(output_scales_device, output_layout)

    block_scales_interleave_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        input_scales_tensor.as_unsafe_any_origin(),
        output_scales_tensor.as_unsafe_any_origin(),
    )

    ctx.synchronize()

    with input_scales_device.map_to_host() as input_host:
        with output_scales_device.map_to_host() as output_host:
            var input_host_tensor = TileTensor(input_host, input_layout)
            var output_host_tensor = TileTensor(output_host, output_layout)

            for row_idx in range(0, align_up(m, SF_MN_GROUP_SIZE)):
                for col_idx in range(0, align_up(n, SF_ATOM_K)):
                    var swizzled_sf = get_scale_factor[
                        SF_VECTOR_SIZE=SF_VECTOR_SIZE
                    ](
                        output_host_tensor.as_unsafe_any_origin(),
                        row_idx,
                        col_idx * SF_VECTOR_SIZE,
                    )
                    if row_idx < m and col_idx < n:
                        var ref_sf = input_host_tensor[row_idx, col_idx]
                        assert_equal(
                            ref_sf.cast[.float64](),
                            swizzled_sf.cast[.float64](),
                        )
                    else:
                        # Compare against the dtype's stored zero representation
                        # instead of raw Float64(0.0). MXFP4's E8M0 scale format
                        # round-trips a different exact bit pattern here.
                        var zero_sf = Scalar[scales_dtype](0.0)
                        assert_equal(
                            zero_sf.cast[.float64](),
                            swizzled_sf.cast[.float64](),
                        )


def main() raises:
    with DeviceContext() as ctx:
        test_block_scales_interleave_fp4[
            NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE, N=4
        ](ctx, 128, 4)
        test_block_scales_interleave_fp4[
            NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE, N=4
        ](ctx, 129, 4)
        test_block_scales_interleave_fp4[
            NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE, N=5
        ](ctx, 129, 5)
        test_block_scales_interleave_fp4[
            NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE, N=1024
        ](ctx, 1024, 1024)
        test_block_scales_interleave_fp4[
            NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE, N=3328
        ](ctx, 16384, 3328)
        test_block_scales_interleave_fp4[
            NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE, N=1024
        ](ctx, 53248, 1024)
        test_block_scales_interleave_fp4[
            MXFP4_SF_DTYPE, MXFP4_SF_VECTOR_SIZE, N=4
        ](ctx, 128, 4)
        test_block_scales_interleave_fp4[
            MXFP4_SF_DTYPE, MXFP4_SF_VECTOR_SIZE, N=4
        ](ctx, 129, 4)
        test_block_scales_interleave_fp4[
            MXFP4_SF_DTYPE, MXFP4_SF_VECTOR_SIZE, N=5
        ](ctx, 129, 5)
        test_block_scales_interleave_fp4[
            MXFP4_SF_DTYPE, MXFP4_SF_VECTOR_SIZE, N=1024
        ](ctx, 1024, 1024)
        test_block_scales_interleave_fp4[
            MXFP4_SF_DTYPE, MXFP4_SF_VECTOR_SIZE, N=3328
        ](ctx, 16384, 3328)
        test_block_scales_interleave_fp4[
            MXFP4_SF_DTYPE, MXFP4_SF_VECTOR_SIZE, N=1024
        ](ctx, 53248, 1024)
