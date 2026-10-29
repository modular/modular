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
from layout import Idx, TileTensor, row_major
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


def test_block_scales_interleave_fp4_zero_rows[
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    N: Int,
](ctx: DeviceContext, m: Int, n: Int) raises:
    """A DP replica with no rows this step hands the interleave a zero-row
    tile; the call returns without launching and the output keeps its
    sentinel. The buffers are sized for one row and one tile so the device
    allocations are real; the views are zero-row. `m` arrives as a runtime
    value so the views carry the same layout types as the non-empty
    cases."""
    var input_layout = row_major(m, Idx[N])
    var output_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(n, SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var tile_elems = (
        ceildiv(n, SF_ATOM_K) * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    var input_scales_device = ctx.enqueue_create_buffer[scales_dtype](n)
    var output_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        tile_elems
    )
    # The reference buffer is filled the same way and never handed to the
    # kernel, so the comparison below catches any write to the output.
    var reference_device = ctx.enqueue_create_buffer[scales_dtype](tile_elems)
    with output_scales_device.map_to_host() as output_host:
        for i in range(len(output_host)):
            output_host[i] = 1
    with reference_device.map_to_host() as reference_host:
        for i in range(len(reference_host)):
            reference_host[i] = 1

    var input_scales_tensor = TileTensor(input_scales_device, input_layout)
    var output_scales_tensor = TileTensor(output_scales_device, output_layout)

    block_scales_interleave_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        input_scales_tensor.as_unsafe_any_origin(),
        output_scales_tensor.as_unsafe_any_origin(),
    )

    ctx.synchronize()

    var bad = 0
    with output_scales_device.map_to_host() as output_host, reference_device.map_to_host() as reference_host:
        var output_host_tensor = TileTensor(output_host, row_major(tile_elems))
        var reference_host_tensor = TileTensor(
            reference_host, row_major(tile_elems)
        )
        for i in range(tile_elems):
            var expected = reference_host_tensor[i].cast[.float64]()
            var got = output_host_tensor[i].cast[.float64]()
            bad += Int(expected != got)
    assert_equal(bad, 0)


def main() raises:
    with DeviceContext() as ctx:
        test_block_scales_interleave_fp4_zero_rows[
            NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE, N=4
        ](ctx, 0, 4)
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
        test_block_scales_interleave_fp4_zero_rows[
            MXFP4_SF_DTYPE, MXFP4_SF_VECTOR_SIZE, N=4
        ](ctx, 0, 4)
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
