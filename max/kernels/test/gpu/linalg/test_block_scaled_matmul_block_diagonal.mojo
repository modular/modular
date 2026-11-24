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
"""Tests the MXFP8 `block_scaled_matmul_block_diagonal` on SM100.

The reference runs `naive_block_scaled_matmul` once per diagonal block, on
strided views of the same buffers: A's and C's column blocks, B's row block
and the matching scale tiles. C starts as a sentinel, so a block that is
skipped or written to the wrong columns fails. M covers the small-BN path
(M <= 16), the heuristic path, and M not a multiple of the 128-row scale
tile.
"""

from std.math import align_up, ceildiv
from std.random import rand, random_ui64

from internal_utils import assert_almost_equal
from layout import Coord, Idx, MixedLayout, TileTensor, row_major
from linalg.block_scaled_quantization import naive_block_scaled_matmul
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    set_scale_factor,
)
from linalg.matmul.gpu.sm100.block_scaled_dispatch import (
    block_scaled_matmul_block_diagonal,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from max.gpu.host import DeviceContext


def _random_scales[
    scales_type: DType
](scales: TileTensor[mut=True, scales_type, ...], rows: Int, cols: Int):
    """Sets every live scale to a random power of two and the padding to 0."""
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    for row in range(align_up(rows, SF_MN_GROUP_SIZE)):
        for col in range(
            0, align_up(cols, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            var value = Scalar[scales_type](0.0)
            if row < rows and col < cols:
                value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_type]()
                )
            set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                scales, row, col, value
            )


def test_block_diagonal[
    groups: Int, N: Int, K: Int
](ctx: DeviceContext, m: Int) raises:
    comptime in_type = DType.float8_e4m3fn
    comptime out_type = DType.bfloat16
    comptime scales_type = MXFP8_SF_DTYPE
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    comptime sf_k = K // (SF_VECTOR_SIZE * SF_ATOM_K)
    comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    comptime n_tiles = N // SF_MN_GROUP_SIZE
    print(t"groups={groups} M={m} N={N} K={K}")

    var a_shape = row_major(m, Idx[groups * K])
    var b_shape = row_major(Idx[groups * N], Idx[K])
    var c_shape = row_major(m, Idx[groups * N])
    var m_tiles = ceildiv(m, SF_MN_GROUP_SIZE)
    var a_scales_shape = row_major(
        m_tiles,
        Idx[groups * sf_k],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_scales_shape = row_major(
        Idx[groups * n_tiles],
        Idx[sf_k],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var a_size = m * groups * K
    var b_size = groups * N * K
    var c_size = m * groups * N
    var a_scales_size = m_tiles * groups * sf_k * sf_atom
    var b_scales_size = groups * n_tiles * sf_k * sf_atom

    var a_host = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_size)
    var a_scales_host = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_size
    )
    var b_scales_host = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_size
    )
    var c_host = ctx.enqueue_create_host_buffer[out_type](c_size)
    var c_ref_host = ctx.enqueue_create_host_buffer[out_type](c_size)

    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    _random_scales(TileTensor(a_scales_host, a_scales_shape), m, groups * K)
    _random_scales(TileTensor(b_scales_host, b_scales_shape), groups * N, K)

    var a_device = ctx.enqueue_create_buffer[in_type](a_size)
    var b_device = ctx.enqueue_create_buffer[in_type](b_size)
    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_size)
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_size)
    var c_device = ctx.enqueue_create_buffer[out_type](c_size)
    var c_ref_device = ctx.enqueue_create_buffer[out_type](c_size)
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(a_scales_device, a_scales_host)
    ctx.enqueue_copy(b_scales_device, b_scales_host)
    c_device.enqueue_fill(Scalar[out_type](-1.0e30))
    c_ref_device.enqueue_fill(Scalar[out_type](0.0))

    block_scaled_matmul_block_diagonal[
        groups=groups, SF_VECTOR_SIZE=SF_VECTOR_SIZE
    ](
        TileTensor(c_device, c_shape),
        TileTensor(a_device, a_shape),
        TileTensor(b_device, b_shape),
        TileTensor(a_scales_device, a_scales_shape),
        TileTensor(b_scales_device, b_scales_shape),
        Float32(1.0),
        ctx,
    )

    for g in range(groups):
        var a_g = TileTensor(
            a_device.unsafe_ptr().unsafe_offset(g * K),
            MixedLayout(
                Coord(Int64(m), Idx[K]), Coord(Idx[groups * K], Idx[1])
            ),
        )
        var b_g = TileTensor(
            b_device.unsafe_ptr().unsafe_offset(g * N * K),
            row_major(Idx[N], Idx[K]),
        )
        var c_g = TileTensor(
            c_ref_device.unsafe_ptr().unsafe_offset(g * N),
            MixedLayout(
                Coord(Int64(m), Idx[N]), Coord(Idx[groups * N], Idx[1])
            ),
        )
        var a_scales_g = TileTensor(
            a_scales_device.unsafe_ptr().unsafe_offset(g * sf_k * sf_atom),
            MixedLayout(
                Coord(
                    Int64(m_tiles),
                    Idx[sf_k],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                ),
                Coord(
                    Idx[groups * sf_k * sf_atom],
                    Idx[sf_atom],
                    Idx[SF_ATOM_M[1] * SF_ATOM_K],
                    Idx[SF_ATOM_K],
                    Idx[1],
                ),
            ),
        )
        var b_scales_g = TileTensor(
            b_scales_device.unsafe_ptr().unsafe_offset(
                g * n_tiles * sf_k * sf_atom
            ),
            row_major(
                Idx[n_tiles],
                Idx[sf_k],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            ),
        )
        naive_block_scaled_matmul[
            scaling_kind=UMMAKind.KIND_MXF8F6F4,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=True,
        ](c_g, a_g, b_g, a_scales_g, b_scales_g, ctx)

    ctx.enqueue_copy(c_host, c_device)
    ctx.enqueue_copy(c_ref_host, c_ref_device)
    ctx.synchronize()

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        c_size,
        atol=1e-2,
        rtol=1e-2,
    )
    print("  -> PASSED")

    _ = a_device^
    _ = b_device^
    _ = a_scales_device^
    _ = b_scales_device^
    _ = c_device^
    _ = c_ref_device^


def main() raises:
    with DeviceContext() as ctx:
        # The block-diagonal `bgk,grk->bgr` projection of 8 groups.
        for m in [1, 16, 96, 129, 200]:
            test_block_diagonal[8, 1024, 4096](ctx, m)
        # Few blocks and a short K, where N tiles past a block's end would
        # read the next block's rows if the views leaked.
        # 9473 is past the prebuilt configs' M sweep, and the heuristic picks
        # a config for it that none of them is.
        for m in [7, 144, 9473]:
            test_block_diagonal[3, 256, 512](ctx, m)
