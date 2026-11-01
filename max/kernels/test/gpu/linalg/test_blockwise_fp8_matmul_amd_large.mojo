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
"""Correctness test for the large-M blockwise-scaled FP8 matmul (gfx950).

Compares `blockwise_scaled_fp8_matmul_amd_large`, launched directly and
through `blockwise_scaled_fp8_matmul_amd`'s dispatch (runtime M, as in
production), against `naive_blockwise_scaled_fp8_matmul` at the GLM-5.3
prefill shapes. The cases cover a partial last row tile, column tiles with
idle waves (N not a multiple of 256), N = 128, K = 12288 (the slot-pair
loop), an odd M (unaligned scale rows), a partial second group of the tile
walk, a padded A-scale row stride, and a padded C row stride (which the
dispatch must send to the tiled kernel).

The scales are drawn LOG-UNIFORM in `[2^-4, 2^4]` per block: a constant
scale hides a scale-indexing bug, the most likely failure mode.
"""

from std.math import ceildiv, exp2

from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from std.random import rand
from std.testing import assert_true
from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.gpu.amd.blockwise_scaled_fp8_matmul_amd import (
    LARGE_M_MIN,
    blockwise_scaled_fp8_matmul_amd,
)
from linalg.matmul.gpu.amd.blockwise_scaled_fp8_matmul_amd_large import (
    blockwise_scaled_fp8_matmul_amd_large,
    blockwise_scaled_fp8_matmul_amd_large_supported,
)

from std.utils.numerics import inf, isnan, nan

comptime FP8 = DType.float8_e4m3fn
comptime BLOCK = 128


def test_large[
    N: Int, K: Int
](
    ctx: DeviceContext,
    M: Int,
    via_dispatch: Bool,
    scale_pad: Int = 0,
    c_pad: Int = 0,
) raises:
    """`scale_pad`: extra elements per row of A's scales. `c_pad`: extra
    elements per row of C (dispatch only: the large-M kernel needs a dense C,
    so the dispatch must fall back). C starts as NaN, so an output a kernel
    skips, or a padding element it writes, fails the check."""
    comptime KB = K // BLOCK
    comptime NB = ceildiv(N, BLOCK)
    print(
        t"== test_large M={M} N={N} K={K} via_dispatch={via_dispatch}"
        t" scale_pad={scale_pad} c_pad={c_pad}"
    )
    assert_true(via_dispatch or c_pad == 0, "the kernel needs a dense C")
    assert_true(
        blockwise_scaled_fp8_matmul_amd_large_supported[.bfloat16, N, K]()
    )
    if via_dispatch:
        assert_true(
            M >= LARGE_M_MIN, "the dispatch would keep the tiled kernel"
        )

    var sa_stride = M + scale_pad
    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var c_ld = N + c_pad
    var as_size = KB * sa_stride
    var bs_size = NB * KB

    var a_host = ctx.enqueue_create_host_buffer[FP8](a_size)
    var b_host = ctx.enqueue_create_host_buffer[FP8](b_size)
    var as_host = ctx.enqueue_create_host_buffer[.float32](as_size)
    var bs_host = ctx.enqueue_create_host_buffer[.float32](bs_size)
    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(as_host.unsafe_ptr(), as_size)
    rand(bs_host.unsafe_ptr(), bs_size)
    # Rewrite the [0, 1) samples as LOG-UNIFORM scales in [2^-4, 2^4).
    for i in range(as_size):
        as_host[i] = exp2(Float32(8.0) * as_host[i] - 4.0)
    for i in range(bs_size):
        bs_host[i] = exp2(Float32(8.0) * bs_host[i] - 4.0)
    # The padding of each block's A-scale row: a valid output that reads it
    # comes out inf or NaN.
    for kb in range(KB):
        for j in range(M, sa_stride):
            as_host[kb * sa_stride + j] = inf[DType.float32]()

    var a_dev = ctx.enqueue_create_buffer[FP8](a_size)
    var b_dev = ctx.enqueue_create_buffer[FP8](b_size)
    var as_dev = ctx.enqueue_create_buffer[.float32](as_size)
    var bs_dev = ctx.enqueue_create_buffer[.float32](bs_size)
    var c_dev = ctx.enqueue_create_buffer[.bfloat16](M * c_ld)
    var ref_dev = ctx.enqueue_create_buffer[.float32](c_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(as_dev, as_host)
    ctx.enqueue_copy(bs_dev, bs_host)
    ctx.enqueue_memset(c_dev, nan[DType.bfloat16]())

    var a_tt = TileTensor(a_dev, row_major(Coord(M, Idx[K])))
    var b_tt = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var bs_tt = TileTensor(bs_dev, row_major(Coord(Idx[NB], Idx[KB])))
    if via_dispatch:
        blockwise_scaled_fp8_matmul_amd[
            transpose_b=True, N_SCALE=BLOCK, K_SCALE=BLOCK
        ](
            TileTensor(
                c_dev, TileLayout(Coord(M, Idx[N]), Coord(c_ld, Idx[1]))
            ),
            a_tt,
            b_tt,
            TileTensor(
                as_dev, TileLayout(Coord(Idx[KB], M), Coord(sa_stride, Idx[1]))
            ),
            bs_tt,
            ctx,
        )
    else:
        blockwise_scaled_fp8_matmul_amd_large[N, K](
            ctx,
            Int(c_dev.unsafe_ptr()),
            Int(a_dev.unsafe_ptr()),
            Int(as_dev.unsafe_ptr()),
            Int(b_dev.unsafe_ptr()),
            Int(bs_dev.unsafe_ptr()),
            M,
            sa_stride,
        )

    # The reference reads the scales densely: give it an unpadded copy.
    var as_ref_dev = ctx.enqueue_create_buffer[.float32](KB * M)
    var as_ref_host = ctx.enqueue_create_host_buffer[.float32](KB * M)
    for kb in range(KB):
        for j in range(M):
            as_ref_host[kb * M + j] = as_host[kb * sa_stride + j]
    ctx.enqueue_copy(as_ref_dev, as_ref_host)
    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=True,
        scales_granularity_mnk=Coord[Int, Int, Int](1, BLOCK, BLOCK),
    ](
        TileTensor(ref_dev, row_major(Coord(M, Idx[N]))),
        a_tt,
        b_tt,
        TileTensor(as_ref_dev, row_major(Coord(Idx[KB], M))),
        bs_tt,
        ctx,
    )

    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * c_ld)
    var ref_host = ctx.enqueue_create_host_buffer[.float32](c_size)
    var c_f32 = ctx.enqueue_create_host_buffer[.float32](c_size)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()
    for m in range(M):
        for n in range(N):
            c_f32[m * N + n] = c_host[m * c_ld + n].cast[.float32]()
        for n in range(N, c_ld):
            assert_true(isnan(c_host[m * c_ld + n]), "a write past a row of C")

    # The bf16 band of `test_blockwise_fp8_matmul_amd`.
    assert_almost_equal(
        c_f32.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        c_size,
        "large-M vs naive",
        atol=2e-1,
        rtol=4e-2,
    )
    print("   PASSED")


def main() raises:
    with DeviceContext() as ctx:
        # MLA q_a + kv_a (fused): a partial last row tile, and a last column
        # tile 64 wide (six of its eight waves idle).
        test_large[2624, 6144](ctx, 1000, via_dispatch=True)
        test_large[2624, 6144](ctx, 1000, via_dispatch=False)
        # MLA q_b: whole tiles.
        test_large[2048, 2048](ctx, 2048, via_dispatch=True)
        # MLA o_proj: odd M, so A-scale rows start unaligned.
        test_large[6144, 2048](ctx, 777, via_dispatch=True)
        # Indexer wk: N = 128, one column tile with half its waves idle.
        test_large[128, 6144](ctx, 300, via_dispatch=True)
        # Indexer wq_b.
        test_large[4096, 2048](ctx, 513, via_dispatch=True)
        # Dense MLP down: K = 12288 takes the slot-pair loop; the smallest M
        # the dispatch routes here.
        test_large[6144, 12288](ctx, LARGE_M_MIN, via_dispatch=True)
        # Nine row tiles: a partial second group of the tile walk.
        test_large[2048, 2048](ctx, 1100, via_dispatch=True)
        # A padded A-scale row stride, launched directly and dispatched.
        test_large[2048, 2048](ctx, 333, via_dispatch=False, scale_pad=3)
        test_large[2048, 2048](ctx, 333, via_dispatch=True, scale_pad=3)
        # A padded C row stride: the dispatch keeps the tiled kernel.
        test_large[2048, 2048](ctx, 333, via_dispatch=True, c_pad=64)
