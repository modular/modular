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

"""Bandwidth benches for the blockwise-scaled FP8 matmul on sm_12x.

Times the tiled kernel at the Qwen3.8-27B dense projection shapes (K=5120, N
in {5120, 10240, 17408}) as back-to-back launches between `perf_counter()` and
`synchronize()`, beside the naive kernel on the same operands and the forced
tile-width and split variants. The operands are filled once and every launch
re-reads the same weights, so a shape whose `N * K` weights fit in L2 measures
above the DRAM wall; the largest is 89 MB, past this part's L2. The single-CTA
case at the end is the launch-overhead floor.

Local only: a bandwidth number needs a quiet box, so BUILD tags this `manual`
and presubmit skips it.
"""

from std.math import ceildiv, exp2

from max.gpu.host import DeviceContext
from std.random import rand
from std.time import perf_counter
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.gpu.blockwise_scaled_fp8_matmul_nvidia import (
    blockwise_scaled_fp8_matmul_nvidia,
)

comptime _BLOCK_SCALE_M = 1
comptime _BLOCK_SCALE_N = 128
comptime _BLOCK_SCALE_K = 128

comptime _WALL_GBPS = 250.1
"""Measured wall bandwidth of this box (GB10, 121.7 GiB unified), GB/s."""


def bench_blockwise_fp8_matmul_nvidia[
    NType: CoordLike,
    KType: CoordLike,
    bn: Int = 0,
    split: Int = 0,
    //,
    iters: Int = 20,
](ctx: DeviceContext, M: Int, n: NType, k: KType) raises:
    """Times the tiled kernel at one (M, N, K) and prints ms and GB/s.

    The operands are filled once and every launch re-reads the same weights,
    so a shape whose `N * K` weights fit in L2 can measure above the DRAM
    wall. The largest N here is 89 MB, well past this part's L2.

    Parameters:
        NType: Type of the N argument (static `Idx` or dynamic `Int`).
        KType: Type of the K argument.
        iters: Timed launches, after 3 warmups.

    Args:
        ctx: Device context.
        M: Number of rows.
        n: Number of columns of the output.
        k: Reduction depth.
    """
    comptime input_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16

    var N = Int(n)
    var K = Int(k)
    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scale_size = ceildiv(K, _BLOCK_SCALE_K) * M
    var b_scale_size = ceildiv(N, _BLOCK_SCALE_N) * ceildiv(K, _BLOCK_SCALE_K)

    var a_host = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_type](b_size)
    var a_scale_host = ctx.enqueue_create_host_buffer[.float32](a_scale_size)
    var b_scale_host = ctx.enqueue_create_host_buffer[.float32](b_scale_size)
    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(a_scale_host.unsafe_ptr(), a_scale_size)
    rand(b_scale_host.unsafe_ptr(), b_scale_size)
    for i in range(a_scale_size):
        a_scale_host[i] = exp2(Float32(8.0) * a_scale_host[i] - 4.0)
    for i in range(b_scale_size):
        b_scale_host[i] = exp2(Float32(8.0) * b_scale_host[i] - 4.0)

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var a_scale_device = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var b_scale_device = ctx.enqueue_create_buffer[.float32](b_scale_size)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(a_scale_device, a_scale_host)
    ctx.enqueue_copy(b_scale_device, b_scale_host)

    var a_dev = TileTensor(a_device, row_major(Coord(M, k)))
    var b_dev = TileTensor(b_device, row_major(Coord(n, k)))
    var c_dev = TileTensor(c_device, row_major(Coord(M, n)))
    var a_scale_dev = TileTensor(
        a_scale_device, row_major(Coord(ceildiv(K, _BLOCK_SCALE_K), M))
    )
    var b_scale_dev = TileTensor(
        b_scale_device,
        row_major(
            Coord(ceildiv(N, _BLOCK_SCALE_N), ceildiv(K, _BLOCK_SCALE_K))
        ),
    )

    for _ in range(3):
        blockwise_scaled_fp8_matmul_nvidia[
            transpose_b=True, bn_override=bn, split_override=split
        ](c_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)
    ctx.synchronize()

    var t0 = perf_counter()
    for _ in range(iters):
        blockwise_scaled_fp8_matmul_nvidia[
            transpose_b=True, bn_override=bn, split_override=split
        ](c_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)
    ctx.synchronize()
    var t1 = perf_counter()

    var ms = (t1 - t0) * 1000.0 / Float64(iters)
    var weight_gbps = Float64(N * K) / (ms * 1e6)
    var m_blocks = ceildiv(M, 8)
    print(
        (
            t"bench M={M} N={N} K={K} bn={bn} split={split}"
            t" iters={iters} ms_per_launch={ms}"
        ),
        sep="",
    )
    print(
        (
            t"   unique-weight GB/s={weight_gbps}"
            t" ({weight_gbps / _WALL_GBPS * 100.0}% of the"
            t" {_WALL_GBPS} GB/s wall), all-traffic GB/s="
            t"{weight_gbps * Float64(m_blocks)} (b read"
            t" {m_blocks}x for M={M})"
        ),
        sep="",
    )

    # The kernel this one replaces, on the same operands: the naive path is
    # one thread per output element with scalar loads, so this is the number
    # the tiled kernel has to beat.
    var naive_iters = 1 if M > 512 else 3
    comptime naive_kernel = naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=True,
        scales_granularity_mnk=Coord[Int, Int, Int](
            _BLOCK_SCALE_M, _BLOCK_SCALE_N, _BLOCK_SCALE_K
        ),
    ]
    naive_kernel(c_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)
    ctx.synchronize()
    var nt0 = perf_counter()
    for _ in range(naive_iters):
        naive_kernel(c_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)
    ctx.synchronize()
    var nt1 = perf_counter()
    var naive_ms = (nt1 - nt0) * 1000.0 / Float64(naive_iters)
    var naive_gbps = Float64(N * K) / (naive_ms * 1e6)
    print(
        (
            t"   naive ms_per_launch={naive_ms} ({weight_gbps / naive_gbps}"
            t"x slower, {naive_gbps} GB/s, "
            t"{naive_gbps / _WALL_GBPS * 100.0}% of wall)"
        ),
        sep="",
    )


def main() raises:
    with DeviceContext() as ctx:
        # Decode: the four real projection shapes of Qwen3.8-27B-FP8, M=1, at
        # whatever tile width and split the launcher picks.
        print("-- decode, launcher's choice", sep="")
        bench_blockwise_fp8_matmul_nvidia(ctx, 1, Idx[5120], Idx[5120])
        bench_blockwise_fp8_matmul_nvidia(ctx, 1, Idx[10240], Idx[5120])
        bench_blockwise_fp8_matmul_nvidia(ctx, 1, Idx[17408], Idx[5120])
        bench_blockwise_fp8_matmul_nvidia(ctx, 1, Idx[5120], Idx[17408])

        # Each lever separately, on the largest and the smallest grid.
        print("-- decode, forced variants", sep="")
        bench_blockwise_fp8_matmul_nvidia[bn=128, split=1](
            ctx, 1, Idx[5120], Idx[5120]
        )
        bench_blockwise_fp8_matmul_nvidia[bn=64, split=1](
            ctx, 1, Idx[5120], Idx[5120]
        )
        bench_blockwise_fp8_matmul_nvidia[bn=128, split=2](
            ctx, 1, Idx[5120], Idx[5120]
        )
        bench_blockwise_fp8_matmul_nvidia[bn=128, split=4](
            ctx, 1, Idx[5120], Idx[5120]
        )
        bench_blockwise_fp8_matmul_nvidia[bn=128, split=1](
            ctx, 1, Idx[17408], Idx[5120]
        )
        bench_blockwise_fp8_matmul_nvidia[bn=64, split=1](
            ctx, 1, Idx[17408], Idx[5120]
        )
        bench_blockwise_fp8_matmul_nvidia[bn=128, split=1](
            ctx, 1, Idx[5120], Idx[17408]
        )
        bench_blockwise_fp8_matmul_nvidia[bn=64, split=2](
            ctx, 1, Idx[5120], Idx[17408]
        )

        # Prefill: 4088 rows, the b re-read cost at one token of a 4088-token
        # chunk (511 m blocks, i.e. 511 passes over the weights).
        print("-- prefill", sep="")
        bench_blockwise_fp8_matmul_nvidia[iters=3](
            ctx, 4088, Idx[5120], Idx[5120]
        )

        # Launch-overhead floor: one CTA and one k slab.
        bench_blockwise_fp8_matmul_nvidia[iters=100](ctx, 1, Idx[128], Idx[128])
