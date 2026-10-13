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
"""Microbenchmark for the tiled blockwise-scaled FP8 dense matmul (gfx950).

Reports median ms and TFLOP/s for the tiled kernel and the naive
reference at the GLM dense shapes. Manual target (not run in CI):

    bazel test --config=remote-mi355 \\
      //max/kernels/test/gpu/linalg:test_blockwise_fp8_matmul_amd_bench.mojo.test
"""

from std.math import ceildiv

from max.gpu.host import DeviceContext
from std.random import rand
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.gpu.amd.blockwise_scaled_fp8_matmul_amd import (
    blockwise_scaled_fp8_matmul_amd,
)

from std.utils.index import Index


def _median(var xs: List[Float64]) -> Float64:
    # Selection sort in place (tiny list); return the middle element.
    for i in range(len(xs)):
        var min_j = i
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[min_j]:
                min_j = j
        var tmp = xs[i]
        xs[i] = xs[min_j]
        xs[min_j] = tmp
    return xs[len(xs) // 2]


def bench_shape[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime input_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime transpose_b = True
    comptime BLOCK_SCALE_K = 128
    comptime BLOCK_SCALE_N = 128
    comptime iters = 20
    comptime warmup = 5
    comptime batches = 5

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scale_size = ceildiv(K, BLOCK_SCALE_K) * M
    var b_scale_size = ceildiv(N, BLOCK_SCALE_N) * ceildiv(K, BLOCK_SCALE_K)

    var a_host = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_type](b_size)
    var as_host = ctx.enqueue_create_host_buffer[.float32](a_scale_size)
    var bs_host = ctx.enqueue_create_host_buffer[.float32](b_scale_size)
    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(as_host.unsafe_ptr(), a_scale_size)
    rand(bs_host.unsafe_ptr(), b_scale_size)

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_my_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_device = ctx.enqueue_create_buffer[.float32](c_size)
    var as_device = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var bs_device = ctx.enqueue_create_buffer[.float32](b_scale_size)
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(as_device, as_host)
    ctx.enqueue_copy(bs_device, bs_host)

    var a_dev = TileTensor(a_device, row_major(Coord(m, k)))
    var b_dev = TileTensor(b_device, row_major(Coord(n, k)))
    var c_my_dev = TileTensor(c_my_device, row_major(Coord(m, n)))
    var c_ref_dev = TileTensor(c_ref_device, row_major(Coord(m, n)))
    var as_dev = TileTensor(
        as_device, row_major(Coord(ceildiv(K, BLOCK_SCALE_K), m))
    )
    var bs_dev = TileTensor(
        bs_device,
        row_major(Coord(ceildiv(N, BLOCK_SCALE_N), ceildiv(K, BLOCK_SCALE_K))),
    )

    def run_mine(c: DeviceContext) raises {imm}:
        blockwise_scaled_fp8_matmul_amd[
            transpose_b=transpose_b,
            N_SCALE=BLOCK_SCALE_N,
            K_SCALE=BLOCK_SCALE_K,
        ](c_my_dev, a_dev, b_dev, as_dev, bs_dev, c)

    def run_naive(c: DeviceContext) raises {imm}:
        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
            scales_granularity_mnk=Index(1, BLOCK_SCALE_N, BLOCK_SCALE_K),
        ](
            c_ref_dev,
            a_dev,
            b_dev,
            as_dev,
            bs_dev,
            c,
        )

    for _ in range(warmup):
        run_mine(ctx)
        run_naive(ctx)
    ctx.synchronize()

    var mine_ms = List[Float64]()
    var naive_ms = List[Float64]()
    for _ in range(batches):
        var t_mine = Float64(ctx.execution_time(run_mine, iters)) / (
            1.0e6 * Float64(iters)
        )
        var t_naive = Float64(ctx.execution_time(run_naive, iters)) / (
            1.0e6 * Float64(iters)
        )
        mine_ms.append(t_mine)
        naive_ms.append(t_naive)

    var med_mine = _median(mine_ms^)
    var med_naive = _median(naive_ms^)
    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var tflops_mine = flops / (med_mine * 1.0e9)
    var tflops_naive = flops / (med_naive * 1.0e9)

    print(
        "MxNxK=",
        M,
        "x",
        N,
        "x",
        K,
        " | tiled: ",
        med_mine,
        " ms (",
        tflops_mine,
        " TFLOP/s) | naive: ",
        med_naive,
        " ms (",
        tflops_naive,
        " TFLOP/s) | speedup: ",
        med_naive / med_mine,
        "x",
        sep="",
    )


def main() raises:
    with DeviceContext() as ctx:
        # GLM q_b and o_proj dense shapes.
        bench_shape(ctx, Idx[2048], Idx[2048], Idx[2048])
        bench_shape(ctx, Idx[2048], Idx[6144], Idx[2048])

        # Single-stream decode: M=1 and M=8 at the o_proj (K=2048 -> N=6144)
        # and q_a (K=6144 -> N=2048) shapes.
        bench_shape(ctx, Idx[1], Idx[6144], Idx[2048])
        bench_shape(ctx, Idx[8], Idx[6144], Idx[2048])
        bench_shape(ctx, Idx[1], Idx[2048], Idx[6144])
        bench_shape(ctx, Idx[8], Idx[2048], Idx[6144])
