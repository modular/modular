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
"""Microbenchmark for the tiled blockwise-scaled FP8 grouped matmul (gfx950).

Reports median ms and TFLOP/s for the tiled grouped kernel and the naive
reference at the GLM MoE shapes. Manual target (not run in CI):

    bazel test --config=remote-mi355 \\
      //max/kernels/test/gpu/linalg:test_blockwise_fp8_grouped_matmul_amd_bench.mojo.test
"""

from std.math import ceildiv

from max.gpu.host import DeviceContext
from std.random import rand
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from linalg.matmul.gpu.amd.blockwise_scaled_fp8_grouped_matmul_amd import (
    blockwise_scaled_fp8_grouped_matmul_amd,
)

from std.utils.index import Index


def _median(var xs: List[Float64]) -> Float64:
    for i in range(len(xs)):
        var min_j = i
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[min_j]:
                min_j = j
        var tmp = xs[i]
        xs[i] = xs[min_j]
        xs[min_j] = tmp
    return xs[len(xs) // 2]


def bench_grouped_shape[
    c_type: DType,
    num_experts: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext, total_rows: Int, num_active: Int) raises:
    comptime input_type = DType.float8_e4m3fn
    comptime transpose_b = True
    comptime BLOCK_SCALE_N = 128
    comptime BLOCK_SCALE_K = 128
    comptime N_BLOCKS = N // BLOCK_SCALE_N
    comptime K_BLOCKS = K // BLOCK_SCALE_K
    comptime iters = 20
    comptime warmup = 3
    comptime batches = 3

    # Even token split across the active experts (last takes the remainder).
    var base = total_rows // num_active
    var rem = total_rows - base * num_active
    var max_tokens = base + (1 if rem > 0 else 0)

    var a_size = total_rows * K
    var b_size = num_experts * N * K
    var c_size = total_rows * N
    var a_scale_size = K_BLOCKS * total_rows
    var b_scale_size = num_experts * N_BLOCKS * K_BLOCKS

    var a_host = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_type](b_size)
    var as_host = ctx.enqueue_create_host_buffer[.float32](a_scale_size)
    var bs_host = ctx.enqueue_create_host_buffer[.float32](b_scale_size)
    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(as_host.unsafe_ptr(), a_scale_size)
    rand(bs_host.unsafe_ptr(), b_scale_size)

    var a_off_host = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var eids_host = ctx.enqueue_create_host_buffer[.int32](num_active)
    var running = 0
    for i in range(num_active):
        a_off_host[i] = UInt32(running)
        running += base + (1 if i < rem else 0)
        eids_host[i] = Int32(i)
    a_off_host[num_active] = UInt32(running)

    var a_dev_buf = ctx.enqueue_create_buffer[input_type](a_size)
    var b_dev_buf = ctx.enqueue_create_buffer[input_type](b_size)
    var c_my_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_buf = ctx.enqueue_create_buffer[.float32](c_size)
    var as_buf = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var bs_buf = ctx.enqueue_create_buffer[.float32](b_scale_size)
    var a_off_buf = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var eids_buf = ctx.enqueue_create_buffer[.int32](num_active)
    ctx.enqueue_copy(a_dev_buf, a_host)
    ctx.enqueue_copy(b_dev_buf, b_host)
    ctx.enqueue_copy(as_buf, as_host)
    ctx.enqueue_copy(bs_buf, bs_host)
    ctx.enqueue_copy(a_off_buf, a_off_host)
    ctx.enqueue_copy(eids_buf, eids_host)

    var a_dev = TileTensor(a_dev_buf, row_major(Coord(total_rows, Idx[K])))
    var b_dev = TileTensor(b_dev_buf, row_major[num_experts, N, K]())
    var c_my_dev = TileTensor(c_my_buf, row_major(Coord(total_rows, Idx[N])))
    var c_ref_dev = TileTensor(c_ref_buf, row_major(Coord(total_rows, Idx[N])))
    var as_dev = TileTensor(as_buf, row_major(Coord(Idx[K_BLOCKS], total_rows)))
    var bs_dev = TileTensor(
        bs_buf, row_major[num_experts, N_BLOCKS, K_BLOCKS]()
    )
    var a_off_dev = TileTensor(a_off_buf, row_major(Coord(num_active + 1)))
    var eids_dev = TileTensor(eids_buf, row_major(Coord(num_active)))

    def run_mine(c: DeviceContext) raises {imm}:
        blockwise_scaled_fp8_grouped_matmul_amd[
            transpose_b=transpose_b,
            N_SCALE=BLOCK_SCALE_N,
            K_SCALE=BLOCK_SCALE_K,
        ](
            c_my_dev,
            a_dev,
            b_dev,
            as_dev,
            bs_dev,
            a_off_dev,
            eids_dev,
            max_num_tokens_per_expert=max_tokens,
            num_active_experts=num_active,
            ctx=c,
        )

    def run_naive(c: DeviceContext) raises {imm}:
        naive_blockwise_scaled_fp8_grouped_matmul[
            BLOCK_DIM_M=16,
            BLOCK_DIM_N=16,
            transpose_b=transpose_b,
            scales_granularity_mnk=Index(1, BLOCK_SCALE_N, BLOCK_SCALE_K),
        ](
            c_ref_dev,
            a_dev,
            b_dev,
            as_dev,
            bs_dev,
            a_off_dev,
            eids_dev,
            max_tokens,
            num_active,
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
    var flops = 2.0 * Float64(total_rows) * Float64(N) * Float64(K)
    var tflops_mine = flops / (med_mine * 1.0e9)
    var tflops_naive = flops / (med_naive * 1.0e9)

    print(
        "rows=",
        total_rows,
        " NxK=",
        N,
        "x",
        K,
        " experts=",
        num_active,
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
        # GLM gate_up prefill: 16384 tokens, N=4096, K=6144.
        bench_grouped_shape[.bfloat16, num_experts=8, N=4096, K=6144](
            ctx, 16384, 8
        )
        # GLM down prefill: 16384 tokens, N=6144, K=2048.
        bench_grouped_shape[.bfloat16, num_experts=8, N=6144, K=2048](
            ctx, 16384, 8
        )
        # GLM down decode: 64 tokens, N=6144, K=2048.
        bench_grouped_shape[.bfloat16, num_experts=8, N=6144, K=2048](
            ctx, 64, 8
        )

        # Single-stream decode: top-8 over 256 experts, 1 row each (8 tokens);
        # 16 and 32 cover a few concurrent single-stream requests.
        bench_grouped_shape[.bfloat16, num_experts=8, N=6144, K=2048](ctx, 8, 8)
        bench_grouped_shape[.bfloat16, num_experts=8, N=6144, K=2048](
            ctx, 16, 8
        )
        bench_grouped_shape[.bfloat16, num_experts=8, N=6144, K=2048](
            ctx, 32, 8
        )
