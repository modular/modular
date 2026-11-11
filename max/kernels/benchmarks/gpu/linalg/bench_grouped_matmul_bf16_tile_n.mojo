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
# Times the SM100 BF16 grouped matmul at a routed-rows shape for several token
# tile widths (MMA_N), each checked against the naive kernel and against the
# MMA_N=32 build bit for bit before it is timed. Time per call is a graph of
# REPS copies replayed back to back, the median over ROUNDS with the arm order
# alternating. The last arm (mma_n=0) is `grouped_matmul`'s own choice.

from std.math import ceildiv
from std.sys import get_defined_int

from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from internal_utils import arg_parse
from layout import Idx, TileTensor, row_major
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.grouped_matmul_sm100 import grouped_matmul_sm100_persistent
from linalg.utils_gpu import MatmulConfig

comptime NUM_EXPERTS = get_defined_int["num_experts", 256]()
comptime K = get_defined_int["K", 4096]()
comptime REPS = 20
comptime REPLAYS = 10
comptime ROUNDS = 5
comptime dtype = DType.bfloat16


def fill_kernel(
    ptr: MutPointer[Scalar[dtype], MutAnyOrigin], n: Int64, stream: UInt32
):
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < Int(n):
        var x = UInt32(i & 0xFFFFFFFF) * 0x9E3779B9 + stream * 0x85EBCA6B
        x ^= UInt32(i >> 32) * 0xC2B2AE35
        x ^= x >> 16
        x *= 0x7FEB352D
        x ^= x >> 15
        # [-1, 1) in steps of 1/128: exact in bf16.
        ptr[i] = Scalar[dtype](Float32(Int32(x >> 24) - 128) / 128.0)
        i += stride


def fill(ctx: DeviceContext, buf: DeviceBuffer[dtype], stream: UInt32) raises:
    ctx.enqueue_function[fill_kernel](
        buf.unsafe_ptr(), Int64(len(buf)), stream, grid_dim=4096, block_dim=256
    )


def parse_list(s: String) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        out.append(Int(part))
    return out^


def run[
    N: Int, mma_n: Int, cta_group: Int
](
    ctx: DeviceContext,
    c: DeviceBuffer[dtype],
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
    offsets: DeviceBuffer[DType.uint32],
    ids: DeviceBuffer[DType.int32],
    stats: DeviceBuffer[DType.uint32],
    M: Int,
) raises:
    comptime if mma_n == 0:
        grouped_matmul(
            TileTensor(
                c.unsafe_ptr().unsafe_mut_cast[True](), row_major(M, Idx[N])
            ),
            TileTensor(a, row_major(M, Idx[K])).as_imm(),
            TileTensor(b, row_major[NUM_EXPERTS, N, K]()).as_imm(),
            TileTensor(offsets, row_major(NUM_EXPERTS + 1)).as_imm(),
            TileTensor(ids, row_major(NUM_EXPERTS)).as_imm(),
            TileTensor(stats, row_major(2)).as_imm(),
            ctx,
        )
        return
    comptime BK = TensorMapSwizzle.SWIZZLE_128B.bytes() // 2
    comptime config = MatmulConfig[dtype, dtype, dtype, True](
        block_tile_shape=(128, mma_n // cta_group, BK),
        mma_shape=(128 * cta_group, mma_n, 16),
        cluster_shape=(cta_group, 1, 1),
    )
    grouped_matmul_sm100_persistent[
        c_type=dtype,
        a_type=dtype,
        b_type=dtype,
        transpose_b=True,
        config=config,
        cta_group=cta_group,
    ](
        TileTensor(
            c.unsafe_ptr().unsafe_mut_cast[True](), row_major(M, Idx[N])
        ),
        TileTensor(a, row_major(M, Idx[K])).as_imm(),
        TileTensor(offsets, row_major(NUM_EXPERTS + 1)).as_imm(),
        TileTensor(b, row_major[NUM_EXPERTS, N, K]()).as_imm(),
        TileTensor(ids, row_major(NUM_EXPERTS)).as_imm(),
        TileTensor(stats, row_major(2)).as_imm(),
        ctx,
    )


def graph_us[
    F: def(DeviceContext) raises -> None
](ctx: DeviceContext, f: F) raises -> Float64:
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        var rec = builder.recording_context()
        for _ in range(REPS):
            f(rec)

    var graph = DeviceGraph.create(ctx, build)

    def replay() raises {imm}:
        graph.replay()

    for _ in range(3):
        replay()
    ctx.synchronize()
    return (
        Float64(ctx.execution_time(replay, REPLAYS))
        / Float64(REPLAYS * REPS)
        / 1e3
    )


def median(var xs: List[Float64]) -> Float64:
    sort(xs)
    return xs[len(xs) // 2]


def bench[
    N: Int
](
    ctx: DeviceContext, label: String, counts: List[Int], expert_ids: List[Int]
) raises:
    var num_active = len(counts)
    var M = 0
    var max_count = 0
    for n in counts:
        M += n
        max_count = max(max_count, n)

    var a = ctx.enqueue_create_buffer[dtype](M * K)
    var b = ctx.enqueue_create_buffer[dtype](NUM_EXPERTS * N * K)
    var c_ref = ctx.enqueue_create_buffer[dtype](M * N)
    var c = ctx.enqueue_create_buffer[dtype](M * N)
    var offsets = ctx.enqueue_create_buffer[DType.uint32](NUM_EXPERTS + 1)
    var ids = ctx.enqueue_create_buffer[DType.int32](NUM_EXPERTS)
    var stats = ctx.enqueue_create_buffer[DType.uint32](2)
    fill(ctx, a, 1)
    fill(ctx, b, 2)
    with offsets.map_to_host() as h:
        h[0] = 0
        for i in range(NUM_EXPERTS):
            h[i + 1] = h[i] + UInt32(counts[i] if i < num_active else 0)
    with ids.map_to_host() as h:
        for i in range(NUM_EXPERTS):
            h[i] = Int32(expert_ids[i] if i < num_active else -1)
    with stats.map_to_host() as h:
        h[0] = UInt32(max_count)
        h[1] = UInt32(num_active)

    naive_grouped_matmul(
        TileTensor(c_ref, row_major(M, Idx[N])),
        TileTensor(a, row_major(M, Idx[K])).as_imm(),
        TileTensor(b, row_major[NUM_EXPERTS, N, K]()).as_imm(),
        TileTensor(offsets, row_major(NUM_EXPERTS + 1)).as_imm(),
        TileTensor(ids, row_major(NUM_EXPERTS)).as_imm(),
        max_count,
        num_active,
        ctx,
    )
    var ref_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    var base_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    var out_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.enqueue_copy(ref_host, c_ref)
    ctx.synchronize()
    var ref_abs_sum: Float64 = 0
    for i in range(M * N):
        ref_abs_sum += Float64(abs(ref_host[i].cast[DType.float32]()))
    print(t"REF {label} N={N} mean_abs={ref_abs_sum / Float64(M * N)}")
    if ref_abs_sum == 0:
        raise Error("reference is all zeros")

    comptime cta = 2 if N % 256 == 0 else 1
    # Token tile widths 32 << w.
    comptime NW = 3
    comptime NARMS = NW + 1
    var times = List[List[Float64]]()
    for _ in range(NARMS):
        times.append(List[Float64]())

    comptime for w in range(NARMS):

        def call(cx: DeviceContext) raises {imm}:
            run[N, (32 << w) if w < NW else 0, cta](
                cx, c, a, b, offsets, ids, stats, M
            )

        ctx.enqueue_memset(c, 0)
        call(ctx)
        ctx.enqueue_copy(out_host if w > 0 else base_host, c)
        ctx.synchronize()
        var bad = 0
        var differ = 0
        var max_err: Float32 = 0
        for i in range(M * N):
            var got = (out_host if w > 0 else base_host)[i].cast[
                DType.float32
            ]()
            var want = ref_host[i].cast[DType.float32]()
            var err = abs(got - want)
            max_err = max(max_err, err)
            if err > 0.02 * abs(want) + 0.5:
                bad += 1
            if w > 0 and out_host[i] != base_host[i]:
                differ += 1
        print(
            t"CHECK {label} N={N} mma_n={(32 << w) if w < NW else 0} cta={cta}"
            t" bad={bad} max_abs_err={max_err} differ_from_32={differ}"
        )
        if bad != 0 or differ != 0:
            raise Error("check failed")

    for r in range(ROUNDS):
        comptime for j in range(NARMS):
            # Odd rounds run the arms in reverse.
            var w = j if r % 2 == 0 else NARMS - 1 - j
            comptime for v in range(NARMS):
                if v == w:

                    def timed(cx: DeviceContext) raises {imm}:
                        run[N, (32 << v) if v < NW else 0, cta](
                            cx, c, a, b, offsets, ids, stats, M
                        )

                    times[v].append(graph_us(ctx, timed))

    for w in range(NARMS):
        print(
            t"TIME {label} N={N} K={K} M={M} active={num_active}"
            t" mma_n={(32 << w) if w < NW else 0} cta={cta} us={median(times[w].copy())}"
        )
    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^
    _ = offsets^
    _ = ids^
    _ = stats^


def copy_bandwidth(ctx: DeviceContext) raises:
    # HBM floor for the weight stream: a device copy reads and writes each byte.
    comptime n = 1 << 30
    var src = ctx.enqueue_create_buffer[DType.uint8](n)
    var dst = ctx.enqueue_create_buffer[DType.uint8](n)

    def copy(cx: DeviceContext) raises {imm}:
        cx.enqueue_copy(dst, src)

    var times = List[Float64]()
    for _ in range(ROUNDS):
        times.append(graph_us(ctx, copy))
    var us = median(times^)
    print(
        t"BW copy"
        t" bytes={n} us={us} read+write_TBps={2.0 * Float64(n) / us / 1e6}"
    )
    _ = src^
    _ = dst^


def main() raises:
    var label = arg_parse("label", "shape")
    var counts = parse_list(arg_parse("counts", "1"))
    var expert_ids = parse_list(arg_parse("ids", "0"))
    var ctx = DeviceContext()
    copy_bandwidth(ctx)
    bench[8192](ctx, label, counts, expert_ids)
    bench[4096](ctx, label, counts, expert_ids)
