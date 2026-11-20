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
"""Times the SM100 W4A8 grouped matmul at MiMo-V2 MoE decode shapes.

Two shapes, both as one TP4 device sees them: the routed experts' gate/up
projection (N = 1024 gate+up rows, K = 4096) and their down projection
(N = 4096 hidden, K = 512, the expert width split four ways). Activations are
MXFP8 and weights nibble-packed MXFP4, which is what a MiMo-V2 routed expert
runs at decode.

Each shape is timed over a grid of launch configurations: the production
dispatcher's own pick, then the same tiling with the k grouping, the pipeline
depth and the number of CTAs each SM walks varied. The arm set is fixed at
compile time; the routed-row shape comes from the command line
(`--counts`/`--ids`), so one binary covers c1 through c8.

Every arm's output is checked against the production arm bit for bit before it
is timed: a configuration change may change the schedule, never the
arithmetic.

What it shows (B200, one GPU; c1 is 58 active experts carrying 64 routed rows,
which is MiMo-V2-Flash's c1 shape under DFlash K=7 and top-8; the second point
is 232 experts, which is what c8 looks like). Absolute times drift a few
percent between runs; the ratio is what the schedule is worth:

| shape            | one k-tile per stage | shipped schedule   |
|------------------|----------------------|--------------------|
| gate/up, c1      | 31.1 us, 3.92 TB/s   | 26.7 us, 4.56 TB/s |
| gate/up, 232 exp | 91.6 us, 5.31 TB/s   | 84.9 us, 5.73 TB/s |
| down, c1         | 16.0 us, 3.81 TB/s   | 13.4 us, 4.55 TB/s |
| down, 232 exp    | 53.9 us, 4.52 TB/s   | 48.9 us, 4.97 TB/s |

The launch is not bandwidth-bound at these shapes: Nsight Compute reports
`dram__bytes_read` at the checkpoint's own expert bytes, `dram__throughput` at
49% of peak, `launch__occupancy_limit_shared_mem` 1 and
`waves_per_multiprocessor` 1. A tile's k-loop is a chain of mbarrier arrivals
-- one per 128-element k-tile -- so covering several k-tiles per pipeline stage
is what moves the launch toward memory rate. That is as far as the schedule
goes: two CTAs per SM (arms 6-8, 10) and three (arm 9) lose at every tile
count measured, and so does halving the depth to pay for them, which is why the
shipped row walks one CTA per SM at the deepest pipeline the budget allows.
"""

from std.math import ceildiv
from std.sys import get_defined_int

from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from max.gpu.primitives.grid_controls import PDLLevel
from internal_utils import arg_parse
from internal_utils._utils import _init_block_scaled_scales_launch
from layout import Idx, TileTensor, row_major

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_mxfp8_dispatch,
)

comptime num_experts = get_defined_int["num_experts", 256]()
comptime REPS = 8
comptime REPLAYS = 10
comptime ROUNDS = 5
comptime NARMS = 11

comptime SF_DTYPE = MXFP8_SF_DTYPE
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
comptime SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]


def _hash(i: Int, stream: UInt32) -> UInt32:
    """A position-dependent 32-bit hash, so a permuted read cannot agree."""
    var x = UInt32(i & 0xFFFFFFFF) * 0x9E3779B9 + stream * 0x85EBCA6B
    x ^= UInt32((i >> 32) & 0xFFFFFFFF) * 0xC2B2AE35
    x ^= x >> 16
    x *= 0x7FEB352D
    x ^= x >> 15
    return x


def fill_bytes_kernel(
    ptr: MutPointer[UInt8, MutUntrackedOrigin],
    n: Int64,
    stream: UInt32,
    e4m3: UInt8,
):
    """Fills with a pattern, so an arm that reads the wrong row cannot agree.

    A constant fill would let a misread pass the bit-for-bit gate. In E4M3 the
    exponent is held in `1..7`, which keeps every byte a finite number below
    2**9; a nibble-packed E2M1 weight byte needs no such care because neither
    of its two values is ever NaN.
    """
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < Int(n):
        var x = _hash(i, stream)
        if e4m3 != 0:
            ptr[i] = (
                UInt8((x >> 31) & 1) << 7
                | UInt8(4 + ((x >> 8) % 4)) << 3
                | UInt8((x >> 16) & 7)
            )
        else:
            ptr[i] = UInt8(x >> 24)
        i += stride


def parse_list(s: String) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        out.append(Int(part))
    return out^


def median(var xs: List[Float64]) -> Float64:
    sort(xs)
    return xs[len(xs) // 2]


def graph_us[
    F: def(DeviceContext) raises -> None
](ctx: DeviceContext, f: F) raises -> Float64:
    """Microseconds per call of `f`, replayed back to back."""

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


def bench_shape[
    N: Int, K: Int
](ctx: DeviceContext, counts: List[Int], expert_ids: List[Int]) raises:
    """Times every arm at one (N, K), all checked against the production arm."""
    comptime packed_K = K // 2
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)

    var num_active = len(counts)
    var M = 0
    for n in counts:
        M += n

    var a = ctx.enqueue_create_buffer[DType.float8_e4m3fn](M * K)
    var b = ctx.enqueue_create_buffer[DType.uint8](num_experts * N * packed_K)
    var c = ctx.enqueue_create_buffer[DType.bfloat16](M * N)

    ctx.enqueue_function[fill_bytes_kernel](
        a.unsafe_ptr().unsafe_bitcast[UInt8](),
        Int64(M * K),
        UInt32(1),
        UInt8(1),
        grid_dim=4096,
        block_dim=256,
    )
    ctx.enqueue_function[fill_bytes_kernel](
        b.unsafe_ptr(),
        Int64(num_experts * N * packed_K),
        UInt32(2),
        UInt8(0),
        grid_dim=4096,
        block_dim=256,
    )

    var a_offsets_host = List[Int](length=num_active + 1, fill=0)
    var a_scale_offsets_host = List[Int](length=num_active, fill=0)
    var a_scale_dim0 = 0
    for i in range(num_active):
        a_scale_offsets_host[i] = a_scale_dim0 - a_offsets_host[i] // (
            SF_MN_GROUP_SIZE
        )
        a_offsets_host[i + 1] = a_offsets_host[i] + counts[i]
        a_scale_dim0 += ceildiv(counts[i], SF_MN_GROUP_SIZE)

    var a_scales = ctx.enqueue_create_buffer[SF_DTYPE](
        a_scale_dim0 * k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    var b_scales = ctx.enqueue_create_buffer[SF_DTYPE](
        num_experts
        * n_groups
        * k_groups
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_offsets = ctx.enqueue_create_buffer[DType.uint32](num_active + 1)
    var a_scale_offsets = ctx.enqueue_create_buffer[DType.uint32](num_active)
    var ids = ctx.enqueue_create_buffer[DType.int32](num_active)
    var expert_scales = ctx.enqueue_create_buffer[DType.float32](num_experts)

    var a_offsets_u32 = List[UInt32](length=num_active + 1, fill=UInt32(0))
    var a_scale_offsets_u32 = List[UInt32](length=num_active, fill=UInt32(0))
    var ids_i32 = List[Int32](length=num_active, fill=Int32(0))
    var expert_scales_f32 = List[Float32](length=num_experts, fill=Float32(0))
    for i in range(num_active + 1):
        a_offsets_u32[i] = UInt32(a_offsets_host[i])
    for i in range(num_active):
        a_scale_offsets_u32[i] = UInt32(a_scale_offsets_host[i])
        ids_i32[i] = Int32(expert_ids[i])
    for i in range(num_experts):
        expert_scales_f32[i] = Float32(1.0) + Float32(i + 1) / Float32(
            num_experts
        )

    ctx.enqueue_copy(a_offsets, a_offsets_u32)
    ctx.enqueue_copy(a_scale_offsets, a_scale_offsets_u32)
    ctx.enqueue_copy(ids, ids_i32)
    ctx.enqueue_copy(expert_scales, expert_scales_f32)
    _init_block_scaled_scales_launch[SF_DTYPE](a_scales, len(a_scales), ctx)
    _init_block_scaled_scales_launch[SF_DTYPE](b_scales, len(b_scales), ctx)

    var a_dev = TileTensor(a, row_major(M, Idx[K]))
    var b_dev = TileTensor(
        b, row_major(Idx[num_experts], Idx[N], Idx[packed_K])
    )
    var c_dev = TileTensor(c, row_major(M, Idx[N]))
    var a_scales_tt = TileTensor(
        a_scales,
        row_major(
            Int64(a_scale_dim0),
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    )
    var b_scales_tt = TileTensor(
        b_scales,
        row_major(
            Idx[num_experts],
            Idx[n_groups],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    )
    var a_offsets_dev = TileTensor(a_offsets, row_major(Int64(num_active) + 1))
    var a_scale_offsets_dev = TileTensor(
        a_scale_offsets, row_major(Int64(num_active))
    )
    var ids_dev = TileTensor(ids, row_major(Int64(num_active)))
    var expert_scales_dev = TileTensor(
        expert_scales, row_major(Int64(num_experts))
    )

    # `num_pipeline_stages` = 0 means "the classifier's own pick".
    @inline(.always)
    def launch[
        mma_bn: Int,
        cta_group: Int,
        stages: Int,
        k_group: Int,
        grid: Int,
        override: Bool,
    ](cx: DeviceContext) raises {imm}:
        grouped_matmul_mxfp8_dispatch[
            transpose_b=True,
            override=override,
            mma_bn=mma_bn,
            cta_group=cta_group,
            num_pipeline_stages=stages,
            k_group_size=k_group,
            grid_scale=grid,
            pdl_level=PDLLevel.ON,
        ](
            c_dev,
            a_dev,
            b_dev,
            a_scales_tt,
            b_scales_tt,
            a_offsets_dev,
            a_scale_offsets_dev,
            ids_dev,
            expert_scales_dev,
            num_active,
            M,
            cx,
        )

    @inline(.always)
    def arm[arm_idx: Int](cx: DeviceContext) raises {imm}:
        if arm_idx == 0:
            # The production dispatcher's choice, whatever it currently is.
            launch[8, 1, 0, 1, 1, False](cx)
        elif arm_idx == 1:
            # The previous schedule: one k-tile per pipeline stage.
            launch[8, 1, 0, 1, 1, True](cx)
        elif arm_idx == 2:
            launch[8, 1, 0, 2, 1, True](cx)
        elif arm_idx == 3:
            launch[8, 1, 0, 4, 1, True](cx)
        elif arm_idx == 4:
            # Half the depth with one k-tile per stage, for the depth A/B.
            launch[8, 1, 6, 1, 1, True](cx)
        elif arm_idx == 5:
            launch[8, 1, 8, 1, 1, True](cx)
        elif arm_idx == 6:
            # Two CTAs per SM: a second prologue to ride the first's mainloop.
            launch[8, 1, 0, 2, 2, True](cx)
        elif arm_idx == 7:
            launch[8, 1, 0, 4, 2, True](cx)
        elif arm_idx == 8:
            launch[8, 1, 6, 4, 2, True](cx)
        elif arm_idx == 9:
            # Three CTAs per SM.
            launch[8, 1, 0, 2, 3, True](cx)
        elif arm_idx == 10:
            launch[8, 1, 0, 1, 2, True](cx)

    comptime WEIGHT_BYTES = N * packed_K

    ctx.enqueue_memset(c, 0.0)
    launch[8, 1, 0, 1, 1, False](ctx)
    ctx.synchronize()
    var base = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    ctx.enqueue_copy(base, c)
    ctx.synchronize()

    var nonzero = 0
    var total: Float32 = 0
    for i in range(M * N):
        total += abs(base[i].cast[DType.float32]())
        if base[i] != 0:
            nonzero += 1
    print(t"BASE N={N} K={K} nonzero={nonzero} abs_sum={total}")
    var out = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    for a_idx in range(NARMS):
        comptime for v in range(NARMS):
            if a_idx == v:
                ctx.enqueue_memset(c, 0.0)
                arm[v](ctx)
                ctx.enqueue_copy(out, c)
                ctx.synchronize()
                var differ = 0
                for i in range(M * N):
                    if out[i] != base[i]:
                        differ += 1
                print(t"CHECK N={N} K={K} arm={v} differ={differ}")
                if differ != 0:
                    raise Error("arm is not bit-identical to production")

    var times = List[List[Float64]]()
    for _ in range(NARMS):
        times.append(List[Float64]())

    for r in range(ROUNDS):
        for j in range(NARMS):
            var a_idx = j if r % 2 == 0 else NARMS - 1 - j
            comptime for v in range(NARMS):
                if a_idx == v:

                    def timed(cx: DeviceContext) raises {imm}:
                        arm[v](cx)

                    times[v].append(graph_us(ctx, timed))

    for a_idx in range(NARMS):
        comptime for v in range(NARMS):
            if a_idx == v:
                var us = median(times[v].copy())
                print(
                    t"TIME N={N} K={K} M={M} active={num_active} arm={v}"
                    t" us={us}"
                    t" TBps={Float64(num_active * WEIGHT_BYTES) / us / 1e6}"
                )

    _ = a^
    _ = b^
    _ = c^
    _ = a_scales^
    _ = b_scales^
    _ = a_offsets^
    _ = a_scale_offsets^
    _ = ids^
    _ = expert_scales^
    _ = a_dev
    _ = b_dev
    _ = c_dev
    _ = a_scales_tt
    _ = b_scales_tt
    _ = a_offsets_dev
    _ = a_scale_offsets_dev
    _ = ids_dev
    _ = expert_scales_dev
    _ = base^
    _ = out^
    _ = a_offsets_host^
    _ = a_scale_offsets_host^
    _ = a_offsets_u32^
    _ = a_scale_offsets_u32^
    _ = ids_i32^
    _ = expert_scales_f32^


def copy_bandwidth(ctx: DeviceContext) raises:
    """The device's own HBM ceiling, read and written once per byte."""
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
        t"BW copy bytes={n} us={us}"
        t" read+write_TBps={2.0 * Float64(n) / us / 1e6}"
    )
    _ = src^
    _ = dst^


def main() raises:
    var counts = parse_list(arg_parse("counts", "1"))
    var expert_ids = parse_list(arg_parse("ids", "0"))
    var which = arg_parse("shapes", "both")
    with DeviceContext() as ctx:
        copy_bandwidth(ctx)
        if which == "both" or which == "gateup":
            bench_shape[1024, 4096](ctx, counts, expert_ids)
        if which == "both" or which == "down":
            bench_shape[4096, 512](ctx, counts, expert_ids)
