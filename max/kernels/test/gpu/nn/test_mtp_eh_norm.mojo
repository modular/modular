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
"""Tests for the MTP draft layer's fused input projection."""

from std.math import rsqrt, sqrt
from std.memory import bitcast
from std.random import rand, seed
from std.sys import argv
from std.testing import assert_equal, assert_true
from std.time import monotonic

from layout import TensorEngine, TensorLayout, TileTensor, row_major
from max.gpu.host import (
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    HostBuffer,
)
from max.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)

from nn.normalization import block_reduce_dual_sum

from nn.mtp_eh_norm import mtp_eh_norm_kernel

comptime BENCH_ROUNDS = 5
comptime GRAPH_WARMUP = 5
comptime GRAPH_REPS = 100


def _reference_row[
    hidden_size: Int
](
    src: HostBuffer[.bfloat16],
    weight: HostBuffer[.bfloat16],
    row: Int,
    col: Int,
    eps: Float64,
    multiply_before_cast: Bool,
) -> Float64:
    """One normalized channel, modelling the kernel's roundings exactly.

    Everything outside a rounding runs in f64, but each bf16 rounding the
    kernel performs is reproduced, so the expected value is exact rather than
    approximate. `multiply_before_cast` selects between the two orderings
    `ops.rms_norm` supports; the kernel implements the `False` arm.
    """
    var acc = Float64(0)
    for i in range(hidden_size):
        var v = Float64(src[row * hidden_size + i].cast[.float32]())
        acc += v * v
    var rrms = 1.0 / sqrt(acc / Float64(hidden_size) + eps)
    var x = Float64(src[row * hidden_size + col].cast[.float32]())
    var w = Float64(weight[col].cast[.float32]())
    if multiply_before_cast:
        return Float64(
            Scalar[.bfloat16]((x * rrms * w).cast[.float32]()).cast[.float32]()
        )
    var normed = Float64(
        Scalar[.bfloat16]((x * rrms).cast[.float32]()).cast[.float32]()
    )
    return Float64(
        Scalar[.bfloat16]((normed * w).cast[.float32]()).cast[.float32]()
    )


@__name(t"mtp_eh_norm_base_{hidden_size}_{max_warps_per_block}")
def mtp_eh_norm_base_kernel[
    dtype: DType,
    OutLayoutType: TensorLayout,
    out_origin: MutOrigin,
    EmbedLayoutType: TensorLayout,
    embed_origin: ImmOrigin,
    PrevLayoutType: TensorLayout,
    prev_origin: ImmOrigin,
    EWLayoutType: TensorLayout,
    ew_origin: ImmOrigin,
    HWLayoutType: TensorLayout,
    hw_origin: ImmOrigin,
    hidden_size: Int,
    max_warps_per_block: Int,
    OutEngine: TensorEngine,
    EmbedEngine: TensorEngine,
    PrevEngine: TensorEngine,
    EWEngine: TensorEngine,
    HWEngine: TensorEngine,
](
    out_buf: TileTensor[dtype, OutLayoutType, out_origin, Engine=OutEngine],
    embed: TileTensor[
        mut=False, dtype, EmbedLayoutType, embed_origin, Engine=EmbedEngine
    ],
    prev: TileTensor[
        mut=False, dtype, PrevLayoutType, prev_origin, Engine=PrevEngine
    ],
    enorm_weight: TileTensor[
        mut=False, dtype, EWLayoutType, ew_origin, Engine=EWEngine
    ],
    hnorm_weight: TileTensor[
        mut=False, dtype, HWLayoutType, hw_origin, Engine=HWEngine
    ],
    epsilon: Float32,
    num_tokens: Int32,
):
    """The implementation this change replaces, frozen for the bench.

    The bench times the new kernel against it and holds the two to the same
    output bits, so the comparison is a measurement of the change and not of
    a shape or an input.

    Parameters:
        dtype: Element type of the inputs, the weights and the output.
        OutLayoutType: Layout of `out_buf`.
        out_origin: Origin of `out_buf`.
        EmbedLayoutType: Layout of `embed`.
        embed_origin: Origin of `embed`.
        PrevLayoutType: Layout of `prev`.
        prev_origin: Origin of `prev`.
        EWLayoutType: Layout of `enorm_weight`.
        ew_origin: Origin of `enorm_weight`.
        HWLayoutType: Layout of `hnorm_weight`.
        hw_origin: Origin of `hnorm_weight`.
        hidden_size: Channels per input; the output row is twice this.
        max_warps_per_block: Power-of-two ceiling on the block's warp count,
            not the launch's actual warp count. `block_reduce_dual_sum` reduces
            over a power-of-two lane group; slots past the real warp count hold
            zero.
        OutEngine: Engine policy of `out_buf`.
        EmbedEngine: Engine policy of `embed`.
        PrevEngine: Engine policy of `prev`.
        EWEngine: Engine policy of `enorm_weight`.
        HWEngine: Engine policy of `hnorm_weight`.

    Args:
        out_buf: Output `[num_tokens, 2 * hidden_size]`. Columns
            `[0, hidden_size)` hold the normalized embedding, the rest the
            normalized hidden state.
        embed: Token embeddings `[num_tokens, hidden_size]`.
        prev: Target hidden states `[num_tokens, hidden_size]`.
        enorm_weight: Embedding norm weight `[hidden_size]`.
        hnorm_weight: Hidden-state norm weight `[hidden_size]`.
        epsilon: Added inside the square root, matching `ops.rms_norm`.
        num_tokens: Rows to process.
    """
    var token = block_idx.x
    if token >= Int(num_tokens):
        return

    # `block_reduce_dual_sum` reads one partial sum per whole warp, so a block
    # width that is not a multiple of `WARP_SIZE` drops its last partial warp.
    debug_assert(
        block_dim.x % WARP_SIZE == 0,
        "mtp_eh_norm block width must be a whole number of warps",
    )

    var tid = thread_idx.x
    var stride = block_dim.x
    var row = token * hidden_size
    var embed_flat = embed.reshape(row_major(embed.num_elements()))
    var prev_flat = prev.reshape(row_major(prev.num_elements()))

    # Reduce in float32 regardless of the input dtype, matching `ops.rms_norm`.
    var acc_e = Float32(0)
    var acc_h = Float32(0)
    var col = tid
    while col < hidden_size:
        var e = embed_flat.load[width=1]((row + col,)).cast[.float32]()
        var p = prev_flat.load[width=1]((row + col,)).cast[.float32]()
        acc_e += e * e
        acc_h += p * p
        col += stride

    var sums = block_reduce_dual_sum[.float32, max_warps_per_block](
        acc_e, acc_h
    )
    var rrms_e = rsqrt(sums[0] / Float32(hidden_size) + epsilon)
    var rrms_h = rsqrt(sums[1] / Float32(hidden_size) + epsilon)

    var out_row = token * 2 * hidden_size
    var out_flat = out_buf.reshape(row_major(out_buf.num_elements()))
    var enorm_flat = enorm_weight.reshape(row_major[hidden_size]())
    var hnorm_flat = hnorm_weight.reshape(row_major[hidden_size]())
    col = tid
    while col < hidden_size:
        # Cast before the multiply: `multiply_before_cast=False`.
        var e = embed_flat.load[width=1]((row + col,)).cast[.float32]() * rrms_e
        var p = prev_flat.load[width=1]((row + col,)).cast[.float32]() * rrms_h
        out_flat.store(
            (out_row + col,),
            e.cast[dtype]() * enorm_flat.load[width=1]((col,)),
        )
        out_flat.store(
            (out_row + hidden_size + col,),
            p.cast[dtype]() * hnorm_flat.load[width=1]((col,)),
        )
        col += stride


def test_mtp_eh_norm[
    hidden_size: Int, block_threads: Int
](num_tokens: Int, ctx: DeviceContext) raises:
    """Checks every output channel of every token against an f64 reference.

    Parameters:
        hidden_size: Channels per input.
        block_threads: Threads per block; the row is walked in strides of it.

    Args:
        num_tokens: Rows to normalize.
        ctx: Device context.
    """
    comptime eps32 = Float32(1e-5)
    # A power-of-two ceiling, the same value the graph op passes.
    comptime max_warps = (
        ctx.default_device_info.max_thread_block_size
        // ctx.default_device_info.warp_size
    )
    var eps = Float64(1e-5)

    print(
        "mtp_eh_norm: hidden=",
        hidden_size,
        " tokens=",
        num_tokens,
        " threads=",
        block_threads,
    )

    var embed_h = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var prev_h = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * hidden_size
    )
    var ew_h = ctx.enqueue_create_host_buffer[.bfloat16](hidden_size)
    var hw_h = ctx.enqueue_create_host_buffer[.bfloat16](hidden_size)
    var out_h = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * 2 * hidden_size
    )
    ctx.synchronize()

    # The expectations below are exact, so the inputs have to be fixed.
    seed(0)
    rand(embed_h.unsafe_ptr(), num_tokens * hidden_size)
    rand(prev_h.unsafe_ptr(), num_tokens * hidden_size)
    rand(ew_h.unsafe_ptr(), hidden_size)
    rand(hw_h.unsafe_ptr(), hidden_size)
    # Different scales per half, so reusing one row's sum for both shows up.
    for i in range(num_tokens * hidden_size):
        embed_h[i] = (embed_h[i] - 0.5) * 2.0
        prev_h[i] = (prev_h[i] - 0.5) * 6.0
    for i in range(hidden_size):
        ew_h[i] = ew_h[i] + 0.5
        hw_h[i] = hw_h[i] + 0.5

    var embed_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * hidden_size)
    var prev_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * hidden_size)
    var ew_d = ctx.enqueue_create_buffer[.bfloat16](hidden_size)
    var hw_d = ctx.enqueue_create_buffer[.bfloat16](hidden_size)
    var out_d = ctx.enqueue_create_buffer[.bfloat16](
        num_tokens * 2 * hidden_size
    )
    ctx.enqueue_copy(embed_d, embed_h)
    ctx.enqueue_copy(prev_d, prev_h)
    ctx.enqueue_copy(ew_d, ew_h)
    ctx.enqueue_copy(hw_d, hw_h)
    ctx.synchronize()

    var out_t = TileTensor(out_d, row_major(num_tokens, 2 * hidden_size))
    var embed_t = TileTensor(embed_d, row_major(num_tokens, hidden_size))
    var prev_t = TileTensor(prev_d, row_major(num_tokens, hidden_size))
    var ew_t = TileTensor(ew_d, row_major(hidden_size))
    var hw_t = TileTensor(hw_d, row_major(hidden_size))

    comptime kernel = mtp_eh_norm_kernel[
        .bfloat16,
        out_t.LayoutType,
        out_t.origin,
        type_of(embed_t.as_imm()).LayoutType,
        ImmOrigin(embed_t.origin),
        type_of(prev_t.as_imm()).LayoutType,
        ImmOrigin(prev_t.origin),
        type_of(ew_t.as_imm()).LayoutType,
        ImmOrigin(ew_t.origin),
        type_of(hw_t.as_imm()).LayoutType,
        ImmOrigin(hw_t.origin),
        hidden_size,
        block_threads,
        max_warps,
        out_t.Engine,
        type_of(embed_t.as_imm()).Engine,
        type_of(prev_t.as_imm()).Engine,
        type_of(ew_t.as_imm()).Engine,
        type_of(hw_t.as_imm()).Engine,
    ]
    ctx.enqueue_function[kernel](
        out_t,
        embed_t.as_imm(),
        prev_t.as_imm(),
        ew_t.as_imm(),
        hw_t.as_imm(),
        eps32,
        Int32(num_tokens),
        grid_dim=(num_tokens, 1, 1),
        block_dim=(block_threads, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # The exact check above only pins the multiply ordering on channels where
    # the two orderings actually produce different bits, so count those.
    var orderings_differ = 0

    for t in range(num_tokens):
        for c in range(hidden_size):
            var want_e = _reference_row[hidden_size](
                embed_h, ew_h, t, c, eps, False
            )
            var got_e = Float64(out_h[t * 2 * hidden_size + c].cast[.float32]())
            assert_equal(
                got_e,
                want_e,
                String("embed half, token ", t, " channel ", c, " mismatch"),
            )

            var want_h = _reference_row[hidden_size](
                prev_h, hw_h, t, c, eps, False
            )
            var got_h = Float64(
                out_h[t * 2 * hidden_size + hidden_size + c].cast[.float32]()
            )
            assert_equal(
                got_h,
                want_h,
                String("hidden half, token ", t, " channel ", c, " mismatch"),
            )

            var other = _reference_row[hidden_size](
                embed_h, ew_h, t, c, eps, True
            )
            if other != want_e:
                orderings_differ += 1

    assert_true(
        orderings_differ > 0,
        (
            "the two multiply orderings agreed everywhere on this data, so"
            " matching one of them says nothing -- pick inputs that separate"
            " them"
        ),
    )
    print(
        "  matched the cast-before-multiply ordering on",
        num_tokens * 2 * hidden_size,
        "channels;",
        orderings_differ,
        "would have differed under the other ordering",
    )

    _ = embed_d
    _ = prev_d
    _ = ew_d
    _ = hw_d
    _ = out_d


def _mix64(x0: UInt64) -> UInt64:
    var x = x0
    x ^= x >> 30
    x *= 0xBF58476D1CE4E5B9
    x ^= x >> 27
    x *= 0x94D049BB133111EB
    x ^= x >> 31
    return x


def _hashed(seed_value: Int, stream: UInt64, i: Int) -> Float32:
    """`mask_harness.gen_value`: k/8 for k in [-15, 15], exact in bf16."""
    return (
        Float32(
            Int(
                _mix64(
                    UInt64(seed_value) * 0x9E3779B97F4A7C15
                    + stream * 0xD1B54A32D192ED03
                    + UInt64(i)
                )
                % 31
            )
            - 15
        )
        * 0.125
    )


def _replay_us(ctx: DeviceContext, graph: DeviceGraph) raises -> Float64:
    """The mean of `GRAPH_REPS` back-to-back replays of `graph` after
    `GRAPH_WARMUP` warmup ones, in microseconds: the rule the campaign and
    the engines use to report kernel time (vLLM's `time_graph`), so a number
    here is comparable with a `graph100` number there."""
    for _ in range(GRAPH_WARMUP):
        graph.replay()
    ctx.synchronize()
    var t0 = monotonic()
    for _ in range(GRAPH_REPS):
        graph.replay()
    ctx.synchronize()
    # `monotonic` is nanoseconds; round to nearest the way the harness does.
    return (
        Float64(monotonic() - t0 + GRAPH_REPS // 2)
        / Float64(GRAPH_REPS)
        / 1000.0
    )


def bench_case[
    hidden_size: Int, block_threads: Int
](counts: List[Int], ctx: DeviceContext) raises:
    """Times the new kernel (`PDLLevel.ON`, with the launch attribute)
    against the implementation it replaces (`PDLLevel.OFF` on the frozen
    body), on the same buffers, alternating the arm order every round and
    reporting the fastest round of each; the two are held to the same output
    bits first.

    The two arms differ in exactly two things: the kernel's call to
    `wait_on_dependent_grids`/`launch_dependent_grids`, and the launch
    attribute. `graph100`'s 100 back-to-back replays are 100 consecutive
    grids in one stream, so what the attribute buys -- the next launch
    starting while the current grid drains -- is on the critical path here.
    """
    comptime eps32 = Float32(1e-5)
    comptime max_warps = (
        ctx.default_device_info.max_thread_block_size
        // ctx.default_device_info.warp_size
    )
    print(
        "mtp_eh_norm bench: hidden ",
        hidden_size,
        " threads ",
        block_threads,
        " rounds ",
        BENCH_ROUNDS,
        " reps ",
        GRAPH_REPS,
        sep="",
    )
    for num_tokens in counts:
        # Hashed values (k/8): their squares and every partial sum are exact
        # in f32, so the two kernels' different groupings of the row reduce
        # to the same sum and the bit gate below is a property of the
        # arithmetic rather than of these particular inputs.
        var embed_h = ctx.enqueue_create_host_buffer[.bfloat16](
            num_tokens * hidden_size
        )
        var prev_h = ctx.enqueue_create_host_buffer[.bfloat16](
            num_tokens * hidden_size
        )
        var ew_h = ctx.enqueue_create_host_buffer[.bfloat16](hidden_size)
        var hw_h = ctx.enqueue_create_host_buffer[.bfloat16](hidden_size)
        ctx.synchronize()
        for i in range(num_tokens * hidden_size):
            embed_h[i] = Scalar[.bfloat16](_hashed(1, 1, i))
            prev_h[i] = Scalar[.bfloat16](_hashed(1, 2, i))
        for i in range(hidden_size):
            ew_h[i] = Scalar[.bfloat16](_hashed(1, 3, i))
            hw_h[i] = Scalar[.bfloat16](_hashed(1, 4, i))

        var embed_d = ctx.enqueue_create_buffer[.bfloat16](
            num_tokens * hidden_size
        )
        var prev_d = ctx.enqueue_create_buffer[.bfloat16](
            num_tokens * hidden_size
        )
        var ew_d = ctx.enqueue_create_buffer[.bfloat16](hidden_size)
        var hw_d = ctx.enqueue_create_buffer[.bfloat16](hidden_size)
        var out_d = ctx.enqueue_create_buffer[.bfloat16](
            num_tokens * 2 * hidden_size
        )
        ctx.enqueue_copy(embed_d, embed_h)
        ctx.enqueue_copy(prev_d, prev_h)
        ctx.enqueue_copy(ew_d, ew_h)
        ctx.enqueue_copy(hw_d, hw_h)
        ctx.synchronize()

        var out_t = TileTensor(out_d, row_major(num_tokens, 2 * hidden_size))
        var embed_t = TileTensor(embed_d, row_major(num_tokens, hidden_size))
        var prev_t = TileTensor(prev_d, row_major(num_tokens, hidden_size))
        var ew_t = TileTensor(ew_d, row_major(hidden_size))
        var hw_t = TileTensor(hw_d, row_major(hidden_size))

        comptime base = mtp_eh_norm_base_kernel[
            .bfloat16,
            out_t.LayoutType,
            out_t.origin,
            type_of(embed_t.as_imm()).LayoutType,
            ImmOrigin(embed_t.origin),
            type_of(prev_t.as_imm()).LayoutType,
            ImmOrigin(prev_t.origin),
            type_of(ew_t.as_imm()).LayoutType,
            ImmOrigin(ew_t.origin),
            type_of(hw_t.as_imm()).LayoutType,
            ImmOrigin(hw_t.origin),
            hidden_size,
            max_warps,
            out_t.Engine,
            type_of(embed_t.as_imm()).Engine,
            type_of(prev_t.as_imm()).Engine,
            type_of(ew_t.as_imm()).Engine,
            type_of(hw_t.as_imm()).Engine,
        ]
        comptime pdl = mtp_eh_norm_kernel[
            .bfloat16,
            out_t.LayoutType,
            out_t.origin,
            type_of(embed_t.as_imm()).LayoutType,
            ImmOrigin(embed_t.origin),
            type_of(prev_t.as_imm()).LayoutType,
            ImmOrigin(prev_t.origin),
            type_of(ew_t.as_imm()).LayoutType,
            ImmOrigin(ew_t.origin),
            type_of(hw_t.as_imm()).LayoutType,
            ImmOrigin(hw_t.origin),
            hidden_size,
            block_threads,
            max_warps,
            out_t.Engine,
            type_of(embed_t.as_imm()).Engine,
            type_of(prev_t.as_imm()).Engine,
            type_of(ew_t.as_imm()).Engine,
            type_of(hw_t.as_imm()).Engine,
            pdl_level=PDLLevel.ON,
        ]

        def build_base(mut b: DeviceGraphBuilder) raises {imm}:
            b.recording_context().enqueue_function[base](
                out_t,
                embed_t.as_imm(),
                prev_t.as_imm(),
                ew_t.as_imm(),
                hw_t.as_imm(),
                eps32,
                Int32(num_tokens),
                grid_dim=(num_tokens, 1, 1),
                block_dim=(block_threads, 1, 1),
            )

        def build_pdl(mut b: DeviceGraphBuilder) raises {imm}:
            b.recording_context().enqueue_function[pdl](
                out_t,
                embed_t.as_imm(),
                prev_t.as_imm(),
                ew_t.as_imm(),
                hw_t.as_imm(),
                eps32,
                Int32(num_tokens),
                grid_dim=(num_tokens, 1, 1),
                block_dim=(block_threads, 1, 1),
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        comptime off_kernel = mtp_eh_norm_kernel[
            .bfloat16,
            out_t.LayoutType,
            out_t.origin,
            type_of(embed_t.as_imm()).LayoutType,
            ImmOrigin(embed_t.origin),
            type_of(prev_t.as_imm()).LayoutType,
            ImmOrigin(prev_t.origin),
            type_of(ew_t.as_imm()).LayoutType,
            ImmOrigin(ew_t.origin),
            type_of(hw_t.as_imm()).LayoutType,
            ImmOrigin(hw_t.origin),
            hidden_size,
            block_threads,
            max_warps,
            out_t.Engine,
            type_of(embed_t.as_imm()).Engine,
            type_of(prev_t.as_imm()).Engine,
            type_of(ew_t.as_imm()).Engine,
            type_of(hw_t.as_imm()).Engine,
            pdl_level=PDLLevel.OFF,
        ]

        def build_off_kernel(mut b: DeviceGraphBuilder) raises {imm}:
            b.recording_context().enqueue_function[off_kernel](
                out_t,
                embed_t.as_imm(),
                prev_t.as_imm(),
                ew_t.as_imm(),
                hw_t.as_imm(),
                eps32,
                Int32(num_tokens),
                grid_dim=(num_tokens, 1, 1),
                block_dim=(block_threads, 1, 1),
            )

        var graph_base = DeviceGraph.create(ctx, build_base)
        var graph_off_kernel = DeviceGraph.create(ctx, build_off_kernel)
        var graph_pdl = DeviceGraph.create(ctx, build_pdl)
        var off_us = Float64(1.0e30)
        var base_out = ctx.enqueue_create_host_buffer[.bfloat16](
            num_tokens * 2 * hidden_size
        )
        var new_out = ctx.enqueue_create_host_buffer[.bfloat16](
            num_tokens * 2 * hidden_size
        )
        var base_us = Float64(1.0e30)
        var pdl_us = Float64(1.0e30)
        for r in range(BENCH_ROUNDS):
            if r % 2 == 1:
                base_us = min(base_us, _replay_us(ctx, graph_base))
                off_us = min(off_us, _replay_us(ctx, graph_off_kernel))
                pdl_us = min(pdl_us, _replay_us(ctx, graph_pdl))
            else:
                pdl_us = min(pdl_us, _replay_us(ctx, graph_pdl))
                off_us = min(off_us, _replay_us(ctx, graph_off_kernel))
                base_us = min(base_us, _replay_us(ctx, graph_base))
        # Correctness gates the speed, on this vendor: the port must produce
        # the bits of the kernel that ships, element for element.
        graph_base.replay()
        ctx.synchronize()
        ctx.enqueue_copy(base_out, out_d)
        # Poison the output before the second replay: an element the new
        # kernel fails to write would otherwise still hold the base arm's
        # bits and compare equal.
        var poison_h = ctx.enqueue_create_host_buffer[.bfloat16](
            num_tokens * 2 * hidden_size
        )
        for i in range(num_tokens * 2 * hidden_size):
            poison_h[i] = Scalar[.bfloat16](
                bitcast[DType.float32](UInt32(0x7FC00000))
            )
        ctx.enqueue_copy(out_d, poison_h)
        ctx.synchronize()
        graph_pdl.replay()
        ctx.synchronize()
        ctx.enqueue_copy(new_out, out_d)
        ctx.synchronize()
        var diff = 0
        for i in range(num_tokens * 2 * hidden_size):
            if base_out[i] != new_out[i]:
                diff += 1
        assert_true(
            diff == 0,
            String(
                "the new kernel differs from the one it replaces on ",
                diff,
                " of ",
                num_tokens * 2 * hidden_size,
                " elements",
            ),
        )
        print(
            "  bits: new == replaced on ",
            num_tokens * 2 * hidden_size,
            " elements",
            sep="",
        )
        print(
            "  ",
            hidden_size,
            "x",
            num_tokens,
            " base_us=",
            base_us,
            " off_us=",
            off_us,
            " pdl_us=",
            pdl_us,
            " rewrite=",
            base_us / off_us,
            " overlap=",
            off_us / pdl_us,
            sep="",
        )
        _ = embed_d
        _ = prev_d
        _ = ew_d
        _ = hw_d
        _ = out_d
        _ = embed_h
        _ = prev_h
        _ = ew_h
        _ = hw_h


def main() raises:
    with DeviceContext() as ctx:
        var args = argv()
        if len(args) > 1 and String(args[1]) == "--bench":
            var counts = List[Int]()
            for i in range(2, len(args)):
                counts.append(atol(args[i]))
            if len(counts) == 0:
                counts = [1, 5, 128, 256]
            bench_case[hidden_size=4096, block_threads=256](counts, ctx)
            return
        comptime warp_size = ctx.default_device_info.warp_size
        # GLM-5.3-Flash / DeepSeek-V3.2 draft width.
        test_mtp_eh_norm[hidden_size=4096, block_threads=256](
            num_tokens=3, ctx=ctx
        )
        # 384 is not a multiple of 256: the last strided pass is ragged.
        test_mtp_eh_norm[hidden_size=384, block_threads=256](
            num_tokens=2, ctx=ctx
        )
        # Three warps, stated in `warp_size` so it stays three on every
        # target. `block_reduce_dual_sum` reduces over a power-of-two lane
        # group, so a non-power-of-two warp count can drop a warp's sum. The
        # other shapes here all have power-of-two warp counts.
        test_mtp_eh_norm[hidden_size=384, block_threads=3 * warp_size](
            num_tokens=2, ctx=ctx
        )
        # Five and seven warps, for the same reason.
        test_mtp_eh_norm[hidden_size=1024, block_threads=5 * warp_size](
            num_tokens=2, ctx=ctx
        )
        test_mtp_eh_norm[hidden_size=2048, block_threads=7 * warp_size](
            num_tokens=2, ctx=ctx
        )
        # One warp per block: the cross-warp path is skipped.
        test_mtp_eh_norm[hidden_size=512, block_threads=warp_size](
            num_tokens=2, ctx=ctx
        )
        print("\nAll tests passed!")
