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
"""The multicast residual boundary: allreduce + add + RMSNorm (+ FP8) in one launch.

The tensor-parallel residual boundary of a decoder layer. Each rank multicasts
its own 128-bit pack of a row once into its own rank slot of the team's
symmetric window (the fabric performs the fan-out, so the store costs one
transaction however wide the team is), polls its own copy until no peer's slot
still holds the "not arrived" sentinel, reduces the copies in fp32 in global
rank order, and runs the residual add, the RMSNorm and the 1x128-group E4M3
quant in the same launch.

This is the shipped v2 protocol of the multicast boundary component. The kernel
takes raw pointers and indexes them itself; only the `mcar_rmsnorm` wrapper sees
TileTensors. The Moka component
`Kernels/test/gpu/Moka/kernels/comm/mcar_plan.mojo` models that protocol on its
own, as an independent model rather than a gate on this kernel's bytes: no test
runs the two on the same inputs. Only this kernel has the cast-first weight
order, PDL, and a runtime column count and team size; the component's quantized
epilogue also stores the bf16 normed row, and its call state has a different
layout. The statement sequence, one CTA per row, is

1. probe: thread 0 reads this rank's call number and the written extent of the
   generation two calls ahead, publishes both into block scratch, and a barrier
   carries them to the block. Nothing after the probe reads the state word, so
   the arrival below can advance it safely;
2. fill: each thread sanitizes its own pack (`-0.0` lanes become `+0.0`, so real
   data can never read as the sentinel) and multicasts it;
3. arrival: thread 0 tallies on the launch's counter, and the last CTA advances
   the call number, records this call's extent for the generation it used, and
   resets the tally. The tally's fetch-add is acquire-release: the flip reads
   nothing this launch wrote, but it must be ordered against every CTA's probe
   read, which a relaxed or release-only add does not guarantee. A one-CTA
   launch writes the flip directly. That ordering is the one property of the
   kernel no GPU test can observe -- a relaxed add passes every correctness
   test -- so `test_mcar_codegen.mojo` compiles the tally for sm_100 and holds
   the emitted atomic to its acq_rel qualifier;
4. clear: one multicast store of the sentinel per pack into the generation
   reused two calls ahead, over the extent that generation was written with. A
   narrow call (fewer than `MCAR_CLEAR_ROWS` rows) clears inside its exchange,
   where the pass rides in the slack of a latency-bound wait; a wide call clears
   after its epilogue, so the sentinel traffic never queues in front of a peer's
   fill. Exactly one of the two runs per call;
5. poll and fold, then the epilogue: residual add, the pre-norm stream, the
   row's sum of squares, `rsqrt`, the norm weight, and either the normed row or
   the group-scaled FP8 row plus its scales.

The quant reads the normed values the norm already holds, rounded to the message
dtype first, so the fused epilogue quantizes exactly the bytes the unfused pair
would have written and read back. Two details make that exact rather than close.
The norm weight's additive offset is already rounded to the message dtype at the
dispatch boundary (`weight_offset.cast[dtype]()`), and every arm -- this one,
`_rms_norm_rows`, the Lamport arm and the warp-tiling one -- adds
`gamma + offset` in fp32, so the effective weight is one value here and in the
unfused norm at any offset. And the norm's epsilon is applied in fp32, which is
`_rms_norm_rows`'s precision; the shipped fused kernels take epsilon as
`Scalar[dtype]` and round it to the message dtype first, so on that one input the
port agrees with the unfused reference rather than with them.

Four generations ring the window, and each rank clears only its own rank slot
in the generation it reused two calls ago: a rank's calls are ordered on its
stream, so clearing a slot it last read is safe without knowing anything about
its peers, and a rank can be at most one call ahead because its call `n` waits
for the peer's call `n` pack. Four rather than the Lamport exchange's three,
which is where the count would otherwise come from: the launch that writes
generation `g` and the launch that clears it are separated by a generation flip
on one rank's stream, so the extra slot keeps a rank that is a call ahead from
reusing a slot its peer has not cleared. The call number is device-resident, so
a captured graph replays it with no host argument.

Numerics: the sum is fp32 in global rank order, the residual stream is
`sum + residual` rounded once to the message dtype, and the quant's amax is over
the bf16-rounded normed row. The sum of squares is over the *unrounded* fp32
stream, as in the shipped fused arms this one replaces (the 1-stage kernel in
`comm.allreduce_residual_rmsnorm` and the Lamport arm in
`comm.allreduce_lamport_rmsnorm`); `_rms_norm_rows` squares the rounded values
instead, so it is the reference only for epsilon's precision and the weight's
order. The norm weight is applied in the order the call asks for: in fp32 before
the single cast to the message dtype (`multiply_before_cast=True`) or after it,
on the rounded value and in the message dtype. Epsilon is fp32, as it is in
`_rms_norm_rows`; `comm.allreduce_lamport_rmsnorm` takes epsilon as
`Scalar[dtype]` and rounds it to the message dtype first, so on that input this
kernel matches the reference the unfused chain writes, not that arm.

Geometry: one CTA per row, one thread per 16-byte pack, so a row's pack count is
the block size and has to be a compile-time geometry (`mcar_supports_cols`). For
the shipped MiMo-V2 boundary that is 512 threads and a 4096-wide hidden.
"""

from std.atomic import Atomic, Ordering
from std.math import abs, max, min, rsqrt
from std.utils import StaticTuple
from std.utils.numerics import max_finite, min_finite

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import Scope
from max.gpu.memory import Consistency, multimem_st
from max.gpu.primitives import warp
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import barrier
from std.memory import bitcast, unsafe_stack_allocation
from std.os import abort

from internal_utils.fp8_utils import guarded_inv_scale

from .lamport import has_neg_zero, remove_neg_zero, set_neg_zero
from .multicast_window import (
    MCAR_GENERATIONS,
    MCAR_STATE_ARRIVAL,
    MCAR_STATE_EXTENT,
    MCAR_STATE_FLAG,
    McarWindowRank,
    mcar_payload_index,
)
from .sync import MAX_GPUS

comptime MCAR_PACK = 8
"""Lanes of the 128-bit pack the protocol's atomicity and sentinel need."""

comptime MCAR_GROUP = 128
"""Columns per FP8 scale group: the 1x128 block the consuming GEMM expects."""

comptime MCAR_PACKS_PER_GROUP = MCAR_GROUP // MCAR_PACK
"""Packs in one scale group: 16 adjacent threads, half a warp."""

comptime MCAR_FP8_MAX = Float32(448.0)
"""E4M3's largest magnitude, the divisor of a dynamic group scale."""

comptime MCAR_CLEAR_ROWS = 16
"""Rows at which the clear moves from inside the exchange to after the epilogue.

Measured on four B200s at TP4 and a 4096-wide hidden: the deferred clear is
0.3-0.5 us slower per call at rows 1-16, where the call is latency-bound and the
sentinel pass rides in the slack of the wait, and 0.7 us (rows 64) to 1.5 us
(rows 128) faster from 32 rows up, where the fabric is the limit and a clear in
front of the poll delays every peer's arrival.
"""


@inline(.always)
def mcar_supports_cols[ngpus: Int](cols: Int) -> Bool:
    """Returns whether a `cols`-wide row has a geometry the arm instantiates.

    A row is covered by one CTA, so its pack count is the launch's block size
    and has to be a compile-time geometry: 1024, 2048, 4096 or 8192 bf16
    columns, whole warps and whole 1x128 scale groups.

    Parameters:
        ngpus: Number of participating ranks.

    Args:
        cols: Row width.
    """
    comptime if ngpus < 2 or ngpus > MAX_GPUS:
        return False
    if cols % MCAR_PACK != 0:
        return False
    var packs = cols // MCAR_PACK
    return packs == 128 or packs == 256 or packs == 512 or packs == 1024


@inline(.always)
def _sentinel_pack() -> SIMD[DType.bfloat16, MCAR_PACK]:
    """The "nothing arrived" pattern, from the arm that shares it with the
    Lamport exchange so the two cannot disagree on what a written slot is."""
    return set_neg_zero[DType.bfloat16, MCAR_PACK]()


@inline(.always)
def _is_sentinel(v: SIMD[DType.bfloat16, MCAR_PACK]) -> Bool:
    """Whether any lane of `v` is still the `-0.0` sentinel."""
    return has_neg_zero[DType.bfloat16, MCAR_PACK](v)


@inline(.always)
def _sanitize(
    v: SIMD[DType.bfloat16, MCAR_PACK]
) -> SIMD[DType.bfloat16, MCAR_PACK]:
    """Maps every `-0.0` lane to `+0.0`, leaving every other lane bit-exact.

    Load-bearing: this rank's own pack must never read as "not arrived", or the
    poll consumes a slot before its writer filled it. A `-0.0` input lane is
    real data (a negatively signed zero) and becomes `+0.0`.
    """
    return remove_neg_zero[DType.bfloat16, MCAR_PACK](v)


@inline(.always)
def _multicast_pack(
    mc: Pointer[Scalar[DType.bfloat16], _, address_space=.GLOBAL],
    index: Int,
    v: SIMD[DType.bfloat16, MCAR_PACK],
):
    """One multicast store of a whole 128-bit pack, as two 64-bit stores.

    The fabric takes no vector qualifier wider than eight bytes from this
    toolchain (`multimem.st` with a 128-bit vector operand is rejected), so a
    pack is two `.v2.f32` stores, which is what the shared helper emits for a
    two-lane float32 value.
    """
    comptime words = MCAR_PACK // 2
    var values = bitcast[DType.float32, words](v)
    var base = Pointer[
        Scalar[DType.float32], MutUntrackedOrigin, address_space=.GLOBAL
    ](unsafe_from_address=Int(mc)).unsafe_offset(index // 2)
    comptime for i in range(words // 2):
        multimem_st[
            .float32,
            simd_width=2,
            scope=Scope.SYSTEM,
            consistency=Consistency.RELAXED,
        ](base.unsafe_offset(2 * i), values.slice[2, offset=2 * i]())


@inline(.always)
def _row_block_sum[
    warps: Int
](
    v: Float32, red: Pointer[Float32, MutUntrackedOrigin, address_space=.SHARED]
) -> Float32:
    """The row's total of `v`, the same bits in every thread of the block.

    Each warp publishes its own sum and every warp then reduces the published
    words in the same lane order, so the result is identical everywhere without
    a second meeting or a broadcast.
    """
    var ws = warp.sum(v)
    var tid = Int(thread_idx.x)
    if tid % 32 == 0:
        red.unsafe_store[width=1](tid // 32, ws)
    barrier()
    var t = Float32(0)
    var l = Int(lane_id())
    if l < warps:
        t = red.unsafe_load[width=1](l)
    return warp.sum(t)


@inline(.always)
def _group_max(v: Float32) -> Float32:
    """The max of `v` over its scale group: 16 adjacent lanes, half a warp, so a
    four-step butterfly stays inside it."""
    var m = v
    comptime for i in range(4):
        m = max(m, warp.shuffle_xor(m, UInt32(1 << i)))
    return m


@inline(.always)
def _set_word(base: MutPointer[UInt32, MutUntrackedOrigin], w: Int, v: Int):
    base.unsafe_store[width=1, volatile=True](w, UInt32(v))


@inline(.always)
def _flip(state: MutPointer[UInt32, MutUntrackedOrigin], gen: Int, packs: Int):
    """Writes this call's outcome: the extent is recorded for the generation the
    call used, the call number advances, and the tally resets.

    Only the CTA the arrival gate selects calls this, so the three stores are
    one CTA's, in program order.
    """
    _set_word(state, MCAR_STATE_EXTENT + gen, packs)
    _set_word(
        state,
        MCAR_STATE_FLAG,
        Int(state.unsafe_load[width=1, volatile=True](MCAR_STATE_FLAG)) + 1,
    )
    _set_word(state, MCAR_STATE_ARRIVAL, 0)


@inline(.always)
def _arrive_and_flip(
    state: MutPointer[UInt32, MutUntrackedOrigin], gen: Int, packs: Int
):
    """This CTA's arrival on the launch's tally, and the flip if it is the last.

    The fetch-add is acquire-release: the flip an arriving CTA can trigger
    mutates the word every CTA of the launch read at its probe, so the release
    half publishes this CTA's read and the acquire half orders the last CTA's
    flip after the others' reads. A grid of one has nothing to wait for.
    """
    if Int(grid_dim.x) == 1:
        _flip(state, gen, packs)
        return
    var arrived = Atomic[UInt32, scope="device"].fetch_add[
        ordering=Ordering.ACQUIRE_RELEASE
    ](
        Pointer[UInt32, MutUntrackedOrigin](
            unsafe_from_address=Int(state) + 4 * MCAR_STATE_ARRIVAL
        ),
        UInt32(1),
    )
    if Int(arrived) == Int(grid_dim.x) - 1:
        _flip(state, gen, packs)


@inline(.always)
def _clear_generation[
    ngpus: Int, packs: Int
](
    mc: Pointer[Scalar[DType.bfloat16], _, address_space=.GLOBAL],
    rank: Int,
    row: Int,
    rows: Int,
    prev: Int,
    clear_gen: Int,
    cols: Int,
    rows_max: Int,
):
    """Multicasts the sentinel into the generation this call reuses two calls
    ahead, over the extent that generation was written with.

    `prev` is the extent the call two ahead recorded, not this call's: the two
    differ whenever a shape changes between calls. Packs past this call's grid (a
    larger earlier call) are walked grid-stride.
    """
    comptime pack = MCAR_PACK
    var tid = Int(thread_idx.x)
    var local = row * packs + tid
    var sentinel = _sentinel_pack()
    if local < prev:
        _multicast_pack(
            mc,
            mcar_payload_index(
                clear_gen, row, rank, tid * pack, rows_max, ngpus, cols
            ),
            sentinel,
        )
    var cur = rows * packs
    var gtid = Int(block_idx.x) * packs + tid
    var nthreads = Int(grid_dim.x) * packs
    var gp = cur + gtid
    while gp < prev:
        var g_row = gp // packs
        var g_col = (gp % packs) * pack
        _multicast_pack(
            mc,
            mcar_payload_index(
                clear_gen, g_row, rank, g_col, rows_max, ngpus, cols
            ),
            sentinel,
        )
        gp += nthreads


@inline(.always)
def _poll_fold[
    ngpus: Int
](
    own: Pointer[Scalar[DType.bfloat16], _, address_space=.GLOBAL],
    gen: Int,
    row: Int,
    col: Int,
    cols: Int,
    rows_max: Int,
) -> SIMD[DType.float32, MCAR_PACK]:
    """Polls this rank's own copy until no rank's slot holds the sentinel, and
    folds the trip the wait ends on in fp32 in global rank order.

    One trip reads every slot, so the wait ends on the last peer's arrival and
    not after `ngpus - 1` serial round trips.
    """
    comptime pack = MCAR_PACK
    var acc = SIMD[DType.float32, pack](0)
    var done = False
    while not done:
        done = True
        comptime for g in range(ngpus):
            var v = own.unsafe_load[width=pack, alignment=16, volatile=True](
                mcar_payload_index(gen, row, g, col, rows_max, ngpus, cols)
            )
            if _is_sentinel(v):
                done = False
            if g == 0:
                acc = v.cast[DType.float32]()
            else:
                acc += v.cast[DType.float32]()
    return acc


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(packs))
)
@__name(t"mcar_rmsnorm_{ngpus}_{packs}_{quant}_{multiply_before_cast}")
def _mcar_rmsnorm_kernel[
    ngpus: Int,
    packs: Int,
    *,
    quant: Bool,
    multiply_before_cast: Bool,
](
    win: MutPointer[UInt32, MutUntrackedOrigin],
    mc: MutPointer[UInt32, MutUntrackedOrigin],
    state: MutPointer[UInt32, MutUntrackedOrigin],
    src: ImmPointer[Scalar[DType.bfloat16], ImmUntrackedOrigin],
    res: ImmPointer[Scalar[DType.bfloat16], ImmUntrackedOrigin],
    pre: MutPointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    normed: MutPointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    gamma: ImmPointer[Scalar[DType.bfloat16], ImmUntrackedOrigin],
    q: MutPointer[Scalar[DType.float8_e4m3fn], MutUntrackedOrigin],
    qs: MutPointer[Float32, MutUntrackedOrigin],
    rows_dev: Int32,
    cols_dev: Int32,
    scale_stride_dev: Int32,
    rows_max_dev: Int32,
    epsilon: Float32,
    weight_offset: Float32,
    scale_ub: Float32,
    my_rank_dev: Int32,
):
    """One row of the multicast residual boundary: one CTA per row, one thread
    per 128-bit pack, so the block covers the whole row."""
    comptime pack = MCAR_PACK
    comptime warps = packs // 32
    comptime local_words = warps + 2
    comptime gen_word = warps
    comptime prev_word = warps + 1

    var rows = Int(rows_dev)
    var cols = Int(cols_dev)
    var scale_stride = Int(scale_stride_dev)
    var rows_max = Int(rows_max_dev)
    var rank = Int(my_rank_dev)
    var row = block_idx.x
    # The window's row capacity is the arm's capacity, and the launcher refuses
    # a call past it before this grid is built. A CTA here without a row has
    # nothing to write and nothing to read, but it must not return quietly: its
    # peers' arrival tally counts every CTA of the grid, so a silent return
    # hangs them. Trap instead.
    if row >= rows_max:
        abort()

    var tid = Int(thread_idx.x)
    var col = tid * pack
    var off = row * cols + col

    # PDL: wait for the producer before reading its output, then release the
    # consumer so the following GEMM overlaps this boundary.
    wait_on_dependent_grids()
    launch_dependent_grids()

    var state_words = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(state)
    )
    var core = unsafe_stack_allocation[
        local_words, UInt32, address_space=.SHARED
    ]()
    var red = unsafe_stack_allocation[warps, Float32, address_space=.SHARED]()
    var own = Pointer[
        Scalar[DType.bfloat16], MutUntrackedOrigin, address_space=.GLOBAL
    ](unsafe_from_address=Int(win))
    var mcast = Pointer[
        Scalar[DType.bfloat16], MutUntrackedOrigin, address_space=.GLOBAL
    ](unsafe_from_address=Int(mc))

    # (1) The probe: this rank's call number and the extent the call two ahead
    # recorded, read once per block and published into the block's scratch.
    if tid == 0:
        var gen = (
            Int(state.unsafe_load[width=1, volatile=True](MCAR_STATE_FLAG))
            % MCAR_GENERATIONS
        )
        core.unsafe_store[width=1](gen_word, UInt32(gen))
        core.unsafe_store[width=1](
            prev_word,
            state.unsafe_load[width=1, volatile=True](
                MCAR_STATE_EXTENT + (gen + 2) % MCAR_GENERATIONS
            ),
        )
    barrier()
    var gen = Int(core.unsafe_load[width=1](gen_word))
    var prev = Int(core.unsafe_load[width=1](prev_word))
    var clear_gen = (gen + 2) % MCAR_GENERATIONS

    # This pack's own address in the window: the generation's own rank slot.
    var generation_base = mcar_payload_index(
        gen, row, rank, 0, rows_max, ngpus, cols
    )
    # The epilogue's independent operands, issued before the wait so their
    # latency hides in the poll.
    var residual_pack = res.unsafe_load[width=pack, alignment=16](off).cast[
        DType.float32
    ]()
    var gamma_pack = (
        gamma.unsafe_load[width=pack, alignment=16](col).cast[DType.float32]()
        + weight_offset
    )

    # (2) The fill: sanitize this rank's pack and multicast it into its own rank
    # slot of every copy.
    _multicast_pack(
        mcast,
        generation_base + col,
        _sanitize(src.unsafe_load[width=pack, alignment=16](off)),
    )

    # (3) The arrival, once the fill is away: it lands in the shadow of the poll
    # instead of in front of the fills it would delay.
    if tid == 0:
        _arrive_and_flip(state_words, gen, rows * packs)

    # (4a) A narrow call's clear, inside its exchange.
    if rows < MCAR_CLEAR_ROWS:
        _clear_generation[ngpus, packs](
            mcast,
            rank,
            row,
            rows,
            prev,
            clear_gen,
            cols,
            rows_max,
        )

    # (5) The poll and fold, then the epilogue.
    var stream = (
        _poll_fold[ngpus](own, gen, row, col, cols, rows_max) + residual_pack
    )
    pre.unsafe_store[width=pack, alignment=16](
        off, stream.cast[DType.bfloat16]()
    )
    # The sum of squares is over the unrounded fp32 stream, reduced over the row.
    var sumsq = _row_block_sum[warps]((stream * stream).reduce_add(), red)
    var norm = rsqrt(sumsq / Float32(cols) + epsilon)
    var normed_value: SIMD[DType.float32, pack]
    comptime if multiply_before_cast:
        normed_value = (stream * norm) * gamma_pack
    else:
        # `ops.rms_norm`'s cast-first order: round to the message dtype, then
        # apply the weight in it, which is what the unfused norm writes.
        normed_value = (
            (stream * norm).cast[DType.bfloat16]()
            * gamma_pack.cast[DType.bfloat16]()
        ).cast[DType.float32]()
    comptime if quant:
        # The quant consumes a bf16 row and rounds to it first: running the amax
        # off the fp32 registers would move the elements on a rounding boundary
        # by one fp8 step.
        var xq = normed_value.cast[DType.bfloat16]().cast[DType.float32]()
        var group = tid // MCAR_PACKS_PER_GROUP
        var scale = (
            min(_group_max(abs(xq).reduce_max()), scale_ub) / MCAR_FP8_MAX
        )
        if tid % MCAR_PACKS_PER_GROUP == 0:
            qs.unsafe_store[width=1](group * scale_stride + row, scale)
        var y = (xq * guarded_inv_scale[DType.float32](scale)).clamp(
            Float32(min_finite[DType.float8_e4m3fn]()),
            Float32(max_finite[DType.float8_e4m3fn]()),
        )
        q.unsafe_store[width=pack, alignment=pack](
            off, y.cast[DType.float8_e4m3fn]()
        )
    else:
        normed.unsafe_store[width=pack, alignment=16](
            off, normed_value.cast[DType.bfloat16]()
        )

    # (4b) A wide call's clear: after everything this call's peers wait on, so
    # the sentinel pass is issued into an idle fabric.
    if rows >= MCAR_CLEAR_ROWS:
        _clear_generation[ngpus, packs](
            mcast,
            rank,
            row,
            rows,
            prev,
            clear_gen,
            cols,
            rows_max,
        )


def allreduce_mcast_rmsnorm[
    ngpus: Int,
    *,
    quant: Bool,
    multiply_before_cast: Bool = True,
](
    src: ImmPointer[Scalar[DType.bfloat16], _],
    residual: ImmPointer[Scalar[DType.bfloat16], _],
    residual_output: MutPointer[Scalar[DType.bfloat16], _],
    normed: MutPointer[Scalar[DType.bfloat16], _],
    gamma: ImmPointer[Scalar[DType.bfloat16], _],
    q: MutPointer[Scalar[DType.float8_e4m3fn], _],
    scales: MutPointer[Float32, _],
    scale_stride: Int,
    rows: Int,
    cols: Int,
    epsilon: Float32,
    weight_offset: Float32,
    scale_ub: Float32,
    my_rank: Int,
    window: McarWindowRank,
    ctx: DeviceContext,
    pdl_level: PDLLevel = PDLLevel(),
) raises:
    """Launches the multicast residual boundary on this rank's GPU.

    One CTA per row, one thread per 128-bit pack of it. The window is the
    session's symmetric allocation (`comm.multicast_window`); its call state is
    device-resident, so a captured graph replays with no host argument.

    Parameters:
        ngpus: Number of participating ranks.
        quant: Whether the epilogue is the normed row (`False`) or the
            group-scaled FP8 row plus its scales (`True`).
        multiply_before_cast: `ops.rms_norm`'s weight order. True applies gamma
            in float32 before the single cast to the message dtype; False
            casts the normalized value first and applies gamma in it. A
            quantized epilogue applies gamma before its amax and its cast, so
            False is the normed-row epilogue's alone.

    Args:
        src: This rank's `[rows, cols]` partial sum, the GEMM's output.
        residual: `[rows, cols]` residual, replicated on every rank.
        residual_output: `[rows, cols]` pre-norm stream, `sum + residual`.
        normed: `[rows, cols]` normed output, bf16 (the `quant=False` output).
        gamma: `[cols]` RMSNorm weight.
        q: `[rows, cols]` FP8 output (the `quant=True` output).
        scales: `[cols / 128, scale_stride]` group scales, the layout the
            consuming block-scaled GEMM reads.
        scale_stride: The scale tensor's row stride: its second dimension, which
            a block-scaled consumer pads past `rows`.
        rows: Rows (tokens) this call reduces.
        cols: Row width; RMSNorm runs over it.
        epsilon: RMSNorm epsilon.
        weight_offset: Additive offset for the RMS norm weight.
        scale_ub: Upper bound for the group scale.
        my_rank: This rank's index in the team.
        window: This rank's views of the team's symmetric window.
        ctx: Device context for this rank.
        pdl_level: Programmatic dependent launch level.

    Raises:
        If the shape has no geometry this kernel instantiates, if its width is
        not the window's, or if the row count is past the window's capacity.
        The caller's fit test refuses those first; the raise is the backstop.
    """
    comptime pack = MCAR_PACK
    # The quantized epilogue takes its amax and its scale off the fp32 value,
    # so it is the multiply-first order's; the fused arms assert the same of
    # every quantized call.
    comptime assert multiply_before_cast or not quant, (
        "the grouped FP8 epilogue applies gamma before the cast, so"
        " multiply_before_cast=False is the normed-row epilogue's only"
    )
    if rows == 0:
        return
    if cols % pack != 0:
        raise Error("mcar_rmsnorm requires cols % 8 == 0 (whole 128-bit packs)")
    var packs = cols // pack
    if cols != window.cols:
        raise Error(
            t"mcar_rmsnorm call of {cols} columns does not match the window's"
            t" {window.cols}"
        )
    if rows > window.rows:
        raise Error(
            t"mcar_rmsnorm call of {rows} rows exceeds the window's"
            t" {window.rows} rows"
        )

    def launch[geometry: Int]() raises {imm}:
        comptime kernel = _mcar_rmsnorm_kernel[
            ngpus,
            geometry,
            quant=quant,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function[kernel](
            window.payload,
            window.payload_multicast,
            window.state,
            src.unsafe_origin_cast[ImmUntrackedOrigin](),
            residual.unsafe_origin_cast[ImmUntrackedOrigin](),
            residual_output.unsafe_origin_cast[MutUntrackedOrigin](),
            normed.unsafe_origin_cast[MutUntrackedOrigin](),
            gamma.unsafe_origin_cast[ImmUntrackedOrigin](),
            q.unsafe_origin_cast[MutUntrackedOrigin](),
            scales.unsafe_origin_cast[MutUntrackedOrigin](),
            Int32(rows),
            Int32(cols),
            Int32(scale_stride),
            Int32(window.rows),
            epsilon,
            weight_offset,
            scale_ub,
            Int32(my_rank),
            grid_dim=rows,
            block_dim=geometry,
            attributes=pdl_launch_attributes(pdl_level),
        )

    if packs == 128:
        launch[128]()
    elif packs == 256:
        launch[256]()
    elif packs == 512:
        launch[512]()
    elif packs == 1024:
        launch[1024]()
    else:
        raise Error(t"mcar_rmsnorm has no geometry for {packs} packs a row")
