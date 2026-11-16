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
"""The multicast residual boundary against an independent host oracle.

`comm.allreduce_mcast_rmsnorm` is the tensor-parallel residual boundary: each
rank multicasts its own pack into the team's symmetric window, polls its own
copy for its peers' slots, and runs the residual add, the RMSNorm and the
1x128-group E4M3 quant in the same launch. This test pins each claim against a
host reference computed independently of the kernel:

- the residual stream, bit-exact on every rank -- bit patterns, so the sign of a
  zero counts -- against an fp32 reference that adds the ranks in global rank
  order;
- the normed row, within one bf16 ulp of that reference's `rms_norm`, with no
  element's bytes expected to differ from it: a tolerance would admit either
  epilogue order there too, since the two agree within a step on most elements,
  so the rung also counts the elements whose bytes differ from the
  multiply-first reference, against the same small budget the cast-first rung
  allows;
- the cast-first order of the same epilogue (`multiply_before_cast=False`,
  `ops.rms_norm`'s default and what most architectures build) against both
  reference orders at once: inside two ulps of the cast-first one, and off the
  multiply-first one on a large fraction of the row. The second half is what
  makes the rung decisive, because the two orders agree within a step on most
  elements and a tolerance admits either;
- the FP8 row and the group scales, bit-exact against `internal_utils.fp8_utils`
  applied to the normed row the norm arm stored, with a padded scale stride
  whose padding is poisoned before every call;
- calls enqueued back to back with no synchronize between them, eagerly and as
  chained graph replays, so a rank can run a call ahead of its peers while they
  still poll, and each call clears an extent a call of another size wrote;
- the window those calls share, created inside a device graph that is
  recording, which is where serving's warmup creates it, so what the replays run
  against is a window a capture made rather than one an eager call made first.

Every claim takes the row count and the width at runtime, so one instantiation
covers the whole shape ladder. Two widths are needed because the window's row
capacity is the 1 MiB band divided by the row's bytes: a row count past 128 is
only reachable at 2048 columns, and the CTA count has to pass the SM count to
put a call in a second wave.

What the data can and cannot see. Every input is a bf16 value, so a team of at
most eight of them sums exactly in fp32 whatever order the reduce takes: the
stream check pins which rank's slots were folded and which residual was added,
not the addition order. That is the whole observable content of the contract
through a bf16 output: an fp32 reordering moves the sum by well under one part
in 2**24, and the store keeps 2**-9.
"""

from comm.allreduce_mcast_rmsnorm import allreduce_mcast_rmsnorm
from comm.multicast_window import McarWindow, mcar_window
from internal_utils.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    HostBuffer,
)
from std.math import abs, max, rsqrt
from std.memory import bitcast
from std.testing import assert_true

comptime EPS = Float32(1e-6)
"""RMSNorm epsilon, the value the MiMo-V2 boundary runs."""

comptime GROUP = 128
"""Columns per FP8 scale group: the 1x128 block the consuming GEMM expects."""

comptime SCALE_STRIDE = 260
"""A padded scale stride.

The consuming block-scaled GEMM pads the scale tensor's row dimension, so the
kernel must address `[group, row]` with the tensor's stride and never with the
call's row count.
"""

comptime SCALE_UB = Float32(1200.0)
"""Scale ceiling: above the normed row's amax, so only a denormal group's
reciprocal reaches `guarded_inv_scale`'s guard."""

comptime NEG_ZERO = Int(-1)
"""Row index that turns a fill's degenerate-row case off."""


@fieldwise_init
struct Bufs:
    """One rank's device buffers, each sized for the window's row capacity."""

    var src: DeviceBuffer[DType.bfloat16]
    var res: DeviceBuffer[DType.bfloat16]
    var pre: DeviceBuffer[DType.bfloat16]
    var normed: DeviceBuffer[DType.bfloat16]
    var q: DeviceBuffer[DType.float8_e4m3fn]
    var qs: DeviceBuffer[DType.float32]
    var gamma: DeviceBuffer[DType.bfloat16]


@fieldwise_init
struct Shape:
    """One call's width and row count, and the degenerate rows it carries."""

    var rows: Int
    var cols: Int
    var negative_zero_row: Int
    """Row whose every lane is `-0.0`, or `NEG_ZERO`."""

    var tiny_group_row: Int
    """Row whose first scale group is bf16 denormals in every rank's input and
    in the residual, so the normed group's dynamic scale is small enough that
    its reciprocal overflows; or `NEG_ZERO`."""


@inline(.always)
def _bf16(v: Float32) -> Float32:
    """`v` rounded once to bfloat16, as an fp32 value."""
    return v.cast[DType.bfloat16]().cast[DType.float32]()


@inline(.always)
def _bits(v: Scalar[DType.bfloat16]) -> Int:
    """The bf16 bit pattern of `v`, so `-0.0` and `+0.0` differ."""
    return Int(bitcast[DType.uint16](v))


@inline(.always)
def _positive_zero(v: Scalar[DType.bfloat16]) -> Float32:
    """`v` in fp32 with a `-0.0` mapped to `+0.0`.

    This is the sanitize the kernel's fill applies to this rank's own pack, and
    the reference has to apply it too: a `-0.0` input lane is an ordinary value
    that contributes zero, not the "not arrived" pattern a peer's absence
    leaves in the window.
    """
    if _bits(v) == 0x8000:
        return Float32(0)
    return v.cast[DType.float32]()


@inline(.always)
def _denormal() -> Float32:
    """`2**-130`: eight units of bf16's `2**-133` denormal step.

    A scale group of these has an amax that survives to bf16 and a dynamic scale
    small enough that its reciprocal overflows fp32, which is the case
    `guarded_inv_scale` exists for.
    """
    return bitcast[DType.float32](UInt32(1 << 19))


@inline(.always)
def _ulp_up(want: Float32) -> Float32:
    """The spacing from the bf16 value `want` up to the next one."""
    var a = abs(want)
    var exponent = (Int(bitcast[DType.uint32](a)) >> 23) & 0xFF
    if exponent < 8:
        return Float32(2.0) ** -133.0
    return bitcast[DType.float32](UInt32((exponent - 7) << 23))


def _fill_rank(
    src: HostBuffer[DType.bfloat16],
    rank: Int,
    shape: Shape,
    call_index: Int,
):
    """Writes one rank's `[rows, cols]` partial sum for one call.

    Full-mantissa bf16 values, distinct per rank and per call, so a dropped,
    duplicated or stale contribution moves the sum. Every row carries `-0.0`
    lanes (a real value the fill has to sanitize), one row can be nothing but
    `-0.0` packs, and one row's first scale group can hold bf16 denormals.
    """
    var cols = shape.cols
    for i in range(shape.rows * cols):
        var row = i // cols
        var col = i % cols
        var h = (i * 2654435761 + rank * 7919 + call_index * 104729) % 16777213
        var v = _bf16(Float32(h) / Float32(8388608) - Float32(1.0))
        if col % 97 == 0 or row == shape.negative_zero_row:
            src[i] = Scalar[DType.bfloat16](Float32(-0.0))
        elif row == shape.tiny_group_row and col < GROUP:
            src[i] = Scalar[DType.bfloat16](_denormal())
        else:
            src[i] = v.cast[DType.bfloat16]()


def _fill_residual(res: HostBuffer[DType.bfloat16], rank: Int, shape: Shape):
    """Writes one rank's residual: rank-distinct bf16 values, never zero.

    Distinct per rank, so a rank that adds a peer's residual instead of its own
    fails instead of cancelling. The denormal row keeps its denormals here too:
    the stream the quant reads is `sum + residual`, so a normal residual would
    put the group's amax back in range.
    """
    for i in range(shape.rows * shape.cols):
        var row = i // shape.cols
        var col = i % shape.cols
        var h = (i * 2246822519 + rank * 3266489917) % 16777213
        var v = _bf16(
            Float32(h) / Float32(8388608) - Float32(1.0) + Float32(0.25)
        )
        if row == shape.tiny_group_row and col < GROUP:
            res[i] = Scalar[DType.bfloat16](_denormal())
        else:
            res[i] = v.cast[DType.bfloat16]()


def _load_inputs[
    ngpus: Int
](
    ctxs: List[DeviceContext[]],
    bufs: List[Bufs],
    shape: Shape,
    call_index: Int,
    src_host: List[HostBuffer[DType.bfloat16]],
    res_host: List[HostBuffer[DType.bfloat16]],
    team_sum: HostBuffer[DType.float32],
) raises:
    """Fills every rank's `src` and residual, and the rank-order fp32 sum.

    `team_sum` accumulates rank 0 first, then rank 1, and so on, which is the
    order the kernel's fold takes. Each rank keeps its own host buffers: a copy
    from host memory may be read after the call returns, so one buffer shared by
    every rank would race with the next rank's fill.
    """
    var count = shape.rows * shape.cols
    for i in range(count):
        team_sum[i] = Float32(0)
    for r in range(ngpus):
        _fill_rank(src_host[r], r, shape, call_index)
        _fill_residual(res_host[r], r, shape)
    for r in range(ngpus):
        bufs[r].src.enqueue_copy_from(src_host[r])
        bufs[r].res.enqueue_copy_from(res_host[r])
        for i in range(count):
            team_sum[i] = team_sum[i] + _positive_zero(src_host[r][i])
    for r in range(ngpus):
        ctxs[r].synchronize()


def _check_stream(
    ctx: DeviceContext,
    got_dev: DeviceBuffer[DType.bfloat16],
    got: HostBuffer[DType.bfloat16],
    team_sum: HostBuffer[DType.float32],
    res: HostBuffer[DType.bfloat16],
    count: Int,
    cols: Int,
) raises -> Int:
    """The residual stream, bit-exact including the sign of zero."""
    ctx.enqueue_copy(got, got_dev)
    ctx.synchronize()
    var bad = 0
    for i in range(count):
        var want = Scalar[DType.bfloat16](
            team_sum[i] + res[i].cast[DType.float32]()
        )
        if _bits(got[i]) != _bits(want):
            bad += 1
            if bad <= 4:
                print(
                    t"  stream mismatch at [{i // cols}, {i % cols}] : got"
                    t" {got[i]} , want {want}"
                )
    return bad


def _quant_reference(
    normed: HostBuffer[DType.bfloat16],
    q: HostBuffer[DType.float8_e4m3fn],
    qs: HostBuffer[DType.float32],
    rows: Int,
    cols: Int,
) raises -> Int:
    """The FP8 row and its group scales, bit-exact against `fp8_utils`.

    The reference quantizes the normed row the norm arm stored, which is the
    bf16 the kernel's own quant consumes; the scales are read with the padded
    stride the consumer uses, so a kernel that writes `[group, row]` with the
    call's row count reads back the poison instead.
    """
    var groups = cols // GROUP
    var bad = 0
    for g in range(groups):
        for r in range(rows):
            var amax = Float32(0)
            for c in range(GROUP):
                amax = max(
                    amax,
                    abs(normed[r * cols + g * GROUP + c].cast[DType.float32]()),
                )
            var pair = compute_dynamic_fp8_scale[DType.float8_e4m3fn](
                amax, SCALE_UB
            )
            var got_scale = qs[g * SCALE_STRIDE + r]
            if bitcast[DType.uint32](got_scale) != bitcast[DType.uint32](
                pair[0]
            ):
                bad += 1
                if bad <= 4:
                    print(
                        t"  scale mismatch at [group {g}, row {r}] : got"
                        t" {got_scale} , want {pair[0]} , group amax {amax}"
                    )
            for c in range(GROUP):
                var idx = r * cols + g * GROUP + c
                var want = fp8_quantize[DType.float8_e4m3fn](
                    SIMD[DType.float32, 1](normed[idx].cast[DType.float32]()),
                    pair[1],
                )
                if bitcast[DType.uint8](q[idx]) != bitcast[DType.uint8](
                    want[0]
                ):
                    bad += 1
                    if bad <= 4:
                        print(
                            t"  fp8 mismatch at [group {g}, row {r}, col"
                            t" {c}] : got {q[idx]} , want {want[0]} , value"
                            t" {normed[idx]}"
                        )
        # Rows past this call's are padding the consumer allocated: nothing may
        # have written them.
        for r in range(rows, SCALE_STRIDE):
            if qs[g * SCALE_STRIDE + r] != Float32(-7.5):
                bad += 1
                if bad <= 4:
                    print(
                        t"  scale padding clobbered at [group {g}, row {r}]"
                        t" : {qs[g * SCALE_STRIDE + r]}"
                    )
    return bad


def _launch_rank[
    ngpus: Int, quant: Bool, *, multiply_before_cast: Bool = True
](
    ctx: DeviceContext[],
    bufs_r: Bufs,
    window: Pointer[McarWindow[ngpus], MutUntrackedOrigin],
    rank: Int,
    shape: Shape,
) raises:
    """Enqueues one rank's launch of the boundary.

    `ctx` is either the rank's device context or a recording context from a
    `DeviceGraphBuilder`, which is what lets the same call record into a graph
    as well as run eagerly.
    """
    allreduce_mcast_rmsnorm[
        ngpus, quant=quant, multiply_before_cast=multiply_before_cast
    ](
        rebind[ImmPointer[Scalar[DType.bfloat16], ImmutAnyOrigin]](
            bufs_r.src.unsafe_ptr()
        ),
        rebind[ImmPointer[Scalar[DType.bfloat16], ImmutAnyOrigin]](
            bufs_r.res.unsafe_ptr()
        ),
        rebind[MutPointer[Scalar[DType.bfloat16], MutAnyOrigin]](
            bufs_r.pre.unsafe_ptr()
        ),
        rebind[MutPointer[Scalar[DType.bfloat16], MutAnyOrigin]](
            bufs_r.normed.unsafe_ptr()
        ),
        rebind[ImmPointer[Scalar[DType.bfloat16], ImmutAnyOrigin]](
            bufs_r.gamma.unsafe_ptr()
        ),
        rebind[MutPointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin]](
            bufs_r.q.unsafe_ptr()
        ),
        rebind[MutPointer[Float32, MutAnyOrigin]](bufs_r.qs.unsafe_ptr()),
        SCALE_STRIDE,
        shape.rows,
        shape.cols,
        EPS,
        Float32(0),
        SCALE_UB,
        rank,
        window[].rank(rank),
        ctx,
    )


def _launch[
    ngpus: Int, quant: Bool, *, multiply_before_cast: Bool = True
](
    ctxs: List[DeviceContext[]],
    bufs: List[Bufs],
    window: Pointer[McarWindow[ngpus], MutUntrackedOrigin],
    shape: Shape,
) raises:
    """Runs one call of the boundary on every rank, as a serving graph does.

    Every rank's launch is enqueued before any rank is synchronized, which is
    what makes the calls overlap: each rank's poll waits on its peers' fills.
    """
    for r in range(ngpus):
        _launch_rank[ngpus, quant, multiply_before_cast=multiply_before_cast](
            ctxs[r], bufs[r], window, r, shape
        )
    for r in range(ngpus):
        ctxs[r].synchronize()


@fieldwise_init
struct Hosts:
    """The host-side oracle: one input pair and one normed row per rank, plus
    the buffers every rank's comparison reads back through."""

    var src: List[HostBuffer[DType.bfloat16]]
    var res: List[HostBuffer[DType.bfloat16]]
    var normed: List[HostBuffer[DType.bfloat16]]
    var team_sum: HostBuffer[DType.float32]
    var got: HostBuffer[DType.bfloat16]
    var q: HostBuffer[DType.float8_e4m3fn]
    var qs: HostBuffer[DType.float32]
    var poison: HostBuffer[DType.float32]
    var gamma: HostBuffer[DType.bfloat16]


def _check_call[
    ngpus: Int
](
    ctxs: List[DeviceContext[]],
    bufs: List[Bufs],
    hosts: Hosts,
    window: Pointer[McarWindow[ngpus], MutUntrackedOrigin],
    shape: Shape,
    call_index: Int,
) raises -> Int:
    """One call's claims, with its own input set."""
    _load_inputs[ngpus](
        ctxs,
        bufs,
        shape,
        call_index,
        hosts.src,
        hosts.res,
        hosts.team_sum,
    )
    var count = shape.rows * shape.cols
    _launch[ngpus, quant=False](ctxs, bufs, window, shape)

    var failures = 0
    # Elements whose bytes may differ from the multiply-first reference: none is
    # expected, and the count is the same budget the cast-first rung allows.
    var budget = count // 65536 + 4
    for r in range(ngpus):
        var stream = _check_stream(
            ctxs[r],
            bufs[r].pre,
            hosts.got,
            hosts.team_sum,
            hosts.res[r],
            count,
            shape.cols,
        )
        var normed = _check_stream_normed(
            ctxs[r],
            bufs[r].normed,
            hosts.normed[r],
            hosts.team_sum,
            hosts.res[r],
            hosts.gamma,
            shape.rows,
            shape.cols,
        )
        # A tolerance cannot carry this rung either: the two epilogue orders
        # agree within one bf16 step on most elements, so a kernel that always
        # computed cast-first would pass the one-ulp check. The bytes are what
        # tell them apart, so the multiply-first rung requires the kernel's rows
        # to *be* the multiply-first reference, not merely near it.
        failures += stream + normed.outside
        if normed.outside != 0 or normed.bits_off > budget:
            print(
                t"  call {call_index} (rows {shape.rows}, cols {shape.cols})"
                t" rank {r}: {stream} stream, {normed.outside} normed outside"
                t" one ulp, {normed.bits_off} of {count} differing from the"
                t" multiply-first reference (budget {budget})"
            )
            return failures + 1
        if call_index == 0 and r == 0:
            print(
                t"  multiply-first rung (rows {shape.rows}, cols"
                t" {shape.cols}): {normed.outside} of {count} outside one ulp,"
                t" {normed.bits_off} differ from the multiply-first reference"
            )

    # The quant arm runs the same statement sequence over the same inputs, so its
    # epilogue quantizes exactly the normed bytes the norm arm stored. Poison
    # every scale word first: the kernel must rewrite `[group, row]` for the rows
    # this call reduces and leave the padding alone.
    for r in range(ngpus):
        for i in range((shape.cols // GROUP) * SCALE_STRIDE):
            hosts.poison[i] = Float32(-7.5)
        bufs[r].qs.enqueue_copy_from(hosts.poison)
        ctxs[r].synchronize()
    _launch[ngpus, quant=True](ctxs, bufs, window, shape)
    for r in range(ngpus):
        ctxs[r].enqueue_copy(hosts.q, bufs[r].q)
        ctxs[r].enqueue_copy(hosts.qs, bufs[r].qs)
        ctxs[r].synchronize()
        var quant = _quant_reference(
            hosts.normed[r], hosts.q, hosts.qs, shape.rows, shape.cols
        )
        failures += quant
        if quant != 0:
            print(
                t"  call {call_index} (rows {shape.rows}, cols {shape.cols})"
                t" rank {r}: {quant} quant"
            )
            return failures

    # The cast-first norm (`ops.rms_norm`'s default, and what most architectures
    # build) applies gamma on the rounded value: same exchange, same stream, a
    # different epilogue order, so the arm answers that one too or a plain-norm
    # composite would silently write multiply-first bytes. It runs last because
    # it overwrites the normed rows the quant rung just used as its reference.
    _launch[ngpus, quant=False, multiply_before_cast=False](
        ctxs, bufs, window, shape
    )
    # A tolerance alone cannot carry this rung: the two orders agree within one
    # bf16 step on most elements, so a multiply-first kernel also passes a loose
    # check. The rung therefore also counts how many elements the kernel moved
    # off the *multiply-first* reference, which a cast-first kernel does on a
    # large fraction of the row and a multiply-first one never does.
    comptime ORDER_WITNESS_PERCENT = 20
    """Least share of the row a cast-first result must move off the
    multiply-first reference. Measured at a quarter of the row on this data."""
    # Elements allowed outside two ulps of the cast-first reference: the bf16
    # boundary cases where the reference's fp32 reduction order moves the value,
    # a handful a call at this size. The multiply-first rung's budget above is
    # the same number.
    for r in range(ngpus):
        var check = _check_cast_first(
            ctxs[r],
            bufs[r].normed,
            hosts.normed[r],
            hosts.team_sum,
            hosts.res[r],
            hosts.gamma,
            shape.rows,
            shape.cols,
        )
        if (
            check.strict > budget
            or check.witness * 100 < count * ORDER_WITNESS_PERCENT
            or check.witness * 10 > count * 9
        ):
            print(
                t"  call {call_index} (rows {shape.rows}, cols {shape.cols})"
                t" rank {r}: cast-first {check.strict} of {count} outside two"
                t" ulps (budget {budget}), {check.witness} off the"
                t" multiply-first reference, wanted at least"
                t" {count * ORDER_WITNESS_PERCENT // 100}"
            )
            return failures + 1
        if call_index == 0 and r == 0:
            print(
                t"  cast-first rung (rows {shape.rows}, cols {shape.cols}):"
                t" {check.strict} of {count} outside two ulps, {check.witness}"
                t" differ from the multiply-first order"
            )
    return failures


@fieldwise_init
struct NormedRowCheck:
    """A normed-row rung's two counts, both against the reference it names.

    `outside` counts elements more than the rung's tolerance from the reference.
    `bits_off` counts elements whose bytes differ from it. The second count is
    what makes the multiply-first rung fail when the kernel computes the other
    order: the two orders agree within one bf16 step on roughly three quarters of
    the row, so a tolerance admits either, and only the bytes separate them.
    """

    var outside: Int
    var bits_off: Int


def _check_stream_normed(
    ctx: DeviceContext,
    got_dev: DeviceBuffer[DType.bfloat16],
    got_normed: HostBuffer[DType.bfloat16],
    team_sum: HostBuffer[DType.float32],
    res: HostBuffer[DType.bfloat16],
    gamma: HostBuffer[DType.bfloat16],
    rows: Int,
    cols: Int,
) raises -> NormedRowCheck:
    """The normed row against `rms_norm`'s multiply-first order: one bf16 ulp,
    and how many elements differ from it in bytes.

    The reference applies the weight in fp32 before the single cast to bf16,
    which is `multiply_before_cast=True`. The cast-first order has its own check
    in `_check_cast_first`, because there the two counts face opposite references.
    """
    ctx.enqueue_copy(got_normed, got_dev)
    ctx.synchronize()
    var check = NormedRowCheck(0, 0)
    for r in range(rows):
        var sumsq = Float64(0)
        for c in range(cols):
            var s = Float64(
                team_sum[r * cols + c] + res[r * cols + c].cast[DType.float32]()
            )
            sumsq += s * s
        var norm = rsqrt(Float32(sumsq / Float64(cols)) + EPS)
        for c in range(cols):
            var stream = (
                team_sum[r * cols + c] + res[r * cols + c].cast[DType.float32]()
            )
            var weight = gamma[c].cast[DType.float32]()
            var want = _bf16((stream * norm) * weight)
            var got = got_normed[r * cols + c]
            var d = abs(got.cast[DType.float32]() - want)
            if d > _ulp_up(want):
                check.outside += 1
                if check.outside <= 4:
                    print(
                        t"  normed mismatch at [{r}, {c}] : got"
                        t" {got_normed[r * cols + c]} , want {want}"
                    )
            if _bits(got) != _bits(Scalar[DType.bfloat16](want)):
                check.bits_off += 1
    return check^


@fieldwise_init
struct OrderCheck:
    """A cast-first rung's two counts, one per reference.

    `strict` counts elements outside two bf16 steps of the cast-first reference,
    which the reference's own fp32 reduction order can move on a few boundary
    cases. `witness` counts elements whose bytes differ from the *multiply-first*
    reference, which is what tells the two orders apart: a cast-first kernel
    moves a large fraction of the row, a multiply-first one moves none.
    """

    var strict: Int
    var witness: Int


def _check_cast_first(
    ctx: DeviceContext,
    got_dev: DeviceBuffer[DType.bfloat16],
    got_normed: HostBuffer[DType.bfloat16],
    team_sum: HostBuffer[DType.float32],
    res: HostBuffer[DType.bfloat16],
    gamma: HostBuffer[DType.bfloat16],
    rows: Int,
    cols: Int,
) raises -> OrderCheck:
    """The normed row against both epilogue orders, counted for a decision.

    The cast-first reference rounds `stream * norm` to bf16 and applies the
    weight to that, in bf16; the product of two bf16 values is exact in fp32, so
    computing it there and rounding once is the same value. The multiply-first
    reference rounds once, at the end. Both are computed per element, so the
    caller can require that the kernel sits inside the first and away from the
    second; a tolerance alone cannot tell them apart, since they agree within
    one bf16 step on most elements.
    """
    ctx.enqueue_copy(got_normed, got_dev)
    ctx.synchronize()
    var check = OrderCheck(0, 0)
    for r in range(rows):
        var sumsq = Float64(0)
        for c in range(cols):
            var s = Float64(
                team_sum[r * cols + c] + res[r * cols + c].cast[DType.float32]()
            )
            sumsq += s * s
        var norm = rsqrt(Float32(sumsq / Float64(cols)) + EPS)
        for c in range(cols):
            var stream = (
                team_sum[r * cols + c] + res[r * cols + c].cast[DType.float32]()
            )
            var weight = gamma[c].cast[DType.float32]()
            var cast_first = _bf16(
                _bf16(stream * norm).cast[DType.float32]() * weight
            )
            var multiply_first = _bf16((stream * norm) * weight)
            var got = got_normed[r * cols + c]
            var d = abs(got.cast[DType.float32]() - cast_first)
            if d > _ulp_up(cast_first) * Float32(2):
                check.strict += 1
                if check.strict <= 4:
                    print(
                        t"  cast-first mismatch at [{r}, {c}] : got {got} ,"
                        t" want {cast_first} , multiply-first {multiply_first}"
                    )
            if _bits(got) != _bits(Scalar[DType.bfloat16](multiply_first)):
                check.witness += 1
    return check^


def _record[
    ngpus: Int, quant: Bool
](
    ctx: DeviceContext[],
    bufs_r: Bufs,
    window: Pointer[McarWindow[ngpus], MutUntrackedOrigin],
    rank: Int,
    shape: Shape,
) raises -> DeviceGraph:
    """Records one rank's launch of the boundary into a device graph.

    The call number, the generation and the clear extent all live in device
    memory, so the recorded launch takes no host argument that a replay would
    have to supply.
    """

    def build(
        mut builder: DeviceGraphBuilder[_],
    ) raises {imm bufs_r, imm window, imm rank, imm shape}:
        _launch_rank[ngpus, quant=quant](
            builder.recording_context(), bufs_r, window, rank, shape
        )

    return DeviceGraph.create(ctx, build)


def _graph_replays[
    ngpus: Int
](
    ctxs: List[DeviceContext[]],
    bufs: List[Bufs],
    hosts: Hosts,
    window: Pointer[McarWindow[ngpus], MutUntrackedOrigin],
    shape: Shape,
    first_call: Int,
    replays: Int,
) raises -> Int:
    """Replays a recorded call `replays` times, each on its own input set.

    This is the shape a captured serving graph runs: the launches are recorded
    once and replayed with new buffer contents, and every replay is checked, so
    the device-resident call sequence has to advance by exactly one per replay
    and each call's clear has to land on the extent it wrote two calls earlier.
    Both epilogues are recorded: the grouped FP8 one is what the graph compiler
    folds, and its scale writes share the launch with the exchange.
    """
    var graphs = List[DeviceGraph](capacity=ngpus)
    var quant_graphs = List[DeviceGraph](capacity=ngpus)
    for r in range(ngpus):
        graphs.append(
            _record[ngpus, quant=False](ctxs[r], bufs[r], window, r, shape)
        )
        quant_graphs.append(
            _record[ngpus, quant=True](ctxs[r], bufs[r], window, r, shape)
        )
    for r in range(ngpus):
        ctxs[r].synchronize()

    for call in range(replays):
        _load_inputs[ngpus](
            ctxs,
            bufs,
            shape,
            first_call + call,
            hosts.src,
            hosts.res,
            hosts.team_sum,
        )
        # The same budget the explicit calls allow: a replay's rows must be the
        # multiply-first reference's bytes, not merely within a ulp of them.
        var budget = shape.rows * shape.cols // 65536 + 4
        for graph in graphs:
            graph.replay()
        for r in range(ngpus):
            ctxs[r].synchronize()
        for r in range(ngpus):
            var stream = _check_stream(
                ctxs[r],
                bufs[r].pre,
                hosts.got,
                hosts.team_sum,
                hosts.res[r],
                shape.rows * shape.cols,
                shape.cols,
            )
            var normed = _check_stream_normed(
                ctxs[r],
                bufs[r].normed,
                hosts.normed[r],
                hosts.team_sum,
                hosts.res[r],
                hosts.gamma,
                shape.rows,
                shape.cols,
            )
            if stream + normed.outside != 0 or normed.bits_off > budget:
                print(
                    t"  graph replay {call} (rows {shape.rows}, cols"
                    t" {shape.cols}) rank {r}: {stream} stream,"
                    t" {normed.outside} normed outside one ulp,"
                    t" {normed.bits_off} differing from the multiply-first"
                    t" reference (budget {budget})"
                )
                return stream + normed.outside + normed.bits_off
        # The grouped epilogue's replay quantizes the normed rows just checked.
        for r in range(ngpus):
            ctxs[r].enqueue_memset(bufs[r].qs, Float32(-7.5))
        for graph in quant_graphs:
            graph.replay()
        for r in range(ngpus):
            ctxs[r].enqueue_copy(hosts.q, bufs[r].q)
            ctxs[r].enqueue_copy(hosts.qs, bufs[r].qs)
            ctxs[r].synchronize()
            var quant = _quant_reference(
                hosts.normed[r], hosts.q, hosts.qs, shape.rows, shape.cols
            )
            if quant != 0:
                print(
                    t"  quant graph replay {call} (rows {shape.rows}, cols"
                    t" {shape.cols}) rank {r}: {quant} quant"
                )
                return quant
    print(
        t"mcar_rmsnorm: {replays} graph replays of each epilogue at"
        t" {shape.cols} columns, rows {shape.rows}: PASS"
    )
    return 0


def _back_to_back[
    ngpus: Int
](
    ctxs: List[DeviceContext[]],
    bufs: List[Bufs],
    hosts: Hosts,
    window: Pointer[McarWindow[ngpus], MutUntrackedOrigin],
    cols: Int,
    call_index: Int,
) raises -> Int:
    """Several calls a rank, enqueued with no synchronize between them.

    Every other rung synchronizes the team after each call, so no rank ever
    runs ahead of a peer. Here each rank's stream holds the whole sequence, so
    a rank can clear two calls ahead while a slow peer still polls, and the
    sequence mixes row counts on both sides of the clear's 16-row split, so a
    call clears an extent a call of another size wrote. It runs once eagerly
    and once as chained graph replays.

    The inputs are one set loaded for the widest call; a narrower call reduces
    a prefix of its rows, and every row's result depends on that row alone, so
    the widest call's reference covers them all. Each call writes its own
    output buffers, so every call is checked after the sequence.
    """
    var rows_list: List[Int] = [17, 1, 64, 5]
    var widest = Shape(64, cols, 9, 63)
    _load_inputs[ngpus](
        ctxs, bufs, widest, call_index, hosts.src, hosts.res, hosts.team_sum
    )
    # Sized like every other buffer: the readback copies the whole host buffer.
    var capacity = window[].rows() * cols
    var outs = List[Bufs](capacity=len(rows_list) * ngpus)
    for _ in range(len(rows_list)):
        for r in range(ngpus):
            outs.append(
                Bufs(
                    src=bufs[r].src,
                    res=bufs[r].res,
                    pre=ctxs[r].create_buffer_sync[DType.bfloat16](capacity),
                    normed=ctxs[r].create_buffer_sync[DType.bfloat16](capacity),
                    q=bufs[r].q,
                    qs=bufs[r].qs,
                    gamma=bufs[r].gamma,
                )
            )

    var graphs = List[DeviceGraph](capacity=len(rows_list) * ngpus)
    for c in range(len(rows_list)):
        for r in range(ngpus):
            graphs.append(
                _record[ngpus, quant=False](
                    ctxs[r],
                    outs[c * ngpus + r],
                    window,
                    r,
                    Shape(rows_list[c], cols, NEG_ZERO, NEG_ZERO),
                )
            )
    for r in range(ngpus):
        ctxs[r].synchronize()

    for replayed in range(2):
        for i in range(len(outs)):
            var owner = i % ngpus
            ctxs[owner].enqueue_memset(
                outs[i].pre, Scalar[DType.bfloat16](-7.5)
            )
            ctxs[owner].enqueue_memset(
                outs[i].normed, Scalar[DType.bfloat16](-7.5)
            )
        for r in range(ngpus):
            ctxs[r].synchronize()
        for c in range(len(rows_list)):
            for r in range(ngpus):
                if replayed == 1:
                    graphs[c * ngpus + r].replay()
                else:
                    _launch_rank[ngpus, quant=False](
                        ctxs[r],
                        outs[c * ngpus + r],
                        window,
                        r,
                        Shape(rows_list[c], cols, NEG_ZERO, NEG_ZERO),
                    )
        for r in range(ngpus):
            ctxs[r].synchronize()

        var how = "graph" if replayed == 1 else "eager"
        for c in range(len(rows_list)):
            var rows = rows_list[c]
            var budget = rows * cols // 65536 + 4
            for r in range(ngpus):
                var stream = _check_stream(
                    ctxs[r],
                    outs[c * ngpus + r].pre,
                    hosts.got,
                    hosts.team_sum,
                    hosts.res[r],
                    rows * cols,
                    cols,
                )
                var normed = _check_stream_normed(
                    ctxs[r],
                    outs[c * ngpus + r].normed,
                    hosts.normed[r],
                    hosts.team_sum,
                    hosts.res[r],
                    hosts.gamma,
                    rows,
                    cols,
                )
                if stream + normed.outside != 0 or normed.bits_off > budget:
                    print(
                        t"  back-to-back call {c} ({how}, rows {rows}, cols"
                        t" {cols}) rank {r}: {stream} stream,"
                        t" {normed.outside} normed outside one ulp,"
                        t" {normed.bits_off} differing from the multiply-first"
                        t" reference (budget {budget})"
                    )
                    return 1 + stream + normed.outside
    print(t"mcar_rmsnorm: back-to-back calls at {cols} columns: PASS")
    return 0


def _width_ladder(cols: Int) -> List[Shape]:
    """The shapes to run at one width.

    One call per shape, each with its own input set: a lone row, a lone row of
    nothing but `-0.0` packs, both sides of the clear's 16-row split, the band's
    own capacity, a count past the SM count (second wave), then a
    descending-and-regrowing sequence whose calls clear extents written by calls
    of a different size.
    """
    if cols == 2048:
        return [
            Shape(160, cols, 159, 3),
            Shape(17, cols, 4, 0),
        ]
    return [
        Shape(1, cols, NEG_ZERO, NEG_ZERO),
        Shape(1, cols, 0, NEG_ZERO),
        Shape(5, cols, 2, 1),
        Shape(17, cols, 4, 0),
        Shape(128, cols, 17, 100),
        Shape(128, cols, 3, 7),
        Shape(17, cols, 2, 1),
        Shape(5, cols, 1, 4),
        Shape(1, cols, NEG_ZERO, NEG_ZERO),
        Shape(64, cols, 9, 63),
    ]


def _window_inside_a_capture[
    ngpus: Int
](ctxs: List[DeviceContext[]], cols: Int) raises -> Optional[
    Pointer[McarWindow[ngpus], MutUntrackedOrigin]
]:
    """Creates the team's window while a device graph is recording.

    This is where serving creates it: `warmup_pre_ready` captures every decode
    batch size before any eager call, so a geometry's first call for the window
    arrives inside a capture. Creating it there is legal -- capture is RELAXED,
    and the allocation and its fills run on the contexts the window creates for
    itself, not on the recording stream -- so the arm is recorded into that graph
    with the rest of the layer rather than deferred past it.

    Args:
        ctxs: One context per rank, in rank order.
        cols: Row width.

    Returns:
        The window the capture created, or `None` when the device set cannot map
        multicast at all.
    """
    var got = List[Pointer[McarWindow[ngpus], MutUntrackedOrigin]](capacity=1)

    def build(
        mut builder: DeviceGraphBuilder[_],
    ) raises {mut got, imm ctxs, imm cols,}:
        # A copy per rank, and rank 0's replaced by the recording context while
        # the capture is live: one `copy` then one assignment, which is how the
        # other sites put a borrowed context into a team array.
        var team = Array[DeviceContext[], ngpus](
            fill_with_unrolled=lambda [i: Int]() -> DeviceContext[]: ctxs[
                i
            ].copy()
        )
        with builder.recording_context() as rec:
            team[0] = rec.copy()
            if case .Some(w) = mcar_window[ngpus, DType.bfloat16](team, cols):
                got.append(w)

    _ = DeviceGraph.create(ctxs[0], build)
    if len(got) == 1:
        return Optional(got[0])
    return None


def _run_width[ngpus: Int](cols: Int) raises -> Bool:
    """The oracle at `ngpus` ranks and one width."""
    print(t"mcar_rmsnorm oracle: {ngpus} ranks, {cols} columns")

    var ctxs = List[DeviceContext[]](capacity=ngpus)
    for r in range(ngpus):
        ctxs.append(DeviceContext(device_id=r))
    var team = Array[DeviceContext[], ngpus](
        fill_with_unrolled=lambda [i: Int]() -> DeviceContext[]: ctxs[i].copy()
    )
    # Created inside a capture, the way serving's warmup does it, so every call
    # below -- eager ones and recorded replays alike -- runs the arm against a
    # window a device graph created.
    var maybe_window = _window_inside_a_capture[ngpus](ctxs, cols)
    __guard case .Some(window) = maybe_window else:
        print("mcar_rmsnorm: SKIP, the window did not allocate")
        return True
    # The capture kept it: an eager call reaches the same allocation, so the
    # capture recorded no failure against this team and geometry.
    var maybe_again = mcar_window[ngpus, DType.bfloat16](team, cols)
    __guard case .Some(again) = maybe_again else:
        print("mcar_rmsnorm: the window a capture created went missing")
        return False
    if Int(again[].rank(0).state) != Int(window[].rank(0).state):
        print("mcar_rmsnorm: a capture created a second window")
        return False
    var rows_max = window[].rows()
    var capacity = rows_max * cols

    var bufs = List[Bufs](capacity=ngpus)
    for r in range(ngpus):
        bufs.append(
            Bufs(
                src=ctxs[r].create_buffer_sync[DType.bfloat16](capacity),
                res=ctxs[r].create_buffer_sync[DType.bfloat16](capacity),
                pre=ctxs[r].create_buffer_sync[DType.bfloat16](capacity),
                normed=ctxs[r].create_buffer_sync[DType.bfloat16](capacity),
                q=ctxs[r].create_buffer_sync[DType.float8_e4m3fn](capacity),
                qs=ctxs[r].create_buffer_sync[DType.float32](
                    (cols // GROUP) * SCALE_STRIDE
                ),
                gamma=ctxs[r].create_buffer_sync[DType.bfloat16](cols),
            )
        )

    # One norm weight, replicated, and not 1.0 so a dropped weight fails.
    var gamma = ctxs[0].enqueue_create_host_buffer[DType.bfloat16](cols)
    for c in range(cols):
        gamma[c] = _bf16(
            Float32(1) + Float32((c * 37) % 101) / Float32(128)
        ).cast[DType.bfloat16]()
    for r in range(ngpus):
        bufs[r].gamma.enqueue_copy_from(gamma)
    for r in range(ngpus):
        ctxs[r].synchronize()

    var src_host = List[HostBuffer[DType.bfloat16]](capacity=ngpus)
    var res_host = List[HostBuffer[DType.bfloat16]](capacity=ngpus)
    var normed_host = List[HostBuffer[DType.bfloat16]](capacity=ngpus)
    for r in range(ngpus):
        src_host.append(
            ctxs[r].enqueue_create_host_buffer[DType.bfloat16](capacity)
        )
        res_host.append(
            ctxs[r].enqueue_create_host_buffer[DType.bfloat16](capacity)
        )
        normed_host.append(
            ctxs[r].enqueue_create_host_buffer[DType.bfloat16](capacity)
        )
    var hosts = Hosts(
        src=src_host^,
        res=res_host^,
        normed=normed_host^,
        team_sum=ctxs[0].enqueue_create_host_buffer[DType.float32](capacity),
        got=ctxs[0].enqueue_create_host_buffer[DType.bfloat16](capacity),
        q=ctxs[0].enqueue_create_host_buffer[DType.float8_e4m3fn](capacity),
        qs=ctxs[0].enqueue_create_host_buffer[DType.float32](
            (cols // GROUP) * SCALE_STRIDE
        ),
        poison=ctxs[0].enqueue_create_host_buffer[DType.float32](
            (cols // GROUP) * SCALE_STRIDE
        ),
        gamma=gamma,
    )

    var ladder = _width_ladder(cols)
    for call_index in range(len(ladder)):
        var failures = _check_call[ngpus](
            ctxs, bufs, hosts, window, ladder[call_index], call_index
        )
        if failures != 0:
            return False
    # The captured-graph shape of the same call: recorded once, replayed with
    # new inputs every time, every replay checked.
    var graphs = _graph_replays[ngpus](
        ctxs, bufs, hosts, window, Shape(17, cols, 4, 0), len(ladder), 5
    )
    if graphs != 0:
        return False
    if cols == 4096:
        if (
            _back_to_back[ngpus](
                ctxs, bufs, hosts, window, cols, len(ladder) + 5
            )
            != 0
        ):
            return False
    print(t"mcar_rmsnorm: PASS at {cols} columns")
    return True


def _run[ngpus: Int]() raises -> Bool:
    """The oracle at `ngpus` ranks, over every width it needs."""
    DeviceContext.enable_all_peer_access()
    var ctxs0 = DeviceContext(device_id=0)
    if not ctxs0.supports_multicast():
        print("mcar_rmsnorm: SKIP, this device set has no multicast support")
        return True
    if not _run_width[ngpus](4096):
        return False
    # 2048 columns hold 256 rows of the band, which is what a call past 128 rows
    # (and past the SM count) needs.
    return _run_width[ngpus](2048)


comptime TEAMS = (2, 4)
"""The teams the oracle runs, each on a host that has the devices for it.

Four is MiMo-V2's tensor-parallel width, and the widest a one-window team can
be here; two is the floor the boundary supports and every multi-GPU host can
run.
"""


def main() raises:
    var devices = DeviceContext.number_of_devices()
    print("mcar_rmsnorm: devices on this host:", devices)
    assert_true(devices >= 2, "the boundary needs a team of at least two")
    comptime for i in range(len(TEAMS)):
        comptime ngpus = rebind[Int](TEAMS[i])
        if devices < ngpus:
            continue
        if not _run[ngpus]():
            raise Error("the multicast residual boundary failed its oracle")
