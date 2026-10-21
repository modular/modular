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
"""The indexer top-k at multi-turn context lengths, over four distributions.

A conversation of 14k initial context plus twenty 7k increments gives
`N = 14336 + 7168 * i` for `i` in 0..20, so `N` runs 14336..157696. The indexer
top-k runs once per query token, so `rows` is a decode batch or a prefill
chunk's token count. All but the shortest of these `N` are past the widest
register-resident payload, so they land on the streaming select, and `rows`
decides only whether it takes the prefetching variant (below the SM count) or
the occupancy one (at or above it). The first rung is the exception, and only
for the rank-free contracts, which reach further before they have to stream --
so it is also where the two designs are closest.

    decode:  rows=48 (batch),        N=14336..157696, K=2048
    prefill: rows=256+ (one chunk),  N=14336..157696, K=2048

`dist` selects the score distribution, because the select's cost is set by how
many columns share the threshold's coarse bin and that is a property of the
distribution, not of `N`:

    q17     ~17 discrete levels -- the tie plateaus fp8-quantized indexer
            scores collapse to. The threshold bin is a wide plateau.
    normal  standard normal -- what an attention logit actually looks like,
            being a sum of many products. Bins cluster near the mode.
    uniform uniform on [-1, 1] -- bins evenly occupied.
    narrow  every value inside one coarse bin (1 + u/32) with real spread
            inside it, so the threshold bin holds the whole row. Adversarial.

`mode` picks the output contract, and all four combinations of the kernel's two
relaxations are reachable so that the effect of each can be measured separately
rather than asserted:

    ord       descending score, ascending column -- the default.
    ord_nd    ordered, `deterministic=False`. Documented to be the same thing
              as `ord`, since an ordered output is reproducible whatever the
              flag says; here to hold that claim to a measurement.
    unord     the same `K` columns, order unspecified but reproducible.
    unord_nd  the same `K` columns, neither ordered nor reproducible. Above the
              resident width it is the only one that skips the rank -- see
              `_histsel_topk_kernel` for why reproducibility costs the rank
              there.

`dtype` picks the score buffer's element type, `float32` or `bfloat16`, and it
measures the read: bf16 halves the bytes and the sectors of the row scan, which
is what the cost of a long row is made of. Both arms live in one binary because
compiling this file dominates running it by orders of magnitude. Only `ord` and
`unord_nd` are built at bf16.

Three knobs isolate each ingredient of the bf16 path, so each step's effect can
be measured PAIRED inside one process rather than across builds. None of them
feeds the dispatch -- all three reach the kernels only -- so every arm runs the
same instantiation family on the same grid, at one input:

    --sig-bits=32   the pre-round-cap schedule. A bf16 score occupies 16 of the
                    key half's bits and resolves in two rounds where f32 takes
                    three; this forces the three-round schedule. Needs
                    `--phi-bits=32`, a 16-bit payload having no bits below 16 to
                    take a digit from.
    --phi-bits=16   the narrow payload: `phi` carried at 16 bits on a bf16
                    score rather than at the derived 32. With `--sig-bits=0`
                    this isolates the payload width alone.
    --scan-items=8  the narrow scan group. The prefetch arm carries 16 columns
                    at bf16 once the payload is 16 bits, because what the
                    prefetch buys is BYTES in flight; this forces the 8-column
                    group back, which is the narrow payload alone.

All three default to 0, meaning "derive", which is what ships. Each is refused
rather than silently accepted wherever it is not a distinct kernel.

`--sweep=1` runs the whole production grid -- both row counts, five lengths,
four arms -- inside ONE process, and is how the A/B is meant to be taken. The
arms of a cell then run back to back on one device with one allocator state, so
a clock or a neighbour drifting hits all four alike and their ratio is a paired
one; the four `BenchId`s differ, so nothing is averaged together. It also runs
in a single remote-B200 invocation, where forty separate ones would spend more
wall clock on bazel round trips than on the kernel. `rows`, `N` and the three
knobs are ignored under it.

Scores land in the buffer already rounded, so at bf16 the printed
`input_checksum` is over the rounded array. Two arms agreeing on it means they
scored the same input; they will NOT match the f32 arm's, by construction.
"""

from std.memory import bitcast
from std.sys import size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import TileTensor, row_major

from nn.topk_bitonic import (
    _hsel_phi_dtype,
    _hsel_prefetch_scan_items,
    _hsel_sig_bits,
    persistent_topk_block_split,
)


def _get_run_name(
    rows: Int,
    N: Int,
    K: Int,
    dist: String,
    mode: String,
    dtype: DType,
    sig_bits: Int,
    phi_bits: Int,
    scan_items: Int,
) -> String:
    # Every knob that changes the kernel belongs in the id: two arms sharing a
    # `BenchId` land under one entry and get averaged into a number that is
    # neither.
    return String(
        "topk_bitonic_split : rows=",
        rows,
        ", N=",
        N,
        ", K=",
        K,
        ", dist=",
        dist,
        ", mode=",
        mode,
        ", dtype=",
        dtype,
        ", sig_bits=",
        sig_bits,
        ", phi_bits=",
        phi_bits,
        ", scan_items=",
        scan_items,
    )


@inline(.always)
def _mix64(x: UInt64) -> UInt64:
    """SplitMix64's finalizer."""
    var z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@inline(.always)
def _u01(i: UInt64) -> Float64:
    """A uniform on [0, 1) keyed by `i` alone.

    Keyed rather than sequential so a harness in another language can generate the
    *same* array without sharing an RNG: different languages have different
    generators, and a distribution that merely matches in family is not the same
    input. The select's cost is set by how many columns share the threshold's
    coarse bin, which is a property of the actual values, so an honest comparison
    needs the values themselves to be equal -- hence the checksum this bench
    prints, which a comparing harness can be held to.
    """
    var h = _mix64((i + 1) * 0x9E3779B97F4A7C15)
    return Float64(h >> 11) * (1.0 / 9007199254740992.0)


@inline(.always)
def _u_val(r: Int, c: Int, N: Int, k: Int) -> Float64:
    return _u01(((UInt64(r) * UInt64(N) + UInt64(c)) << 2) + UInt64(k))


@inline(.always)
def _u_len(r: Int) -> Float64:
    return _u01(0x8000000000000000 + UInt64(r))


# Irwin-Hall rather than Box-Muller: Box-Muller needs `log` and `cos`, and two
# languages' libms need not agree in the last bit, which is enough to make two
# harnesses generate different arrays from the same key. Sums and a literal scale
# are exact IEEE operations, so this is reproducible anywhere. Variance of the
# terms below is 1/2, hence the scale.
comptime _IH_TERMS: Int = 6
comptime _IH_SCALE: Float64 = 1.4142135623730951


def _sample(dist: String, r: Int, c: Int, N: Int) -> Float32:
    """One score from `dist`. See the module docstring for what each one is for.
    """
    var u = _u_val(r, c, N, 0)
    if dist == "q17":
        return Float32(Int(u * 17.0)) / 16.0
    if dist == "uniform":
        return Float32(u * 2.0 - 1.0)
    if dist == "narrow":
        return Float32(1.0 + u / 32.0)
    # One index computation for the whole sum: only the term number varies.
    var base = (UInt64(r) * UInt64(N) + UInt64(c)) << 2
    var acc = Float64(0.0)
    for k in range(_IH_TERMS):
        acc += _u01(base + UInt64(k))
    return Float32((acc - Float64(_IH_TERMS) / 2.0) * _IH_SCALE)


def execute_topk_bitonic[
    ordered: Bool,
    deterministic: Bool,
    in_dtype: DType,
    sig_bits: Int = _hsel_sig_bits[in_dtype](),
    phi_dtype: DType = _hsel_phi_dtype[in_dtype](),
    scan_items: Int = _hsel_prefetch_scan_items[in_dtype, phi_dtype](),
](
    ctx: DeviceContext,
    mut m: Bench,
    rows: Int,
    N: Int,
    K: Int,
    dist: String,
    mode: String,
) raises:
    """Benches `persistent_topk_block_split` on scores shaped like the DSA
    indexer's: a random-length valid prefix per row (varying `num_keys`
    within a causal chunk), an `-inf` masked suffix, and valid values drawn
    from `dist`.
    """
    var scores_buf = ctx.enqueue_create_buffer[in_dtype](rows * N)
    var idxs_buf = ctx.enqueue_create_buffer[.int32](rows * K)

    var csum = UInt64(0xCBF29CE484222325)
    with scores_buf.map_to_host() as h:
        for r in range(rows):
            var num_keys = N - Int(_u_len(r) * Float64(N) / 8.0)
            for c in range(N):
                var v: Float32
                if c < num_keys:
                    v = _sample(dist, r, c, N)
                else:
                    # `min_or_neg_inf` sentinel. bf16 rounds it to ~-2.996e38,
                    # still below every live score and still finite, so the
                    # masked suffix means the same thing at either width.
                    v = Float32(-3.0e38)
                var stored = v.cast[in_dtype]()
                h[r * N + c] = stored
                # Checksum the value the KERNEL sees, not the one generated:
                # at bf16 those differ, and hashing the generator's array would
                # name an input this run never had.
                csum = (
                    csum ^ UInt64(bitcast[.uint32, 1](Float32(stored)))
                ) * 0x100000001B3
    print("input_checksum=", hex(csum), sep="")

    var scores_t = TileTensor(scores_buf, row_major(rows, N))
    var idxs_t = TileTensor(idxs_buf, row_major(rows, K))
    ctx.synchronize()

    @inline(.always)
    def kernel_launch(c: DeviceContext) raises {mut idxs_t, imm}:
        persistent_topk_block_split[
            ordered=ordered,
            deterministic=deterministic,
            sig_bits=sig_bits,
            phi_dtype=phi_dtype,
            scan_items=scan_items,
        ](
            c,
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](scores_t.ptr),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
            N,
            K,
            rows,
        )

    @inline(.always)
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_func,
        BenchId(
            _get_run_name(
                rows,
                N,
                K,
                dist,
                mode,
                in_dtype,
                sig_bits,
                8 * size_of[phi_dtype](),
                scan_items,
            )
        ),
        [],
    )

    _ = scores_buf
    _ = idxs_buf


def _bf16_arms[
    ordered: Bool, deterministic: Bool
](
    ctx: DeviceContext,
    mut m: Bench,
    rows: Int,
    N: Int,
    K: Int,
    dist: String,
    mode: String,
    sig_bits: Int,
    phi_bits: Int,
    scan_items: Int,
) raises:
    """The four bf16 configurations, selected at runtime from the two knobs.

    One parametric helper rather than an inline tree per contract: two contracts
    times four arms is eight instantiations either way, but only one of them has
    to be read, and `_sweep` reads the same one.
    """
    if sig_bits == 32:
        execute_topk_bitonic[
            ordered,
            deterministic,
            DType.bfloat16,
            sig_bits=32,
            phi_dtype=DType.uint32,
        ](ctx, m, rows, N, K, dist, mode)
    elif scan_items == 8:
        execute_topk_bitonic[
            ordered,
            deterministic,
            DType.bfloat16,
            phi_dtype=DType.uint16,
            scan_items=8,
        ](ctx, m, rows, N, K, dist, mode)
    elif phi_bits == 16:
        execute_topk_bitonic[
            ordered, deterministic, DType.bfloat16, phi_dtype=DType.uint16
        ](ctx, m, rows, N, K, dist, mode)
    else:
        execute_topk_bitonic[ordered, deterministic, DType.bfloat16](
            ctx, m, rows, N, K, dist, mode
        )


def _sweep(ctx: DeviceContext, mut m: Bench, dist: String) raises:
    """The production grid, four arms per cell, interleaved within the cell.

    The contract is `unord_nd` with `K = 2048`, which is what `mla_index_fp8`
    launches, and the row counts straddle the SM count -- the guard that decides
    prefetching-streaming from occupancy-streaming, and so the guard that
    decides whether the wide scan group is even reachable.

    Arm order is fixed rather than randomized: the ratio of interest is within a
    cell, and holding the order constant keeps whatever the first arm of a cell
    pays for a cold allocator identical across cells.
    """
    comptime K = 2048
    for rows in [48, 256]:
        for N in [14336, 32768, 65536, 107232, 157696]:
            # f32 baseline: the denominator every ratio below is stated
            # against.
            execute_topk_bitonic[False, False, DType.float32](
                ctx, m, rows, N, K, dist, "unord_nd"
            )
            # As it ships, then the narrow payload at the wide group it pays
            # for, then the narrow payload alone.
            for knobs in [(0, 0, 0), (0, 16, 0), (0, 16, 8)]:
                _bf16_arms[False, False](
                    ctx,
                    m,
                    rows,
                    N,
                    K,
                    dist,
                    "unord_nd",
                    knobs[0],
                    knobs[1],
                    knobs[2],
                )


def main() raises:
    var rows = arg_parse("rows", 48)
    var N = arg_parse("N", 157696)
    var K = arg_parse("K", 2048)
    var dist = arg_parse("dist", String("q17"))
    var mode = arg_parse("mode", String("ord"))
    var dtype = arg_parse("dtype", String("float32"))
    var sig_bits = arg_parse("sig-bits", 0)
    var phi_bits = arg_parse("phi-bits", 0)
    var scan_items = arg_parse("scan-items", 0)
    var sweep = arg_parse("sweep", 0)

    # An unrecognized mode or distribution is refused rather than falling back to
    # a default. A sweep that asks for a contract this binary does not have would
    # otherwise measure the default four times over and report that the flags
    # change nothing; a misspelled distribution would report the fallback's
    # numbers under the label as typed, which is how the adversarial one comes to
    # look benign. Both are wrong answers that look like findings.
    if dist not in ["q17", "normal", "uniform", "narrow"]:
        raise Error("unknown dist: ", dist)
    if mode not in ["ord", "ord_nd", "unord", "unord_nd"]:
        raise Error("unknown mode: ", mode)
    if dtype not in ["float32", "bfloat16"]:
        raise Error("unknown dtype: ", dtype)
    # 0 derives from the dtype; 32 forces the f32 schedule. 16 is refused rather
    # than defaulted: it IS the derived width at bf16 and wrong at f32.
    if sig_bits not in [0, 32]:
        raise Error("sig-bits must be 0 (derive) or 32 (force), not ", sig_bits)
    if sig_bits == 32 and dtype != "bfloat16":
        raise Error("sig-bits=32 is only a distinct schedule at dtype=bfloat16")
    # Payload width and group width, on the same refuse-don't-default footing.
    # `0` derives; the forced values build these arms in one binary, at one
    # input and one dispatch:
    #
    #   sig-bits 32                the three-round schedule
    #   phi-bits  0                what ships
    #   phi-bits 16, scan-items 8  the narrow payload alone
    #   phi-bits 16, scan-items 0  narrow payload + wide group
    if phi_bits not in [0, 16]:
        raise Error("phi-bits must be 0 (derive) or 16 (force), not ", phi_bits)
    if phi_bits == 16 and dtype != "bfloat16":
        raise Error("phi-bits=16 is only a distinct payload at dtype=bfloat16")
    if scan_items not in [0, 8]:
        raise Error(
            "scan-items must be 0 (derive) or 8 (force the narrow group), not ",
            scan_items,
        )
    if scan_items == 8 and phi_bits != 16:
        raise Error(
            "scan-items=8 is already the derived width except with the narrow"
            " payload, which only bf16 has"
        )
    if sig_bits == 32 and phi_bits == 16:
        raise Error(
            "sig-bits=32 needs the wide payload: a 16-bit one cannot hold a"
            " digit taken from below bit 16"
        )
    # bf16 covers only `ord` and `unord_nd` (what the fp8 indexer calls); the
    # other two would double this binary's launcher specializations to eight.
    if dtype == "bfloat16" and mode not in ["ord", "unord_nd"]:
        raise Error(
            "dtype=bfloat16 is built for mode ord or unord_nd, not ", mode
        )

    var m = Bench()
    with DeviceContext() as ctx:
        if sweep != 0:
            _sweep(ctx, m, dist)
            m.dump_report()
            return
        if dtype == "bfloat16":
            if mode == "unord_nd":
                _bf16_arms[False, False](
                    ctx,
                    m,
                    rows,
                    N,
                    K,
                    dist,
                    mode,
                    sig_bits,
                    phi_bits,
                    scan_items,
                )
            else:
                _bf16_arms[True, True](
                    ctx,
                    m,
                    rows,
                    N,
                    K,
                    dist,
                    mode,
                    sig_bits,
                    phi_bits,
                    scan_items,
                )
        elif mode == "unord_nd":
            execute_topk_bitonic[False, False, DType.float32](
                ctx, m, rows, N, K, dist, mode
            )
        elif mode == "unord":
            execute_topk_bitonic[False, True, DType.float32](
                ctx, m, rows, N, K, dist, mode
            )
        elif mode == "ord_nd":
            execute_topk_bitonic[True, False, DType.float32](
                ctx, m, rows, N, K, dist, mode
            )
        else:
            execute_topk_bitonic[True, True, DType.float32](
                ctx, m, rows, N, K, dist, mode
            )

    m.dump_report()
