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
"""Per-row phase attribution for the indexer top-k radix select.

Stamps `global_perf_counter_ns` at every phase boundary of the select into a
per-row trace buffer and prints it, so the kernel's time splits into "round `r`
row scan", "round `r` split", "append" and "sort" instead of one opaque total.
Reads the same input as `bench_topk_mt`, so a phase attribution describes the
run that was benchmarked rather than a different one.

Traces whichever kernel the decode dispatch would pick for the shape *and* the
output contract -- `histsel_resident_topk` at whichever payload width holds the
row, `histsel_topk` with prefetch and narrow refine digits otherwise. Tracing a
configuration the dispatch never launches attributes phases that nothing runs,
which is why the contract is a flag here and not a fixed choice.

`--dtype` picks the score buffer's element type, `float32` or `bfloat16`, as
`bench_topk_mt` does -- the two read widths take different round schedules
(`_hsel_sig_bits`), so a phase attribution at one dtype does not describe the
other. `--sig-bits=32` forces the f32 schedule onto a bf16 buffer, which is the
pre-Stage-A code at an unchanged input: it is what makes "does bf16 run MORE
rounds than f32" answerable from the dump rather than from the schedule table. The `SCHED` header line reports the schedule the traced arm actually
ran, because the slot layout depends on it: round `r`'s slots are `[1 + 2r]`
and `[2 + 2r]` over the DENSE round index, and the score half may be shorter
than the column half.

    trace_topk_bitonic --rows=48 --N=14336 --K=2048 --dist=q17 \
        --unordered=True --deterministic=False --dtype=bfloat16
"""

from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, FuncAttribute
from internal_utils import arg_parse
from layout import TileTensor, row_major
from structured_kernels.trace_buf import GmemTrace

from nn.topk_bitonic import (
    HSEL_TRACE_EVENTS,
    _histsel_resident_kernel,
    _histsel_topk_kernel,
    _hsel_half_rounds,
    _hsel_sig_bits,
    _HSEL_RANK_BITS,
    _HSEL_RES_BLOCK,
    _HSEL_RES_MAX,
    _HSEL_RES_MAX_WIDE,
    _HSEL_RES_VECS,
    _HSEL_RES_VECS_WIDE,
    _HSEL_SEL_CAP,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _PTOPK_BLOCK,
    _PTOPK_TOTAL,
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
    var h = _mix64((i + 1) * 0x9E3779B97F4A7C15)
    return Float64(h >> 11) * (1.0 / 9007199254740992.0)


@inline(.always)
def _u_val(r: Int, c: Int, N: Int, k: Int) -> Float64:
    return _u01(((UInt64(r) * UInt64(N) + UInt64(c)) << 2) + UInt64(k))


@inline(.always)
def _u_len(r: Int) -> Float64:
    return _u01(0x8000000000000000 + UInt64(r))


comptime _IH_TERMS: Int = 6
comptime _IH_SCALE: Float64 = 1.4142135623730951


def _sample(dist: String, r: Int, c: Int, N: Int) -> Float32:
    """One score, generated exactly as `bench_topk_mt` generates it.

    Carried verbatim rather than approximated: the select's cost is set by how
    many columns share the threshold's coarse bin, so a phase attributed on a
    distribution that merely resembles the benchmarked one describes a different
    kernel behaviour. `_u_len` reproduces the same masked-suffix lengths too.
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


# Which instantiation to trace. Named once and shared by the launch and the
# `SCHED` header, so the two cannot disagree about which kernel the trace slots
# belong to. Keyed on `N` and the contract only: the launcher additionally gates
# the resident arms on rows under the SM count, which this ignores so a resident
# arm stays traceable at any row count.
comptime _ARM_RES_UNORD: Int = 0
comptime _ARM_RES_UNORD_WIDE: Int = 1
comptime _ARM_RES_ORD_BINDIGIT: Int = 2
comptime _ARM_RES_ORD: Int = 3
comptime _ARM_STREAM_UNORD_ND: Int = 4
comptime _ARM_STREAM_RANKED: Int = 5


def _pick_arm(N: Int, K: Int, unordered: Bool, deterministic: Bool) -> Int:
    if unordered:
        if N <= _HSEL_RES_MAX:
            return _ARM_RES_UNORD
        if N <= _HSEL_RES_MAX_WIDE:
            return _ARM_RES_UNORD_WIDE
    if N <= _HSEL_RES_MAX and N < K + K // 2:
        return _ARM_RES_ORD_BINDIGIT
    if N <= _HSEL_RES_MAX:
        return _ARM_RES_ORD
    if unordered and not deterministic:
        return _ARM_STREAM_UNORD_ND
    return _ARM_STREAM_RANKED


def _arm_runs_column_half(arm: Int) -> Bool:
    """Whether the arm resolves the column half at all.

    The resident select skips it under the cheap contract -- it cuts the tie
    plateau by counting instead -- so its trace holds score-half rounds only.
    """
    return arm not in [_ARM_RES_UNORD, _ARM_RES_UNORD_WIDE]


def _launch[
    in_dtype: DType,
    //,
    *,
    unordered: Bool = False,
    deterministic: Bool = True,
    sched: Int = _hsel_sig_bits[in_dtype](),
](
    ctx: DeviceContext,
    scores_t: TileTensor[in_dtype, ...],
    idxs_t: TileTensor[.int32, ...],
    trace_ptr: MutPointer[UInt64, MutUntrackedOrigin],
    N: Int,
    K: Int,
    rows: Int,
) raises:
    var arm = _pick_arm(N, K, unordered, deterministic)
    comptime if unordered:

        @inline(.always)
        def resident[res_vecs: Int]() raises {imm}:
            ctx.enqueue_function[
                _histsel_resident_kernel[
                    GmemTrace,
                    enable_trace=True,
                    sel_cap=_PTOPK_TOTAL,
                    ordered=False,
                    deterministic=deterministic,
                    res_vecs=res_vecs,
                    in_dtype=in_dtype,
                    sig_bits=sched,
                ]
            ](
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    scores_t.ptr
                ),
                rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
                Int32(N),
                Int32(K),
                GmemTrace(trace_ptr),
                grid_dim=rows,
                block_dim=_HSEL_RES_BLOCK,
            )

        if arm == _ARM_RES_UNORD:
            resident[_HSEL_RES_VECS]()
            return
        if arm == _ARM_RES_UNORD_WIDE:
            resident[_HSEL_RES_VECS_WIDE]()
            return

    if arm == _ARM_RES_ORD_BINDIGIT:
        ctx.enqueue_function[
            _histsel_resident_kernel[
                GmemTrace,
                enable_trace=True,
                bin_digit=True,
                in_dtype=in_dtype,
                sig_bits=sched,
            ]
        ](
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](scores_t.ptr),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
            Int32(N),
            Int32(K),
            GmemTrace(trace_ptr),
            grid_dim=rows,
            block_dim=_HSEL_RES_BLOCK,
        )
        return

    if arm == _ARM_RES_ORD:
        ctx.enqueue_function[
            _histsel_resident_kernel[
                GmemTrace,
                enable_trace=True,
                in_dtype=in_dtype,
                sig_bits=sched,
            ]
        ](
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](scores_t.ptr),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
            Int32(N),
            Int32(K),
            GmemTrace(trace_ptr),
            grid_dim=rows,
            block_dim=_HSEL_RES_BLOCK,
        )
        return

    # Past the resident widths the fully relaxed contract streams as well, and it
    # is the only one that skips the rank there. Tracing it with the ordered
    # instantiation would attribute a rank the dispatch never launches, which is
    # the one thing this tool exists not to do -- and every rung of the multi-turn
    # benchmark past the first lands here.
    comptime if unordered and not deterministic:
        ctx.enqueue_function[
            _histsel_topk_kernel[
                GmemTrace,
                enable_trace=True,
                prefetch=True,
                tail_bits=_HSEL_TAIL_BITS,
                ordered=False,
                deterministic=False,
                in_dtype=in_dtype,
                sig_bits=sched,
            ]
        ](
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](scores_t.ptr),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
            Int32(N),
            Int32(K),
            GmemTrace(trace_ptr),
            grid_dim=rows,
            block_dim=_PTOPK_BLOCK,
            shared_mem_bytes=_HSEL_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(_HSEL_SMEM_BYTES)
            ),
        )
        return

    ctx.enqueue_function[
        _histsel_topk_kernel[
            GmemTrace,
            enable_trace=True,
            prefetch=True,
            tail_bits=_HSEL_TAIL_BITS,
            rank_bits=_HSEL_RANK_BITS,
            rank_slots=True,
            sel_cap=_HSEL_SEL_CAP,
            in_dtype=in_dtype,
            sig_bits=sched,
        ]
    ](
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](scores_t.ptr),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
        Int32(N),
        Int32(K),
        GmemTrace(trace_ptr),
        grid_dim=rows,
        block_dim=_PTOPK_BLOCK,
        shared_mem_bytes=_HSEL_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(_HSEL_SMEM_BYTES)
        ),
    )


def _run[
    in_dtype: DType, sched: Int = _hsel_sig_bits[in_dtype]()
](
    ctx: DeviceContext,
    rows: Int,
    N: Int,
    K: Int,
    dist: String,
    unordered: Bool,
    deterministic: Bool,
    iters: Int,
) raises:
    var scores_buf = ctx.enqueue_create_buffer[in_dtype](rows * N)
    var idxs_buf = ctx.enqueue_create_buffer[.int32](rows * K)
    var trace_buf = ctx.enqueue_create_buffer[.uint64](rows * HSEL_TRACE_EVENTS)

    # Keyed by index, so the traced run and the benchmarked run are the same
    # array. Stored at the READ width: at bf16 the trace sees the rounded input
    # the kernel sees, not the one generated.
    with scores_buf.map_to_host() as h:
        for r in range(rows):
            var nk = N - Int(_u_len(r) * Float64(N) / 8.0)
            for c in range(N):
                var v: Float32
                if c < nk:
                    v = Float32(_sample(dist, r, c, N))
                else:
                    v = Float32(-3.0e38)
                h[r * N + c] = v.cast[in_dtype]()
    ctx.enqueue_memset(trace_buf, 0)

    var scores_t = TileTensor(scores_buf, row_major(rows, N))
    var idxs_t = TileTensor(idxs_buf, row_major(rows, K))
    ctx.synchronize()

    var trace_ptr = rebind[MutPointer[UInt64, MutUntrackedOrigin]](
        trace_buf.unsafe_ptr()
    )

    # The contract is a comptime parameter of `_launch`, so the three arms are
    # three instantiations and the runtime flags only choose between them.
    @inline(.always)
    def sweep(reps: Int) raises {var}:
        for _ in range(reps):
            if unordered and not deterministic:
                _launch[unordered=True, deterministic=False, sched=sched](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )
            elif unordered:
                _launch[unordered=True, sched=sched](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )
            else:
                _launch[unordered=False, sched=sched](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )

    # Warm up before the traced launch: a cold launch's first row scan
    # measures instruction-cache and page-table misses, not the kernel.
    sweep(20)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    sweep(iters)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("TIME_MS", Float64(t1 - t0) / Float64(iters) / 1.0e6)

    # Round `r`'s pass and split are slots `[1 + 2r]` and `[2 + 2r]` over the
    # DENSE round index, so the column half's rounds start at `VAL_ROUNDS`; a
    # decoder assuming one stride misattributes them and still looks plausible.
    comptime val_rounds = _hsel_half_rounds[_HSEL_TAIL_BITS, sched]()
    comptime col_rounds = _hsel_half_rounds[_HSEL_TAIL_BITS, 32]()
    var arm = _pick_arm(N, K, unordered, deterministic)
    print(
        "SCHED arm",
        arm,
        "tail_bits",
        _HSEL_TAIL_BITS,
        "sig_bits",
        sched,
        "VAL_ROUNDS",
        val_rounds,
        "COL_ROUNDS",
        col_rounds if _arm_runs_column_half(arm) else 0,
    )

    print("TRACE", rows, N, K, dist, HSEL_TRACE_EVENTS, unordered, in_dtype)
    with trace_buf.map_to_host() as h:
        for r in range(rows):
            var line = String("T ")
            line += String(r)
            for e in range(HSEL_TRACE_EVENTS):
                line += " "
                line += String(h[r * HSEL_TRACE_EVENTS + e])
            print(line)
    print("TRACEDONE")

    _ = scores_buf
    _ = idxs_buf
    _ = trace_buf


def main() raises:
    var rows = arg_parse("rows", 48)
    var N = arg_parse("N", 107228)
    var K = arg_parse("K", 2048)
    var dist = arg_parse("dist", String("q17"))
    var unordered = arg_parse("unordered", False)
    var deterministic = arg_parse("deterministic", True)
    var iters = arg_parse("iters", 20)
    var dtype = arg_parse("dtype", String("float32"))
    var sig_bits = arg_parse("sig-bits", 0)

    # Refused rather than defaulted: a misspelled flag that keeps its default
    # reports the default's phases under the label as typed.
    if dtype not in ["float32", "bfloat16"]:
        raise Error("unknown dtype: ", dtype)
    # 0 derives from the dtype; 32 forces the f32 schedule, which is only a
    # different kernel at bf16.
    if sig_bits not in [0, 32]:
        raise Error("sig-bits must be 0 (derive) or 32 (force), not ", sig_bits)
    if sig_bits == 32 and dtype != "bfloat16":
        raise Error("sig-bits=32 is only a distinct schedule at dtype=bfloat16")

    with DeviceContext() as ctx:
        if dtype == "bfloat16":
            if sig_bits == 32:
                _run[DType.bfloat16, 32](
                    ctx, rows, N, K, dist, unordered, deterministic, iters
                )
            else:
                _run[DType.bfloat16](
                    ctx, rows, N, K, dist, unordered, deterministic, iters
                )
        else:
            _run[DType.float32](
                ctx, rows, N, K, dist, unordered, deterministic, iters
            )
