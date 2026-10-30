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
"""PDL race test for the row-wise kernels behind `mo.reduce.rms_norm`,
layer norm, softmax and the reduce-shaped argmax.

A real PDL chain with no host sync between the launches:

1. `slow_producer` releases its dependents at entry, spins, and only then
   publishes the activation. It starts from a stale sentinel, so a norm that
   reads its input before its own `wait_on_dependent_grids()` computes the
   wrong row statistics.
2. The row-wise kernel is launched with the PDL attribute at each tested
   level (`ON` releases dependents at exit, `OVERLAP_AT_BEGINNING` right
   after the wait). It becomes resident while the producer still spins.
3. `consumer` waits on the row-wise kernel and copies its output out. Its own
   wait makes an early release unobservable here, so this only confirms the
   output is complete; the test checks that the row-wise kernel does not read
   its input before its own wait, not when it releases dependents. The `ON`
   arm runs the same check and is a harness baseline for `OVERLAP_AT_BEGINNING`.

The producer's activation is not scale-invariant against the sentinel, so a
stale read changes every output element.

When the row-wise kernel releases its dependents is checked separately, with
`release_probe`: a dependent of the row-wise grid that marks itself resident
before its own wait. Every input load of the row-wise kernel is held
(`_gate`) until the probe is resident or a timeout passes. The probe
can only be resident while the grid runs if the grid released it before its
threads finished (a CTA releases its dependents when its first thread issues
the release, so every load is held), which makes the hold end in time only for
an early release. `OVERLAP_AT_BEGINNING` must end it; `ON` (release at exit)
and `OFF` must time out, which shows the harness can tell the two apart.
"""

from std.math import ceildiv, exp, rsqrt
from std.sys import align_of, get_defined_int
from std.sys.intrinsics import llvm_intrinsic
from std.time import global_perf_counter_ns

from builtin_kernels.rowwise_normalization import layer_norm, rms_norm
from builtin_kernels.rowwise_reductions import reduce_argmax
from builtin_kernels.rowwise_softmax import softmax
from layout import Coord, TileTensor, row_major
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.utils.index import Index

comptime dtype = DType.bfloat16
comptime EPS = Float32(0.001)


def slow_producer[
    dt: DType
](
    act: MutPointer[Scalar[dt], MutAnyOrigin],
    real: MutPointer[Scalar[dt], MutAnyOrigin],
    length: Int32,
    delay_us: Int32,
):
    """Releases the dependent grid at entry, spins, then writes `act`."""
    launch_dependent_grids()
    # nanosleep takes nanoseconds and caps at ~1ms per call.
    var spins = Int(delay_us)
    while spins > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        spins -= 1000
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(length), stride):
        act[i] = real[i]


def consumer[
    dt: DType
](
    src: MutPointer[Scalar[dt], MutAnyOrigin],
    dst: MutPointer[Scalar[dt], MutAnyOrigin],
    length: Int32,
):
    """Waits for the row-wise grid, then copies its output backwards."""
    wait_on_dependent_grids()
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(length), stride):
        dst[Int(length) - 1 - i] = src[Int(length) - 1 - i]


comptime GATE_TIMEOUT_NS = 30_000_000
"""How long a gated load waits for the probe: far longer than a launch."""


def release_probe(flags: MutPointer[Int32, MutAnyOrigin]):
    """A dependent of the row-wise grid: marks itself resident before it
    waits, so it can only do so while that grid still runs if the grid
    released it early."""
    if thread_idx.x == 0:
        flags.unsafe_store[volatile=True](Int32(1))
    wait_on_dependent_grids()


@always_inline
def _gate(flags: MutPointer[Int32, MutAnyOrigin]):
    """Holds the calling thread until the probe is resident (`flags[0]`) or
    `GATE_TIMEOUT_NS` has passed, which it records in `flags[1]`."""
    var gave_up = flags.unsafe_offset(1)
    if gave_up.unsafe_load[volatile=True, invariant=False]() != 0:
        return
    var deadline = global_perf_counter_ns() + UInt64(GATE_TIMEOUT_NS)
    while flags.unsafe_load[volatile=True, invariant=False]() == 0:
        if global_perf_counter_ns() > deadline:
            gave_up.unsafe_store[volatile=True](Int32(1))
            return


def _x(r: Int, c: Int) -> Float64:
    return 0.5 + Float64((c * 7 + r * 3) % 11) * 0.25


def _gamma(c: Int) -> Float64:
    return 0.5 + Float64(c % 5) * 0.25


def _beta(c: Int) -> Float64:
    return Float64(c % 3 - 1) * 0.25


def _close(got: Float64, want: Float64, atol: Float64) -> Bool:
    return abs(got - want) <= atol + 0.03 * abs(want)


struct Kind:
    comptime RMS = 0
    comptime LAYER = 1
    comptime SOFTMAX = 2


def run_case[
    kind: Int,
    pdl_level: PDLLevel,
](
    ctx: DeviceContext,
    label: String,
    rows: Int,
    cols: Int,
    iters: Int,
    delay_us: Int,
    mut failures: Int,
) raises:
    var n = rows * cols
    var real_h = ctx.enqueue_create_host_buffer[dtype](n)
    var stale_h = ctx.enqueue_create_host_buffer[dtype](n)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var result_h = ctx.enqueue_create_host_buffer[dtype](n)
    for r in range(rows):
        for c in range(cols):
            real_h[r * cols + c] = _x(r, c).cast[dtype]()
            stale_h[r * cols + c] = Scalar[dtype](1.0)
    for c in range(cols):
        gamma_h[c] = _gamma(c).cast[dtype]()
        beta_h[c] = _beta(c).cast[dtype]()

    var real_d = ctx.enqueue_create_buffer[dtype](n)
    var act_d = ctx.enqueue_create_buffer[dtype](n)
    var out_d = ctx.enqueue_create_buffer[dtype](n)
    var result_d = ctx.enqueue_create_buffer[dtype](n)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)
    ctx.enqueue_copy(real_d, real_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    var shape = Index(rows, cols)
    var act_t = TileTensor(act_d, row_major(Coord(shape)))
    var out_t = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(cols))))
    var beta = TileTensor(beta_d, row_major(Coord(Index(cols))))

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var act_t} -> SIMD[dtype, width]:
        return act_t.load[width=width, alignment=alignment * align_of[dtype]()](
            coords
        )

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_t}:
        out_t.store[alignment=alignment * align_of[dtype]()](coords, val)

    var failed = False
    for it in range(iters):
        ctx.enqueue_copy(act_d, stale_h)
        ctx.enqueue_memset[dtype](out_d, 0)
        ctx.enqueue_memset[dtype](result_d, 0)

        ctx.enqueue_function[slow_producer[dtype]](
            act_d,
            real_d,
            Int32(n),
            Int32(delay_us),
            grid_dim=4,
            block_dim=64,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )

        comptime if kind == Kind.RMS:
            rms_norm[
                dtype,
                2,
                target="gpu",
                epsilon=EPS,
                weight_offset=Float32(0),
                pdl_level=pdl_level,
            ](
                input_fn,
                output_fn,
                Coord(shape),
                Scalar[DType.int](cols),
                gamma,
                ctx,
            )
        elif kind == Kind.LAYER:
            layer_norm[dtype, 2, target="gpu"](
                input_fn,
                output_fn,
                Coord(shape),
                Int(cols),
                gamma,
                beta,
                EPS.cast[dtype](),
                ctx,
            )
        else:
            softmax[dtype, 2, target="gpu", pdl_level=pdl_level](
                input_fn, Coord(shape), cols, out_t, 1, ctx
            )

        ctx.enqueue_function[consumer[dtype]](
            out_d,
            result_d,
            Int32(n),
            grid_dim=ceildiv(n, 256),
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )
        ctx.synchronize()
        ctx.enqueue_copy(result_h, result_d)
        ctx.synchronize()

        var bad = 0
        var first = -1
        for r in range(rows):
            var mean = Float64(0)
            var ms = Float64(0)
            var mx = Float64(-1e30)
            for c in range(cols):
                var v = Float64(_x(r, c).cast[dtype]())
                mean += v
                ms += v * v
                mx = max(mx, v)
            mean /= Float64(cols)
            ms /= Float64(cols)
            var denom = Float64(0)
            for c in range(cols):
                denom += exp(Float64(_x(r, c).cast[dtype]()) - mx)
            for c in range(cols):
                var v = Float64(_x(r, c).cast[dtype]())
                var want: Float64
                comptime if kind == Kind.RMS:
                    want = v * rsqrt(ms + Float64(EPS)) * _gamma(c)
                elif kind == Kind.LAYER:
                    var var_ = ms - mean * mean
                    want = (v - mean) * rsqrt(var_ + Float64(EPS)) * _gamma(
                        c
                    ) + _beta(c)
                else:
                    want = exp(v - mx) / denom
                if not _close(
                    Float64(result_h[r * cols + c]),
                    want,
                    # Softmax outputs are ~1/cols, so only a relative check
                    # tells the real row from the uniform stale one.
                    Float64(0) if kind == Kind.SOFTMAX else Float64(0.03),
                ):
                    if first < 0:
                        first = r * cols + c
                    bad += 1
        if bad > 0:
            print(
                "FAIL:",
                label,
                "iter",
                it,
                ":",
                bad,
                "/",
                n,
                "mismatches; first at",
                first,
                "got",
                Float64(result_h[first]),
            )
            failures += 1
            failed = True
            break
    if not failed:
        print("PASS:", label, "x", iters)
    _ = real_d^
    _ = act_d^
    _ = out_d^
    _ = result_d^
    _ = gamma_d^
    _ = beta_d^


def _v(a: Int, o: Int) -> Float32:
    return Float32((a * 37 + o * 11) % 1009)


def run_argmax_case[
    reduce_dim: Int
](
    ctx: DeviceContext,
    label: String,
    rows: Int,
    cols: Int,
    iters: Int,
    delay_us: Int,
    mut failures: Int,
) raises:
    """Reduce-shaped argmax over a delayed float32 activation.

    The sentinel is all zeros, whose argmax is index 0 for every output, so a
    read before the wait changes nearly every result. `reduce_dim` 1 on a
    long, few-row shape takes the split-K tier; `reduce_dim` 0 lands on the
    tiled tiers (SIMD, scalar fallback) or the cooperative block tier by shape.
    """
    comptime adt = DType.float32
    var n = rows * cols
    var axis = rows if reduce_dim == 0 else cols
    var outs = cols if reduce_dim == 0 else rows
    var real_h = ctx.enqueue_create_host_buffer[adt](n)
    var stale_h = ctx.enqueue_create_host_buffer[adt](n)
    var result_h = ctx.enqueue_create_host_buffer[DType.int64](outs)
    for r in range(rows):
        for c in range(cols):
            var a = r if reduce_dim == 0 else c
            var o = c if reduce_dim == 0 else r
            real_h[r * cols + c] = _v(a, o)
            stale_h[r * cols + c] = 0
    var real_d = ctx.enqueue_create_buffer[adt](n)
    var act_d = ctx.enqueue_create_buffer[adt](n)
    var out_d = ctx.enqueue_create_buffer[DType.int64](outs)
    var result_d = ctx.enqueue_create_buffer[DType.int64](outs)
    ctx.enqueue_copy(real_d, real_h)
    var act_t = TileTensor(act_d, row_major(Coord(Index(rows, cols))))
    var out_ptr = out_d.unsafe_ptr()

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var act_t} -> SIMD[adt, width]:
        return act_t.load[width=width, alignment=alignment * align_of[adt]()](
            coords
        )

    @inline(.always)
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[DType.int64, width]) {var out_ptr}:
        # One value per output; the reduced coordinate is pinned to 0.
        out_ptr.unsafe_store[width=width](Int(coords[1 - reduce_dim]), val)

    var failed = False
    for it in range(iters):
        ctx.enqueue_copy(act_d, stale_h)
        ctx.enqueue_memset[DType.int64](out_d, 0)
        ctx.enqueue_memset[DType.int64](result_d, 0)
        ctx.enqueue_function[slow_producer[adt]](
            act_d,
            real_d,
            Int32(n),
            Int32(delay_us),
            grid_dim=4,
            block_dim=64,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )
        reduce_argmax[adt, target="gpu", reduce_dim=reduce_dim](
            input_fn, output_fn, Coord(Index(rows, cols)), ctx
        )
        ctx.enqueue_function[consumer[DType.int64]](
            out_d,
            result_d,
            Int32(outs),
            grid_dim=ceildiv(outs, 256),
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )
        ctx.synchronize()
        ctx.enqueue_copy(result_h, result_d)
        ctx.synchronize()
        var bad = 0
        var first = -1
        for o in range(outs):
            # Lower index wins ties, as the monoid does.
            var best = 0
            var best_v = _v(0, o)
            for a in range(1, axis):
                var v = _v(a, o)
                if v > best_v:
                    best_v = v
                    best = a
            if Int(result_h[o]) != best:
                if first < 0:
                    first = o
                bad += 1
        if bad > 0:
            print(
                "FAIL:",
                label,
                "iter",
                it,
                ":",
                bad,
                "/",
                outs,
                "mismatches; first at",
                first,
            )
            failures += 1
            failed = True
            break
    if not failed:
        print("PASS:", label, "x", iters)
    _ = real_d^
    _ = act_d^
    _ = out_d^
    _ = result_d^


def _report_release(
    label: String,
    flags_h: HostBuffer[DType.int32],
    expect_early: Bool,
    mut failures: Int,
):
    """Checks the probe ran and whether the gated load gave up on it."""
    var gave_up = flags_h[1] != 0
    if flags_h[0] == 0 or gave_up == expect_early:
        print(
            "FAIL:",
            label,
            "probe ran",
            flags_h[0] != 0,
            "gate gave up",
            gave_up,
            "expected early release",
            expect_early,
        )
        failures += 1
    else:
        print("PASS:", label)


def run_release_case[
    kind: Int,
    pdl_level: PDLLevel,
](
    ctx: DeviceContext,
    label: String,
    rows: Int,
    cols: Int,
    expect_early: Bool,
    mut failures: Int,
) raises:
    """Checks that the row-wise grid releases its dependent early iff
    `pdl_level` says so (the default level for layer norm, which takes none)."""
    var n = rows * cols
    var act_h = ctx.enqueue_create_host_buffer[dtype](n)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var flags_h = ctx.enqueue_create_host_buffer[DType.int32](2)
    for r in range(rows):
        for c in range(cols):
            act_h[r * cols + c] = _x(r, c).cast[dtype]()
    for c in range(cols):
        gamma_h[c] = _gamma(c).cast[dtype]()
        beta_h[c] = _beta(c).cast[dtype]()
    var act_d = ctx.enqueue_create_buffer[dtype](n)
    var out_d = ctx.enqueue_create_buffer[dtype](n)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)
    var flags_d = ctx.enqueue_create_buffer[DType.int32](2)
    ctx.enqueue_copy(act_d, act_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    var shape = Index(rows, cols)
    var act_t = TileTensor(act_d, row_major(Coord(shape)))
    var out_t = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(cols))))
    var beta = TileTensor(beta_d, row_major(Coord(Index(cols))))
    var flags = (
        flags_d.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var act_t, var flags} -> SIMD[dtype, width]:
        _gate(flags)
        return act_t.load[width=width, alignment=alignment * align_of[dtype]()](
            coords
        )

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_t}:
        out_t.store[alignment=alignment * align_of[dtype]()](coords, val)

    ctx.enqueue_memset[DType.int32](flags_d, 0)
    ctx.synchronize()
    comptime if kind == Kind.RMS:
        rms_norm[
            dtype,
            2,
            target="gpu",
            epsilon=EPS,
            weight_offset=Float32(0),
            pdl_level=pdl_level,
        ](
            input_fn,
            output_fn,
            Coord(shape),
            Scalar[DType.int](cols),
            gamma,
            ctx,
        )
    elif kind == Kind.LAYER:
        layer_norm[dtype, 2, target="gpu"](
            input_fn,
            output_fn,
            Coord(shape),
            Int(cols),
            gamma,
            beta,
            EPS.cast[dtype](),
            ctx,
        )
    else:
        softmax[dtype, 2, target="gpu", pdl_level=pdl_level](
            input_fn, Coord(shape), cols, out_t, 1, ctx
        )
    ctx.enqueue_function[release_probe](
        flags,
        grid_dim=1,
        block_dim=32,
        attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
    )
    ctx.enqueue_copy(flags_h, flags_d)
    ctx.synchronize()
    _report_release(label, flags_h, expect_early, failures)


def run_release_argmax_case[
    reduce_dim: Int
](
    ctx: DeviceContext,
    label: String,
    rows: Int,
    cols: Int,
    mut failures: Int,
) raises:
    """The same release check on a reduce-shaped argmax: `reduce_dim` 1 on a
    long, few-row shape takes the split-K tier, `reduce_dim` 0 the tiled tier.
    Both release early."""
    comptime adt = DType.float32
    var n = rows * cols
    var act_h = ctx.enqueue_create_host_buffer[adt](n)
    var flags_h = ctx.enqueue_create_host_buffer[DType.int32](2)
    for r in range(rows):
        for c in range(cols):
            act_h[r * cols + c] = _v(
                r if reduce_dim == 0 else c, c if reduce_dim == 0 else r
            )
    var outs = cols if reduce_dim == 0 else rows
    var act_d = ctx.enqueue_create_buffer[adt](n)
    var out_d = ctx.enqueue_create_buffer[DType.int64](outs)
    var flags_d = ctx.enqueue_create_buffer[DType.int32](2)
    ctx.enqueue_copy(act_d, act_h)
    var act_t = TileTensor(act_d, row_major(Coord(Index(rows, cols))))
    var out_ptr = out_d.unsafe_ptr()
    var flags = (
        flags_d.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var act_t, var flags} -> SIMD[adt, width]:
        _gate(flags)
        return act_t.load[width=width, alignment=alignment * align_of[adt]()](
            coords
        )

    @inline(.always)
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[DType.int64, width]) {var out_ptr}:
        out_ptr.unsafe_store[width=width](Int(coords[1 - reduce_dim]), val)

    ctx.enqueue_memset[DType.int32](flags_d, 0)
    ctx.synchronize()
    reduce_argmax[adt, target="gpu", reduce_dim=reduce_dim](
        input_fn, output_fn, Coord(Index(rows, cols)), ctx
    )
    ctx.enqueue_function[release_probe](
        flags,
        grid_dim=1,
        block_dim=32,
        attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
    )
    ctx.enqueue_copy(flags_h, flags_d)
    ctx.synchronize()
    _report_release(label, flags_h, True, failures)


def run_levels[
    kind: Int
](
    ctx: DeviceContext,
    name: String,
    rows: Int,
    cols: Int,
    iters: Int,
    delay_us: Int,
    mut failures: Int,
) raises:
    # Layer norm takes the scaffolder's default level only.
    comptime if kind != Kind.LAYER:
        run_case[kind, PDLLevel.OFF](
            ctx, name + " OFF", rows, cols, iters, delay_us, failures
        )
        run_case[kind, PDLLevel.ON](
            ctx, name + " ON", rows, cols, iters, delay_us, failures
        )
    run_case[kind, PDLLevel.OVERLAP_AT_BEGINNING](
        ctx, name + " OVERLAP", rows, cols, iters, delay_us, failures
    )


def main() raises:
    var iters = get_defined_int["ITERS", 8]()
    # The producer keeps spinning for DELAY_US after releasing the row-wise
    # grid, so the grid is resident long
    # before the activation lands.
    var delay_us = get_defined_int["DELAY_US", 3000]()
    var failures = 0
    with DeviceContext() as ctx:
        # Warp tier (row <= 2048 bf16), single row and several blocks.
        run_levels[Kind.RMS](
            ctx, "rms warp 4x1024", 4, 1024, iters, delay_us, failures
        )
        run_levels[Kind.RMS](
            ctx, "rms warp 300x1536", 300, 1536, iters, delay_us, failures
        )
        # Block tier, cached row (Llama-3-8B hidden) and a ragged width.
        run_levels[Kind.RMS](
            ctx, "rms block 3x4096", 3, 4096, iters, delay_us, failures
        )
        run_levels[Kind.RMS](
            ctx, "rms block 200x8192", 200, 8192, iters, delay_us, failures
        )
        run_levels[Kind.RMS](
            ctx, "rms ragged 5x3000", 5, 3000, iters, delay_us, failures
        )
        run_levels[Kind.LAYER](
            ctx, "layer warp 4x1024", 4, 1024, iters, delay_us, failures
        )
        run_levels[Kind.LAYER](
            ctx, "layer block 3x4096", 3, 4096, iters, delay_us, failures
        )
        # Softmax: block tier, and the per-element split-K tier whose K+1
        # launches each wait on the previous phase.
        run_levels[Kind.SOFTMAX](
            ctx, "softmax block 32x4096", 32, 4096, iters, delay_us, failures
        )
        run_levels[Kind.SOFTMAX](
            ctx, "softmax splitk 2x131072", 2, 131072, iters, delay_us, failures
        )
        # Reduce-shaped argmax, one case per tier that takes the early
        # release: split-K (few long rows), SIMD tiled, scalar tiled
        # (innermost width not a multiple of the SIMD width), and the
        # cooperative block tier on a non-inner axis. The split-K case
        # cannot see a missing entry wait: the launch enqueues a memset of
        # its counters first, so the kernel's predecessor is that memset and
        # never overlaps the producer. It still checks the output through
        # the consumer.
        run_argmax_case[1](
            ctx, "argmax splitk 2x65536", 2, 65536, iters, delay_us, failures
        )
        run_argmax_case[0](
            ctx, "argmax tiled 16x4096", 16, 4096, iters, delay_us, failures
        )
        run_argmax_case[0](
            ctx,
            "argmax tiled scalar 16x4098",
            16,
            4098,
            iters,
            delay_us,
            failures,
        )
        run_argmax_case[0](
            ctx, "argmax coop 4096x8", 4096, 8, iters, delay_us, failures
        )
        # When the grid releases its dependent: early at the default level
        # and at OVERLAP_AT_BEGINNING, at exit at ON, never at OFF. The warp
        # and block tiers of the norms and softmax, and the split-K and tiled
        # argmax tiers; the split-K softmax phases always release at exit.
        # The probe's module is loaded first: a cold first launch takes
        # longer than a gated load waits for it.
        var warm_d = ctx.enqueue_create_buffer[DType.int32](2)
        ctx.enqueue_memset[DType.int32](warm_d, 0)
        ctx.enqueue_function[release_probe](
            warm_d,
            grid_dim=1,
            block_dim=32,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )
        ctx.synchronize()
        run_release_case[Kind.RMS, PDLLevel.OVERLAP_AT_BEGINNING](
            ctx,
            "release rms warp 1024x1024 OVERLAP",
            1024,
            1024,
            True,
            failures,
        )
        run_release_case[Kind.RMS, PDLLevel.ON](
            ctx, "release rms warp 1024x1024 ON", 1024, 1024, False, failures
        )
        run_release_case[Kind.RMS, PDLLevel.OFF](
            ctx, "release rms warp 1024x1024 OFF", 1024, 1024, False, failures
        )
        run_release_case[Kind.RMS, PDLLevel.OVERLAP_AT_BEGINNING](
            ctx, "release rms block 3x4096 OVERLAP", 3, 4096, True, failures
        )
        run_release_case[Kind.RMS, PDLLevel.ON](
            ctx, "release rms block 3x4096 ON", 3, 4096, False, failures
        )
        run_release_case[Kind.LAYER, PDLLevel.OVERLAP_AT_BEGINNING](
            ctx, "release layer block 3x4096 default", 3, 4096, True, failures
        )
        run_release_case[Kind.SOFTMAX, PDLLevel.OVERLAP_AT_BEGINNING](
            ctx,
            "release softmax block 32x4096 OVERLAP",
            32,
            4096,
            True,
            failures,
        )
        run_release_case[Kind.SOFTMAX, PDLLevel.ON](
            ctx, "release softmax block 32x4096 ON", 32, 4096, False, failures
        )
        run_release_argmax_case[1](
            ctx, "release argmax splitk 2x65536", 2, 65536, failures
        )
        run_release_argmax_case[0](
            ctx, "release argmax tiled 16x4096", 16, 4096, failures
        )
    if failures > 0:
        raise Error(String(failures, " row-wise PDL race case(s) failed"))
    print("All row-wise PDL race tests passed!")
