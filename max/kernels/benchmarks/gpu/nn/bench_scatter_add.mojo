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
"""Benchmarks the accumulating scatter: `scatter_elements` with `_atomic_add`.

Modes:
- `memcpy`: the device-to-device copy of the output volume on its own. It is the
  ceiling for the `input -> output` copy the op issues internally, and no
  patch touches it, so both arms of an A/B must report the same number here.
- `emb`: embedding-gradient shape, `indices` constant down the columns so
  every update of an index row lands in one output row: many updates into
  few destination rows.
- `spread`: every element picks its own destination out of a much larger
  table: few updates into many destination rows, no duplicates.
- `dupblk`: groups of `_RUNS` index rows repeat one destination row, so
  several threads of one block hit the same addresses.
- `seg`: the 1-D histogram MAX's ragged and speculative paths build every
  step, `scatter_add(zeros[T + 1], ones[b], ends, axis=0)`. `rows` is the
  slot count `T`, `num_idx` is the segment count `b`; `T` is odd, so the
  segment count is not a power of two.
- `det`: reports how many distinct byte patterns repeated `emb` runs produce.

The timed region is the whole op, which includes the internal
`input -> output` device copy. Reported bytes are
`2 * input + 2 * updates + indices`, the same accounting
`bench_scatter_nd.mojo` uses, so the achievable ceiling is the
device-to-device memcpy bandwidth.
"""

from std.sys import get_defined_dtype, get_defined_int, size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchId,
    BenchMetric,
    Bencher,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext, HostBuffer
from internal_utils import arg_parse
from layout import TileTensor, row_major
from nn.gather_scatter import _atomic_add, scatter_elements

comptime itype = DType.int32

comptime _RUNS = 4
"""Rows of `indices` sharing one destination row in `dupblk`.

These rows sit `cols` apart in flat order, so they land in different blocks:
the arm measures repeated destinations across blocks. A collision inside one
block is not reachable along axis 0, because consecutive elements of an index
row address different output elements.
"""

comptime _HOT = 64
"""Destination rows touched in `emb`, `dupblk`, and `det`."""

comptime _DET_RUNS = 8
"""Reruns of the same op the `det` mode compares."""


@inline(.always)
def _wrap32(v: Int) -> Int:
    return v & 0xFFFFFFFF


@inline(.always)
def _host_index(mode: String, i: Int, j: Int, rows: Int) -> Int:
    """Destination row of update `(i, j)`; always inside `[0, rows)`."""
    if mode == "emb":
        # Constant down the columns, so one index row lands in one output row.
        return _wrap32(i * 2654435761) % _HOT
    if mode == "dupblk":
        # Hash the *group* number, so index rows `_RUNS * g .. _RUNS * g +
        # _RUNS - 1` all name one destination row and several threads of one
        # block hit the same addresses. Hashing `i` and then dividing would
        # leave every row on its own destination.
        return _wrap32((i // _RUNS) * 2654435761) % _HOT
    # Every element picks its own destination across the whole table.
    return _wrap32(i * 2654435761 + j * 40503 + 17) % rows


def _check_bounds[
    it: DType
](host: HostBuffer[it], n: Int, bound: Int, what: String) raises:
    """Fails the bench if a generated index leaves `[0, bound)`.

    The kernel applies these indices with an unchecked atomic write, so a
    generator that computes its bound from the wrong quantity corrupts the
    buffer instead of failing, and the timing does not move.
    """
    for k in range(n):
        var v = Int(host[k])
        if v < 0 or v >= bound:
            raise Error(
                String(
                    "bench: ",
                    what,
                    " index ",
                    v,
                    " outside [0, ",
                    bound,
                    ") at ",
                    k,
                )
            )


def _fill_indices[
    mode: String, rows: Int, cols: Int, num_idx: Int
](host: HostBuffer[itype]):
    """Fills `[num_idx, cols]` 2-D indices for `mode`."""
    for i in range(num_idx):
        for j in range(cols):
            host[i * cols + j] = Scalar[itype](_host_index(mode, i, j, rows))


@inline(.never)
def run_2d[
    dtype: DType,
    rows: Int,
    cols: Int,
    num_idx: Int,
    mode: String,
    use_atomic: Bool = True,
](mut m: Bench, ctx: DeviceContext[]) raises:
    """`scatter_add` along axis 0 of a `[rows, cols]` output.

    `use_atomic=False` runs the same index pattern through the op's
    overwriting path instead, which is what an accumulating scatter would
    cost with the atomics removed. Its result is not an accumulation, so it
    is a bound and not a candidate.
    """
    var in_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](num_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](num_idx * cols)

    var idx_host = ctx.enqueue_create_host_buffer[itype](num_idx * cols)
    ctx.synchronize()
    _fill_indices[mode, rows, cols, num_idx](idx_host)
    _check_bounds[itype](idx_host, num_idx * cols, rows, mode)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var in_tt = TileTensor(in_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[num_idx, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[num_idx, cols]())

    @inline(.always)
    def bench_func(
        mut b: Bencher,
    ) raises {var in_tt, var out_tt, var upd_tt, var idx_tt, imm}:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext[]) raises {imm}:
            comptime if use_atomic:
                scatter_elements[target="gpu", atomic_update_fn=_atomic_add](
                    in_tt, idx_tt, upd_tt, 0, out_tt, ctx
                )
            else:
                scatter_elements[target="gpu"](
                    in_tt, idx_tt, upd_tt, 0, out_tt, ctx
                )

        bencher_iter_custom(b, kernel_launch, ctx)

    comptime in_bytes = rows * cols * size_of[dtype]()
    comptime upd_bytes = num_idx * cols * size_of[dtype]()
    comptime idx_bytes = num_idx * cols * size_of[itype]()
    m.bench_function(
        bench_func,
        BenchId(
            String(mode, "_add") if use_atomic else String(mode, "_store"),
            input_id=String(dtype, "/", rows, "x", cols, "/idx=", num_idx),
        ),
        [
            ThroughputMeasure(
                BenchMetric.bytes, 2 * in_bytes + 2 * upd_bytes + idx_bytes
            )
        ],
    )
    ctx.synchronize()

    _ = in_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev
    _ = idx_host


@inline(.never)
def run_seg[
    dtype: DType, it: DType, total: Int, num_seg: Int
](mut m: Bench, ctx: DeviceContext[]) raises:
    """The 1-D histogram the serving paths build every step.

    `hist = scatter_add(zeros"[total]", ones"[num_seg]", clipped_ends, axis=0)`
    where `clipped_ends` is `min(cumulative_sum(lengths), total)`:
    nondecreasing, and saturating at `total` so its tail repeats. The
    ragged callers take `total = 1 + ` the row count, which is why the slot
    counts here are odd. Input is zero and updates are one, as the graph op
    builds them.
    """
    var in_dev = ctx.enqueue_create_buffer[dtype](total)
    var out_dev = ctx.enqueue_create_buffer[dtype](total)
    var upd_dev = ctx.enqueue_create_buffer[dtype](num_seg)
    var idx_dev = ctx.enqueue_create_buffer[it](num_seg)

    var idx_host = ctx.enqueue_create_host_buffer[it](num_seg)
    var zero_host = ctx.enqueue_create_host_buffer[dtype](total)
    var one_host = ctx.enqueue_create_host_buffer[dtype](num_seg)
    ctx.synchronize()
    # Nondecreasing ends of requests whose lengths average `total / num_seg`,
    # clipped into the buffer the way `ops.min(ends, pos[-1])` clips them, so
    # the rows past the last request all pile onto the final slot. Production
    # allocates `T + 1` slots and sinks at `T`, so its highest index is one
    # short of the buffer length; here `total` is the length.
    var through = 1
    comptime avg = total // num_seg
    for i in range(num_seg):
        through += 1 + _wrap32((i + 1) * 2654435761) % max(1, 2 * avg)
        idx_host[i] = Scalar[it](min(through, total - 1))
    _check_bounds[it](idx_host, num_seg, total, "seg")
    for i in range(total):
        zero_host[i] = Scalar[dtype](0)
    for i in range(num_seg):
        one_host[i] = Scalar[dtype](1)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.enqueue_copy(in_dev, zero_host)
    ctx.enqueue_copy(upd_dev, one_host)
    ctx.synchronize()

    var in_tt = TileTensor(in_dev, row_major[total]())
    var out_tt = TileTensor(out_dev, row_major[total]())
    var upd_tt = TileTensor(upd_dev, row_major[num_seg]())
    var idx_tt = TileTensor(idx_dev, row_major[num_seg]())

    @inline(.always)
    def bench_func(
        mut b: Bencher,
    ) raises {var in_tt, var out_tt, var upd_tt, var idx_tt, imm}:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext[]) raises {imm}:
            scatter_elements[target="gpu", atomic_update_fn=_atomic_add](
                in_tt, idx_tt, upd_tt, 0, out_tt, ctx
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    comptime in_bytes = total * size_of[dtype]()
    comptime upd_bytes = num_seg * size_of[dtype]()
    comptime idx_bytes = num_seg * size_of[it]()
    m.bench_function(
        bench_func,
        BenchId(
            String("seg_add"),
            input_id=String(dtype, "/", it, "/T=", total, "/b=", num_seg),
        ),
        [
            ThroughputMeasure(
                BenchMetric.bytes, 2 * in_bytes + 2 * upd_bytes + idx_bytes
            )
        ],
    )
    ctx.synchronize()

    _ = in_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev
    _ = idx_host
    _ = zero_host
    _ = one_host


@inline(.never)
def run_memcpy[
    dtype: DType, rows: Int, cols: Int
](mut m: Bench, ctx: DeviceContext[]) raises:
    """Control: the `input -> output` memcpy on its own, unchanged by a patch.
    """
    var in_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var in_tt = TileTensor(in_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())

    @inline(.always)
    def bench_func(
        mut b: Bencher,
    ) raises {var in_tt, var out_tt, imm}:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext[]) raises {imm}:
            # Exactly the call `scatter_elements` makes internally.
            ctx.enqueue_copy(out_tt.ptr, in_tt.ptr, out_tt.num_elements())

        bencher_iter_custom(b, kernel_launch, ctx)

    comptime in_bytes = rows * cols * size_of[dtype]()
    m.bench_function(
        bench_func,
        BenchId("memcpy", input_id=String(dtype, "/", rows, "x", cols)),
        [ThroughputMeasure(BenchMetric.bytes, 2 * in_bytes)],
    )
    ctx.synchronize()

    _ = in_dev
    _ = out_dev


def _pattern[dtype: DType](host: HostBuffer[dtype], n: Int) -> Int:
    """FNV-1a over the bit patterns, so two runs can be compared as wholes."""
    var h = 0xCBF29CE484222325
    for i in range(n):
        h = (h ^ (Int(host[i].to_bits()) & 0xFFFFFFFFFFFFFFFF)) * 0x100000001B3
        h = h & 0xFFFFFFFFFFFFFFFF
    return h


def _report_determinism[
    dtype: DType, rows: Int, cols: Int, num_idx: Int
](ctx: DeviceContext[]) raises:
    """Reports how many distinct byte patterns repeated `emb` runs produce.

    The accumulating path adds each update with a device atomic, so the order
    the adds land in is whatever the scheduler chose. `input` is zero and
    `updates` holds values whose sum is not exact under a reordering, so a
    result that depends on that order shows up as more than one pattern.
    """
    var in_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](num_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](num_idx * cols)

    var idx_host = ctx.enqueue_create_host_buffer[itype](num_idx * cols)
    var upd_host = ctx.enqueue_create_host_buffer[dtype](num_idx * cols)
    var in_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var ref_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var cmp_host = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.synchronize()

    _fill_indices["emb", rows, cols, num_idx](idx_host)
    _check_bounds[itype](idx_host, num_idx * cols, rows, "det")
    for k in range(num_idx * cols):
        upd_host[k] = Scalar[dtype](1.0 / Float64(k % 1021 + 1))
    for k in range(rows * cols):
        in_host[k] = Scalar[dtype](0)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.enqueue_copy(upd_dev, upd_host)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var in_tt = TileTensor(in_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[num_idx, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[num_idx, cols]())

    # Every rerun is hashed as a whole, so the report counts distinct results
    # rather than distances from an arbitrary first run: two arms that both
    # differ from their first run can still have the same spread.
    scatter_elements[target="gpu", atomic_update_fn=_atomic_add](
        in_tt, idx_tt, upd_tt, 0, out_tt, ctx
    )
    ctx.enqueue_copy(ref_host, out_dev)
    ctx.synchronize()
    var first = _pattern[dtype](ref_host, rows * cols)
    var patterns = List[Int](capacity=_DET_RUNS + 1)
    patterns.append(first)

    var differing = 0
    var worst_bits = 0
    for _ in range(_DET_RUNS):
        scatter_elements[target="gpu", atomic_update_fn=_atomic_add](
            in_tt, idx_tt, upd_tt, 0, out_tt, ctx
        )
        ctx.enqueue_copy(cmp_host, out_dev)
        ctx.synchronize()
        var diff = 0
        for k in range(rows * cols):
            var delta = Int(cmp_host[k].to_bits()) - Int(ref_host[k].to_bits())
            if delta < 0:
                delta = -delta
            diff = max(diff, delta)
        if diff != 0:
            differing += 1
            worst_bits = max(worst_bits, diff)
        var h = _pattern[dtype](cmp_host, rows * cols)
        var seen = False
        for prior in patterns:
            if prior == h:
                seen = True
                break
        if not seen:
            patterns.append(h)

    print(
        "determinism:",
        String(dtype),
        "emb",
        String(rows, "x", cols, "/idx=", num_idx),
        "--",
        len(patterns),
        "distinct results in",
        _DET_RUNS + 1,
        "reruns;",
        differing,
        "differ from the first; worst bit delta",
        worst_bits,
    )

    _ = in_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev
    _ = idx_host
    _ = upd_host
    _ = in_host
    _ = ref_host
    _ = cmp_host


def _report_determinism_seg[
    dtype: DType, it: DType, total: Int, num_seg: Int
](ctx: DeviceContext[]) raises:
    """The same count for the rank-1 histogram the serving paths build.

    Integral accumulation is exact, so every rerun must agree bit for bit
    under any ordering. This is the arm that says whether the production
    geometry depends on the ordering at all.
    """
    var in_dev = ctx.enqueue_create_buffer[dtype](total)
    var out_dev = ctx.enqueue_create_buffer[dtype](total)
    var upd_dev = ctx.enqueue_create_buffer[dtype](num_seg)
    var idx_dev = ctx.enqueue_create_buffer[it](num_seg)
    var idx_host = ctx.enqueue_create_host_buffer[it](num_seg)
    var zero_host = ctx.enqueue_create_host_buffer[dtype](total)
    var one_host = ctx.enqueue_create_host_buffer[dtype](num_seg)
    var ref_host = ctx.enqueue_create_host_buffer[dtype](total)
    var cmp_host = ctx.enqueue_create_host_buffer[dtype](total)
    ctx.synchronize()

    var through = 1
    comptime avg = total // num_seg
    for i in range(num_seg):
        through += 1 + _wrap32((i + 1) * 2654435761) % max(1, 2 * avg)
        idx_host[i] = Scalar[it](min(through, total - 1))
    _check_bounds[it](idx_host, num_seg, total, "det_seg")
    for i in range(total):
        zero_host[i] = Scalar[dtype](0)
    for i in range(num_seg):
        one_host[i] = Scalar[dtype](1)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.enqueue_copy(in_dev, zero_host)
    ctx.enqueue_copy(upd_dev, one_host)
    ctx.synchronize()

    var in_tt = TileTensor(in_dev, row_major[total]())
    var out_tt = TileTensor(out_dev, row_major[total]())
    var upd_tt = TileTensor(upd_dev, row_major[num_seg]())
    var idx_tt = TileTensor(idx_dev, row_major[num_seg]())

    scatter_elements[target="gpu", atomic_update_fn=_atomic_add](
        in_tt, idx_tt, upd_tt, 0, out_tt, ctx
    )
    ctx.enqueue_copy(ref_host, out_dev)
    ctx.synchronize()
    var patterns = List[Int](capacity=_DET_RUNS + 1)
    patterns.append(_pattern[dtype](ref_host, total))

    var differing = 0
    for _ in range(_DET_RUNS):
        scatter_elements[target="gpu", atomic_update_fn=_atomic_add](
            in_tt, idx_tt, upd_tt, 0, out_tt, ctx
        )
        ctx.enqueue_copy(cmp_host, out_dev)
        ctx.synchronize()
        var same = True
        for k in range(total):
            if cmp_host[k].to_bits() != ref_host[k].to_bits():
                same = False
                break
        if not same:
            differing += 1
        var h = _pattern[dtype](cmp_host, total)
        var seen = False
        for prior in patterns:
            if prior == h:
                seen = True
                break
        if not seen:
            patterns.append(h)

    print(
        "determinism:",
        String(dtype),
        "seg",
        String(it, " T=", total, "/b=", num_seg),
        "--",
        len(patterns),
        "distinct results in",
        _DET_RUNS + 1,
        "reruns;",
        differing,
        "differ from the first",
    )

    _ = in_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev
    _ = idx_host
    _ = zero_host
    _ = one_host
    _ = ref_host
    _ = cmp_host


def main() raises:
    var mode = arg_parse("mode", "memcpy")
    comptime dtype = get_defined_dtype["dtype", .float32]()
    comptime rows = get_defined_int["rows", 131072]()
    comptime cols = get_defined_int["cols", 1024]()
    comptime num_idx = get_defined_int["num_idx", 4096]()

    var m = Bench()
    with DeviceContext() as ctx:
        if mode == "memcpy":
            run_memcpy[dtype, rows, cols](m, ctx)
        elif mode == "emb":
            run_2d[dtype, rows, cols, num_idx, "emb"](m, ctx)
        elif mode == "spread":
            run_2d[dtype, rows, cols, num_idx, "spread"](m, ctx)
        elif mode == "dupblk":
            run_2d[dtype, rows, cols, num_idx, "dupblk"](m, ctx)
        elif mode == "emb_store":
            run_2d[dtype, rows, cols, num_idx, "emb", use_atomic=False](m, ctx)
        elif mode == "spread_store":
            run_2d[dtype, rows, cols, num_idx, "spread", use_atomic=False](
                m, ctx
            )
        elif mode == "seg":
            run_seg[dtype, itype, rows, num_idx](m, ctx)
        elif mode == "seg64":
            run_seg[dtype, DType.int64, rows, num_idx](m, ctx)
        elif mode == "det":
            _report_determinism[dtype, rows, cols, num_idx](ctx)
            run_2d[dtype, rows, cols, num_idx, "emb"](m, ctx)
        elif mode == "det_seg":
            _report_determinism_seg[dtype, itype, rows, num_idx](ctx)
            run_seg[dtype, itype, rows, num_idx](m, ctx)
        else:
            raise Error("unknown mode: " + mode)
    m.dump_report()
