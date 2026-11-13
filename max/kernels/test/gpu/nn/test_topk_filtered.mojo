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
"""Tests the filtered top-k select (`nn.topk_filtered`) against a host oracle.

A row the filtered launch selects must hold exactly the reference set:
descending score, then ascending column among the columns tied at the K-th
score. A row it flags goes to the full select, whose contract lets tied
columns differ, so that row must only hold valid top-K scores. Each scenario
also requires a number of rows to take the filtered launch, so a select that
flags every row cannot pass. Then each planted fault (`FILTER_POISONS`) must
make some scenario fail.
"""

from std.builtin.sort import sort
from std.collections import Set
from std.testing import assert_equal, assert_true

from max.gpu.host import DeviceContext
from nn.topk_bitonic import RowBoundsPtr
from nn.topk_filtered import (
    FILTER_POISONS,
    FILTER_SAMPLES,
    filtered_topk_block_split,
    filtered_topk_launch,
)

comptime K = 2048


def _lcg(n: Int, sd: UInt32, scale: Float32) -> List[Float32]:
    var state = sd
    var out = List[Float32](capacity=n)
    for _ in range(n):
        state = UInt32(1664525) * state + UInt32(1013904223)
        out.append(Float32(Int32(state)) / Float32(2**31) * scale)
    return out^


def _as_read[dtype: DType](scores: List[Float32]) -> List[Float32]:
    """The scores as the kernel sees them, rounded to `dtype`."""
    var out = List[Float32](capacity=len(scores))
    for v in scores:
        out.append(Float32(v.cast[dtype]()))
    return out^


@fieldwise_init
struct Outcome(ImplicitlyCopyable):
    var bad_rows: Int
    var filtered_rows: Int


def _check_row(
    seen: List[Float32],
    got: List[Int],
    base: Int,
    count: Int,
    exact: Bool,
) -> Bool:
    """Whether `got` is the row's top-K: the reference set when `exact`, else
    any K columns with the reference's scores. Slots past a short row's
    columns must be -1."""
    var live = min(K, count)
    for k in range(live, K):
        if got[k] != -1:
            return False
    if live == 0:
        return True
    var row = List[Float32](capacity=count)
    for c in range(count):
        row.append(seen[base + c])
    sort(row)
    var t = row[count - live]
    var above = 0
    for c in range(count):
        above += Int(seen[base + c] > t)
    var ties_needed = live - above
    var picked = Set[Int]()
    var got_above = 0
    for k in range(live):
        var c = got[k]
        if c < 0 or c >= count or c in picked:
            return False
        picked.add(c)
        var s = seen[base + c]
        if s > t:
            got_above += 1
        elif s < t:
            return False
    if got_above != above:
        return False
    if exact:
        # The tied columns taken must be the lowest ones.
        var taken = 0
        for c in range(count):
            if taken == ties_needed:
                break
            if seen[base + c] == t:
                if c not in picked:
                    return False
                taken += 1
    return True


def _run[
    dtype: DType, poison: Int = 0
](
    ctx: DeviceContext,
    scores: List[Float32],
    N: Int,
    bounds: List[Int],
    split: Bool = False,
) raises -> Outcome:
    """Runs one launch over `len(bounds)` rows and checks every row.

    With `split`, the call goes through `filtered_topk_block_split`, whose
    route the shape decides, and every row is held to the score-valid check.
    """
    var B = len(bounds)
    var scores_dev = ctx.enqueue_create_buffer[dtype](B * N)
    # A spare row past the last: a planted fault that writes one key too many
    # spills into it, not past the buffer.
    var idxs_dev = ctx.enqueue_create_buffer[.int32]((B + 1) * K)
    idxs_dev.enqueue_fill(Int32(-2))
    var bounds_dev = ctx.enqueue_create_buffer[.int32](B)
    var flags_dev = ctx.enqueue_create_buffer[.uint8](B)
    flags_dev.enqueue_fill(UInt8(1))
    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = scores[i].cast[dtype]()
    with bounds_dev.map_to_host() as buf:
        for b in range(B):
            buf[b] = Int32(bounds[b])
    var rb = RowBoundsPtr(
        rebind[ImmPointer[Int32, ImmutAnyOrigin]](bounds_dev.unsafe_ptr())
    )
    var in_ptr = rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
        scores_dev.unsafe_ptr()
    )
    var out_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr())
    if split:
        filtered_topk_block_split(ctx, in_ptr, out_ptr, N, K, B, rb)
    else:
        filtered_topk_launch[poison=poison](
            ctx,
            in_ptr,
            out_ptr,
            N,
            K,
            B,
            rb,
            rebind[MutPointer[UInt8, MutAnyOrigin]](flags_dev.unsafe_ptr()),
        )
    var idxs = ctx.enqueue_create_host_buffer[.int32]((B + 1) * K)
    var flags = ctx.enqueue_create_host_buffer[.uint8](B)
    ctx.enqueue_copy(dst_buf=idxs, src_buf=idxs_dev)
    ctx.enqueue_copy(dst_buf=flags, src_buf=flags_dev)
    ctx.synchronize()

    var seen = _as_read[dtype](scores)
    var bad = 0
    var filtered = 0
    for b in range(B):
        var exact = (not split) and flags[b] == 0
        filtered += Int(exact)
        var got = List[Int](capacity=K)
        for k in range(K):
            got.append(Int(idxs[b * K + k]))
        if not _check_row(seen, got, b * N, min(N, bounds[b]), exact):
            bad += 1
    comptime if poison == 0:
        for k in range(K):
            assert_equal(idxs[B * K + k], Int32(-2), "write past the last row")
    _ = scores_dev
    _ = bounds_dev
    return Outcome(bad, filtered)


def _bounds(B: Int, lo: Int, hi: Int) -> List[Int]:
    var out = List[Int](capacity=B)
    for b in range(B):
        out.append(lo + (b * (hi - lo)) // (B - 1))
    return out^


def _pad(mut scores: List[Float32], N: Int, bounds: List[Int]):
    """Fills each row past its bound with a score above every live one, so a
    select that reads past a bound picks it and fails."""
    for b in range(len(bounds)):
        for i in range(bounds[b], N):
            scores[b * N + i] = 1.0e30


struct Scenarios(Movable):
    """Hy4's prefill rows on a 32K history: `N` 40961 and causal bounds from
    32769, 160 rows to fill a B200, each scenario one way the bound can go.

    - `lcg`: the ordinary case, plus a short row (8191), a row below
      `FILTER_MIN_RATIO * K` (4000), a row below `K` (1500) and a zero row,
      which the full select takes in the same launch.
    - `fewties`: 4096 score levels, so a few tied columns at the threshold
      take the tie list.
    - `levels`: 64 levels, so hundreds tie and the column rounds pick.
    - `sampled`: large scores at 1500 sampled columns only, so the bound sits
      above all but 1500 columns and the row falls back.
    - `plateau`: 1000 columns above a plateau holding the rest of the row, so
      the strict pass and the plateau's tie scan run.
    """

    comptime N = 40961
    comptime B = 160
    var bounds: List[Int]
    var lcg: List[Float32]
    var fewties: List[Float32]
    var levels: List[Float32]
    var sampled: List[Float32]
    var plateau: List[Float32]

    def __init__(out self):
        comptime N = Self.N
        comptime B = Self.B
        self.bounds = _bounds(B, 32769, N)
        self.bounds[5] = 8191
        self.bounds[6] = 4000
        self.bounds[7] = 1500
        self.bounds[8] = 0
        self.lcg = _lcg(B * N, 0xF117E400, 100.0)
        self.fewties = _lcg(B * N, 0xF117E404, 1.0)
        self.levels = _lcg(B * N, 0xF117E401, 1.0)
        for i in range(B * N):
            self.fewties[i] = Float32(Int(abs(self.fewties[i]) * 4096.0))
            self.levels[i] = Float32(Int(abs(self.levels[i]) * 64.0)) * 1.5 - 40
        self.sampled = _lcg(B * N, 0xF117E402, 1.0)
        self.plateau = List[Float32](length=B * N, fill=Float32(3.0))
        var noise = _lcg(B * N, 0xF117E403, 1.0)
        for b in range(B):
            var count = self.bounds[b]
            for s in range(1500):
                self.sampled[
                    b * N + (s * count) // FILTER_SAMPLES
                ] = 50.0 + Float32(s)
            for c in range(1000):
                self.plateau[b * N + 17 + c * 23] = 10.0 + noise[b * N + c]
        _pad(self.lcg, N, self.bounds)
        _pad(self.fewties, N, self.bounds)
        _pad(self.levels, N, self.bounds)
        _pad(self.sampled, N, self.bounds)
        _pad(self.plateau, N, self.bounds)


def _expect[
    dtype: DType
](
    ctx: DeviceContext,
    scores: List[Float32],
    bounds: List[Int],
    N: Int,
    min_filtered: Int,
    label: String,
) raises:
    var o = _run[dtype](ctx, scores, N, bounds)
    print(label, dtype, "bad", o.bad_rows, "filtered", o.filtered_rows)
    assert_equal(o.bad_rows, 0, label)
    assert_true(o.filtered_rows >= min_filtered, String(label, " filtered"))


def test_scenarios[dtype: DType](ctx: DeviceContext, s: Scenarios) raises:
    comptime N = Scenarios.N
    # Rows 5-8 cannot take the filtered select; the rest must, but where the
    # bound is meant to fail (`sampled`) or the select is checked on its
    # fallback alone.
    _expect[dtype](ctx, s.lcg, s.bounds, N, Scenarios.B - 4, "lcg")
    _expect[dtype](ctx, s.fewties, s.bounds, N, Scenarios.B - 4, "fewties")
    _expect[dtype](ctx, s.levels, s.bounds, N, Scenarios.B - 4, "levels")
    _expect[dtype](ctx, s.sampled, s.bounds, N, 0, "sampled")
    _expect[dtype](ctx, s.plateau, s.bounds, N, Scenarios.B - 4, "plateau")


def test_edges[dtype: DType](ctx: DeviceContext) raises:
    """A prime stride, bounds at the filtered select's edges (exactly
    `FILTER_MIN_RATIO * K`, one below and one above), every row equal, and
    negative scores."""
    comptime N = 12289
    var bounds = _bounds(149, 8193, N)
    bounds[0] = 8192
    bounds[1] = 8191
    bounds[2] = 8193
    bounds[3] = 1
    bounds[4] = K
    var scores = _lcg(149 * N, 0xED6E0001, 7.0)
    for i in range(149 * N):
        scores[i] = scores[i] - 20.0
    for i in range(N):
        scores[10 * N + i] = -5.0
    _pad(scores, N, bounds)
    _expect[dtype](ctx, scores, bounds, N, 140, "edges")


def test_largest_and_empty(ctx: DeviceContext) raises:
    """The largest row the server sends (a 4096-token chunk on a 139,675-token
    history, the P90 context) on enough rows to fill a B200, a lone row, and
    an empty launch, which must do nothing."""
    comptime N = 139675 + 4096
    var bounds = _bounds(148, 139676, N)
    var scores = _lcg(148 * N, 0x1A56E001, 4.0)
    _pad(scores, N, bounds)
    _expect[DType.bfloat16](ctx, scores, bounds, N, 148, "largest")
    var one: List[Int] = [N]
    var o = _run[DType.bfloat16](ctx, scores, N, one, split=True)
    assert_equal(o.bad_rows, 0, "one row")
    var idxs = ctx.enqueue_create_buffer[DType.int32](1)
    var dummy = ctx.enqueue_create_buffer[DType.bfloat16](8)
    var rows = ctx.enqueue_create_buffer[DType.int32](1)
    idxs.enqueue_fill(Int32(-7))
    filtered_topk_block_split(
        ctx,
        rebind[ImmPointer[BFloat16, ImmutAnyOrigin]](dummy.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs.unsafe_ptr()),
        N,
        K,
        0,
        RowBoundsPtr(
            rebind[ImmPointer[Int32, ImmutAnyOrigin]](rows.unsafe_ptr())
        ),
    )
    with idxs.map_to_host() as h:
        assert_equal(h[0], Int32(-7), "an empty launch wrote")


def test_split_route(ctx: DeviceContext, s: Scenarios) raises:
    """The launcher's routes: 160 rows take the filtered launches, 8 rows the
    shared select (they under-fill the GPU), and a stride of `4 * K` too."""
    var o = _run[DType.bfloat16](ctx, s.lcg, Scenarios.N, s.bounds, split=True)
    assert_equal(o.bad_rows, 0, "split route, 160 rows")
    var few = List[Int]()
    for b in range(8):
        few.append(s.bounds[b * 20])
    o = _run[DType.bfloat16](ctx, s.lcg, Scenarios.N, few, split=True)
    assert_equal(o.bad_rows, 0, "split route, 8 rows")
    comptime N2 = 4 * K
    var narrow = _lcg(160 * N2, 0x5911E001, 3.0)
    o = _run[DType.bfloat16](
        ctx, narrow, N2, _bounds(160, 2049, N2), split=True
    )
    assert_equal(o.bad_rows, 0, "split route, stride 4K")


def _poisoned[p: Int](ctx: DeviceContext, s: Scenarios) raises -> Int:
    comptime N = Scenarios.N
    comptime BF = DType.bfloat16
    var bad = 0
    bad += _run[BF, p](ctx, s.lcg, N, s.bounds).bad_rows
    bad += _run[BF, p](ctx, s.fewties, N, s.bounds).bad_rows
    bad += _run[BF, p](ctx, s.levels, N, s.bounds).bad_rows
    bad += _run[BF, p](ctx, s.plateau, N, s.bounds).bad_rows
    print("poison", p, "bad rows", bad)
    return bad


def test_poisons(ctx: DeviceContext, s: Scenarios) raises:
    comptime assert FILTER_POISONS == 4
    assert_true(_poisoned[1](ctx, s) > 0, "poison 1")
    assert_true(_poisoned[2](ctx, s) > 0, "poison 2")
    assert_true(_poisoned[3](ctx, s) > 0, "poison 3")
    assert_true(_poisoned[4](ctx, s) > 0, "poison 4")


def main() raises:
    with DeviceContext() as ctx:
        var s = Scenarios()
        test_scenarios[DType.bfloat16](ctx, s)
        test_scenarios[DType.float32](ctx, s)
        test_edges[DType.bfloat16](ctx)
        test_edges[DType.float32](ctx)
        test_split_route(ctx, s)
        test_largest_and_empty(ctx)
        test_poisons(ctx, s)
        print("PASS test_topk_filtered")
