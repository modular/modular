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
"""The deterministic cluster top-k against the ranked select it replaces.

`topk_cluster_det` promises the ranked streaming select's output slot for
slot: the same columns in the same slots, ties at the threshold included.
That select is the oracle here, launched directly with the parameters the
dispatch gave it for few rows over a long stride, so a wrong tie cut, a
misplaced column or an unwritten slot shows as a differing slot, and is
checked at every cluster size the kernel takes:

- Strides: Mistral Large 4's capture-frozen 262144 plus a decode query, and
  two shorter ones past the resident select's widest row.
- Rows: a decode, a speculative verify, a batch, and the most rows a cluster
  of two keeps in one wave on B200.
- Bounds per row: none, one, a partial 16 B group, `K - 1`, `K`, `K + 1`,
  just past one group of the slice split, mid-length, the stride itself,
  then random.
- Scores: Gaussian; 17 levels, so the threshold is a wide plateau; all
  equal, so the column alone decides; signed zeros, which are distinct keys;
  all negative; a few infinities; uniform.

A smaller `K` repeats the comparison on a subset. Then two inputs launched
1000 times each must give one result, and the dispatch's route is pinned on
the predicate it decides by: the cluster kernel up to a 32-row batch, the
ranked select at 128 rows, both writing the oracle's output.
"""

from std.testing import assert_equal, assert_true

from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from structured_kernels.trace_buf import NullTrace

from nn.attention.gpu.nvidia.topk_cluster_det import (
    topk_cluster_det,
    topk_cluster_det_cluster_size,
)
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    RowBoundsPtr,
    _histsel_topk_bounded_kernel,
    _HSEL_RANK_BITS,
    _HSEL_SEL_CAP,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _PTOPK_BLOCK,
    persistent_topk_block_split,
    topk_det_cluster_route,
)

comptime K = PERSISTENT_TOPK_MAX_N
comptime FROZEN_N = 262144 + 8


def _dists() -> List[String]:
    return [
        "normal",
        "q17",
        "equal",
        "zeros",
        "negative",
        "inf",
        "uniform",
    ]


@inline(.always)
def _mix(x: UInt64) -> UInt64:
    """SplitMix64's finalizer: a counter-keyed generator, so a case's input
    is a function of its key alone and costs one hash per draw."""
    var z = (x + 1) * 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@inline(.always)
def _u01(x: UInt64) -> Float64:
    return Float64(_mix(x) >> 11) * (1.0 / 9007199254740992.0)


def _score(dist: String, key: UInt64) -> Float32:
    var k = key << 3
    if dist == "q17":
        return Float32(Int(_u01(k) * 17.0)) / 16.0
    if dist == "equal":
        return 0.5
    if dist == "zeros":
        var u = _mix(k) % 16
        if u == 0:
            return 1.0
        return Float32(0.0) if u % 2 == 0 else Float32(-0.0)
    if dist == "uniform":
        return Float32(_u01(k) * 2.0 - 1.0)
    var acc = Float64(0.0)
    for t in range(6):
        acc += _u01(k + UInt64(t))
    var v = Float32((acc - 3.0) * 1.4142135623730951)
    if dist == "negative":
        return -abs(v) - 1.0
    if dist == "inf":
        var u = _mix(k + 7) % 1000
        if u == 0:
            return Float32.MAX * 2.0
        if u == 1:
            return -Float32.MAX * 2.0
    return v


def _bound(r: Int, N: Int, k: Int, salt: UInt64) -> Int:
    var edges: List[Int] = [
        0,
        1,
        5,
        k - 1,
        k,
        k + 1,
        k + 9,
        3 * k + 7,
        N // 2 + 3,
        N - 1,
        N,
    ]
    if r < len(edges):
        return edges[r]
    return Int(_mix(salt * 1000003 + UInt64(r)) % UInt64(N + 1))


def _ranked(
    ctx: DeviceContext,
    scores: ImmPointer[BFloat16, ImmutAnyOrigin],
    dst: MutPointer[Int32, MutAnyOrigin],
    N: Int,
    k: Int,
    rows: Int,
    bounds: RowBoundsPtr,
) raises:
    ctx.enqueue_function[
        _histsel_topk_bounded_kernel[
            NullTrace,
            RB=RowBoundsPtr,
            prefetch=True,
            tail_bits=_HSEL_TAIL_BITS,
            rank_bits=_HSEL_RANK_BITS,
            rank_slots=True,
            sel_cap=_HSEL_SEL_CAP,
            in_dtype=.bfloat16,
        ]
    ](
        scores,
        dst,
        Int32(N),
        Int32(k),
        bounds,
        NullTrace(),
        grid_dim=rows,
        block_dim=_PTOPK_BLOCK,
        shared_mem_bytes=_HSEL_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(_HSEL_SMEM_BYTES)
        ),
    )


struct _Case:
    """One input on the device and the oracle's output for it."""

    var rows: Int
    var N: Int
    var k: Int
    var scores: DeviceBuffer[DType.bfloat16]
    var bounds_buf: DeviceBuffer[DType.int32]
    var out: DeviceBuffer[DType.int32]
    var expect: List[Int32]

    def __init__(
        out self,
        ctx: DeviceContext,
        rows: Int,
        N: Int,
        dist: String,
        salt: UInt64,
        k: Int = K,
    ) raises:
        self.rows = rows
        self.N = N
        self.k = k
        self.scores = ctx.enqueue_create_buffer[.bfloat16](rows * N)
        self.bounds_buf = ctx.enqueue_create_buffer[.int32](rows)
        self.out = ctx.enqueue_create_buffer[.int32](rows * k)
        self.expect = List[Int32](capacity=rows * k)
        # Columns past a row's bound are never selected, so only its live
        # prefix is written; the kernels may read the rest, never use it.
        with self.scores.map_to_host() as h:
            with self.bounds_buf.map_to_host() as b:
                for r in range(rows):
                    var bound = _bound(r, N, k, salt)
                    b[r] = Int32(bound)
                    var base = ((salt << 24) + UInt64(r)) << 20
                    for c in range(bound):
                        h[r * N + c] = _score(dist, base + UInt64(c)).cast[
                            .bfloat16
                        ]()
        var ref_buf = ctx.enqueue_create_buffer[.int32](rows * k)
        _ranked(ctx, self.ptr(), _mut_ptr(ref_buf), N, k, rows, self.rb())
        with ref_buf.map_to_host() as h:
            for i in range(rows * k):
                self.expect.append(h[i])

    def ptr(self) -> ImmPointer[BFloat16, ImmutAnyOrigin]:
        return rebind[ImmPointer[BFloat16, ImmutAnyOrigin]](
            self.scores.unsafe_ptr()
        )

    def rb(self) -> RowBoundsPtr:
        return RowBoundsPtr(
            rebind[ImmPointer[Int32, ImmutAnyOrigin]](
                self.bounds_buf.unsafe_ptr()
            )
        )

    def differing(self) raises -> Int:
        """Slots of `self.out` that differ from the oracle's."""
        var n = 0
        with self.out.map_to_host() as h:
            for i in range(self.rows * self.k):
                if h[i] != self.expect[i]:
                    n += 1
        return n


def _mut_ptr(buf: DeviceBuffer[DType.int32]) -> MutPointer[Int32, MutAnyOrigin]:
    return rebind[MutPointer[Int32, MutAnyOrigin]](buf.unsafe_ptr())


def _check_all_sizes(ctx: DeviceContext, tc: _Case, dist: String) raises -> Int:
    """Runs every cluster size on `tc` against its oracle; returns the slots
    compared."""
    for cs in [16, 8, 4, 2, 1]:
        tc.out.enqueue_fill(Int32(-7))
        topk_cluster_det(
            ctx,
            rebind[ImmPointer[BFloat16, ImmUntrackedOrigin]](tc.ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](_mut_ptr(tc.out)),
            tc.N,
            tc.k,
            tc.rows,
            tc.rb(),
            cs,
        )
        var bad = tc.differing()
        assert_equal(
            bad,
            0,
            String(
                t"N={tc.N} rows={tc.rows} K={tc.k} dist={dist} cs={cs}: {bad}"
                t" slots differ from the ranked select"
            ),
        )
    return 5 * tc.rows * tc.k


def test_matches_ranked(ctx: DeviceContext) raises:
    var shapes: List[Tuple[Int, Int]] = [
        (FROZEN_N, 1),
        (FROZEN_N, 4),
        (FROZEN_N, 32),
        (131080, 74),
        (32776, 11),
    ]
    var checked = 0
    for shape in shapes:
        for dist in _dists():
            var tc = _Case(
                ctx, shape[1], shape[0], dist, UInt64(shape[0] + shape[1])
            )
            checked += _check_all_sizes(ctx, tc, dist)
    # The gate admits any `K` up to 2048; at a smaller one the ranks past the
    # selection receive no keys and write only the `-1` tail.
    var small_k: List[Tuple[Int, Int]] = [
        (FROZEN_N, 1),
        (FROZEN_N, 4),
        (131080, 74),
    ]
    for shape in small_k:
        for dist in ["normal", "q17", "equal"]:
            var tc = _Case(
                ctx,
                shape[1],
                shape[0],
                String(dist),
                UInt64(shape[0] + shape[1] + 1),
                k=1024,
            )
            checked += _check_all_sizes(ctx, tc, String(dist))
    print("PASS test_matches_ranked:", checked, "slots")


def test_reproducible(ctx: DeviceContext) raises:
    """1000 launches of one input give one output."""
    var inputs: List[Tuple[Int, String]] = [(1, "normal"), (4, "q17")]
    for inp in inputs:
        var rows = inp[0]
        var tc = _Case(ctx, rows, FROZEN_N, inp[1], UInt64(7 + rows))
        var cs = topk_cluster_det_cluster_size[B200.sm_count](rows)
        var strays = 0
        for _ in range(1000):
            tc.out.enqueue_fill(Int32(-7))
            topk_cluster_det(
                ctx,
                rebind[ImmPointer[BFloat16, ImmUntrackedOrigin]](tc.ptr()),
                rebind[MutPointer[Int32, MutUntrackedOrigin]](_mut_ptr(tc.out)),
                FROZEN_N,
                K,
                rows,
                tc.rb(),
                cs,
            )
            if tc.differing() != 0:
                strays += 1
        assert_equal(
            strays,
            0,
            String(rows, " rows: launches disagreed with the oracle"),
        )
    print("PASS test_reproducible: 2 inputs x 1000 launches, 1 result each")


def test_dispatch(ctx: DeviceContext) raises:
    """The route the dispatch takes, and its output.

    `topk_det_cluster_route` is the predicate `persistent_topk_block_split`
    decides by, so asserting on it pins the route rather than re-deriving
    it: decode, verify and a 32-row batch take the cluster kernel, 128 rows
    (a 32-request speculative verify) keep the ranked select, and so does a
    launch without row bounds. Both routes must write the oracle's output.
    """
    for rows in [1, 4, 32, 128]:
        var cs = topk_det_cluster_route[
            DType.bfloat16, ordered=False, deterministic=True
        ](ctx, FROZEN_N, K, rows, True)
        if rows <= 32:
            assert_equal(
                cs,
                topk_cluster_det_cluster_size[B200.sm_count](rows),
                String(rows, " rows: cluster size"),
            )
            assert_true(cs >= 2, String(rows, " rows should take a cluster"))
        else:
            assert_equal(
                cs, 0, String(rows, " rows should keep the ranked select")
            )
        var tc = _Case(ctx, rows, FROZEN_N, "normal", UInt64(11 + rows))
        tc.out.enqueue_fill(Int32(-7))
        persistent_topk_block_split[deterministic=True](
            ctx,
            tc.ptr(),
            _mut_ptr(tc.out),
            FROZEN_N,
            K,
            rows,
            Optional(tc.rb()),
        )
        assert_equal(tc.differing(), 0, String(rows, " rows"))
    assert_equal(
        topk_det_cluster_route[
            DType.bfloat16, ordered=False, deterministic=True
        ](ctx, FROZEN_N, K, 1, False),
        0,
        "an unbounded launch should keep the ranked select",
    )
    assert_equal(
        topk_det_cluster_route[
            DType.bfloat16, ordered=False, deterministic=False
        ](ctx, FROZEN_N, K, 1, True),
        0,
        "the default contract should not take the deterministic kernel",
    )
    print("PASS test_dispatch")


def main() raises:
    with DeviceContext() as ctx:
        test_matches_ranked(ctx)
        test_reproducible(ctx)
        test_dispatch(ctx)
