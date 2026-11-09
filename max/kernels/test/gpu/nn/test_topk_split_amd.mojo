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
"""The AMD row-split top-k against the single-block select it replaces.

`persistent_topk_block_split`'s default contract (`ordered=False`,
`deterministic=False`) routes bounded rows that under-fill the GPU to
`_histsel_split_topk` on gfx950. Every case runs both that and the streaming
select the route used before (`_histsel_topk_bounded_kernel`, launched with
the launcher's own parameters) on the same input, and requires per row:

- the same slots hold `-1`, exactly `[min(K, bound), K)`;
- the same columns, sorted, element for element;
- both equal the CPU top-`K` by descending score then ascending column, which
  pins which members of a tied plateau survive.

Where the launcher is exercised and the merge is resident, a split route must
also match a direct `_histsel_split_topk` call slot for slot: that merge's slot
order is deterministic and the single-block select's is not, so this is what
shows the route was taken rather than only that the set is right. A merge too
wide for the resident select streams, and its slot order is arrival order.

Columns at or past each bound hold a score above every live one, so a kernel
that reads past its bound selects it and fails. NaN is outside the contract;
`-inf` is in it and appears as masked columns.
"""

from std.builtin.sort import sort
from std.math import cos, log, sqrt
from max.gpu.host import (
    DeviceAttribute,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    FuncAttribute,
)
from std.testing import assert_equal
from std.utils.numerics import neg_inf

from nn.topk_bitonic import (
    RowBoundsPtr,
    _HSEL_RES_MAX_WIDE,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _PTOPK_BLOCK,
    _histsel_split_topk,
    _histsel_topk_bounded_kernel,
    _hsel_phi_dtype,
    _hsel_prefetch_scan_items,
    _hsplit_plan,
    _phi,
    persistent_topk_block_split,
)
from structured_kernels.trace_buf import NullTrace


def _lcg(mut state: UInt32) -> Float32:
    state = 1664525 * state + 1013904223
    return Float32(Int32(state)) / Float32(2**31)


def _poison(mut scores: List[Float32], N: Int, bounds: List[Int]):
    for b in range(len(bounds)):
        for i in range(max(0, bounds[b]), N):
            scores[b * N + i] = Float32(1.0e30)


def _random_scores(B: Int, N: Int, sd: UInt32) -> List[Float32]:
    var state = sd
    var out = List[Float32](capacity=B * N)
    for _ in range(B * N):
        out.append(_lcg(state) * 100.0)
    return out^


def _ragged_bounds(B: Int, N: Int, K: Int, sd: UInt32) -> List[Int]:
    """Full rows, a row just past `K`, rows at or under it, an empty row, and
    the rest uniform over `(K, N]`."""
    var fixed: List[Int] = [N, K + 1, K, K - 1, 0, N - 3]
    var state = sd
    var out = List[Int](capacity=B)
    for b in range(B):
        if b < len(fixed) and B > 1:
            out.append(max(0, min(N, fixed[b])))
        elif B == 1:
            out.append(N)
        else:
            var u = (_lcg(state) + 1.0) * 0.5
            out.append(min(N, K + 1 + Int(u * Float32(N - K))))
    return out^


def _cpu_topk(row: List[Float32], bound: Int, K: Int) -> List[Int32]:
    """Top `K` of `row[:bound]` by descending score, then ascending column,
    sorted by column."""
    var keys = List[UInt64](capacity=bound)
    for c in range(bound):
        keys.append(
            (UInt64(_phi(row[c])) << 32)
            | UInt64(UInt32(0xFFFFFFFF) - UInt32(c))
        )
    sort(keys)
    var out = List[Int32]()
    for i in range(min(K, bound)):
        out.append(
            Int32(
                Int(
                    UInt32(0xFFFFFFFF)
                    - UInt32(keys[bound - 1 - i] & 0xFFFFFFFF)
                )
            )
        )
    sort(out)
    return out^


def _check_row(
    got: List[Int32], b: Int, K: Int, live: Int, label: String
) raises -> List[Int32]:
    var cols = List[Int32]()
    for k in range(K):
        var c = got[b * K + k]
        if (c == -1) != (k >= live):
            raise Error(label, " row ", b, " slot ", k, ": ", c)
        if c != -1:
            cols.append(c)
    sort(cols)
    return cols^


def _run_old[
    in_dtype: DType
](
    ctx: DeviceContext,
    scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    dst: MutPointer[Int32, MutAnyOrigin],
    N: Int,
    K: Int,
    B: Int,
    bounds: RowBoundsPtr,
) raises:
    """The select this shape took before the split, with its launch."""
    comptime phi = _hsel_phi_dtype[in_dtype]()
    ctx.enqueue_function[
        _histsel_topk_bounded_kernel[
            NullTrace,
            RB=RowBoundsPtr,
            prefetch=True,
            scan_items=_hsel_prefetch_scan_items[in_dtype, phi](),
            tail_bits=_HSEL_TAIL_BITS,
            ordered=False,
            deterministic=False,
            in_dtype=in_dtype,
        ]
    ](
        scores,
        dst,
        Int32(N),
        Int32(K),
        bounds,
        NullTrace(),
        grid_dim=B,
        block_dim=_PTOPK_BLOCK,
        shared_mem_bytes=_HSEL_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(_HSEL_SMEM_BYTES)
        ),
    )


def _case[
    in_dtype: DType = .float32
](
    ctx: DeviceContext,
    var scores: List[Float32],
    N: Int,
    K: Int,
    bounds: List[Int],
    label: String,
    slices: Int = 0,
    slice_len: Int = 0,
    graph: Bool = False,
    split: Bool = True,
) raises:
    """Old select vs the split on one input, both against the CPU top-`K`.

    `slices == 0` takes the public launcher, which must route to the split
    exactly when `split` says (the plan, and for a split the slot-for-slot
    match against a direct call); otherwise the split runs with the given
    geometry, so shapes the planner would not choose are covered too. `graph`
    records that launch in a device graph and replays it twice.
    """
    var B = len(bounds)
    _poison(scores, N, bounds)
    var scores_dev = ctx.enqueue_create_buffer[in_dtype](B * N)
    var bounds_dev = ctx.enqueue_create_buffer[.int32](B)
    var old_dev = ctx.enqueue_create_buffer[.int32](B * K)
    var new_dev = ctx.enqueue_create_buffer[.int32](B * K)
    with scores_dev.map_to_host() as h:
        for i in range(B * N):
            h[i] = scores[i].cast[in_dtype]()
    with bounds_dev.map_to_host() as h:
        for b in range(B):
            h[b] = Int32(bounds[b])
    old_dev.enqueue_fill(Int32(-2))
    new_dev.enqueue_fill(Int32(-2))

    var sp = rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
        scores_dev.unsafe_ptr()
    )
    var rb = RowBoundsPtr(
        rebind[ImmPointer[Int32, ImmutAnyOrigin]](bounds_dev.unsafe_ptr())
    )
    var old_p = rebind[MutPointer[Int32, MutAnyOrigin]](old_dev.unsafe_ptr())
    var new_p = rebind[MutPointer[Int32, MutAnyOrigin]](new_dev.unsafe_ptr())
    _run_old(ctx, sp, old_p, N, K, B, rb)

    var ns = slices
    var sl = slice_len
    if ns == 0:
        var cu = ctx.device().get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        var plan = _hsplit_plan(B, N, K, cu)
        ns = plan[0]
        sl = plan[1]
        assert_equal(ns >= 2, split, String(label, ": planner route"))
    var n_cand = max(1, B * ns * K)
    var cols_dev = ctx.enqueue_create_buffer[.int32](n_cand)
    var cand_dev = ctx.enqueue_create_buffer[in_dtype](n_cand)
    var cols_p = rebind[MutPointer[Int32, MutAnyOrigin]](cols_dev.unsafe_ptr())
    var cand_p = rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](
        cand_dev.unsafe_ptr()
    )
    var replays = 1
    if graph:
        replays = 2

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            _histsel_split_topk(
                builder.recording_context(),
                sp,
                new_p,
                cols_p,
                cand_p,
                N,
                K,
                B,
                rb,
                ns,
                sl,
            )

        var g = DeviceGraph.create(ctx, build)
        g.replay()
        g.replay()
        ctx.synchronize()
        _ = g
    elif slices == 0:
        persistent_topk_block_split(
            ctx, sp, new_p, N, K, B, row_bounds=Optional(rb)
        )
        if split and ns * K <= _HSEL_RES_MAX_WIDE:
            var direct_dev = ctx.enqueue_create_buffer[.int32](B * K)
            _histsel_split_topk(
                ctx,
                sp,
                rebind[MutPointer[Int32, MutAnyOrigin]](
                    direct_dev.unsafe_ptr()
                ),
                cols_p,
                cand_p,
                N,
                K,
                B,
                rb,
                ns,
                sl,
            )
            with direct_dev.map_to_host() as d:
                with new_dev.map_to_host() as h:
                    for i in range(B * K):
                        if d[i] != h[i]:
                            raise Error(
                                label,
                                ": launcher slot ",
                                i,
                                " is ",
                                h[i],
                                ", the split's is ",
                                d[i],
                            )
    else:
        _histsel_split_topk(ctx, sp, new_p, cols_p, cand_p, N, K, B, rb, ns, sl)
    ctx.synchronize()

    var old_h = List[Int32](length=B * K, fill=0)
    var new_h = List[Int32](length=B * K, fill=0)
    with old_dev.map_to_host() as h:
        for i in range(B * K):
            old_h[i] = h[i]
    with new_dev.map_to_host() as h:
        for i in range(B * K):
            new_h[i] = h[i]

    for b in range(B):
        var bound = max(0, min(N, bounds[b]))
        var live = min(K, bound)
        var row = List[Float32](capacity=bound)
        for c in range(bound):
            row.append(Float32(scores[b * N + c].cast[in_dtype]()))
        var ref_cols = _cpu_topk(row, bound, K)
        var old_cols = _check_row(old_h, b, K, live, String(label, " old"))
        var new_cols = _check_row(new_h, b, K, live, String(label, " new"))
        assert_equal(len(old_cols), live)
        assert_equal(len(new_cols), live)
        for i in range(live):
            if old_cols[i] != ref_cols[i] or new_cols[i] != old_cols[i]:
                raise Error(
                    label,
                    " row ",
                    b,
                    " sorted rank ",
                    i,
                    ": cpu ",
                    ref_cols[i],
                    " old ",
                    old_cols[i],
                    " new ",
                    new_cols[i],
                )
    print(
        t"PASS {label} N={N} K={K} rows={B} slices={ns} slice_len={sl}"
        t" replays={replays}"
    )
    _ = scores_dev
    _ = bounds_dev
    _ = cols_dev
    _ = cand_dev


def test_plan() raises:
    """The route is decided by host-known shape alone."""
    assert_equal(_hsplit_plan(4, 16384, 2048, 256)[0], 0)
    assert_equal(_hsplit_plan(0, 84000, 2048, 256)[0], 0)
    assert_equal(_hsplit_plan(4, 84000, 0, 256)[0], 0)
    var p = _hsplit_plan(4, 84000, 2048, 256)
    assert_equal(p[0], 7)
    assert_equal(p[1], 13116)
    # Exactly the resident floor: eight slices of 16384, merged resident.
    assert_equal(_hsplit_plan(4, 131072, 2048, 256)[0], 8)
    # The resident floor (13 slices at 200k) does not fit 20 rows' share.
    assert_equal(_hsplit_plan(20, 200000, 2048, 256)[0], 0)
    assert_equal(_hsplit_plan(16, 200000, 2048, 256)[0], 13)


def test_random_shapes(ctx: DeviceContext) raises:
    comptime K = 2048
    for N in [16400, 84000, 131072, 200000]:
        for B in [1, 4, 16]:
            var sd = UInt32(N * 31 + B)
            _case(
                ctx,
                _random_scores(B, N, sd),
                N,
                K,
                _ragged_bounds(B, N, K, sd),
                "random",
            )
    # Twenty rows cannot fit 200k's thirteen resident slices each on one
    # wave, so the launcher keeps the single-block select.
    _case(
        ctx,
        _random_scores(20, 200000, 13),
        200000,
        K,
        _ragged_bounds(20, 200000, K, 13),
        "not-split",
        split=False,
    )
    # A stride that is not a 16 B multiple takes the scalar loads.
    _case(
        ctx,
        _random_scores(4, 84001, 7),
        84001,
        K,
        _ragged_bounds(4, 84001, K, 7),
        "odd-stride",
    )
    for B in [2, 8]:
        _case(
            ctx,
            _random_scores(B, 84000, UInt32(B)),
            84000,
            K,
            _ragged_bounds(B, 84000, K, UInt32(B)),
            "rows",
        )


def test_other_k(ctx: DeviceContext) raises:
    for K in [512, 1000, 1]:
        _case(
            ctx,
            _random_scores(4, 84000, UInt32(K)),
            84000,
            K,
            _ragged_bounds(4, 84000, max(K, 2), UInt32(K)),
            "k",
        )


def test_ties(ctx: DeviceContext) raises:
    """Plateaus at the threshold, where which tied columns survive matters."""
    comptime K = 2048
    comptime N = 84000
    comptime B = 4
    var bounds: List[Int] = [N, N - 1, 50000, 30001]
    # Every score equal: the lowest K columns, all from the first slice.
    var flat = List[Float32](length=B * N, fill=1.0)
    _case(ctx, flat^, N, K, bounds, "constant")
    # 17 levels, as fp8-quantized scores collapse to.
    var state = UInt32(17)
    var q = List[Float32](capacity=B * N)
    for _ in range(B * N):
        q.append(Float32(Int((_lcg(state) + 1.0) * 8.5)))
    _case(ctx, q^, N, K, bounds, "q17")
    # 1500 distinct leaders and a 5000-column plateau spread over every slice,
    # so the cut runs through the plateau in every slice.
    var p = List[Float32](length=B * N, fill=-5.0)
    for b in range(B):
        for i in range(1500):
            p[b * N + (i * 37) % bounds[b]] = Float32(10 + i)
        for i in range(5000):
            var c = (i * 16 + 3) % bounds[b]
            if p[b * N + c] < 10.0:
                p[b * N + c] = 7.0
    _case(ctx, p^, N, K, bounds, "plateau-straddles-k")
    # Masked rows: few finite scores, the rest `-inf`, so `-inf` columns tie
    # at the threshold; and +0 against -0, which order apart.
    var m = List[Float32](length=B * N, fill=neg_inf[DType.float32]())
    for b in range(B):
        for i in range(700 * (b + 1)):
            m[b * N + (i * 101) % bounds[b]] = Float32(i % 9) - 4.0
        m[b * N + 11] = -0.0
        m[b * N + 12] = 0.0
    _case(ctx, m^, N, K, bounds, "masked-inf")
    # Ascending and descending rows put all K in the last or first slice.
    var asc = List[Float32](capacity=B * N)
    for b in range(B):
        for i in range(N):
            asc.append(Float32(i) if b % 2 == 0 else Float32(N - i))
    _case(ctx, asc^, N, K, bounds, "sorted")


def _shaped_scores(
    B: Int, N: Int, kind: String, spread: Float64, sd: UInt32
) -> List[Float32]:
    """Scores clustered the way indexer rows are, which is what decides how
    the select's histograms concentrate: `gauss` is `1 + spread * z`,
    `neg-gauss` its mirror below zero, `narrow` a sliver above one."""
    var state = sd
    var out = List[Float32](capacity=B * N)
    for _ in range(B * N):
        var u = (Float64(_lcg(state)) + 1.0) * 0.5
        if kind == "narrow":
            out.append(Float32(1.0 + u / 32.0))
        elif kind == "uniform":
            out.append(Float32(u * 200.0 - 100.0))
        else:
            var u2 = (Float64(_lcg(state)) + 1.0) * 0.5
            var z = sqrt(-2.0 * log(max(u, 1.0e-12))) * cos(
                6.283185307179586 * u2
            )
            var v = 1.0 + spread * z
            out.append(Float32(-v if kind == "neg-gauss" else v))
    return out^


def test_distributions(ctx: DeviceContext) raises:
    """Score distributions that move the round-0 window: clustered, mirrored
    below zero, one extreme outlier per row, and the top `K` confined to a few
    warps so the K-th key lands in a bin clamped to the window's edge. Every
    row set also carries rows under `K` and an empty row."""
    comptime K = 2048
    comptime N = 84000
    comptime B = 8
    var bounds = _ragged_bounds(B, N, K, 41)
    var kinds: List[String] = [
        "uniform",
        "gauss",
        "gauss",
        "gauss",
        "narrow",
        "neg-gauss",
    ]
    var spreads: List[Float64] = [0.0, 0.25, 0.1, 0.03, 0.0, 0.1]
    for i in range(len(kinds)):
        _case(
            ctx,
            _shaped_scores(B, N, kinds[i], spreads[i], UInt32(100 + i)),
            N,
            K,
            bounds,
            String(kinds[i], " ", spreads[i]),
        )

    # One huge score per row, and in odd rows one hugely negative one too:
    # either alone stretches a min/max window over the whole key range.
    var o = _shaped_scores(B, N, "narrow", 0.0, 7)
    for b in range(B):
        var live = max(1, bounds[b])
        o[b * N + (b * 7919 + 13) % live] = Float32(1.0e30)
        if b % 2 == 1:
            o[b * N + (b * 104729 + 5) % live] = Float32(-1.0e30)
    _case(ctx, o^, N, K, bounds, "outlier")
    var og = _shaped_scores(B, N, "gauss", 0.03, 8)
    for b in range(B):
        og[b * N + (b * 31 + 1) % max(1, bounds[b])] = Float32(3.0e4)
    _case(ctx, og^, N, K, bounds, "gauss-outlier")

    # 3000 columns from 1e3 to 1e6 near the row start, over a tight cluster:
    # only the warps holding them see the tail, so the window covers the
    # cluster and the K-th key sits in the clamped top bin.
    var h = _shaped_scores(B, N, "gauss", 0.01, 9)
    var state = UInt32(99)
    for b in range(B):
        for i in range(min(3000, max(0, bounds[b] - 100))):
            var u = (Float64(_lcg(state)) + 1.0) * 0.5
            h[b * N + 100 + i] = Float32(10.0 ** (3.0 + 3.0 * u))
    _case(ctx, h^, N, K, bounds, "tail-in-few-warps")

    # Ties at the threshold inside a clustered row: 4 levels around one.
    var t = _shaped_scores(B, N, "gauss", 0.1, 10)
    for i in range(B * N):
        if i % 3 == 0:
            t[i] = Float32(1.0 + 0.0625 * Float64(i % 4))
    _case(ctx, t^, N, K, bounds, "clustered-ties")


def test_forced_geometry(ctx: DeviceContext) raises:
    """Geometries the planner does not choose: a streaming merge (more than
    the resident width of candidates), one slice, and rows past the CU count.
    """
    comptime K = 2048
    _case(
        ctx,
        _random_scores(4, 84000, 3),
        84000,
        K,
        _ragged_bounds(4, 84000, K, 3),
        "streaming-merge",
        slices=12,
        slice_len=7000,
    )
    # Rows shorter than `slices` aligned steps leave trailing slices empty.
    _case(
        ctx,
        _random_scores(6, 4096, 11),
        4096,
        64,
        _ragged_bounds(6, 4096, 64, 11),
        "empty-slices",
        slices=64,
        slice_len=1,
    )
    _case(
        ctx,
        _random_scores(3, 16000, 5),
        16000,
        K,
        _ragged_bounds(3, 16000, K, 5),
        "one-slice",
        slices=1,
        slice_len=16000,
    )
    var B = (
        ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT) + 13
    )
    _case(
        ctx,
        _random_scores(B, 20000, 9),
        20000,
        K,
        _ragged_bounds(B, 20000, K, 9),
        "rows-past-cus",
        slices=2,
        slice_len=10000,
    )


def test_bf16(ctx: DeviceContext) raises:
    _case[.bfloat16](
        ctx,
        _random_scores(4, 84000, 21),
        84000,
        2048,
        _ragged_bounds(4, 84000, 2048, 21),
        "bf16",
    )


def test_graph_replay(ctx: DeviceContext) raises:
    var p = _hsplit_plan(4, 84000, 2048, 256)
    _case(
        ctx,
        _random_scores(4, 84000, 77),
        84000,
        2048,
        _ragged_bounds(4, 84000, 2048, 77),
        "graph",
        slices=p[0],
        slice_len=p[1],
        graph=True,
    )


def main() raises:
    test_plan()
    with DeviceContext() as ctx:
        test_random_shapes(ctx)
        test_other_k(ctx)
        test_ties(ctx)
        test_distributions(ctx)
        test_forced_geometry(ctx)
        test_bf16(ctx)
        test_graph_replay(ctx)
    print("ALL TESTS PASSED")
