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
"""AMD indexer top-k at decode: the single-block select vs the row split.

The GLM-5.3 indexer's top-k (`ordered=False`, `deterministic=False`, bounded
rows) at decode batch sizes, K = 2048:

  old : `_histsel_topk_bounded_kernel` with the launcher's prefetch-arm
        parameters, one block per row -- what the route took before the split.
  new : `_histsel_split_topk` at `_hsplit_plan`'s geometry (two launches).

Each arm records `REPS` calls into one `DeviceGraph`; a timed sample is
`ITERS` replays, so a sample's us/call includes the launch boundaries the arm
really has (one per call for old, two for new). Arms alternate inside every
round, the order flipping each round, and the table reports the median of
`ROUNDS` samples. Both arms are checked to select the same set on every row
before anything is timed.

Arguments: `--dist=q17` scores on 17 levels, `--dist=narrow` every score in
one coarse bin (`1 + u/32`, the select's adversarial case), instead of
uniform;
`--slice-len=L` overrides the planner's slice length (the slice count follows
from it as in `_hsplit_plan`).

Real scores: `--scores=PATH --rows=R --n=N [--bounds=PATH]` times one shape on
a dumped buffer instead of the synthetic grid. `PATH` holds `R * N` raw
little-endian float32 (row-major `[R, N]`, the indexer's `scores_buf` with
`N = max_num_keys`); `--bounds` holds `R` raw int32 live lengths, and without
it every row is taken as fully live. Columns at or past a row's bound are never
read, so they may hold whatever the dump captured.

Dumping one from a GLM-5.3 decode step (needs a local debug patch): in
`mla_index_fp8.mojo`, just before `Topk.select(...)` in the AMD scorer's chunk
loop, `ctx.synchronize()`, copy `scores_buf` (`chunk_rows * max_num_keys` f32)
to a host buffer and write its bytes to a file; for the bounds, launch a
one-block-per-row kernel that stores `row_bound.row(r, out)[0]` (the same
`_IndexerRowBound` the top-k reads) to an int32 buffer and write that too. For
a plain decode step (one token per request, `kpool = 1`, causal) the bound is
also `cache_length + 1` per request, readable on the host. The step has to run
uncaptured, since a host sync cannot sit inside a captured graph; dump one
indexer layer of a step at the context length of interest.
"""

from std.builtin.sort import sort
from std.math import ceildiv
from std.memory import bitcast

from max.gpu.host import (
    DeviceAttribute,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    FuncAttribute,
)
from internal_utils import arg_parse
from nn.topk_bitonic import (
    RowBoundsPtr,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _HSPLIT_SLICE_MAX,
    _PTOPK_BLOCK,
    _histsel_split_topk,
    _histsel_topk_bounded_kernel,
    _hsel_phi_dtype,
    _hsel_prefetch_scan_items,
    _hsplit_plan,
)
from structured_kernels.trace_buf import NullTrace

comptime K = 2048
comptime REPS = 20
comptime ITERS = 10
comptime ROUNDS = 7


def _sorted_row(h: List[Int32], b: Int) -> List[Int32]:
    var out = List[Int32]()
    for k in range(K):
        out.append(h[b * K + k])
    sort(out)
    return out^


def _load_words(path: String, count: Int) raises -> List[UInt32]:
    """`count` raw little-endian 32-bit words from `path`, which must hold
    exactly that many."""
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes()
    if len(raw) != 4 * count:
        raise Error(path, " holds ", len(raw), " bytes, expected ", 4 * count)
    var out = List[UInt32](capacity=count)
    for i in range(count):
        out.append(
            UInt32(raw[4 * i])
            | (UInt32(raw[4 * i + 1]) << 8)
            | (UInt32(raw[4 * i + 2]) << 16)
            | (UInt32(raw[4 * i + 3]) << 24)
        )
    return out^


def _median(var v: List[Float64]) -> Float64:
    sort(v)
    return v[len(v) // 2]


def _bench(
    ctx: DeviceContext,
    rows: Int,
    N: Int,
    dist: String,
    slice_len_arg: Int,
    scores_path: String = "",
    bounds_path: String = "",
) raises:
    var cu = ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var plan = _hsplit_plan(rows, N, K, cu)
    var slices = plan[0]
    var slice_len = plan[1]
    if slice_len_arg > 0:
        slice_len = slice_len_arg
        slices = min(
            cu // rows,
            max(ceildiv(N, slice_len), ceildiv(N, _HSPLIT_SLICE_MAX)),
        )
    if slices < 2:
        print(t"rows={rows} N={N}: not split by the planner, skipped")
        return

    var scores = ctx.enqueue_create_buffer[.float32](rows * N)
    var bounds = ctx.enqueue_create_buffer[.int32](rows)
    var old_out = ctx.enqueue_create_buffer[.int32](rows * K)
    var new_out = ctx.enqueue_create_buffer[.int32](rows * K)
    var cand_cols = ctx.enqueue_create_buffer[.int32](rows * slices * K)
    var cand_scores = ctx.enqueue_create_buffer[.float32](rows * slices * K)
    var state = UInt32(rows * 7919 + N)
    var file_words = List[UInt32]()
    if scores_path != "":
        file_words = _load_words(scores_path, rows * N)
    with scores.map_to_host() as h:
        for i in range(rows * N):
            if scores_path != "":
                h[i] = bitcast[.float32, 1](file_words[i])[0]
                continue
            state = 1664525 * state + 1013904223
            var u = Float32(Int32(state)) / Float32(2**31)
            if dist == "q17":
                h[i] = Float32(Int((u + 1.0) * 8.5))
            elif dist == "narrow":
                h[i] = 1.0 + (u + 1.0) / 64.0
            else:
                h[i] = u * 100.0
    var bound_words = List[UInt32]()
    if bounds_path != "":
        bound_words = _load_words(bounds_path, rows)
    with bounds.map_to_host() as h:
        for b in range(rows):
            h[b] = Int32(bound_words[b]) if bounds_path != "" else Int32(N)

    var sp = rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores.unsafe_ptr())
    var rb = RowBoundsPtr(
        rebind[ImmPointer[Int32, ImmutAnyOrigin]](bounds.unsafe_ptr())
    )
    var op = rebind[MutPointer[Int32, MutAnyOrigin]](old_out.unsafe_ptr())
    var np = rebind[MutPointer[Int32, MutAnyOrigin]](new_out.unsafe_ptr())
    var cp = rebind[MutPointer[Int32, MutAnyOrigin]](cand_cols.unsafe_ptr())
    var csp = rebind[MutPointer[Float32, MutAnyOrigin]](
        cand_scores.unsafe_ptr()
    )
    comptime old_kernel = _histsel_topk_bounded_kernel[
        NullTrace,
        RB=RowBoundsPtr,
        prefetch=True,
        scan_items=_hsel_prefetch_scan_items[
            .float32, _hsel_phi_dtype[.float32]()
        ](),
        tail_bits=_HSEL_TAIL_BITS,
        ordered=False,
        deterministic=False,
        in_dtype=.float32,
    ]

    def run_old(c: DeviceContext) raises {imm}:
        c.enqueue_function[old_kernel](
            sp,
            op,
            Int32(N),
            Int32(K),
            rb,
            NullTrace(),
            grid_dim=rows,
            block_dim=_PTOPK_BLOCK,
            shared_mem_bytes=_HSEL_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(_HSEL_SMEM_BYTES)
            ),
        )

    def run_new(c: DeviceContext) raises {imm}:
        _histsel_split_topk(
            c, sp, np, cp, csp, N, K, rows, rb, slices, slice_len
        )

    run_old(ctx)
    run_new(ctx)
    ctx.synchronize()
    var oh = List[Int32](length=rows * K, fill=0)
    var nh = List[Int32](length=rows * K, fill=0)
    with old_out.map_to_host() as h:
        for i in range(rows * K):
            oh[i] = h[i]
    with new_out.map_to_host() as h:
        for i in range(rows * K):
            nh[i] = h[i]
    for b in range(rows):
        var o = _sorted_row(oh, b)
        var n = _sorted_row(nh, b)
        for i in range(K):
            if o[i] != n[i]:
                raise Error("row ", b, " sets differ at sorted rank ", i)

    def build_old(mut builder: DeviceGraphBuilder) raises {imm}:
        for _ in range(REPS):
            run_old(builder.recording_context())

    def build_new(mut builder: DeviceGraphBuilder) raises {imm}:
        for _ in range(REPS):
            run_new(builder.recording_context())

    var g_old = DeviceGraph.create(ctx, build_old)
    var g_new = DeviceGraph.create(ctx, build_new)
    for _ in range(3):
        g_old.replay()
        g_new.replay()
    ctx.synchronize()

    def replay_old() raises {imm}:
        g_old.replay()

    def replay_new() raises {imm}:
        g_new.replay()

    comptime per_call = Float64(ITERS * REPS) * 1000.0
    var t_old = List[Float64]()
    var t_new = List[Float64]()
    for r in range(ROUNDS):
        if r % 2 == 0:
            t_old.append(
                Float64(ctx.execution_time(replay_old, ITERS)) / per_call
            )
            t_new.append(
                Float64(ctx.execution_time(replay_new, ITERS)) / per_call
            )
        else:
            t_new.append(
                Float64(ctx.execution_time(replay_new, ITERS)) / per_call
            )
            t_old.append(
                Float64(ctx.execution_time(replay_old, ITERS)) / per_call
            )
    var mo = _median(t_old^)
    var mn = _median(t_new^)
    print(
        t"| {rows} | {N} | {slices} | {slice_len} | {mo} | {mn} | {mn / mo} |"
    )
    _ = scores
    _ = bounds
    _ = old_out
    _ = new_out
    _ = cand_cols
    _ = cand_scores


def main() raises:
    var dist = String(arg_parse("dist", "uniform"))
    if dist != "uniform" and dist != "q17" and dist != "narrow":
        raise Error("unknown --dist ", dist)
    var slice_len = arg_parse("slice-len", 0)
    var scores_path = String(arg_parse("scores", ""))
    var bounds_path = String(arg_parse("bounds", ""))
    if scores_path != "":
        dist = String("file:", scores_path)
    with DeviceContext() as ctx:
        print(
            "dist=",
            dist,
            t" K={K} REPS={REPS} ITERS={ITERS} ROUNDS={ROUNDS}, us/call median",
        )
        print("| rows | N | slices | slice_len | old us | new us | new/old |")
        print("|---|---|---|---|---|---|---|")
        if scores_path != "":
            var rows = arg_parse("rows", 0)
            var n = arg_parse("n", 0)
            if rows <= 0 or n <= 0:
                raise Error("--scores needs --rows and --n")
            _bench(ctx, rows, n, dist, slice_len, scores_path, bounds_path)
            return
        for N in [84000, 131072]:
            for rows in [1, 4, 8, 16]:
                _bench(ctx, rows, N, dist, slice_len)
