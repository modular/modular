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
"""Benchmark for the paged MLA FP8 indexer (`mla_indexer_ragged_float8_paged`).

Unlike `bench_fp8_index` (contiguous scorer only), this drives the full
production op — scores alloc/fill, SM100 scorer, top-k, and invalid-fill —
against a paged K cache, at MTP decode shapes.

The `frozen_cache_len` knob sets the cache collection's `max_cache_length`
metadata independently of the actual per-row cache lengths. Captured decode
device graphs bake grid dims, the score-buffer allocation, its `-inf` fill,
and the top-k scan width from this metadata at capture time (one captured
graph serves every cache length that maps to the same dispatch key, and it is
captured at the largest), so replay does metadata-proportional work no matter
the batch's real lengths. Sweeping `frozen_cache_len` at a fixed actual
`cache_len` measures exactly that gap.

Prefill shapes exercise a second axis: past the score budget the op scores the
matrix one row window at a time instead of materializing it, so the whole
memory/latency trade shows up here. The cost tracks the ROWS PER CHUNK the budget
works out to (`budget / (max_num_keys * size_of[scores_dtype])`), not the chunk
count -- below ~500 rows it collapses.

Three axes are swept from ONE process, so every ratio is paired and carries no
cross-run clock or cache drift:

- `scores_dtype`, f32 and bf16, as two comptime arms;
- `--budgets`, a comma list of MB, iterated innermost so a cell's ladder is
  adjacent in time. It is a RUNTIME argument of the op, so the ladder costs no
  recompiles;
- `--spread`, cache-depth raggedness in percent, as in `bench_fp8_index_prefill`.

Every arm also prints an `MLAIDX` line carrying the chunk geometry it ran
(`rows_per_chunk`, chunk count, chunk bytes against the device's L2) keyed by the
same name as its `BenchId`, so the analysis joins on that rather than on a parse
of the timing table. Note the budget is denominated in BYTES: at bf16 a given
budget holds twice the rows, so a fixed-budget dtype pair also moves
`rows_per_chunk` -- and `rows_per_chunk >= sm_count` is the top-k's own dispatch
boundary. Pair a bf16 arm at HALF the f32 budget to hold the rows fixed instead.

    ... -- --batch_size=1 --seq_len=4096 --cache_len=103136 \\
        --frozen_cache_len=103136 --budgets=0,512,128,64 --label=prefill
"""

from std.random import rand, seed
from std.sys import get_defined_int, size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from max.gpu.host import DeviceAttribute, DeviceContext
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Coord,
    TileTensor,
    row_major,
)
from nn.attention.gpu.mla_index_fp8 import (
    _SCORES_BUDGET_BYTES,
    mla_indexer_ragged_float8_paged,
)
from nn.topk_bitonic import _elems_per_16b
from nn.attention.mha_mask import MaskName
from std.math import align_up, ceildiv, clamp
from std.utils.index import IndexList


def _budget_list(s: String) raises -> List[Int]:
    """Parse `a,b,c` into MB budgets; `0` means the op's own default.

    Args:
        s: Comma-separated list of megabyte budgets.

    Returns:
        The parsed budgets, in the order given.

    Raises:
        If a token is not an integer. Refusing beats defaulting: a silently
        dropped token would leave the sweep reporting a ladder it never ran.
    """
    var out = List[Int]()
    for tok in s.split(","):
        var t = tok.strip()
        if t.byte_length() == 0:
            continue
        out.append(Int(t))
    if len(out) == 0:
        raise Error("--budgets parsed to an empty list: " + s)
    return out^


def _run_name[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    scores_dtype: DType,
](
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    frozen_cache_len: Int,
    spread: Int,
    budget_mb: Int,
    label: String,
) -> String:
    var kind = String("mla_indexer_fp8_paged")
    if label.byte_length() > 0:
        kind += "/" + label
    # fmt: off
    return String(
        kind, " : ",
        "num_heads=", num_heads, ", ",
        "depth=", depth, ", ",
        "page_size=", page_size, ", ",
        "top_k=", top_k, " : ",
        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "cache_len=", cache_len, ", ",
        "frozen_cache_len=", frozen_cache_len, ", ",
        "spread=", spread, ", ",
        "scores_budget_mb=", budget_mb, ", ",
        "scores_dtype=", scores_dtype,
    )
    # fmt: on


def execute_mla_indexer_paged[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    scores_dtype: DType,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    frozen_cache_len: Int,
    spread: Int,
    budgets: List[Int],
    label: String,
    run_benchmark: Bool,
) raises:
    """Benchmark the paged indexer at one shape, over a ladder of budgets.

    Parameters:
        num_heads: Query index heads.
        depth: Per-head key dimension.
        page_size: Tokens per KV cache page.
        top_k: Keys selected per token.
        scores_dtype: Element type of the transient score matrix.

    Args:
        ctx: Device context.
        m: Bench harness collecting results.
        batch_size: Number of sequences (decode requests).
        seq_len: New tokens per sequence (1 + num_speculative_tokens for MTP).
        cache_len: Actual cached tokens per sequence; the batch MEAN once
            `spread` is nonzero.
        frozen_cache_len: The `max_cache_length` metadata the op sees. Must be
            >= the deepest entry. Equal reproduces eager execution; larger
            reproduces a decode graph captured at that cache length.
        spread: Cache-depth raggedness, in percent of `cache_len`. 0 is a
            uniform batch.
        budgets: Score-buffer budgets in MB, run innermost so a cell's ladder is
            adjacent in time. 0 takes the op's own default.
        label: Which group this shape came from; reporting only.
        run_benchmark: False leaves ONE launch per budget for `ncu` to replay,
            which a `Bench` loop's hundreds of launches make impractical.
    """
    var lo_cache = cache_len - (cache_len * spread) // 100
    var hi_cache = cache_len + (cache_len * spread) // 100
    if frozen_cache_len < hi_cache:
        raise Error("frozen_cache_len must be >= the deepest cache length")
    var total_seq_len = batch_size * seq_len

    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=depth, is_mla=True
    )
    comptime num_layers = 1

    # Pool holds the actual tokens; the LUT view is as wide as the frozen
    # metadata implies (mirroring capture-time `runtime_inputs`), with unused
    # tail slots pointing at block 0. The kernel never dereferences past each
    # row's real key count, so the tail is address-safety padding only.
    #
    # Sized on the DEEPEST entry, not the mean: every allocation here is a batch
    # maximum, exactly as production's captured metadata is, so a ragged batch
    # must not shrink the pool under its own deepest row.
    var real_keys_per_seq = hi_cache + seq_len
    var real_pages_per_seq = ceildiv(real_keys_per_seq, page_size)
    var lut_pages_per_seq = ceildiv(frozen_cache_len + seq_len, page_size)
    var num_blocks = batch_size * real_pages_per_seq + 1

    var q_size = total_seq_len * num_heads * depth
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    with q_device.map_to_host() as q_host:
        rand(q_host.as_span())

    var qs_size = total_seq_len * num_heads
    var qs_device = ctx.enqueue_create_buffer[.float32](qs_size)
    with qs_device.map_to_host() as qs_host:
        rand(qs_host.as_span())

    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    with input_row_offsets_device.map_to_host() as iro_host:
        for i in range(batch_size + 1):
            iro_host[i] = UInt32(i * seq_len)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for i in range(batch_size):
            # Linear grade lo -> hi across the batch; a single-entry batch takes
            # the mean, so `spread` cannot change a batch of one.
            if batch_size > 1:
                cl_host[i] = UInt32(
                    lo_cache + (hi_cache - lo_cache) * i // (batch_size - 1)
                )
            else:
                cl_host[i] = UInt32(cache_len)

    var k_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var k_block_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        k_shape.flattened_length()
    )
    with k_block_device.map_to_host() as k_block_host:
        rand(k_block_host.as_span())

    comptime head_dim_granularity = 1
    var ks_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_dim_granularity,
    )
    var ks_block_device = ctx.enqueue_create_buffer[.float32](
        ks_shape.flattened_length()
    )
    with ks_block_device.map_to_host() as ks_block_host:
        rand(ks_block_host.as_span())

    var paged_lut_shape = IndexList[2](batch_size, lut_pages_per_seq)
    var k_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    with k_lut_device.map_to_host() as k_lut_host:
        for bs in range(batch_size):
            for page_idx in range(lut_pages_per_seq):
                var block_idx = 0
                if page_idx < real_pages_per_seq:
                    block_idx = 1 + bs * real_pages_per_seq + page_idx
                k_lut_host[bs * lut_pages_per_seq + page_idx] = UInt32(
                    block_idx
                )

    comptime Collection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ]
    comptime blocks_layout_type = Collection.blocks_tt_layout
    var blocks_shape = Coord[*blocks_layout_type.shape_types]()
    blocks_shape[0] = Int64(num_blocks)
    blocks_shape[2] = Int64(num_layers)
    var blocks_strides = Coord[*blocks_layout_type.stride_types]()
    blocks_strides[1] = blocks_shape[2] * Int64(blocks_strides[2].value())
    blocks_strides[0] = Int64(blocks_shape[1].value()) * blocks_strides[1]
    var blocks = TileTensor(
        k_block_device, blocks_layout_type(blocks_shape, blocks_strides)
    ).as_unsafe_any_origin()
    comptime assert Collection.scale_dtype == DType.float32
    comptime scales_layout_type = Collection.scales_tt_layout
    var scales_shape = Coord[*scales_layout_type.shape_types]()
    scales_shape[0] = Int64(num_blocks)
    scales_shape[2] = Int64(num_layers)
    var scales_strides = Coord[*scales_layout_type.stride_types]()
    scales_strides[1] = scales_shape[2] * Int64(scales_strides[2].value())
    scales_strides[0] = Int64(scales_shape[1].value()) * scales_strides[1]
    var scales = (
        TileTensor(
            ks_block_device, scales_layout_type(scales_shape, scales_strides)
        )
        .bitcast[Collection.scale_dtype]()
        .as_unsafe_any_origin()
    )
    var cache_lengths = (
        TileTensor(cache_lengths_device, row_major(Coord(Int64(batch_size))))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var lookup_table = (
        TileTensor(
            k_lut_device,
            row_major(Coord(Int64(batch_size), Int64(lut_pages_per_seq))),
        )
        .as_imm()
        .as_unsafe_any_origin()
    )
    var k_collection = Collection(
        blocks,
        cache_lengths,
        lookup_table,
        UInt32(seq_len),
        UInt32(frozen_cache_len),
        scales,
    )

    var o_device = ctx.enqueue_create_buffer[.int32](total_seq_len * top_k)

    var q_tile = TileTensor(
        q_device, row_major(total_seq_len, num_heads, depth)
    )
    var qs_tile = TileTensor(qs_device, row_major(total_seq_len, num_heads))
    var input_row_offsets_tile = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )
    var o_tile = TileTensor(o_device, row_major(total_seq_len, top_k))

    # Mirrors the op's own sizing, so the geometry printed beside a timing is
    # the one that produced it: frozen metadata, not the batch's real depths.
    comptime scores_align = _elems_per_16b[scores_dtype]()
    var max_num_keys = align_up(frozen_cache_len + seq_len, scores_align)
    var row_bytes = max_num_keys * size_of[scores_dtype]()

    var l2_bytes = ctx.get_attribute(DeviceAttribute.L2_CACHE_SIZE)
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    for budget_mb in budgets:
        var budget_bytes = (
            _SCORES_BUDGET_BYTES if budget_mb <= 0 else budget_mb * 1024 * 1024
        )
        var rows_per_chunk = clamp(budget_bytes // row_bytes, 1, total_seq_len)

        @inline(.always)
        def kernel_launch(
            launch_ctx: DeviceContext,
        ) raises {mut o_tile, imm}:
            mla_indexer_ragged_float8_paged[
                DType.float8_e4m3fn,
                type_of(k_collection),
                num_heads,
                depth,
                top_k,
                MaskName.CAUSAL.name,
                scores_dtype,
            ](
                o_tile,
                q_tile,
                qs_tile,
                input_row_offsets_tile,
                k_collection,
                UInt32(0),
                launch_ctx,
                budget_bytes,
            )

        @inline(.always)
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, kernel_launch, ctx)

        var name = _run_name[num_heads, depth, page_size, top_k, scores_dtype](
            batch_size,
            seq_len,
            cache_len,
            frozen_cache_len,
            spread,
            budget_bytes // (1024 * 1024),
            label,
        )
        # `topk_fills_gpu` is the top-k's own dispatch boundary, and the budget
        # ladder can cross it: a chunk under `sm_count` rows switches the select
        # from its prefill arm to the prefetching decode one, which is a
        # different kernel and not an apples-to-apples ladder step.
        # fmt: off
        print(
            "MLAIDX ", name,
            " | max_num_keys=", max_num_keys,
            " rows_per_chunk=", rows_per_chunk,
            " chunks=", ceildiv(total_seq_len, rows_per_chunk),
            " chunk_bytes=", rows_per_chunk * row_bytes,
            " l2_bytes=", l2_bytes,
            " chunk_over_l2=",
            Float64(rows_per_chunk * row_bytes) / Float64(l2_bytes),
            " topk_fills_gpu=", rows_per_chunk >= sm_count,
        )
        # fmt: on

        if run_benchmark:
            m.bench_function(bench_func, BenchId(name))
        else:
            kernel_launch(ctx)
            ctx.synchronize()

    _ = q_device
    _ = qs_device
    _ = input_row_offsets_device
    _ = cache_lengths_device
    _ = k_block_device
    _ = ks_block_device
    _ = k_lut_device
    _ = o_device


def main() raises:
    # The indexer is REPLICATED per tensor-parallel rank, not sharded: the
    # `Indexer` layer computes an `n_local_heads` but never uses it, reshaping
    # to the full `index_n_heads` instead. So GLM 5.2 puts 32 heads through this
    # kernel on every rank and DeepSeek V3.2 puts 64, whatever the TP degree.
    # 4 and 8 are reachable only where a caller shards the heads itself.
    comptime num_heads = get_defined_int["num_heads", 32]()
    comptime depth = get_defined_int["depth", 128]()
    comptime page_size = get_defined_int["page_size", 128]()
    comptime top_k = get_defined_int["top_k", 2048]()

    var batch_size = arg_parse("batch_size", 8)
    # 1 + num_speculative_tokens: the MTP verify width GLM 5.2 decodes at.
    var seq_len = arg_parse("seq_len", 6)
    var cache_len = arg_parse("cache_len", 76000)
    # 0 sweeps {cache_len, GLM recipe pin 163840, GLM max_position 1048576}.
    var frozen_cache_len = arg_parse("frozen_cache_len", 0)
    # Cache-depth raggedness, in percent of `cache_len`; 0 is the uniform batch.
    var spread = arg_parse("spread", 0)
    # Which group this shape came from; reporting only.
    var label = String(arg_parse("label", ""))
    # Score-buffer budgets in MB, the innermost axis. 0 takes the op's own
    # default, so a ladder that includes it cannot drift from production when
    # that default changes.
    var budgets = _budget_list(String(arg_parse("budgets", "0")))
    # False leaves a single launch per budget for `ncu` to replay.
    var run_benchmark = arg_parse("run_benchmark", True)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        # Read, never assumed: every budget in a sweep is stated as a fraction
        # of these two, and both differ across Blackwell parts.
        var l2_bytes = ctx.get_attribute(DeviceAttribute.L2_CACHE_SIZE)
        var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        print("MLAIDXDEV l2_bytes=", l2_bytes, " sm_count=", sm_count)

        var frozen_sweep = List[Int]()
        if frozen_cache_len != 0:
            frozen_sweep.append(frozen_cache_len)
        else:
            frozen_sweep.append(cache_len)
            frozen_sweep.append(163840)
            frozen_sweep.append(1048576)

        var deepest = cache_len + (cache_len * spread) // 100
        for frozen in frozen_sweep:
            if frozen < deepest:
                continue
            # Both score dtypes in ONE process, back to back within a cell, so
            # the ratio is paired and carries no cross-run clock or cache drift.
            # The budget ladder sits inside each arm, so a step's neighbour is
            # always the adjacent budget.
            comptime for scores_dtype in [DType.float32, DType.bfloat16]:
                execute_mla_indexer_paged[
                    num_heads, depth, page_size, top_k, scores_dtype
                ](
                    ctx,
                    m,
                    batch_size,
                    seq_len,
                    cache_len,
                    frozen,
                    spread,
                    budgets,
                    label,
                    run_benchmark,
                )

    if run_benchmark:
        m.dump_report()
