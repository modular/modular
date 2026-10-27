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
"""The paged MLA FP8 indexer (`mla_indexer_ragged_float8_paged`) as one sweep.

The sweep port of the deleted `gpu/nn/bench_mla_index_fp8_paged.mojo`, which
explained the two effects it measures: frozen `max_cache_length` metadata, which a captured
decode graph bakes in, and the score budget past which the op scores the
matrix one row window at a time. It drives the full production op (scores
alloc/fill, SM100 scorer, top-k, invalid-fill) against a paged K cache.

Runtime axes, each a comma list: `--batch_size`, `--seq_len`, `--cache_len`,
`--spread` (cache-depth raggedness in percent of `cache_len`) and
`--frozen_cache_len`, innermost of the runtime axes. A frozen length of 0 is
the batch's deepest cache length, which is the metadata eager execution sees;
configs whose frozen length is below the deepest are dropped. One shared K
cache, sized for the deepest config, serves the whole sweep; each config
writes a page table as wide as its frozen metadata over the fixture's blocks,
with the columns past the real pages pointing at the null block.

Inside each config, `scores_dtype` (f32, bf16; `-D scores_dtype=` pins it) is a
comptime arm, `--budgets` (MB, 0 for the op's default) is a runtime ladder
inside it, and the remap arm is comptime and innermost, so every ratio is
paired. Each arm prints an `MLAIDX` line with its chunk geometry, keyed by
the same name as its `BenchId`; with `--json`, the same geometry is in each
benchmark's fields. `--run_benchmark=False` leaves one launch per arm for
`ncu` to replay. The measurement flags are `SweepHarness`'s.

    ... -- --batch_size=1 --seq_len=4096 --cache_len=103136 \\
        --frozen_cache_len=0 --budgets=0,512,128,64 --label=prefill
"""

from std.math import align_up, ceildiv, clamp
from std.sys import get_defined_int, size_of

from bench_sweep import (
    Config,
    Fields,
    Footprint,
    PageLayout,
    PagedBlocks,
    PageTable,
    Sweep,
    SweepHarness,
    device_budget,
    dtype_arm_enabled,
    fill_affine,
    fill_lut_view,
    fill_uniform,
    fill_uniform_host_rand,
    parse_int_list,
    plan_groups,
)
from internal_utils import arg_parse
from kv_cache.paged_sparse_kv_index_remap import (
    NoRemap,
    PagedKVRemap,
    paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch,
)
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from nn.attention.gpu.mla_index_fp8 import (
    _SCORES_BUDGET_BYTES,
    mla_indexer_ragged_float8_paged,
)
from nn.attention.mha_mask import MaskName
from nn.topk_bitonic import _elems_per_16b
from std.benchmark import BenchId

comptime fp8 = DType.float8_e4m3fn
# The indexer is REPLICATED per tensor-parallel rank, not sharded, so GLM 5.2
# puts 32 heads through this kernel on every rank and DeepSeek V3.2 puts 64.
comptime NUM_HEADS = get_defined_int["num_heads", 32]()
comptime DEPTH = get_defined_int["depth", 128]()
comptime PAGE_SIZE = get_defined_int["page_size", 128]()
comptime TOP_K = get_defined_int["top_k", 2048]()

# Remap arms, run back to back within each budget so their ratios are paired:
# logical indices only, the physical rows fused into the top-k, the standalone
# remap kernel after a logical-only top-k, and the fused physical rows without
# the logical indices.
comptime _ARM_NONE = 0
comptime _ARM_FUSED = 1
comptime _ARM_SPLIT = 2
comptime _ARM_PHYSICAL = 3
comptime _NUM_ARMS = 4


@fieldwise_init
struct Shape(TrivialRegisterPassable):
    """One config's sizes, with the cache depths and frozen length resolved."""

    var batch_size: Int
    var seq_len: Int
    var cache_len: Int
    var spread: Int
    var lo_cache: Int
    var hi_cache: Int
    var frozen_cache_len: Int

    @staticmethod
    def of(cfg: Config) raises -> Self:
        var cache_len = cfg["cache_len"]
        var spread = cfg["spread"]
        var delta = (cache_len * spread) // 100
        var frozen = cfg["frozen_cache_len"]
        return Self(
            cfg["batch_size"],
            cfg["seq_len"],
            cache_len,
            spread,
            cache_len - delta,
            cache_len + delta,
            frozen if frozen != 0 else cache_len + delta,
        )

    def tokens(self) -> Int:
        return self.batch_size * self.seq_len

    def real_pages(self) -> Int:
        """Returns the pages each sequence's deepest row needs."""
        return ceildiv(self.hi_cache + self.seq_len, PAGE_SIZE)

    def lut_cols(self) -> Int:
        """Returns the page table width the frozen metadata implies."""
        return ceildiv(self.frozen_cache_len + self.seq_len, PAGE_SIZE)

    def touched_bytes(self) -> Int:
        """Returns the keys and scales one launch reads: every key of every
        sequence, at most the deepest's."""
        return self.batch_size * (self.hi_cache + self.seq_len) * (DEPTH + 4)


struct IndexCache(Movable):
    """The indexer's paged FP8 keys and per-row scales, on one page table."""

    var table: PageTable[PAGE_SIZE]
    var keys: PagedBlocks[fp8]
    var scales: PagedBlocks[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        footprint: Footprint,
        layout: PageLayout,
        on_host: Bool,
    ) raises:
        var table = PageTable[PAGE_SIZE].from_footprint(ctx, footprint, layout)
        self.keys = PagedBlocks[fp8](ctx, table, DEPTH, seed=1, on_host=on_host)
        self.scales = PagedBlocks[DType.float32](
            ctx, table, 1, seed=2, on_host=on_host
        )
        self.table = table^

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        return (
            PageTable[PAGE_SIZE].bytes_for(footprint)
            + PagedBlocks[fp8].bytes_for[PAGE_SIZE](footprint, DEPTH)
            + PagedBlocks[DType.float32].bytes_for[PAGE_SIZE](footprint, 1)
        )

    def bytes(self) -> Int:
        return self.table.bytes() + self.keys.bytes() + self.scales.bytes()


struct Workspace(Movable):
    """Per-token buffers and the frozen-width page table, sized per group."""

    var q: DeviceBuffer[fp8]
    var q_scales: DeviceBuffer[DType.float32]
    var row_offsets: DeviceBuffer[DType.uint32]
    var logical: DeviceBuffer[DType.int32]
    var physical: DeviceBuffer[DType.int32]
    var lut: DeviceBuffer[DType.uint32]

    def __init__(
        out self, ctx: DeviceContext, footprint: Footprint, on_host: Bool
    ) raises:
        var tokens = footprint["tokens"]
        self.q = ctx.enqueue_create_buffer[fp8](tokens * NUM_HEADS * DEPTH)
        self.q_scales = ctx.enqueue_create_buffer[DType.float32](
            tokens * NUM_HEADS
        )
        self.row_offsets = ctx.enqueue_create_buffer[DType.uint32](
            footprint["batch"] + 1
        )
        self.logical = ctx.enqueue_create_buffer[DType.int32](tokens * TOP_K)
        self.physical = ctx.enqueue_create_buffer[DType.int32](tokens * TOP_K)
        self.lut = ctx.enqueue_create_buffer[DType.uint32](
            footprint["lut_entries"]
        )
        if on_host:
            fill_uniform_host_rand(self.q)
            fill_uniform_host_rand(self.q_scales)
        else:
            fill_uniform(ctx, self.q, 11)
            fill_uniform(ctx, self.q_scales, 12)

    @staticmethod
    def require(shape: Shape, mut footprint: Footprint, copies: Int):
        footprint.need("tokens", shape.tokens())
        footprint.need(
            "lut_entries", copies * shape.batch_size * shape.lut_cols()
        )

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        var tokens = footprint["tokens"]
        return (
            tokens * (NUM_HEADS * (DEPTH + 4) + 8 * TOP_K)
            + 4 * (footprint["batch"] + 1)
            + 4 * footprint["lut_entries"]
        )


def _budget_bytes(budget_mb: Int) -> Int:
    """Returns a budget in bytes; 0 MB is the op's own default."""
    return _SCORES_BUDGET_BYTES if budget_mb <= 0 else budget_mb * 1024 * 1024


def _arm_label(label: String, arm: Int) -> String:
    var name = StaticString("remap=none")
    if arm == _ARM_FUSED:
        name = "remap=fused"
    elif arm == _ARM_SPLIT:
        name = "remap=split"
    elif arm == _ARM_PHYSICAL:
        name = "remap=physical"
    if label.byte_length() > 0:
        return String(label, "/", name)
    return String(name)


def _run_name[
    scores_dtype: DType
](shape: Shape, budget_mb: Int, label: String) -> String:
    var kind = String("mla_indexer_fp8_paged")
    if label.byte_length() > 0:
        kind += "/" + label
    # fmt: off
    return String(
        kind, " : ",
        "num_heads=", NUM_HEADS, ", ",
        "depth=", DEPTH, ", ",
        "page_size=", PAGE_SIZE, ", ",
        "top_k=", TOP_K, " : ",
        "batch_size=", shape.batch_size, ", ",
        "seq_len=", shape.seq_len, ", ",
        "cache_len=", shape.cache_len, ", ",
        "frozen_cache_len=", shape.frozen_cache_len, ", ",
        "spread=", shape.spread, ", ",
        "scores_budget_mb=", budget_mb, ", ",
        "scores_dtype=", scores_dtype,
    )
    # fmt: on


def run_config[
    scores_dtype: DType
](
    ctx: DeviceContext,
    mut h: SweepHarness,
    mut cache: IndexCache,
    mut ws: Workspace,
    shape: Shape,
    budgets: List[Int],
    label: String,
    run_benchmark: Bool,
    copies: Int,
) raises:
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=DEPTH, is_mla=True
    )
    comptime Collection = PagedKVCacheCollection[
        fp8,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ]
    ref table = cache.table
    var batch_size = shape.batch_size
    var tokens = shape.tokens()
    var lut_cols = shape.lut_cols()
    var num_blocks = table.num_blocks

    h.timer.reset()
    fill_affine(ctx, ws.row_offsets, shape.seq_len, 0, batch_size + 1)
    table.set_cache_len_graded(ctx, batch_size, shape.lo_cache, shape.hi_cache)
    # Each copy reads its own pages, through a frozen-width view of its own
    # page table.
    table.write_copies(ctx, max(copies, 1), batch_size, shape.real_pages())
    var view_entries = batch_size * lut_cols
    for copy in range(max(copies, 1)):
        fill_lut_view(
            ctx,
            DeviceBuffer[DType.uint32](
                ctx,
                ws.lut.unsafe_ptr().unsafe_offset(copy * view_entries),
                view_entries,
                owning=False,
            ),
            table.copy_lut(ctx, copy, batch_size, shape.real_pages()),
            shape.real_pages(),
            batch_size,
            lut_cols,
            shape.real_pages(),
        )

    comptime blocks_layout = Collection.blocks_tt_layout
    var blocks_shape = Coord[*blocks_layout.shape_types]()
    blocks_shape[0] = Int64(num_blocks)
    blocks_shape[2] = Int64(1)
    var blocks_strides = Coord[*blocks_layout.stride_types]()
    blocks_strides[1] = blocks_shape[2] * Int64(blocks_strides[2].value())
    blocks_strides[0] = Int64(blocks_shape[1].value()) * blocks_strides[1]
    comptime scales_layout = Collection.scales_tt_layout
    var scales_shape = Coord[*scales_layout.shape_types]()
    scales_shape[0] = Int64(num_blocks)
    scales_shape[2] = Int64(1)
    var scales_strides = Coord[*scales_layout.stride_types]()
    scales_strides[1] = scales_shape[2] * Int64(scales_strides[2].value())
    scales_strides[0] = Int64(scales_shape[1].value()) * scales_strides[1]
    var lut_base = ws.lut.unsafe_ptr()

    def collection_at(copy: Int) {mut cache, imm} -> Collection:
        return Collection(
            TileTensor(
                cache.keys.buf, blocks_layout(blocks_shape, blocks_strides)
            ).as_unsafe_any_origin(),
            TileTensor(table.cache_lengths, row_major(Int64(batch_size)))
            .as_imm()
            .as_unsafe_any_origin(),
            TileTensor(
                lut_base.unsafe_offset(copy * view_entries),
                row_major(Int64(batch_size), Int64(lut_cols)),
            )
            .as_imm()
            .as_unsafe_any_origin(),
            UInt32(shape.seq_len),
            UInt32(shape.frozen_cache_len),
            TileTensor(
                cache.scales.buf, scales_layout(scales_shape, scales_strides)
            )
            .bitcast[Collection.scale_dtype]()
            .as_unsafe_any_origin(),
        )

    var phys_ptr = ws.physical.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var logical_ptr = ws.logical.unsafe_ptr().as_imm()
    var row_offsets_ptr = ws.row_offsets.unsafe_ptr().as_imm()
    var q_tile = TileTensor(ws.q, row_major(tokens, NUM_HEADS, DEPTH))
    var qs_tile = TileTensor(ws.q_scales, row_major(tokens, NUM_HEADS))
    var row_offsets_tile = TileTensor(ws.row_offsets, row_major(batch_size + 1))
    var o_tile = TileTensor(ws.logical, row_major(tokens, TOP_K))

    # Mirrors the op's own sizing, so the geometry printed beside a timing is
    # the one that produced it: frozen metadata, not the batch's real depths.
    comptime scores_align = _elems_per_16b[scores_dtype]()
    var max_num_keys = align_up(
        shape.frozen_cache_len + shape.seq_len, scores_align
    )
    var row_bytes = max_num_keys * size_of[scores_dtype]()
    var l2_bytes = ctx.get_attribute(DeviceAttribute.L2_CACHE_SIZE)
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    ctx.synchronize()
    h.timer.lap("config_setup")

    for budget_mb in budgets:
        var budget_bytes = _budget_bytes(budget_mb)
        var rows_per_chunk = clamp(budget_bytes // row_bytes, 1, tokens)

        comptime for arm in range(_NUM_ARMS):

            @inline(.always)
            def kernel_launch(
                launch_ctx: DeviceContext, copy: Int
            ) raises {mut o_tile, imm}:
                var k_collection = collection_at(copy)
                var lut_ptr = lut_base.unsafe_offset(copy * view_entries)
                comptime if arm == _ARM_FUSED or arm == _ARM_PHYSICAL:
                    var paged_remap = PagedKVRemap[PAGE_SIZE](
                        phys_ptr,
                        lut_ptr,
                        lut_cols,
                        batch_size,
                        UInt32(num_blocks),
                    )
                    mla_indexer_ragged_float8_paged[
                        fp8,
                        Collection,
                        NUM_HEADS,
                        DEPTH,
                        TOP_K,
                        MaskName.CAUSAL.name,
                        scores_dtype,
                        write_logical=(arm == _ARM_FUSED),
                    ](
                        o_tile,
                        q_tile,
                        qs_tile,
                        row_offsets_tile,
                        k_collection,
                        UInt32(0),
                        launch_ctx,
                        paged_remap,
                        budget_bytes,
                    )
                else:
                    mla_indexer_ragged_float8_paged[
                        fp8,
                        Collection,
                        NUM_HEADS,
                        DEPTH,
                        TOP_K,
                        MaskName.CAUSAL.name,
                        scores_dtype,
                        write_logical=True,
                    ](
                        o_tile,
                        q_tile,
                        qs_tile,
                        row_offsets_tile,
                        k_collection,
                        UInt32(0),
                        launch_ctx,
                        NoRemap(),
                        budget_bytes,
                    )
                comptime if arm == _ARM_SPLIT:
                    paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch[
                        "gpu", PAGE_SIZE
                    ](
                        phys_ptr,
                        logical_ptr,
                        row_offsets_ptr,
                        lut_ptr.as_imm(),
                        tokens * TOP_K,
                        lut_cols,
                        batch_size,
                        TOP_K,
                        UInt32(num_blocks),
                        batch_size,
                        TOP_K,
                        1,
                        launch_ctx,
                    )

            var name = _run_name[scores_dtype](
                shape, budget_bytes // (1024 * 1024), _arm_label(label, arm)
            )
            # `topk_fills_gpu` is the top-k's own dispatch boundary, and the
            # budget ladder can cross it: a chunk under `sm_count` rows
            # switches the select from its prefill arm to the prefetching
            # decode one, a different kernel and not an apples-to-apples step.
            # fmt: off
            print(
                "MLAIDX ", name,
                " | max_num_keys=", max_num_keys,
                " rows_per_chunk=", rows_per_chunk,
                " chunks=", ceildiv(tokens, rows_per_chunk),
                " chunk_bytes=", rows_per_chunk * row_bytes,
                " l2_bytes=", l2_bytes,
                " chunk_over_l2=",
                Float64(rows_per_chunk * row_bytes) / Float64(l2_bytes),
                " topk_fills_gpu=", rows_per_chunk >= sm_count,
            )
            # fmt: on

            h.first_launch(ctx, kernel_launch)
            if run_benchmark:
                var fields = Fields()
                fields.add("kernel", String("mla_indexer_fp8_paged"))
                fields.add("label", label)
                fields.add(
                    "remap", String(_arm_label("", arm).removeprefix("remap="))
                )
                fields.add("num_heads", NUM_HEADS)
                fields.add("depth", DEPTH)
                fields.add("page_size", PAGE_SIZE)
                fields.add("top_k", TOP_K)
                fields.add("batch_size", batch_size)
                fields.add("seq_len", shape.seq_len)
                fields.add("cache_len", shape.cache_len)
                fields.add("frozen_cache_len", shape.frozen_cache_len)
                fields.add("spread", shape.spread)
                fields.add("scores_budget_mb", budget_bytes // (1024 * 1024))
                fields.add("scores_dtype", scores_dtype)
                fields.add("max_num_keys", max_num_keys)
                fields.add("rows_per_chunk", rows_per_chunk)
                fields.add("chunk_bytes", rows_per_chunk * row_bytes)
                fields.add("topk_fills_gpu", rows_per_chunk >= sm_count)
                h.bench(ctx, kernel_launch, BenchId(name), fields^, copies)


def main() raises:
    var h = SweepHarness("index_fp8_paged")

    var sweep = Sweep()
    sweep.axis("batch_size", "8")
    # 1 + num_speculative_tokens: the MTP verify width GLM 5.2 decodes at.
    sweep.axis("seq_len", "6")
    sweep.axis("cache_len", "76000")
    sweep.axis("spread", "0")
    # Eager, the GLM recipe pin, and the GLM max_position.
    sweep.axis("frozen_cache_len", "0,163840,1048576")
    # Score-buffer budgets in MB; 0 takes the op's own default, so a ladder
    # that includes it cannot drift from production when that default moves.
    var budgets = parse_int_list(arg_parse("budgets", "0"))
    if len(budgets) == 0:
        raise Error("--budgets parsed to an empty list")
    var label = String(arg_parse("label", ""))
    var run_benchmark = arg_parse("run_benchmark", True)
    var fill = String(arg_parse("fill", "gpu"))
    var layout = PageLayout.parse(arg_parse("page_layout", "identity"))
    var on_host = fill == "host"

    var max_budget = 0
    for b in budgets:
        max_budget = max(max_budget, _budget_bytes(b))

    def frozen_covers_batch(cfg: Config) raises -> Bool:
        var shape = Shape.of(cfg)
        return shape.frozen_cache_len >= shape.hi_cache

    var configs = sweep.configs(frozen_covers_batch)
    if len(configs) == 0:
        raise Error("every frozen_cache_len is below its deepest cache length")

    h.timer.lap("main_prologue")
    with DeviceContext() as ctx:
        h.timer.lap("device_init")
        h.start(ctx)
        # Read, never assumed: every budget in a sweep is stated as a fraction
        # of these two, and both differ across Blackwell parts.
        print(
            "MLAIDXDEV l2_bytes=",
            ctx.get_attribute(DeviceAttribute.L2_CACHE_SIZE),
            " sm_count=",
            ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
        )
        var rot = h.rotation()

        # One more copy allocates its pages whole: keys and scales.
        def copies(shape: Shape) {imm} -> Int:
            var pages = shape.batch_size * shape.real_pages()
            return rot.copies(
                shape.touched_bytes(), pages * PAGE_SIZE * (DEPTH + 4)
            )

        def require(cfg: Config, mut fp: Footprint) raises {imm}:
            var shape = Shape.of(cfg)
            var n = copies(shape)
            # Sized on the deepest row, not the mean: every allocation here is
            # a batch maximum, as production's captured metadata is.
            PageTable[PAGE_SIZE].require(
                fp, shape.batch_size, shape.hi_cache + shape.seq_len, n
            )
            Workspace.require(shape, fp, max(1, n))

        def bytes(fp: Footprint) raises {imm} -> Int:
            # The op allocates its score buffer itself, up to the budget.
            return (
                IndexCache.bytes_for(fp) + Workspace.bytes_for(fp) + max_budget
            )

        var groups = plan_groups(configs, require, bytes, device_budget(ctx))
        for ref group in groups:
            ref fp = group.footprint
            var cache = IndexCache(ctx, fp, layout, on_host)
            ctx.synchronize()
            h.timer.lap("fixture_fill")
            var ws = Workspace(ctx, fp, on_host)
            ctx.synchronize()
            h.timer.lap("workspace_fill")
            var info = Fields()
            info.add("fixture_bytes", cache.bytes())
            info.add("planned_bytes", group.bytes)
            info.extend(fp)
            info.add("configs", len(group.configs))
            info.add("fill", fill)
            info.add("page_layout", String(layout))
            h.note(info^)
            for ref cfg in group.configs:
                var shape = Shape.of(cfg)
                comptime for scores_dtype in [DType.float32, DType.bfloat16]:
                    comptime if dtype_arm_enabled[
                        "scores_dtype", scores_dtype
                    ]():
                        run_config[scores_dtype](
                            ctx,
                            h,
                            cache,
                            ws,
                            shape,
                            budgets,
                            label,
                            run_benchmark,
                            copies(shape),
                        )
            h.timer.reset()
        h.stop(ctx)
    h.timer.lap("device_teardown")
    h.finish()
