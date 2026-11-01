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
"""DSA attention block (FP8 indexer + top-k, then sparse MLA) over a sweep.

One process runs every (heads, seq_len, cache_len) config against a single
shared paged cache, and charges wall time to each phase so the cost of a sweep
is measured: process start, device init, fixture fill, per-config setup, first
launch, and the benchmark loop split into measured, warmup and harness time.

Runtime axes come from `--seq_len=6,8192 --cache_len=32768,0`; the head
count is a comptime axis, run innermost so both arms of a runtime config share
clocks. The same source builds as one executable per config by fixing the
comptime axes with `-D sweep_h=<heads> -D sweep_seq=<seq_len>`, which is how
the separate-executable cost is measured against it. `--fill=host` reproduces
the host-side `rand` and index loops the per-kernel benchmarks use today, and
`--page_layout=shuffled|fragmented` scatters the page table. The measurement
flags (`--target_time`, `--l2`, `--throttle_check`, `--json`) are
`SweepHarness`'s.

Indexer: 32 heads x 128 (replicated per TP rank). MLA: `h` query heads of the
576-wide FP8 latent, top_k 2048, page 128. seq_len <= 8 takes the sparse
decode, longer prefixes the native-FP8 sparse prefill, as `mla_graph` routes.
The MLA reads synthetic index lists with the indexer's shape and spread, not
the indexer's own output, so a cold prefix cannot hand it -1 rows the decode
kernel would dereference.
"""

from std.math import ceildiv
from std.sys import get_defined_int

from bench_sweep import (
    Config,
    Fields,
    Footprint,
    PageLayout,
    PagedBlocks,
    PageTable,
    Sweep,
    SweepHarness,
    arm_enabled,
    device_budget,
    fill_affine,
    fill_sparse_indices,
    fill_uniform,
    fill_uniform_host_rand,
    plan_groups,
)
from internal_utils import arg_parse
from kv_cache.paged_sparse_kv_index_remap import PagedKVRemap
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.mla_index_fp8 import mla_indexer_ragged_float8_paged
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.attention.mha_mask import CausalMask, MaskName
from nn.attention.mha_utils import MHAConfig
from std.benchmark import BenchId

comptime fp8 = DType.float8_e4m3fn
comptime KV_LATENT_DIM = 512
comptime Q_DEPTH = 576
comptime V_DEPTH = KV_LATENT_DIM
comptime INDEX_HEADS = 32
comptime INDEX_DEPTH = 128
comptime PAGE_SIZE = 128
comptime TOP_K = 2048
comptime MAX_HEADS = 64
comptime DECODE_MAX_SEQ = 8

# 0 compiles every arm into one executable; a value pins that axis.
comptime SWEEP_SEQ = get_defined_int["sweep_seq", 0]()
comptime COMPILE_DECODE = SWEEP_SEQ == 0 or SWEEP_SEQ <= DECODE_MAX_SEQ
comptime COMPILE_PREFILL = SWEEP_SEQ == 0 or SWEEP_SEQ > DECODE_MAX_SEQ


struct DSACache(Movable):
    """Paged FP8 caches for a DSA layer: the MLA latent and the indexer keys.

    Both share one page table, as the per-kernel benchmarks do; production
    allocates both caches' pages together.
    """

    var table: PageTable[PAGE_SIZE]
    var latent: PagedBlocks[fp8]
    var index_k: PagedBlocks[fp8]
    var index_scales: PagedBlocks[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        footprint: Footprint,
        layout: PageLayout,
        on_host: Bool,
    ) raises:
        var table = PageTable[PAGE_SIZE].from_footprint(ctx, footprint, layout)
        self.latent = PagedBlocks[fp8](
            ctx, table, Q_DEPTH, seed=0, on_host=on_host
        )
        self.index_k = PagedBlocks[fp8](
            ctx, table, INDEX_DEPTH, seed=1, on_host=on_host
        )
        self.index_scales = PagedBlocks[DType.float32](
            ctx, table, 1, seed=2, on_host=on_host
        )
        self.table = table^

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        return (
            PageTable[PAGE_SIZE].bytes_for(footprint)
            + PagedBlocks[fp8].bytes_for[PAGE_SIZE](footprint, Q_DEPTH)
            + PagedBlocks[fp8].bytes_for[PAGE_SIZE](footprint, INDEX_DEPTH)
            + PagedBlocks[DType.float32].bytes_for[PAGE_SIZE](footprint, 1)
        )

    def bytes(self) -> Int:
        return (
            self.table.bytes()
            + self.latent.bytes()
            + self.index_k.bytes()
            + self.index_scales.bytes()
        )


struct Workspace(Movable):
    """Per-token buffers, allocated once for the largest config in the sweep.

    Q contents do not affect timing, so they are filled once; index lists
    depend on the shape and are rewritten per config.
    """

    var q_mla: DeviceBuffer[fp8]
    var out: DeviceBuffer[DType.bfloat16]
    var q_index: DeviceBuffer[fp8]
    var q_index_scales: DeviceBuffer[DType.float32]
    var row_offsets: DeviceBuffer[DType.uint32]
    var mla_indices: DeviceBuffer[DType.int32]
    var topk_lengths: DeviceBuffer[DType.uint32]
    var index_out: DeviceBuffer[DType.int32]
    var index_phys: DeviceBuffer[DType.int32]
    var sink: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        max_tokens: Int,
        max_batch: Int,
        max_heads: Int,
        index_slots: Int,
        on_host: Bool,
    ) raises:
        self.q_mla = ctx.enqueue_create_buffer[fp8](
            max_tokens * max_heads * Q_DEPTH
        )
        self.out = ctx.enqueue_create_buffer[DType.bfloat16](
            max_tokens * max_heads * V_DEPTH
        )
        self.q_index = ctx.enqueue_create_buffer[fp8](
            max_tokens * INDEX_HEADS * INDEX_DEPTH
        )
        self.q_index_scales = ctx.enqueue_create_buffer[DType.float32](
            max_tokens * INDEX_HEADS
        )
        self.row_offsets = ctx.enqueue_create_buffer[DType.uint32](
            max_batch + 1
        )
        self.mla_indices = ctx.enqueue_create_buffer[DType.int32](index_slots)
        self.topk_lengths = ctx.enqueue_create_buffer[DType.uint32](max_tokens)
        self.index_out = ctx.enqueue_create_buffer[DType.int32](
            max_tokens * TOP_K
        )
        self.index_phys = ctx.enqueue_create_buffer[DType.int32](
            max_tokens * TOP_K
        )
        self.sink = ctx.enqueue_create_buffer[DType.float32](max_heads)
        if on_host:
            fill_uniform_host_rand(self.q_mla)
            fill_uniform_host_rand(self.q_index)
            fill_uniform_host_rand(self.q_index_scales)
        else:
            fill_uniform(ctx, self.q_mla, 10)
            fill_uniform(ctx, self.q_index, 11)
            fill_uniform(ctx, self.q_index_scales, 12)
        ctx.enqueue_memset(self.topk_lengths, UInt32(TOP_K))
        # A -1e38 sink is a no-op on the output but keeps the prefill kernel
        # on the path the graph always takes.
        ctx.enqueue_memset(self.sink, Float32(-1.0e38))

    @staticmethod
    def bytes_for(footprint: Footprint, max_heads: Int) -> Int:
        var tokens = footprint["tokens"]
        var per_token = (
            max_heads * (Q_DEPTH + 2 * V_DEPTH)
            + INDEX_HEADS * (INDEX_DEPTH + 4)
            + 4 * (2 * TOP_K + 1)
        )
        return (
            tokens * per_token
            + 4 * footprint["index_slots"]
            + 4 * (footprint["batch"] + 1 + max_heads)
        )


def _name[
    h: Int
](kernel: StaticString, batch: Int, seq_len: Int, cache_len: Int) -> String:
    return String(
        "dsa_block/",
        kernel,
        " : h=",
        h,
        ", batch=",
        batch,
        ", seq_len=",
        seq_len,
        ", cache_len=",
        cache_len,
    )


def run_config[
    h: Int
](
    ctx: DeviceContext,
    mut sh: SweepHarness,
    mut cache: DSACache,
    mut ws: Workspace,
    batch: Int,
    seq_len: Int,
    cache_len: Int,
    on_host: Bool,
    indexer_copies: Int,
    mla_copies: Int,
    block_copies: Int,
) raises:
    var is_decode = seq_len <= DECODE_MAX_SEQ
    var tokens = batch * seq_len
    var num_keys = cache_len + seq_len
    var pages = ceildiv(num_keys, PAGE_SIZE)
    var slots = tokens * TOP_K

    ref table = cache.table
    sh.timer.reset()
    fill_affine(ctx, ws.row_offsets, seq_len, 0, batch + 1)
    table.set_cache_len(ctx, cache_len)
    table.write_copies(
        ctx, max(1, indexer_copies, mla_copies, block_copies), batch, pages
    )
    # The MLA index lists name physical rows, so each copy the MLA reads has
    # its own, through its own page table.
    for copy in range(max(1, mla_copies, block_copies)):
        fill_sparse_indices(
            ctx,
            DeviceBuffer[DType.int32](
                ctx,
                ws.mla_indices.unsafe_ptr().unsafe_offset(copy * slots),
                slots,
                owning=False,
            ),
            table.copy_lut(ctx, copy, batch, pages),
            pages,
            PAGE_SIZE,
            not is_decode,
            batch,
            seq_len,
            cache_len,
            TOP_K,
            on_host,
        )

    comptime index_kv_params = KVCacheStaticParams(
        num_heads=1, head_size=INDEX_DEPTH, is_mla=True
    )
    comptime IndexCollection = PagedKVCacheCollection[
        fp8,
        index_kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ]
    comptime mla_kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    comptime MLACollection = PagedKVCacheCollection[
        fp8,
        mla_kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]

    var cache_lengths = (
        TileTensor(table.cache_lengths, row_major(Int64(batch)))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var lut_base = table.lut.unsafe_ptr()
    var lut0 = (
        TileTensor(lut_base, row_major(Int64(batch), Int64(pages)))
        .as_imm()
        .as_unsafe_any_origin()
    )

    def lut_at(copy: Int) {imm} -> type_of(lut0):
        return (
            TileTensor(
                lut_base.unsafe_offset(copy * batch * pages),
                row_major(Int64(batch), Int64(pages)),
            )
            .as_imm()
            .as_unsafe_any_origin()
        )

    comptime ib_layout = IndexCollection.blocks_tt_layout
    var ib_shape = Coord[*ib_layout.shape_types]()
    ib_shape[0] = Int64(table.num_blocks)
    ib_shape[2] = Int64(1)
    var ib_strides = Coord[*ib_layout.stride_types]()
    ib_strides[1] = ib_shape[2] * Int64(ib_strides[2].value())
    ib_strides[0] = Int64(ib_shape[1].value()) * ib_strides[1]
    comptime is_layout = IndexCollection.scales_tt_layout
    var is_shape = Coord[*is_layout.shape_types]()
    is_shape[0] = Int64(table.num_blocks)
    is_shape[2] = Int64(1)
    var is_strides = Coord[*is_layout.stride_types]()
    is_strides[1] = is_shape[2] * Int64(is_strides[2].value())
    is_strides[0] = Int64(is_shape[1].value()) * is_strides[1]

    def index_collection_at(copy: Int) {mut cache, imm} -> IndexCollection:
        return IndexCollection(
            TileTensor(
                cache.index_k.buf, ib_layout(ib_shape, ib_strides)
            ).as_unsafe_any_origin(),
            cache_lengths,
            lut_at(copy),
            UInt32(seq_len),
            UInt32(cache_len),
            TileTensor(cache.index_scales.buf, is_layout(is_shape, is_strides))
            .bitcast[IndexCollection.scale_dtype]()
            .as_unsafe_any_origin(),
        )

    comptime mb_layout = MLACollection.blocks_tt_layout
    var mb_shape = Coord[*mb_layout.shape_types]()
    mb_shape[0] = Int64(table.num_blocks)
    mb_shape[2] = Int64(1)
    var mb_strides = Coord[*mb_layout.stride_types]()
    mb_strides[1] = mb_shape[2] * Int64(mb_strides[2].value())
    mb_strides[0] = Int64(mb_shape[1].value()) * mb_strides[1]

    def mla_cache_at(copy: Int) {mut cache, imm} -> MLACollection.CacheType:
        return MLACollection(
            TileTensor(
                cache.latent.buf, mb_layout(mb_shape, mb_strides)
            ).as_unsafe_any_origin(),
            cache_lengths,
            lut_at(copy),
            UInt32(seq_len),
            UInt32(cache_len if is_decode else num_keys),
        ).get_key_cache(0)

    var q_index = TileTensor(
        ws.q_index.unsafe_ptr(), row_major(tokens, INDEX_HEADS, INDEX_DEPTH)
    )
    var q_index_scales = TileTensor(
        ws.q_index_scales.unsafe_ptr(), row_major(tokens, INDEX_HEADS)
    )
    var row_offsets = TileTensor(
        ws.row_offsets.unsafe_ptr(), row_major(batch + 1)
    )
    var index_out = TileTensor(
        ws.index_out.unsafe_ptr(), row_major(tokens, TOP_K)
    )
    var index_phys = ws.index_phys.unsafe_ptr().unsafe_origin_cast[
        MutAnyOrigin
    ]()

    var q_mla = TileTensor(
        ws.q_mla.unsafe_ptr(), row_major((tokens, Idx[h], Idx[Q_DEPTH]))
    )
    var out_tt = TileTensor(
        ws.out.unsafe_ptr(), row_major((tokens, Idx[h], Idx[V_DEPTH]))
    )
    var mla_indices_ptr = ws.mla_indices.unsafe_ptr().unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var topk_lengths = TileTensor(
        ws.topk_lengths.unsafe_ptr(), row_major(tokens)
    )
    var sink = Optional[Pointer[Float32, ImmutAnyOrigin]](
        ws.sink.unsafe_ptr().as_unsafe_any_origin()
    )
    var mla_args = MLADispatchScalarArgs[num_heads=h, is_fp8_kv=True](
        batch, cache_len, seq_len, ctx
    )
    var mla_scalar_args = mla_args.gpu_tile_tensor()

    comptime cta_group = 2 if h == 128 else 1
    comptime prefill_config = MLASparseConfig[
        DType.bfloat16,
        b_topk_=128 if cta_group == 2 else 64,
        num_mbars_=2 if cta_group == 2 else 4,
        cta_group_=cta_group,
    ](
        num_q_heads=h,
        num_kv_heads=1,
        qk_depth=Q_DEPTH,
        input_qk_depth=Q_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=TOP_K,
        group=h,
    )
    ctx.synchronize()
    sh.timer.lap("config_setup")

    @inline(.always)
    def launch_indexer(
        launch_ctx: DeviceContext, copy: Int
    ) raises {mut index_out, imm}:
        var paged_remap = PagedKVRemap[PAGE_SIZE](
            index_phys,
            table.lut.unsafe_ptr().unsafe_offset(copy * batch * pages),
            pages,
            batch,
            UInt32(table.num_blocks),
        )
        mla_indexer_ragged_float8_paged[
            fp8,
            IndexCollection,
            INDEX_HEADS,
            INDEX_DEPTH,
            TOP_K,
            MaskName.CAUSAL.name,
            DType.float32,
            write_logical=False,
        ](
            index_out,
            q_index,
            q_index_scales,
            row_offsets,
            index_collection_at(copy),
            UInt32(0),
            launch_ctx,
            paged_remap,
        )

    @inline(.always)
    def launch_mla(
        launch_ctx: DeviceContext, copy: Int
    ) raises {mut out_tt, imm}:
        comptime if COMPILE_DECODE:
            if is_decode:
                flare_mla_decoding[
                    rank=3,
                    config=MHAConfig[fp8](h, Q_DEPTH),
                    ragged=True,
                    sparse=True,
                ](
                    out_tt,
                    q_mla,
                    mla_cache_at(copy),
                    CausalMask(),
                    row_offsets,
                    Float32(0.0625),
                    launch_ctx,
                    mla_scalar_args,
                    d_indices=mla_indices_ptr.unsafe_offset(copy * slots),
                    indices_stride=TOP_K,
                )
                return
        comptime if COMPILE_PREFILL:
            if not is_decode:
                mla_prefill_sparse_qkv_fp8[
                    config=prefill_config,
                    group=h,
                    q_depth=Q_DEPTH,
                    scale_block_size=0,
                ](
                    out_tt,
                    q_mla,
                    mla_cache_at(copy),
                    TileTensor(
                        mla_indices_ptr.unsafe_offset(
                            copy * slots
                        ).unsafe_bitcast[UInt32](),
                        row_major(slots),
                    ),
                    topk_lengths,
                    sink,
                    Float32(0.0625),
                    Int32(TOP_K),
                    launch_ctx,
                )
                return
        raise Error("seq_len ", seq_len, " was not compiled in")

    @inline(.always)
    def launch_block(launch_ctx: DeviceContext, copy: Int) raises {imm}:
        launch_indexer(launch_ctx, copy)
        launch_mla(launch_ctx, copy)

    sh.first_launch(ctx, launch_block)

    def fields(kernel: StaticString) {imm} -> Fields:
        var f = Fields()
        f.add("kernel", String(kernel))
        f.add("heads", h)
        f.add("batch", batch)
        f.add("seq_len", seq_len)
        f.add("cache_len", cache_len)
        return f^

    sh.bench(
        ctx,
        launch_indexer,
        BenchId(_name[h]("indexer", batch, seq_len, cache_len)),
        fields("indexer"),
        indexer_copies,
    )
    sh.bench(
        ctx,
        launch_mla,
        BenchId(_name[h]("mla", batch, seq_len, cache_len)),
        fields("mla"),
        mla_copies,
    )
    sh.bench(
        ctx,
        launch_block,
        BenchId(_name[h]("block", batch, seq_len, cache_len)),
        fields("block"),
        block_copies,
    )

    _ = mla_args^


def main() raises:
    var sh = SweepHarness("combined")

    var default_seqs = String("6,8192") if SWEEP_SEQ == 0 else String(SWEEP_SEQ)
    var sweep = Sweep()
    sweep.axis("seq_len", default_seqs)
    sweep.axis("cache_len", "32768,0")
    var decode_batch = arg_parse("decode_batch", 8)
    var prefill_batch = arg_parse("prefill_batch", 1)
    var fill = String(arg_parse("fill", "gpu"))
    var layout = PageLayout.parse(arg_parse("page_layout", "identity"))
    var on_host = fill == "host"

    @inline(.always)
    def batch_of(cfg: Config) raises {imm} -> Int:
        return decode_batch if cfg["seq_len"] <= DECODE_MAX_SEQ else (
            prefill_batch
        )

    sh.timer.lap("main_prologue")
    with DeviceContext() as ctx:
        sh.timer.lap("device_init")
        sh.start(ctx)
        var rot = sh.rotation()

        # Each benchmark spans the cache with its own reads: the indexer
        # scans every index key, the MLA gathers its top-k rows of the
        # latent, and the block does both. One more copy allocates its pages
        # whole, for both caches, and the MLA's index list.
        def copies(cfg: Config, kernel: StaticString) raises {imm} -> Int:
            var batch = batch_of(cfg)
            var keys = cfg["cache_len"] + cfg["seq_len"]
            var index_bytes = batch * keys * (INDEX_DEPTH + 4)
            var mla_keys = keys if cfg["seq_len"] > DECODE_MAX_SEQ else min(
                keys, TOP_K + cfg["seq_len"]
            )
            var mla_bytes = batch * mla_keys * Q_DEPTH
            var page_bytes = (
                batch
                * ceildiv(keys, PAGE_SIZE)
                * PAGE_SIZE
                * (Q_DEPTH + INDEX_DEPTH + 4)
            )
            var list_bytes = 4 * batch * cfg["seq_len"] * TOP_K
            if kernel == "indexer":
                return rot.copies(index_bytes, page_bytes)
            if kernel == "mla":
                return rot.copies(mla_bytes, page_bytes + list_bytes)
            return rot.copies(index_bytes + mla_bytes, page_bytes + list_bytes)

        def most_copies(cfg: Config) raises {imm} -> Int:
            return max(
                1,
                copies(cfg, "indexer"),
                copies(cfg, "mla"),
                copies(cfg, "block"),
            )

        def require(cfg: Config, mut fp: Footprint) raises {imm}:
            var batch = batch_of(cfg)
            var tokens = batch * cfg["seq_len"]
            PageTable[PAGE_SIZE].require(
                fp, batch, cfg["cache_len"] + cfg["seq_len"], most_copies(cfg)
            )
            fp.need("tokens", tokens)
            var lists = max(1, copies(cfg, "mla"), copies(cfg, "block"))
            fp.need("index_slots", lists * tokens * TOP_K)

        def bytes(fp: Footprint) raises -> Int:
            return DSACache.bytes_for(fp) + Workspace.bytes_for(fp, MAX_HEADS)

        var groups = plan_groups(
            sweep.configs(), require, bytes, device_budget(ctx)
        )
        for ref group in groups:
            ref fp = group.footprint
            var cache = DSACache(ctx, fp, layout, on_host)
            ctx.synchronize()
            sh.timer.lap("fixture_fill")
            var ws = Workspace(
                ctx,
                fp["tokens"],
                fp["batch"],
                MAX_HEADS,
                fp["index_slots"],
                on_host,
            )
            ctx.synchronize()
            sh.timer.lap("workspace_fill")
            var info = Fields()
            info.add("fixture_bytes", cache.bytes())
            info.add("planned_bytes", group.bytes)
            info.extend(fp)
            info.add("configs", len(group.configs))
            info.add("fill", fill)
            info.add("page_layout", String(layout))
            sh.note(info^)
            for ref cfg in group.configs:
                comptime for h in [8, 64]:
                    comptime if arm_enabled["sweep_h", h]():
                        run_config[h](
                            ctx,
                            sh,
                            cache,
                            ws,
                            batch_of(cfg),
                            cfg["seq_len"],
                            cfg["cache_len"],
                            on_host,
                            copies(cfg, "indexer"),
                            copies(cfg, "mla"),
                            copies(cfg, "block"),
                        )
            sh.timer.reset()
        sh.stop(ctx)
    sh.timer.lap("device_teardown")
    sh.finish()
