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
"""Native-FP8 sparse MLA prefill over a paged FP8 latent cache, as one sweep.

The sweep port of the deleted `gpu/nn/bench_mla_prefill_sparse_qkv_fp8.mojo`.
`mla_prefill_sparse_qkv_fp8` is what GLM-5.2 / GLM-5.3 prefill runs on B200
with the FP8 latent cache: FP8 Q, K and V, QK^T and P*V in
`tcgen05.mma.kind::f8f6f4`. The tile config below is the one `mla_graph.mojo`
dispatches, so keep the two in sync.

Index lists mirror the graph: `topk_lengths` is always `top_k`, and a query at
position `p` with fewer than `top_k` visible keys pads its list with -1. The
measured region is the attention kernel alone, not the BF16-to-FP8 Q
conversion or the W_UV projection around it.

Runtime axes are `--batch_size`, `--seq_len` and `--cache_len`, each a comma
list; the defaults are the single-prompt group of the old YAML. `num_heads`,
`page_size` and `top_k` are `-D` defines, as before. `--index_stride`, as
before, fixes the gather stride where 0 spreads the picks over the window.
The measurement flags (`--target_time`, `--l2`, `--throttle_check`,
`--json`) are `SweepHarness`'s.
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
    device_budget,
    fill_sparse_indices,
    fill_uniform,
    fill_uniform_host_rand,
    plan_groups,
)
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from std.benchmark import BenchId, BenchMetric, ThroughputMeasure

comptime fp8 = DType.float8_e4m3fn
comptime KV_LATENT_DIM = 512
comptime ROPE_DEPTH = 64
comptime Q_DEPTH = KV_LATENT_DIM + ROPE_DEPTH
comptime V_DEPTH = KV_LATENT_DIM
# 64 attention heads / TP=8, the GLM-5.2 and GLM-5.3 B200 recipes.
comptime NUM_HEADS = get_defined_int["num_heads", 8]()
comptime PAGE_SIZE = get_defined_int["page_size", 128]()
comptime TOP_K = get_defined_int["top_k", 2048]()


struct Workspace(Movable):
    """Per-token buffers, sized for the largest config of a sweep group."""

    var q: DeviceBuffer[fp8]
    var out: DeviceBuffer[DType.bfloat16]
    var indices: DeviceBuffer[DType.int32]
    var topk_lengths: DeviceBuffer[DType.uint32]
    var sink: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, footprint: Footprint, on_host: Bool
    ) raises:
        var tokens = footprint["tokens"]
        self.q = ctx.enqueue_create_buffer[fp8](tokens * NUM_HEADS * Q_DEPTH)
        self.out = ctx.enqueue_create_buffer[DType.bfloat16](
            tokens * NUM_HEADS * V_DEPTH
        )
        self.indices = ctx.enqueue_create_buffer[DType.int32](
            footprint["index_slots"]
        )
        self.topk_lengths = ctx.enqueue_create_buffer[DType.uint32](tokens)
        self.sink = ctx.enqueue_create_buffer[DType.float32](NUM_HEADS)
        if on_host:
            fill_uniform_host_rand(self.q)
        else:
            fill_uniform(ctx, self.q, 10)
        ctx.enqueue_memset(self.topk_lengths, UInt32(TOP_K))
        # The graph always passes a sink; -1e38 makes it a no-op on the output
        # but keeps the kernel on the same code path.
        ctx.enqueue_memset(self.sink, Float32(-1.0e38))

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        return (
            footprint["tokens"] * (NUM_HEADS * (Q_DEPTH + 2 * V_DEPTH) + 4)
            + 4 * footprint["index_slots"]
            + 4 * NUM_HEADS
        )


def _run_name(
    batch_size: Int, seq_len: Int, cache_len: Int, index_stride: Int
) -> String:
    # fmt: off
    var name = String(
        "mla_prefill_sparse_qkv_fp8 : ",
        "num_heads=", NUM_HEADS, ", ",
        "page_size=", PAGE_SIZE, ", ",
        "top_k=", TOP_K, " : ",
        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "cache_len=", cache_len,
    )
    # fmt: on
    if index_stride > 0:
        name += String(", index_stride=", index_stride)
    return name


def _num_valid_pairs(batch_size: Int, seq_len: Int, cache_len: Int) -> Int:
    """Returns the (query, key) pairs that are not -1 padding."""
    var per_seq = 0
    for s in range(seq_len):
        per_seq += min(cache_len + s + 1, TOP_K)
    return batch_size * per_seq


def run_config(
    ctx: DeviceContext,
    mut h: SweepHarness,
    mut table: PageTable[PAGE_SIZE],
    mut latent: PagedBlocks[fp8],
    mut ws: Workspace,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    index_stride: Int,
    on_host: Bool,
    copies: Int,
) raises:
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    comptime Collection = PagedKVCacheCollection[
        fp8,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var tokens = batch_size * seq_len
    var num_keys = cache_len + seq_len
    var pages = ceildiv(num_keys, PAGE_SIZE)
    var slots = tokens * TOP_K

    h.timer.reset()
    table.set_cache_len(ctx, cache_len)
    table.write_copies(ctx, max(copies, 1), batch_size, pages)
    # The index lists name physical rows, so each copy has its own, through
    # its own page table.
    for copy in range(max(copies, 1)):
        fill_sparse_indices(
            ctx,
            DeviceBuffer[DType.int32](
                ctx,
                ws.indices.unsafe_ptr().unsafe_offset(copy * slots),
                slots,
                owning=False,
            ),
            table.copy_lut(ctx, copy, batch_size, pages),
            pages,
            PAGE_SIZE,
            True,
            batch_size,
            seq_len,
            cache_len,
            TOP_K,
            on_host,
            index_stride,
        )

    comptime blocks_layout = Collection.blocks_tt_layout
    var blocks_shape = Coord[*blocks_layout.shape_types]()
    blocks_shape[0] = Int64(table.num_blocks)
    blocks_shape[2] = Int64(1)
    var blocks_strides = Coord[*blocks_layout.stride_types]()
    blocks_strides[1] = blocks_shape[2] * Int64(blocks_strides[2].value())
    blocks_strides[0] = Int64(blocks_shape[1].value()) * blocks_strides[1]

    def cache_at(
        launch_ctx: DeviceContext, copy: Int
    ) raises {mut latent, imm} -> Collection.CacheType:
        var lut = table.copy_lut(launch_ctx, copy, batch_size, pages)
        var kv_collection = Collection(
            TileTensor(
                latent.buf, blocks_layout(blocks_shape, blocks_strides)
            ).as_unsafe_any_origin(),
            TileTensor(table.cache_lengths, row_major(Int64(batch_size)))
            .as_imm()
            .as_unsafe_any_origin(),
            TileTensor(lut, row_major(Int64(batch_size), Int64(pages)))
            .as_imm()
            .as_unsafe_any_origin(),
            UInt32(seq_len),
            UInt32(num_keys),
        )
        return kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        ws.q.unsafe_ptr(), row_major((tokens, Idx[NUM_HEADS], Idx[Q_DEPTH]))
    )
    var out_tt = TileTensor(
        ws.out.unsafe_ptr(), row_major((tokens, Idx[NUM_HEADS], Idx[V_DEPTH]))
    )
    # int32 slots reinterpreted as uint32, as the graph does: -1 becomes
    # 0xFFFFFFFF, which the gather producer rejects.
    var indices_ptr = ws.indices.unsafe_ptr().unsafe_bitcast[UInt32]()
    var topk_lengths_tt = TileTensor(
        ws.topk_lengths.unsafe_ptr(), row_major(tokens)
    )
    var attn_sink = Optional[Pointer[Float32, ImmutAnyOrigin]](
        ws.sink.unsafe_ptr().as_unsafe_any_origin()
    )

    # Same tactic selection as the sparse prefill branches in mla_graph.mojo.
    comptime cta_group = 2 if NUM_HEADS == 128 else 1
    comptime config = MLASparseConfig[
        DType.bfloat16,
        b_topk_=128 if cta_group == 2 else 64,
        num_mbars_=2 if cta_group == 2 else 4,
        cta_group_=cta_group,
    ](
        num_q_heads=NUM_HEADS,
        num_kv_heads=1,
        qk_depth=Q_DEPTH,
        input_qk_depth=Q_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=TOP_K,
        group=NUM_HEADS,
    )
    ctx.synchronize()
    h.timer.lap("config_setup")

    @inline(.always)
    def kernel_launch(
        launch_ctx: DeviceContext, copy: Int
    ) raises {mut out_tt, imm}:
        var indices_tt = TileTensor(
            indices_ptr.unsafe_offset(copy * slots), row_major(slots)
        )
        mla_prefill_sparse_qkv_fp8[
            config=config,
            group=NUM_HEADS,
            q_depth=Q_DEPTH,
            scale_block_size=0,
        ](
            out_tt,
            q_tt,
            cache_at(launch_ctx, copy),
            indices_tt,
            topk_lengths_tt,
            attn_sink,
            Float32(0.0625),
            Int32(TOP_K),
            launch_ctx,
        )

    h.first_launch(ctx, kernel_launch)
    # Counts only the pairs that are not -1 padding, so the early-context
    # positions of a fresh prompt do not inflate the rate.
    var flops = (
        2
        * _num_valid_pairs(batch_size, seq_len, cache_len)
        * NUM_HEADS
        * (Q_DEPTH + V_DEPTH)
    )
    var fields = Fields()
    fields.add("kernel", String("mla_prefill_sparse_qkv_fp8"))
    fields.add("num_heads", NUM_HEADS)
    fields.add("page_size", PAGE_SIZE)
    fields.add("top_k", TOP_K)
    fields.add("batch_size", batch_size)
    fields.add("seq_len", seq_len)
    fields.add("cache_len", cache_len)
    fields.add("index_stride", index_stride)
    h.bench(
        ctx,
        kernel_launch,
        BenchId(_run_name(batch_size, seq_len, cache_len, index_stride)),
        fields^,
        copies,
        [ThroughputMeasure(BenchMetric.flops, flops)],
    )


def main() raises:
    var h = SweepHarness("prefill_sparse")

    var sweep = Sweep()
    sweep.axis("batch_size", "1")
    # One prefill chunk per recipe: max_batch_input_tokens is 2048 for
    # GLM-5.3 FP8 and 8192 for GLM-5.2 / 5.3 NVFP4.
    sweep.axis("seq_len", "2048,8192")
    sweep.axis("cache_len", "0,8192,65536,131072")
    var index_stride = arg_parse("index_stride", 0)
    var fill = String(arg_parse("fill", "gpu"))
    var layout = PageLayout.parse(arg_parse("page_layout", "identity"))
    var on_host = fill == "host"

    h.timer.lap("main_prologue")
    with DeviceContext() as ctx:
        h.timer.lap("device_init")
        h.start(ctx)
        var rot = h.rotation()

        # Picks spread over each position's whole causal window, so a launch
        # reads about every key once.
        # One more copy allocates its pages whole, and its index list.
        def copies(cfg: Config) raises {imm} -> Int:
            var batch = cfg["batch_size"]
            var keys = cfg["cache_len"] + cfg["seq_len"]
            var copy_bytes = (
                batch * ceildiv(keys, PAGE_SIZE) * PAGE_SIZE * Q_DEPTH
                + 4 * batch * cfg["seq_len"] * TOP_K
            )
            return rot.copies(batch * keys * Q_DEPTH, copy_bytes)

        def require(cfg: Config, mut fp: Footprint) raises {imm}:
            var tokens = cfg["batch_size"] * cfg["seq_len"]
            PageTable[PAGE_SIZE].require(
                fp,
                cfg["batch_size"],
                cfg["cache_len"] + cfg["seq_len"],
                copies(cfg),
            )
            fp.need("tokens", tokens)
            fp.need("index_slots", max(1, copies(cfg)) * tokens * TOP_K)

        def bytes(fp: Footprint) raises -> Int:
            return (
                PageTable[PAGE_SIZE].bytes_for(fp)
                + PagedBlocks[fp8].bytes_for[PAGE_SIZE](fp, Q_DEPTH)
                + Workspace.bytes_for(fp)
            )

        var groups = plan_groups(
            sweep.configs(), require, bytes, device_budget(ctx)
        )
        for ref group in groups:
            ref fp = group.footprint
            var table = PageTable[PAGE_SIZE].from_footprint(ctx, fp, layout)
            var latent = PagedBlocks[fp8](
                ctx, table, Q_DEPTH, seed=0, on_host=on_host
            )
            ctx.synchronize()
            h.timer.lap("fixture_fill")
            var ws = Workspace(ctx, fp, on_host)
            ctx.synchronize()
            h.timer.lap("workspace_fill")
            var info = Fields()
            info.add("fixture_bytes", table.bytes() + latent.bytes())
            info.add("planned_bytes", group.bytes)
            info.extend(fp)
            info.add("configs", len(group.configs))
            info.add("fill", fill)
            info.add("page_layout", String(layout))
            h.note(info^)
            for ref cfg in group.configs:
                run_config(
                    ctx,
                    h,
                    table,
                    latent,
                    ws,
                    cfg["batch_size"],
                    cfg["seq_len"],
                    cfg["cache_len"],
                    index_stride,
                    on_host,
                    copies(cfg),
                )
            h.timer.reset()
        h.stop(ctx)
    h.timer.lap("device_teardown")
    h.finish()
