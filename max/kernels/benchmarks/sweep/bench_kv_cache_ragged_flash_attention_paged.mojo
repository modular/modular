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
"""Ragged paged flash attention (MHA / GQA, `flash_attention[ragged=True]`)
as one sweep.

The sweep port of the deleted
`gpu/nn/bench_kv_cache_ragged_flash_attention_paged.mojo`, which timed the
fused QKV ragged attention over a paged KV cache at the llama 3.1 / 3.2 /
3.3-70B serving shapes. `--batch_size`, `--seq_len` and
`--cache_len` are runtime axes, each a comma list; the defaults cross the old
YAML's decode group (`seq_len` 1 over a warm cache) with its prefill group
(`seq_len` 256). The `-D` defines are the old benchmark's: `dtype`,
`head_dim`, `num_q_heads`, `num_kv_heads`, `page_size`, `local_window_size`,
`cross_attention` and `sink`.

`--use_random_seq_lengths`, `--use_random_cache_lengths`, `--max_cache_len`,
`--num_partitions`, `--run_benchmark` and `--verify` keep their old meanings.
The random-length flags still draw on the host, as the old benchmark did:
lengths feed a prefix sum the generator set does not cover yet, and the YAML
never runs them. `--verify` prints the old gross-error digest, taken after
the first launch rather than after the benchmark loop, because a rotating
launch reads a different copy of the cache each iteration.

Inputs are device hash fills by default; `--fill=host` restores
`std.random.rand`. `--page_layout` defaults to `shuffled`, whose scatter is
comparable to the old benchmark's random page draw; `identity` flatters the
cache. The page table the kernel sees is a per-config view whose row stride
is padded for `PagedKVCache.populate`'s SIMD LUT loads, as production
allocates and the old benchmark hand-rolled. The measurement flags
(`--target_time`, `--l2`, `--throttle_check`, `--json`) are
`SweepHarness`'s.
"""

from std.math import ceildiv, isinf, isnan, rsqrt, sqrt
from std.random import random_ui64, seed
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    size_of,
)

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
    fill_affine,
    fill_lut_view,
    fill_uniform,
    fill_uniform_host_rand,
    plan_groups,
)
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask, SlidingWindowCausalMask
from std.benchmark import BenchId, BenchMetric, ThroughputMeasure

comptime dtype = get_defined_dtype["dtype", .bfloat16]()
comptime HEAD_DIM = get_defined_int["head_dim", 128]()
comptime NUM_Q_HEADS = get_defined_int["num_q_heads", 32]()
comptime NUM_KV_HEADS = get_defined_int["num_kv_heads", 8]()
comptime PAGE_SIZE = get_defined_int["page_size", 256]()
comptime LOCAL_WINDOW_SIZE = get_defined_int["local_window_size", -1]()
comptime CROSS_ATTENTION = get_defined_bool["cross_attention", False]()
comptime SINK = get_defined_bool["sink", False]()

# K and V share one block tensor, in disjoint per-page regions.
comptime KV_ROW_ELEMS = 2 * NUM_KV_HEADS * HEAD_DIM
comptime DTYPE_SIZE = size_of[dtype]()


def flops(
    batch: Int, nheads: Int, seqlen_q: Int, seqlen_k: Int, headdim: Int
) raises -> Int:
    var avg_seqlen = Float64(max(seqlen_k - seqlen_q, 0) + seqlen_k) / 2
    return Int(
        Float64(batch * nheads * 2 * seqlen_q)
        * avg_seqlen
        * Float64((headdim + headdim))
    )


def _p_opt(p: Int) -> Optional[Int]:
    # 0 == auto (the `mha.mojo` decode heuristic), >= 1 pins the count.
    return p if p > 0 else Optional[Int]()


def _padded_lut_cols(pages: Int) -> Int:
    # `PagedKVCache.populate` SIMD-loads LUT rows in chunks up to 8 wide, so
    # the row stride must be 8-aligned and a max-width load at any valid
    # `first_lut_idx` must stay in-bounds. Mirrors `_padded_lut_cols`
    # (cache_manager.py) / `padded_lut_cols` (kv_cache_test_utils.mojo).
    return (pages + 7) // 8 * 8 + 16


def _run_name(
    batch_size: Int,
    seq_len: Int,
    use_random_seq_lengths: Bool,
    cache_len: Int,
    use_random_cache_lengths: Bool,
    max_cache_len: Int,
) -> String:
    # The old benchmark's name, verbatim, so kdiff joins old and new results.
    return String(
        "fused_qkv_ragged_flash_attention",
        "(",
        dtype,
        ") : ",
        # head_info
        "num_q_heads=",
        NUM_Q_HEADS,
        ", num_kv_heads=",
        NUM_KV_HEADS,
        ", head_dim=",
        HEAD_DIM,
        " : ",
        "batch_size=",
        batch_size,
        ", seq_len=",
        seq_len,
        ", use_random_seq_lengths=",
        use_random_seq_lengths,
        ", cache_len=",
        cache_len,
        ", use_random_cache_lengths=",
        use_random_cache_lengths,
        ", max_cache_len=",
        max_cache_len,
        ", cross_attention=",
        CROSS_ATTENTION,
        ", sink=",
        SINK,
    )


def _digest(
    ctx: DeviceContext, output: DeviceBuffer[dtype, ...], n: Int
) raises:
    # A GROSS-ERROR detector -- wrong keys, an unmasked tail, NaN -- not an
    # oracle: 8 samples cannot localize a defect. It is sensitive only
    # because the fills above are zero-mean when `--verify` is set; on raw
    # uniform[0,1) it provably could not fail. Absolute correctness stays
    # with `test_mha_sm100_ws_shared_key.mojo`.
    ctx.synchronize()
    var digest_host = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.enqueue_copy(digest_host, output)
    ctx.synchronize()

    var nan_count = 0
    var inf_count = 0
    var sum_sq = Float64(0)
    var sum_abs = Float64(0)
    var max_abs = Float64(0)
    for i in range(n):
        var v = digest_host[i].cast[DType.float64]()
        if isnan(v):
            nan_count += 1
            continue
        if isinf(v):
            inf_count += 1
            continue
        var a = abs(v)
        sum_sq += v * v
        sum_abs += a
        max_abs = max(max_abs, a)

    # `rms` divides by the full element count on purpose: a NaN/Inf run must
    # DEPRESS rms rather than be silently excluded from the denominator, so a
    # partially poisoned output cannot pass the ratio bar by shrinking its
    # own reference.
    print(
        "[digest] n=",
        n,
        " nan=",
        nan_count,
        " inf=",
        inf_count,
        " rms=",
        sqrt(sum_sq / Float64(n)),
        " maxabs=",
        max_abs,
        " meanabs=",
        sum_abs / Float64(n),
        sep="",
    )

    # Eight fixed, shape-derived indices, so two builds of this benchmark are
    # positionally matched without either knowing about the other.
    var stride = max(1, n // 8)
    var samples = String("[digest_samples]")
    for s in range(8):
        var idx = min(n - 1, s * stride + (13 % stride))
        samples += String(
            " i",
            s,
            "=",
            idx,
            " v",
            s,
            "=",
            digest_host[idx].cast[DType.float64](),
        )
    print(samples)


struct MHACache(Movable):
    """K and V in one paged block tensor, sharing one page table.

    `--verify` centers the fill on [-0.5, 0.5), so the digest can see which
    keys were attended; timing is data-independent, so other runs fill
    [0, 1).
    """

    var table: PageTable[PAGE_SIZE]
    var kv: PagedBlocks[dtype]

    def __init__(
        out self,
        ctx: DeviceContext,
        footprint: Footprint,
        layout: PageLayout,
        on_host: Bool,
        verify: Bool,
    ) raises:
        var table = PageTable[PAGE_SIZE].from_footprint(ctx, footprint, layout)
        var lo = Float32(-0.5) if verify else Float32(0.0)
        var hi = Float32(0.5) if verify else Float32(1.0)
        self.kv = PagedBlocks[dtype](
            ctx, table, KV_ROW_ELEMS, seed=0, lo=lo, hi=hi, on_host=on_host
        )
        self.table = table^

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        return PageTable[PAGE_SIZE].bytes_for(footprint) + PagedBlocks[
            dtype
        ].bytes_for[PAGE_SIZE](footprint, KV_ROW_ELEMS)

    def bytes(self) -> Int:
        return self.table.bytes() + self.kv.bytes()


struct Workspace(Movable):
    """Per-token buffers and the padded LUT views, sized per sweep group.

    Q contents do not affect timing, so Q is filled once; row offsets, cache
    lengths and the LUT views depend on the config and are rewritten per
    config. The cross-attention and sink buffers exist in every build, as in
    the old benchmark, but only the comptime arm that reads them fills them.
    """

    var q: DeviceBuffer[dtype]
    var out: DeviceBuffer[dtype]
    var row_offsets: DeviceBuffer[DType.uint32]
    var kv_row_offsets: DeviceBuffer[DType.uint32]
    var sink_weights: DeviceBuffer[dtype]
    var lut: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        ctx: DeviceContext,
        footprint: Footprint,
        on_host: Bool,
        verify: Bool,
    ) raises:
        var tokens = footprint["tokens"]
        var lo = Float32(-0.5) if verify else Float32(0.0)
        var hi = Float32(0.5) if verify else Float32(1.0)
        self.q = ctx.enqueue_create_buffer[dtype](
            tokens * NUM_Q_HEADS * HEAD_DIM
        )
        self.out = ctx.enqueue_create_buffer[dtype](
            tokens * NUM_Q_HEADS * HEAD_DIM
        )
        self.row_offsets = ctx.enqueue_create_buffer[DType.uint32](
            footprint["batch"] + 1
        )
        self.kv_row_offsets = ctx.enqueue_create_buffer[DType.uint32](
            footprint["batch"] + 1
        )
        self.sink_weights = ctx.enqueue_create_buffer[dtype](NUM_Q_HEADS)
        self.lut = ctx.enqueue_create_buffer[DType.uint32](
            footprint["lut_entries"]
        )
        if on_host:
            fill_uniform_host_rand(self.q, Float64(lo), Float64(hi))
        else:
            fill_uniform(ctx, self.q, 10, lo, hi)
        comptime if SINK:
            # A small sink weight gives the seeded `(max_vec, norm_vec)`
            # state a non-trivial sink contribution.
            ctx.enqueue_memset(self.sink_weights, Scalar[dtype](0.05))

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        return (
            footprint["tokens"] * NUM_Q_HEADS * HEAD_DIM * 2 * DTYPE_SIZE
            + 8 * (footprint["batch"] + 1)
            + NUM_Q_HEADS * DTYPE_SIZE
            + 4 * footprint["lut_entries"]
        )


def run_config(
    ctx: DeviceContext,
    mut h: SweepHarness,
    mut cache: MHACache,
    mut ws: Workspace,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    max_cache_len: Int,
    num_partitions: Int,
    use_random_seq_lengths: Bool,
    use_random_cache_lengths: Bool,
    run_benchmark: Bool,
    verify: Bool,
    copies: Int,
) raises:
    comptime kv_params = KVCacheStaticParams(
        num_heads=NUM_KV_HEADS, head_size=HEAD_DIM
    )
    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime num_layers = 1
    comptime layer_idx = 0
    var batch = batch_size
    var tokens = batch * seq_len
    var max_seq_length: UInt32 = UInt32(seq_len)
    var max_context = cache_len + seq_len

    ref table = cache.table
    h.timer.reset()
    if use_random_seq_lengths or use_random_cache_lengths:
        # Host draws, as the old benchmark: lengths feed a prefix sum
        # (row offsets) that the generator set does not cover yet.
        var offsets_host = List(length=batch + 1, fill=UInt32(0))
        var lengths_host = List(length=batch, fill=UInt32(0))
        var total: UInt32 = 0
        for i in range(batch):
            var s: UInt32
            if use_random_seq_lengths:
                s = random_ui64(1, UInt64(seq_len)).cast[DType.uint32]()
            else:
                s = UInt32(seq_len)
            var c: UInt32
            if use_random_cache_lengths:
                c = random_ui64(0, UInt64(cache_len)).cast[DType.uint32]()
            else:
                c = UInt32(cache_len)
            offsets_host[i] = total
            lengths_host[i] = c
            total += s
            max_seq_length = max(max_seq_length, s)
            max_context = max(max_context, Int(c) + Int(s))
        offsets_host[batch] = total
        tokens = Int(total)
        ctx.enqueue_copy(ws.row_offsets, offsets_host)
        ctx.enqueue_copy(table.cache_lengths, lengths_host)
        comptime if CROSS_ATTENTION:
            ctx.enqueue_copy(ws.kv_row_offsets, offsets_host)
    else:
        fill_affine(ctx, ws.row_offsets, seq_len, 0, batch + 1)
        table.set_cache_len(ctx, cache_len)
        comptime if CROSS_ATTENTION:
            fill_affine(ctx, ws.kv_row_offsets, seq_len, 0, batch + 1)

    # The LUT width a captured graph's frozen metadata implies, and the
    # collection's max context with it.
    var lut_context = max_context
    if max_cache_len > 0:
        lut_context = max(lut_context, max_cache_len + seq_len)
    var real_pages = ceildiv(cache_len + seq_len, PAGE_SIZE)
    var lut_cols = _padded_lut_cols(ceildiv(lut_context, PAGE_SIZE))

    # Copies rotate the cache: each reads its own pages through its own
    # padded table view; columns past the real pages point at the null block.
    table.write_copies(ctx, max(copies, 1), batch, real_pages)
    var view_entries = batch * lut_cols
    for copy in range(max(copies, 1)):
        fill_lut_view(
            ctx,
            DeviceBuffer[DType.uint32](
                ctx,
                ws.lut.unsafe_ptr().unsafe_offset(copy * view_entries),
                view_entries,
                owning=False,
            ),
            table.copy_lut(ctx, copy, batch, real_pages),
            real_pages,
            batch,
            lut_cols,
            real_pages,
        )

    comptime blocks_layout = Collection.blocks_tt_layout
    var blocks_shape = Coord[*blocks_layout.shape_types]()
    blocks_shape[0] = Int64(table.num_blocks)
    blocks_shape[2] = Int64(num_layers)
    var blocks_strides = Coord[*blocks_layout.stride_types]()
    blocks_strides[1] = blocks_shape[2] * Int64(blocks_strides[2].value())
    blocks_strides[0] = Int64(blocks_shape[1].value()) * blocks_strides[1]
    var lut_base = ws.lut.unsafe_ptr()

    def kv_at(copy: Int) raises {mut cache, imm} -> Collection:
        return Collection(
            TileTensor(
                cache.kv.buf, blocks_layout(blocks_shape, blocks_strides)
            ).as_unsafe_any_origin(),
            TileTensor(table.cache_lengths, row_major(Int64(batch)))
            .as_imm()
            .as_unsafe_any_origin(),
            TileTensor(
                lut_base.unsafe_offset(copy * view_entries),
                row_major(Int64(batch), Int64(lut_cols)),
            )
            .as_imm()
            .as_unsafe_any_origin(),
            max_seq_length,
            UInt32(lut_context),
        )

    var q_tt = TileTensor(
        ws.q.unsafe_ptr(), row_major((tokens, Idx[NUM_Q_HEADS], Idx[HEAD_DIM]))
    )
    var out_tt = TileTensor(
        ws.out.unsafe_ptr(),
        row_major((tokens, Idx[NUM_Q_HEADS], Idx[HEAD_DIM])),
    )
    var row_offsets_tt = TileTensor(
        ws.row_offsets.unsafe_ptr(), row_major(batch + 1)
    )
    # Cross-attention uses independent KV-side row offsets. The bench sets
    # them equal to the Q-side so the dispatcher uses the cross-attention
    # launcher (`mha_prefill_v2_ragged[cross_attention=True]`) but
    # `num_keys` derives identically; this measures the comptime-
    # monomorphized path, not a true encoder-decoder shape.
    var kv_row_offsets_view = (
        TileTensor(ws.kv_row_offsets.unsafe_ptr(), row_major(Int64(batch + 1)))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var sink_weights_view = (
        TileTensor(ws.sink_weights.unsafe_ptr(), row_major(Int64(NUM_Q_HEADS)))
        .as_imm()
        .as_unsafe_any_origin()
    )
    ctx.synchronize()
    h.timer.lap("config_setup")

    @inline(.always)
    def kernel_launch(
        launch_ctx: DeviceContext, copy: Int
    ) raises {mut out_tt, imm}:
        var kv_collection = kv_at(copy)
        var k_cache = kv_collection.get_key_cache(layer_idx)
        var v_cache = kv_collection.get_value_cache(layer_idx)
        comptime if LOCAL_WINDOW_SIZE > 0:
            comptime assert (
                not SINK
            ), "sliding window mask does not support sink"
            comptime assert (
                not CROSS_ATTENTION
            ), "sliding window mask does not support cross_attention"
            flash_attention[ragged=True](
                out_tt,
                q_tt,
                k_cache,
                v_cache,
                SlidingWindowCausalMask[LOCAL_WINDOW_SIZE](),
                row_offsets_tt,
                rsqrt(Float32(HEAD_DIM)),
                launch_ctx,
                num_partitions=_p_opt(num_partitions),
            )
        else:
            # Sink / cross_attention dispatch: passing `sink_weights=...` to
            # `flash_attention[sink=True]` selects the dispatcher's
            # `comptime if sink:` branch; passing `kv_input_row_offsets=...`
            # selects the runtime `if kv_input_row_offsets:` branch.
            comptime if SINK and CROSS_ATTENTION:
                flash_attention[ragged=True, sink=True](
                    out_tt,
                    q_tt,
                    k_cache,
                    v_cache,
                    CausalMask(),
                    row_offsets_tt,
                    rsqrt(Float32(HEAD_DIM)),
                    launch_ctx,
                    kv_input_row_offsets=kv_row_offsets_view,
                    num_partitions=_p_opt(num_partitions),
                    sink_weights=sink_weights_view,
                )
            elif SINK:
                flash_attention[ragged=True, sink=True](
                    out_tt,
                    q_tt,
                    k_cache,
                    v_cache,
                    CausalMask(),
                    row_offsets_tt,
                    rsqrt(Float32(HEAD_DIM)),
                    launch_ctx,
                    num_partitions=_p_opt(num_partitions),
                    sink_weights=sink_weights_view,
                )
            elif CROSS_ATTENTION:
                flash_attention[ragged=True](
                    out_tt,
                    q_tt,
                    k_cache,
                    v_cache,
                    CausalMask(),
                    row_offsets_tt,
                    rsqrt(Float32(HEAD_DIM)),
                    launch_ctx,
                    kv_input_row_offsets=kv_row_offsets_view,
                    num_partitions=_p_opt(num_partitions),
                )
            else:
                flash_attention[ragged=True](
                    out_tt,
                    q_tt,
                    k_cache,
                    v_cache,
                    CausalMask(),
                    row_offsets_tt,
                    rsqrt(Float32(HEAD_DIM)),
                    launch_ctx,
                    num_partitions=_p_opt(num_partitions),
                )

    h.first_launch(ctx, kernel_launch)
    if verify:
        h.timer.reset()
        _digest(ctx, ws.out, tokens * NUM_Q_HEADS * HEAD_DIM)
        h.timer.lap("verify")
    if not run_benchmark:
        # `--run_benchmark=False` leaves one launch per config for `ncu` to
        # replay, as the old benchmark did.
        return

    var flop_count = flops(
        batch_size, NUM_Q_HEADS, seq_len, cache_len + seq_len, HEAD_DIM
    )
    var fields = Fields()
    fields.add("kernel", String("fused_qkv_ragged_flash_attention"))
    fields.add("dtype", dtype)
    fields.add("num_q_heads", NUM_Q_HEADS)
    fields.add("num_kv_heads", NUM_KV_HEADS)
    fields.add("head_dim", HEAD_DIM)
    fields.add("page_size", PAGE_SIZE)
    fields.add("local_window_size", LOCAL_WINDOW_SIZE)
    fields.add("cross_attention", CROSS_ATTENTION)
    fields.add("sink", SINK)
    fields.add("batch_size", batch_size)
    fields.add("seq_len", seq_len)
    fields.add("use_random_seq_lengths", use_random_seq_lengths)
    fields.add("cache_len", cache_len)
    fields.add("use_random_cache_lengths", use_random_cache_lengths)
    fields.add("max_cache_len", max_cache_len)
    fields.add("num_partitions", num_partitions)
    h.bench(
        ctx,
        kernel_launch,
        BenchId(
            _run_name(
                batch_size,
                seq_len,
                use_random_seq_lengths,
                cache_len,
                use_random_cache_lengths,
                max_cache_len,
            )
        ),
        fields^,
        copies,
        [ThroughputMeasure(BenchMetric.flops, flop_count)],
    )


def main() raises:
    var h = SweepHarness("kv_cache_ragged_flash_attention_paged")

    var sweep = Sweep()
    sweep.axis("batch_size", "20,250")
    # The old YAML's decode group (one new token per sequence) crossed with
    # its prefill group (a chunk of new tokens); a sweep YAML passes each
    # axis as one comma list, a point per process before the port.
    sweep.axis("seq_len", "1,256")
    sweep.axis("cache_len", "215,1024")
    var use_random_seq_lengths = arg_parse("use_random_seq_lengths", False)
    var use_random_cache_lengths = arg_parse("use_random_cache_lengths", False)
    var max_cache_len = Int(arg_parse("max_cache_len", 0))
    var num_partitions = Int(arg_parse("num_partitions", 0))
    var run_benchmark = arg_parse("run_benchmark", True)
    var verify = arg_parse("verify", False)
    var fill = String(arg_parse("fill", "gpu"))
    var layout = PageLayout.parse(arg_parse("page_layout", "shuffled"))
    var on_host = fill == "host"

    # Reproducible host draws for the random-length flags, as before.
    seed(0)

    def lut_cols_of(cfg: Config) raises {imm} -> Int:
        var widest = cfg["cache_len"] + cfg["seq_len"]
        if max_cache_len > 0:
            widest = max(widest, max_cache_len + cfg["seq_len"])
        return _padded_lut_cols(ceildiv(widest, PAGE_SIZE))

    h.timer.lap("main_prologue")
    with DeviceContext() as ctx:
        h.timer.lap("device_init")
        h.start(ctx)
        var rot = h.rotation()

        # One launch reads every key of every sequence at least once: a
        # causal query at position p re-reads the cache plus p new keys, so
        # the batch's union is the whole cache_len + seq_len window. One
        # more copy takes every page it reads fresh, K and V both, plus its
        # LUT view.
        def copies(cfg: Config) raises {imm} -> Int:
            var batch = cfg["batch_size"]
            var keys = cfg["cache_len"] + cfg["seq_len"]
            var touched = batch * keys * KV_ROW_ELEMS * DTYPE_SIZE
            var copy_bytes = batch * ceildiv(
                keys, PAGE_SIZE
            ) * PAGE_SIZE * KV_ROW_ELEMS * DTYPE_SIZE + 4 * batch * lut_cols_of(
                cfg
            )
            return rot.copies(touched, copy_bytes)

        def require(cfg: Config, mut fp: Footprint) raises {imm}:
            var n = max(1, copies(cfg))
            PageTable[PAGE_SIZE].require(
                fp, cfg["batch_size"], cfg["cache_len"] + cfg["seq_len"], n
            )
            fp.need("tokens", cfg["batch_size"] * cfg["seq_len"])
            fp.need("lut_entries", n * cfg["batch_size"] * lut_cols_of(cfg))

        def bytes(fp: Footprint) raises -> Int:
            return MHACache.bytes_for(fp) + Workspace.bytes_for(fp)

        var groups = plan_groups(
            sweep.configs(), require, bytes, device_budget(ctx)
        )
        for ref group in groups:
            ref fp = group.footprint
            var cache = MHACache(ctx, fp, layout, on_host, verify)
            ctx.synchronize()
            h.timer.lap("fixture_fill")
            var ws = Workspace(ctx, fp, on_host, verify)
            ctx.synchronize()
            h.timer.lap("workspace_fill")
            var info = Fields()
            info.add("fixture_bytes", cache.bytes())
            info.add("planned_bytes", group.bytes)
            info.extend(fp)
            info.add("configs", len(group.configs))
            info.add("fill", fill)
            info.add("page_layout", String(layout))
            info.add("verify", verify)
            h.note(info^)
            for ref cfg in group.configs:
                run_config(
                    ctx,
                    h,
                    cache,
                    ws,
                    cfg["batch_size"],
                    cfg["seq_len"],
                    cfg["cache_len"],
                    max_cache_len,
                    num_partitions,
                    use_random_seq_lengths,
                    use_random_cache_lengths,
                    run_benchmark,
                    verify,
                    copies(cfg),
                )
            h.timer.reset()
        h.stop(ctx)
    h.timer.lap("device_teardown")
    h.finish()
