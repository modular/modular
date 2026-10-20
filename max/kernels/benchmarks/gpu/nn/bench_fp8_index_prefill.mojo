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
"""Benchmark the FP8 indexer prefill scorer with a paged key cache.

Drives the vendor's paged scorer directly: on SM100 the warp-specialized
K-streaming kernel (`fp8_index_score_sm100`, which routes the nh=32
long-prefill shape to `sparse_index_fp8_sm100_prefill`), on gfx950 the MFMA
scorer (`fp8_index_score_amd`). Only the launch differs; the cache, shapes,
timing and checks are shared.

One invocation benchmarks one shape. The SM100 shape set lives in the sibling
`bench_fp8_index_prefill.yaml`, whose four labelled groups say what each shape
is there to answer; the gfx950 set lives in `bench_fp8_index_prefill_amd.yaml`.
Run the whole set, or one group:

    kbench max/kernels/benchmarks/gpu/nn/bench_fp8_index_prefill.yaml
    kbench max/kernels/benchmarks/gpu/nn/bench_fp8_index_prefill.yaml \\
        --filter label=ragged_prefill
    kbench max/kernels/benchmarks/gpu/nn/bench_fp8_index_prefill_amd.yaml

For a source-correlated capture, drive one shape directly -- every field is an
argument, so nothing is implied by a default. Pass `--run_benchmark=False`
under `ncu`: the profiler replays the kernel itself for its sampling, so the
benchmark loop on top of it only multiplies capture time.

    ncu --set full mojo bench_fp8_index_prefill.mojo --run_benchmark=False \\
        --batch_size=8 --seq_len=512 --cache_len=2048 --max_num_keys=3584 \\
        --spread=50 --label=ragged_prefill

The bare defaults are the nh=32 pure-prefill route (`cache_len=0`, causal, long
enough to clear the 448-token-tile gate), which is the SM100 configuration
whose 2-CTA/SM residency is too deeply amortized to attribute from nsys.

`--kpool=4` (pooled keys) and `--window_mb` (scoring the rows one window at a
time, as production does past its score budget) are gfx950-only and raise on
SM100.
"""

from std.random import rand, seed
from std.sys import bit_width_of, get_defined_int
from std.builtin.dtype import _uint_type_of_width

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import (
    KVCacheStaticParams,
    KVCollectionT,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.tile_tensor import ImmTileTensor, MutTileTensor
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    _INDEX_SWIZZLE,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from nn.attention.gpu.sparse_index_fp8_sm100_prefill import _FOLD_DTYPE
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    KVCacheScalesMHAOperand,
)
from nn.index_fp8 import (
    _has_gfx950_accelerator,
    fp8_index_naive,
    fp8_index_score_amd,
)
from std.math import ceildiv, clamp, log
from std.memory import bitcast
from std.utils.index import IndexList


comptime _is_amd = _has_gfx950_accelerator()


# Calling the scorer with `k_op`/`ks_op` built from the same cache inlined in
# one function trips a false aliasing error (the two operands share an origin).
# Constructing the operands INSIDE this helper, from a collection whose type is
# a TYPE parameter, makes their origins provably disjoint -- the test's
# `_score_paged_sm100` does the same -- so the scorer call lives here.
def _launch_scorer[
    out_dtype: DType,
    o_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    iro_layout: TensorLayout,
    //,
    num_heads: Int,
    depth: Int,
    KCollectionT: KVCollectionT,
    kpool: Int,
](
    o_tile: MutTileTensor[out_dtype, o_layout, _],
    q_tile: ImmTileTensor[KCollectionT.CacheType.dtype, q_layout, _],
    qs_tile: ImmTileTensor[.float32, qs_layout, _],
    input_row_offsets_tile: ImmTileTensor[.uint32, iro_layout, _],
    k_collection: KCollectionT,
    batch_size: Int,
    seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
    chunk_begin: Int = 0,
) raises:
    var k_cache = k_collection.get_key_cache(0)
    var k_op = KVCacheMHAOperand(k_cache)
    var ks_op = KVCacheScalesMHAOperand(k_cache)
    comptime if _is_amd:
        comptime assert (
            out_dtype == DType.float32
        ), "the gfx950 scorer writes float32 scores"
        fp8_index_score_amd[
            KCollectionT.CacheType.dtype,
            type_of(k_op),
            type_of(ks_op),
            num_heads,
            depth,
            _is_cache_length_accurate=False,
            causal=True,
            kpool=kpool,
            pool_row_tokens=kpool,
        ](
            o_tile.bitcast[DType.float32](),
            q_tile,
            qs_tile,
            k_op,
            ks_op,
            input_row_offsets_tile,
            batch_size,
            seq_len,
            max_num_keys,
            ctx,
            chunk_begin,
        )
    else:
        comptime assert kpool == 1, "the SM100 scorer has no pooled keys"
        if chunk_begin != 0:
            raise Error("the SM100 scorer has no row windows")
        var k_tma_tile = k_cache.create_paged_tma_tile[
            _INDEX_SWIZZLE, BN=_BM_KEY, BK=depth
        ](ctx)
        fp8_index_score_sm100[
            KCollectionT.CacheType.dtype,
            type_of(k_op),
            type_of(ks_op),
            num_heads,
            depth,
            # MLA cache lengths do not include the new tokens, so the scorer
            # adds `seq_len` itself; this matches the test's call.
            _is_cache_length_accurate=False,
            # The production MTP-step tile hint; inert at the prefill width
            # this benchmark defaults to, kept so the routed instantiation set
            # matches what production builds.
            N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
        ](
            o_tile,
            q_tile,
            qs_tile,
            k_op,
            ks_op,
            k_tma_tile,
            input_row_offsets_tile,
            batch_size,
            seq_len,
            max_num_keys,
            True,  # causal: the nh=32 prefill route requires it
            ctx,
        )


def _exp_quantile(mean: Int, i: Int, n: Int) -> Int:
    """Returns the `(i + 0.5) / n` quantile of an exponential with `mean`."""
    return Int(-Float64(mean) * log(1.0 - (Float64(i) + 0.5) / Float64(n)))


def _run_name[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    out_dtype: DType,
](
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    max_num_keys: Int,
    spread: Int,
    label: String,
    kpool: Int = 1,
    windows: Int = 1,
    dist: Int = 0,
    seq_dist: Int = 0,
    cache_cap: Int = 0,
) -> String:
    var kind = String("fp8_index_score_amd_prefill") if _is_amd else String(
        "fp8_index_score_sm100_prefill"
    )
    if label.byte_length() > 0:
        kind += "/" + label
    # fmt: off
    return String(
        kind, " : ",
        "num_heads=", num_heads, ", ",
        "depth=", depth, ", ",
        "page_size=", page_size, " : ",
        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "cache_len=", cache_len, ", ",
        String("kpool=", kpool, ", ") if kpool != 1 else String(),
        String("windows=", windows, ", ") if windows != 1 else String(),
        "max_num_keys=", max_num_keys, ", ",
        "spread=", spread, ", ",
        "out_dtype=", out_dtype,
        String(", dist=", dist) if dist != 0 else String(),
        String(", seq_dist=", seq_dist) if seq_dist != 0 else String(),
        String(", cache_cap=", cache_cap) if cache_cap != 0 else String(),
    )
    # fmt: on


def execute_fp8_index_prefill[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    out_dtype: DType = DType.float32,
    kpool: Int = 1,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    var max_num_keys: Int,
    spread: Int = 0,
    label: String = String(),
    run_benchmark: Bool = True,
    verify: Bool = False,
    digest: Int = 0,
    dist: Int = 0,
    seq_dist: Int = 0,
    cache_cap: Int = 0,
    window_mb: Int = 0,
) raises:
    """Benchmark one scorer invocation at one prefill shape.

    Args:
        ctx: Device context.
        m: Bench harness collecting results.
        batch_size: Number of sequences in the batch.
        seq_len: New (prefill) tokens per sequence. On SM100, long enough to
            clear the nh=32 prefill route's 448-token-tile gate;
            ``cache_len=0`` keeps the pure-prefill causal-cache=0 path that
            gate admits.
        cache_len: Cached tokens per sequence. 0 reproduces a fresh prefill;
            a nonzero value exercises the chunked-prefill-continuation path.
        max_num_keys: Score-buffer row stride (>= any entry's pooled key
            count); 0 takes the deepest entry's.
        spread: Cache-depth raggedness, in percent of `cache_len`. 0 gives the
            uniform batch. Otherwise entry depths run linearly from
            `cache_len * (1 - spread/100)` to `cache_len * (1 + spread/100)`,
            so the MEAN depth -- and hence the total key work -- is unchanged
            and only the distribution differs from the `spread=0` twin.
        label: Free-form group tag echoed into the benchmark name, so a row in
            a kbench sweep says which group of the YAML it came from. Nothing
            branches on it.
        run_benchmark: Time the kernel over the harness' iteration loop. False
            launches it exactly once and reports nothing, which is what `ncu`
            wants -- the profiler does its own replay.
        verify: Compare windowed scores with a full launch bitwise and, when
            its logits fit in 1 GiB, compare scores with fp8_index_naive.
        digest: Nonzero launches once and prints a bit-exact hash of the
            scores instead of timing.
        dist: Cache-depth distribution. 0 is the linear grade `spread`
            describes; 1 draws depths from an exponential distribution with
            mean `cache_len` (deterministic quantiles, deepest entry last),
            and `spread` is ignored.
        seq_dist: New-token distribution. 0 gives every entry `seq_len`; 1
            draws lengths from an exponential distribution with mean
            `seq_len`, so most token blocks of the shorter entries are empty.
        cache_cap: Nonzero clamps every entry's cache depth to at most this,
            bounding an exponential draw's tail.
        window_mb: Score-buffer budget in MiB; 0 uses one window. gfx950 only.
    """
    # Per-entry new-token and cache lengths, then the batch maxima every
    # allocation is sized by, as production's captured-graph metadata is.
    var seq_lens = List[Int](length=batch_size, fill=seq_len)
    var cache_lens = List[Int](length=batch_size, fill=cache_len)
    var lo_cache = cache_len - (cache_len * spread) // 100
    var hi_cache = cache_len + (cache_len * spread) // 100
    for i in range(batch_size):
        if dist == 1:
            cache_lens[i] = _exp_quantile(cache_len, i, batch_size)
        elif batch_size > 1:
            # Linear grade lo -> hi across the batch; a single-entry batch
            # takes the mean, so `spread` cannot change a batch of one.
            cache_lens[i] = lo_cache + (hi_cache - lo_cache) * i // (
                batch_size - 1
            )
        if seq_dist == 1:
            seq_lens[i] = max(1, _exp_quantile(seq_len, i, batch_size))
        if cache_cap > 0:
            cache_lens[i] = min(cache_lens[i], cache_cap)
    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache = 0
    var tokens_per_seq = 0
    for i in range(batch_size):
        total_seq_len += seq_lens[i]
        max_seq_len = max(max_seq_len, seq_lens[i])
        max_cache = max(max_cache, cache_lens[i])
        tokens_per_seq = max(tokens_per_seq, cache_lens[i] + seq_lens[i])
    var keys_per_seq = tokens_per_seq // kpool

    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=depth // kpool, is_mla=True
    )
    comptime num_layers = 1
    comptime score_bytes = bit_width_of[out_dtype]() // 8
    comptime unwritten_score = -Scalar[out_dtype].MAX

    # The page pool, LUT width and cache-collection bound are all sized by the
    # deepest entry, as production's captured-graph metadata is. Score rows are
    # `max_num_keys` apart, so a deeper entry would spill into the next row and
    # race the CTA that owns it: the output stops being a function of the
    # inputs.
    if max_num_keys == 0:
        max_num_keys = keys_per_seq
    if max_num_keys < keys_per_seq:
        raise Error(
            "max_num_keys="
            + String(max_num_keys)
            + " is below the deepest entry's "
            + String(keys_per_seq)
            + " keys"
        )
    var pages_per_seq = ceildiv(tokens_per_seq, page_size)
    var num_blocks = batch_size * pages_per_seq + 1

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
        var off = 0
        for i in range(batch_size):
            iro_host[i] = UInt32(off)
            off += seq_lens[i]
        iro_host[batch_size] = UInt32(off)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for i in range(batch_size):
            cl_host[i] = UInt32(cache_lens[i])

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

    var paged_lut_shape = IndexList[2](batch_size, pages_per_seq)
    var k_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    with k_lut_device.map_to_host() as k_lut_host:
        for bs in range(batch_size):
            for page_idx in range(pages_per_seq):
                k_lut_host[bs * pages_per_seq + page_idx] = UInt32(
                    1 + bs * pages_per_seq + page_idx
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
        TileTensor(cache_lengths_device, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var lookup_table = (
        TileTensor(
            k_lut_device,
            row_major(Int64(batch_size), Int64(pages_per_seq)),
        )
        .as_imm()
        .as_unsafe_any_origin()
    )
    var k_collection = Collection(
        blocks,
        cache_lengths,
        lookup_table,
        UInt32(max_seq_len),
        UInt32(max_cache),
        scales,
    )

    if window_mb < 0:
        raise Error("window_mb must be nonnegative")
    var rows_per_chunk = total_seq_len
    if window_mb > 0:
        rows_per_chunk = clamp(
            window_mb * 1024 * 1024 // (max_num_keys * score_bytes),
            1,
            total_seq_len,
        )
    var windows = ceildiv(total_seq_len, rows_per_chunk)
    var o_size = rows_per_chunk * max_num_keys
    var o_device = ctx.enqueue_create_buffer[out_dtype](o_size)
    var q_tile = TileTensor(
        q_device, row_major(total_seq_len, num_heads, depth)
    )
    var qs_tile = TileTensor(qs_device, row_major(total_seq_len, num_heads))
    var input_row_offsets_tile = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )

    # The scorer leaves forbidden slots untouched, so a buffer reused across
    # iters needs a defined baseline. -inf is the production convention.
    # On device: a host fill is quadratic in the two axes deep-cache shapes
    # raise (`total_seq_len * max_num_keys`), and at `s8192 k131072` it runs
    # past a probe's timeout before the kernel launches once.
    ctx.enqueue_memset(o_device, unwritten_score)

    @inline(.always)
    def kernel_launch(launch_ctx: DeviceContext) raises {mut o_device, imm}:
        for chunk_begin in range(0, total_seq_len, rows_per_chunk):
            var chunk_rows = min(rows_per_chunk, total_seq_len - chunk_begin)
            var o_tile = TileTensor(
                o_device, row_major(chunk_rows, max_num_keys)
            )
            _launch_scorer[num_heads, depth, type_of(k_collection), kpool](
                o_tile,
                q_tile,
                qs_tile,
                input_row_offsets_tile,
                k_collection,
                batch_size,
                max_seq_len,
                max_num_keys,
                launch_ctx,
                chunk_begin,
            )

    @inline(.always)
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    var name = _run_name[num_heads, depth, page_size, out_dtype](
        batch_size,
        seq_len,
        cache_len,
        max_num_keys,
        spread,
        label,
        kpool,
        windows,
        dist,
        seq_dist,
        cache_cap,
    )
    if digest != 0 and windows != 1:
        raise Error("digest requires one score window")
    if digest != 0:
        # One launch over the -inf fill, hashed bit for bit: two builds of the
        # scorer agree iff their digests do, `-inf` slots included.
        kernel_launch(ctx)
        ctx.synchronize()
        var h = UInt64(0xCBF29CE484222325)
        var written = 0
        with o_device.map_to_host() as oh:
            for i in range(o_size):
                var v = oh[i]
                if v != unwritten_score:
                    written += 1
                h = (
                    h
                    ^ UInt64(
                        bitcast[
                            _uint_type_of_width[bit_width_of[out_dtype]()]()
                        ](v)
                    )
                ) * 0x100000001B3
        print(
            "#digest",
            name,
            ":",
            hex(h),
            "written=",
            written,
        )
    elif run_benchmark:
        comptime if _is_amd:
            # The gfx950 scorer loads every 128-key tile for every token;
            # causality guards stores only. SM100 shares K across tokens, so
            # this model would overstate its traffic.
            comptime key_tile_rows = 128
            var effective_k_bytes = 0
            for i in range(batch_size):
                effective_k_bytes += (
                    seq_lens[i]
                    * ceildiv(
                        (cache_lens[i] + seq_lens[i]) // kpool, key_tile_rows
                    )
                    * key_tile_rows
                    * depth
                )
            print("#effective_k_bytes", effective_k_bytes)
            m.bench_function(
                bench_func,
                BenchId(name),
                [ThroughputMeasure(BenchMetric.bytes, effective_k_bytes)],
            )
        else:
            m.bench_function(bench_func, BenchId(name))
    else:
        kernel_launch(ctx)
        ctx.synchronize()

    if verify:
        var full_size = total_seq_len * max_num_keys
        var full_device = ctx.enqueue_create_buffer[out_dtype](full_size)
        ctx.enqueue_memset(full_device, unwritten_score)
        _launch_scorer[num_heads, depth, type_of(k_collection), kpool](
            TileTensor(full_device, row_major(total_seq_len, max_num_keys)),
            q_tile,
            qs_tile,
            input_row_offsets_tile,
            k_collection,
            batch_size,
            max_seq_len,
            max_num_keys,
            ctx,
        )
        ctx.synchronize()
        var got_host = ctx.enqueue_create_host_buffer[out_dtype](full_size)
        for chunk_begin in range(0, total_seq_len, rows_per_chunk):
            var chunk_rows = min(rows_per_chunk, total_seq_len - chunk_begin)
            ctx.enqueue_memset(o_device, unwritten_score)
            _launch_scorer[num_heads, depth, type_of(k_collection), kpool](
                TileTensor(o_device, row_major(chunk_rows, max_num_keys)),
                q_tile,
                qs_tile,
                input_row_offsets_tile,
                k_collection,
                batch_size,
                max_seq_len,
                max_num_keys,
                ctx,
                chunk_begin,
            )
            ctx.synchronize()
            with o_device.map_to_host() as window_host:
                for i in range(chunk_rows * max_num_keys):
                    got_host[chunk_begin * max_num_keys + i] = window_host[i]
        with full_device.map_to_host() as expected:
            var actual = Span(
                unsafe_ptr=got_host.unsafe_ptr().unsafe_bitcast[Byte](),
                length=full_size * score_bytes,
            )
            var reference = Span(
                unsafe_ptr=expected.unsafe_ptr().unsafe_bitcast[Byte](),
                length=full_size * score_bytes,
            )
            for i in range(len(actual)):
                if actual[i] != reference[i]:
                    raise Error("windowed scorer differs from full launch")
        print("#verify windowed scores match full launch bitwise")

        comptime max_naive_bytes = 1024 * 1024 * 1024
        var naive_bytes = (
            batch_size * max_seq_len * max_num_keys * num_heads * 4
        )
        if naive_bytes > max_naive_bytes:
            print("#verify naive skipped: logits buffer exceeds 1 GiB")
        else:
            var total_keys = 0
            for i in range(batch_size):
                total_keys += (cache_lens[i] + seq_lens[i]) // kpool
            var packed_k = ctx.enqueue_create_buffer[.float8_e4m3fn](
                total_keys * depth
            )
            var packed_ks = ctx.enqueue_create_buffer[.float32](total_keys)
            var cro_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
            with k_block_device.map_to_host() as source_k, ks_block_device.map_to_host() as source_ks, packed_k.map_to_host() as dest_k, packed_ks.map_to_host() as dest_ks, cro_device.map_to_host() as cro_host:
                var key_offset = 0
                for bs in range(batch_size):
                    cro_host[bs] = UInt32(key_offset)
                    var num_keys = (cache_lens[bs] + seq_lens[bs]) // kpool
                    for key in range(num_keys):
                        var cache_row = key * kpool
                        var page = (
                            1 + bs * pages_per_seq + cache_row // page_size
                        )
                        var source_row = (
                            page * page_size + cache_row % page_size
                        )
                        dest_ks[key_offset + key] = source_ks[source_row]
                        for d in range(depth):
                            dest_k[(key_offset + key) * depth + d] = source_k[
                                source_row * (depth // kpool) + d
                            ]
                    key_offset += num_keys
                cro_host[batch_size] = UInt32(key_offset)
            var ref_device = ctx.enqueue_create_buffer[.float32](full_size)
            var ref_tile = TileTensor(
                ref_device, row_major(total_seq_len, max_num_keys)
            )
            fp8_index_naive[num_heads, depth](
                ref_tile,
                TileTensor(
                    q_device,
                    row_major(total_seq_len, Idx[num_heads], Idx[depth]),
                ),
                TileTensor(qs_device, row_major(total_seq_len, Idx[num_heads])),
                TileTensor(packed_k, row_major(total_keys, Idx[1], Idx[depth])),
                TileTensor(packed_ks, row_major(total_keys)),
                input_row_offsets_tile,
                TileTensor(cro_device, row_major(batch_size + 1)),
                batch_size,
                max_seq_len,
                max_num_keys,
                ctx,
            )
            ctx.synchronize()
            # gfx950 and an f32 SM100 fold accumulate in f32, so 1% covers a
            # bfloat16 score's rounding (2^-8). The default bf16 fold drifts by
            # design; this matches the shared test's bound for it.
            comptime rel_tolerance = Float32(
                0.01 if _is_amd or _FOLD_DTYPE == DType.float32 else 0.1
            )
            var checked = 0
            var mismatched = 0
            var max_abs = Float32(0)
            var max_rel = Float32(0)
            with full_device.map_to_host() as got, ref_device.map_to_host() as expected:
                var row_offset = 0
                for bs in range(batch_size):
                    for t in range(seq_lens[bs]):
                        var causal_keys = (cache_lens[bs] + t + 1) // kpool
                        for key in range(min(max_num_keys, causal_keys)):
                            var index = (row_offset + t) * max_num_keys + key
                            var a = got[index].cast[DType.float32]()
                            var b = expected[index]
                            var diff = abs(a - b)
                            var rel = diff / max(abs(b), Float32(1))
                            max_abs = max(max_abs, diff)
                            max_rel = max(max_rel, rel)
                            # Negated so a NaN on either side counts.
                            if not rel <= rel_tolerance:
                                mismatched += 1
                            checked += 1
                    row_offset += seq_lens[bs]
            print(
                "#verify naive checked=",
                checked,
                "max_abs=",
                max_abs,
                "max_rel=",
                max_rel,
                "mismatched=",
                mismatched,
            )
            if mismatched > 0:
                raise Error("scorer differs from fp8_index_naive")

    _ = q_device
    _ = qs_device
    _ = input_row_offsets_device
    _ = cache_lengths_device
    _ = k_block_device
    _ = ks_block_device
    _ = k_lut_device
    _ = o_device


def main() raises:
    # GLM-5.2 replicated indexer: 32 heads on every tensor-parallel rank.
    comptime num_heads = get_defined_int["num_heads", 32]()
    comptime depth = get_defined_int["depth", 128]()
    comptime page_size = get_defined_int["page_size", 128]()

    var batch_size = arg_parse("batch_size", 1)
    # Long enough to clear the SM100 nh=32 prefill route's 448-token-tile gate
    # (seq >= 1792); 2048 is the production continuation shape in the test.
    var seq_len = arg_parse("seq_len", 2048)
    # 0 = fresh prefill (causal cache=0), the shape the prefill route admits.
    var cache_len = arg_parse("cache_len", 0)
    # Tokens per pooled key; 4 is GLM-5.3-Flash's k-pool. gfx950 only.
    var kpool = arg_parse("kpool", 1)
    # Row stride of the score buffer; 0 takes the deepest entry's key count.
    var max_num_keys = arg_parse("max_num_keys", 0)
    # Cache-depth raggedness, in percent of `cache_len`; 0 is the uniform batch.
    var spread = arg_parse("spread", 0)
    # Which YAML group this shape came from; reporting only.
    var label = String(arg_parse("label", ""))
    # False leaves a single launch for `ncu` to replay.
    var run_benchmark = arg_parse("run_benchmark", True)
    var verify = arg_parse("verify", False)
    # Nonzero prints a bit-exact hash of one launch's scores instead of timing.
    var digest = arg_parse("digest", 0)
    # 1 draws cache depths / new-token counts from an exponential
    # distribution; see `execute_fp8_index_prefill`.
    var dist = arg_parse("dist", 0)
    var seq_dist = arg_parse("seq_dist", 0)
    # Nonzero clamps every cache depth to at most this.
    var cache_cap = arg_parse("cache_cap", 0)
    # Score-buffer budget in MiB; 0 scores every row in one window.
    var window_mb = arg_parse("window_mb", 0)

    comptime if not _is_amd:
        if kpool != 1:
            raise Error("--kpool is gfx950-only")
        if window_mb != 0:
            raise Error("--window_mb is gfx950-only")
    if kpool != 1 and kpool != 4:
        raise Error("supported kpool values: 1, 4")

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        # On SM100 both score dtypes run in ONE process: the ratio is paired,
        # so it carries no cross-run clock or cache drift. The FOLD dtype is a
        # module-level `-D`, so a fold A/B is still two builds. The gfx950
        # scorer writes float32 only.
        comptime for out_dtype in [DType.float32, DType.bfloat16]:
            comptime if not _is_amd or out_dtype == DType.float32:
                comptime for pooled in [1, 4]:
                    comptime if _is_amd or pooled == 1:
                        if kpool == pooled:
                            execute_fp8_index_prefill[
                                num_heads, depth, page_size, out_dtype, pooled
                            ](
                                ctx,
                                m,
                                batch_size,
                                seq_len,
                                cache_len,
                                max_num_keys,
                                spread,
                                label,
                                run_benchmark,
                                verify,
                                digest,
                                dist,
                                seq_dist,
                                cache_cap,
                                window_mb,
                            )

    if run_benchmark and digest == 0:
        m.dump_report()
