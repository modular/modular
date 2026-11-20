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

"""Convert-once NVFP4 sparse MLA prefill (`mla_prefill_sparse_nvfp4_via_fp8`).

Target family: Blackwell sm_100.

The oracle is the sparse decode kernel over an FP8 cache that the host expands
independently from the packed NVFP4 rows and scales. The convert-once path
expands the batch's whole context to FP8 on the device and runs the same
kernel at the same split on that, so it must match the oracle bit for bit and
any difference is the conversion or the index remap.

The cache has several layers and the cases read a middle one, the page tables
are shuffled and the scales have their own, entries are ragged prefixes plus a
new chunk whose own rows are selected too, and two entries can share their
first pages, as a cached prefix does. `PERF` lines time the oracle (over the
pre-expanded cache) and the convert-once path.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import randn, seed
from std.sys import get_defined_int
from std.testing import assert_equal, assert_false, assert_true

from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from max.gpu.host.info import _is_sm10x_gpu
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.nvidia.sm100.mla_sparse_attention import (
    mla_sparse_attention,
    mla_sparse_attention_flat,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    mla_decode_sm100_dispatch,
)
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import (
    MHAConfig,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
)
from nn.attention.gpu.nvidia.sm100.mla_nvfp4_kv_to_fp8 import (
    mla_prefill_sparse_nvfp4_via_fp8,
    NVFP4PrefillWorkspaceView,
    nvfp4_prefill_workspace,
    nvfp4_prefill_converts_once,
    nvfp4_prefill_scratch_rows,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)

comptime QK_DEPTH = 576  # 512 latent + 64 rope
comptime V_DEPTH = 512
comptime ROPE_DEPTH = 64
comptime PACKED_ROW = V_DEPTH // 2 + ROPE_DEPTH
comptime SF_PER_ROW = V_DEPTH // 16
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 3
comptime LAYER = 1
comptime KV_NUM_HEADS = 1
# `quantization_granularity` counts packed bytes (16 elements = 8 bytes).
comptime QUANT_GRANULARITY = 8

comptime kv_params_fp4 = KVCacheStaticParams(
    num_heads=KV_NUM_HEADS,
    head_size=PACKED_ROW,
    is_mla=True,
    fp8_tail_size=ROPE_DEPTH,
)
comptime CollectionFP4 = PagedKVCacheCollection[
    DType.uint8,
    kv_params_fp4,
    PAGE_SIZE,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
    scale_dtype_=DType.float8_e4m3fn,
    quantization_granularity_=QUANT_GRANULARITY,
]


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int, start: Int = 0) -> Int:
    var candidates = [3, 5, 7, 11, 13, 17, 19, 23, 40503, 40507, 40519]
    for i in range(start, len(candidates)):
        if n <= 1 or _gcd(candidates[i], n) == 1:
            return candidates[i]
    return 1


struct _Lcg:
    """A deterministic generator so a failing case reproduces."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt32:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return UInt32(self.state >> 33)


def _scale_byte(mut rng: _Lcg) -> UInt8:
    """A random E4M3 byte: 2^-5 .. 2^1 times 1 .. 1.875, or zero.

    The range keeps `6 * sf` far below the E4M3 maximum so no product saturates
    and no byte is a NaN.
    """
    var r = rng.next()
    if r % 16 == 0:
        return UInt8(0)
    var exponent = 2 + Int((r >> 4) % 7)
    var mantissa = Int((r >> 8) % 8)
    return UInt8(exponent * 8 + mantissa)


def _us(ns: Int, iters: Int) -> Float64:
    return Float64(ns) / Float64(iters) / 1000.0


def test_gate() raises:
    # Every entry is given the pages of the longest context.
    assert_equal(nvfp4_prefill_scratch_rows[128](1, 8192), 8192)
    assert_equal(nvfp4_prefill_scratch_rows[128](1, 8193), 8320)
    assert_equal(nvfp4_prefill_scratch_rows[128](3, 300), 3 * 384)
    # Rows per query token decide; 0 turns the conversion off.
    assert_true(nvfp4_prefill_converts_once(8192, 8192, 150, 262144))
    assert_true(nvfp4_prefill_converts_once(150 * 100, 100, 150, 262144))
    assert_false(nvfp4_prefill_converts_once(150 * 100 + 1, 100, 150, 262144))
    assert_false(nvfp4_prefill_converts_once(8192, 8192, 0, 262144))
    # Past the scratch the planner reserved, the NVFP4 kernel runs.
    assert_true(nvfp4_prefill_converts_once(262144, 8192, 150, 262144))
    assert_false(nvfp4_prefill_converts_once(262272, 8192, 150, 262144))
    assert_false(nvfp4_prefill_converts_once(0, 8192, 150, 262144))


def run_case[
    num_heads: Int,
    topk: Int = 2048,
    perf_iters: Int = 0,
](
    name: StringLiteral,
    cache_lens: List[Int],
    seq_lens: List[Int],
    ctx: DeviceContext[],
    *,
    valid_topk: Int = -1,
    with_sink: Bool = False,
    shared_prefix_pages: Int = 0,
) raises:
    """Compares both paths on one batch.

    Entry `b` holds `cache_lens[b]` earlier tokens plus `seq_lens[b]` new ones,
    all of which its queries select from. `shared_prefix_pages` gives entry 1
    entry 0's first blocks.
    """
    var n_valid = topk if valid_topk < 0 else valid_topk
    var batch_size = len(cache_lens)
    var context = List[Int]()
    var max_context = 0
    var max_seq = 0
    var total_q = 0
    for b in range(batch_size):
        context.append(cache_lens[b] + seq_lens[b])
        max_context = max(max_context, context[b])
        max_seq = max(max_seq, seq_lens[b])
        total_q += seq_lens[b]
    print(
        t"test: {name}  batch: {batch_size}  heads: {num_heads}  q_tokens:"
        t" {total_q}  max_context: {max_context}  topk: {topk}  valid:"
        t" {n_valid}"
    )

    # Physical blocks: each entry's pages, a shuffled table, one null block.
    var lut_cols = ceildiv(max_context, PAGE_SIZE)
    var total_blocks = 0
    for b in range(batch_size):
        total_blocks += ceildiv(context[b], PAGE_SIZE)
    total_blocks -= shared_prefix_pages
    var null_block = total_blocks
    var pool_blocks = total_blocks + 1
    var lut_size = batch_size * lut_cols
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    var scales_lut_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    var mult = _coprime_multiplier(total_blocks)
    var scale_mult = _coprime_multiplier(total_blocks, start=1)
    var next_block = 0
    for b in range(batch_size):
        for p in range(lut_cols):
            var block = null_block
            if p < ceildiv(context[b], PAGE_SIZE):
                if b == 1 and p < shared_prefix_pages:
                    block = Int(lut_host[p])
                else:
                    block = (next_block * mult + 1) % total_blocks
                    next_block += 1
            lut_host[b * lut_cols + p] = UInt32(block)
            # The scales page in their own block-id space.
            scales_lut_host[b * lut_cols + p] = UInt32(
                block if block
                == null_block else (block * scale_mult + 2) % total_blocks
            )

    # Random packed cache in every layer: e2m1 nibbles, scales, FP8 rope.
    var rows_per_block = NUM_LAYERS * PAGE_SIZE
    var fp4_elems = pool_blocks * rows_per_block * PACKED_ROW
    var sf_elems = pool_blocks * rows_per_block * SF_PER_ROW
    var fp4_host = ctx.enqueue_create_host_buffer[.uint8](fp4_elems)
    var sf_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](sf_elems)
    var rng = _Lcg(UInt64(0x9E3779B97F4A7C15) ^ UInt64(total_q * 131 + topk))
    for row in range(pool_blocks * rows_per_block):
        for j in range(V_DEPTH // 2):
            fp4_host[row * PACKED_ROW + j] = UInt8(rng.next() & 255)
        for j in range(ROPE_DEPTH):
            fp4_host[row * PACKED_ROW + V_DEPTH // 2 + j] = _scale_byte(
                rng
            ) | UInt8(0x80 if rng.next() & 1 == 1 else 0)
        for g in range(SF_PER_ROW):
            sf_host[row * SF_PER_ROW + g] = bitcast[DType.float8_e4m3fn, 1](
                SIMD[DType.uint8, 1](_scale_byte(rng))
            )[0]

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = 0
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(cache_lens[b])
        row_offsets_host[b + 1] = row_offsets_host[b] + UInt32(seq_lens[b])

    var q_size = total_q * num_heads * QK_DEPTH
    var q_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_scratch[i].cast[.float8_e4m3fn]()

    # Each query walks its entry's whole context, new chunk included, from a
    # rotated start. Index rows are `[value rows | scale rows]`: a value row is
    # `block * page_size + offset`, a scale row the scale pool's row, whose
    # blocks are a layer stack apart.
    var wide_size = total_q * 2 * topk
    var wide_host = ctx.enqueue_create_host_buffer[.uint32](wide_size)
    for t in range(total_q):
        var b = 0
        while Int(row_offsets_host[b + 1]) <= t:
            b += 1
        var ctx_len = context[b]
        var sel_mult = _coprime_multiplier(ctx_len, start=8)
        var rotation = (t * 131) % ctx_len
        for i in range(topk):
            var value_row = UInt32(0xFFFFFFFF)
            var scale_row = UInt32(0xFFFFFFFF)
            # A context shorter than top-k pads with unused slots.
            if i < n_valid and i < ctx_len:
                var pos = ((rotation + i) * sel_mult + 1) % ctx_len
                var page, offset = divmod(pos, PAGE_SIZE)
                value_row = UInt32(
                    Int(lut_host[b * lut_cols + page]) * PAGE_SIZE + offset
                )
                scale_row = UInt32(
                    Int(scales_lut_host[b * lut_cols + page]) * rows_per_block
                    + offset
                )
            wide_host[t * 2 * topk + i] = value_row
            wide_host[t * 2 * topk + topk + i] = scale_row

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.uint32](total_q)
    for t in range(total_q):
        topk_lengths_host[t] = UInt32(n_valid)
    var sink_host = ctx.enqueue_create_host_buffer[.float32](num_heads)
    for h in range(num_heads):
        sink_host[h] = Float32(h % 5) - 2.0

    var fp4_dev = ctx.enqueue_create_buffer[.uint8](fp4_elems)
    var sf_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](sf_elems)
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    var row_offsets_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var lut_dev = ctx.enqueue_create_buffer[.uint32](lut_size)
    var scales_lut_dev = ctx.enqueue_create_buffer[.uint32](lut_size)
    var q_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    var wide_dev = ctx.enqueue_create_buffer[.uint32](wide_size)
    var topk_lengths_dev = ctx.enqueue_create_buffer[.uint32](total_q)
    var sink_dev = ctx.enqueue_create_buffer[.float32](num_heads)
    var out_size = total_q * num_heads * V_DEPTH
    var out_ref_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(fp4_dev, fp4_host)
    ctx.enqueue_copy(sf_dev, sf_host)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    ctx.enqueue_copy(row_offsets_dev, row_offsets_host)
    ctx.enqueue_copy(lut_dev, lut_host)
    ctx.enqueue_copy(scales_lut_dev, scales_lut_host)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(wide_dev, wide_host)
    ctx.enqueue_copy(topk_lengths_dev, topk_lengths_host)
    ctx.enqueue_copy(sink_dev, sink_host)
    ctx.synchronize()

    comptime kv_dim2 = 1
    var fp4_tt = TileTensor(
        fp4_dev,
        row_major(
            Int64(pool_blocks),
            Idx[kv_dim2],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[PACKED_ROW],
        ),
    )
    var sf_tt = TileTensor(
        sf_dev,
        row_major(
            Int64(pool_blocks),
            Idx[kv_dim2],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[SF_PER_ROW],
        ),
    )
    var coll = CollectionFP4(
        rebind[CollectionFP4.blocks_tt_type](fp4_tt.as_unsafe_any_origin()),
        TileTensor(cache_lengths_dev, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(Int64(batch_size), Int64(lut_cols)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(max_seq),
        UInt32(max_context),
        rebind[CollectionFP4.scales_tt_type](sf_tt.as_unsafe_any_origin()),
        TileTensor(
            scales_lut_dev, row_major(Int64(batch_size), Int64(lut_cols))
        )
        .as_imm()
        .as_unsafe_any_origin(),
    )
    var kv = coll.get_key_cache(LAYER)
    comptime FP8Collection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        KVCacheStaticParams(num_heads=1, head_size=QK_DEPTH, is_mla=True),
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var fp8_reference = ctx.enqueue_create_buffer[.float8_e4m3fn](
        pool_blocks * rows_per_block * QK_DEPTH
    )
    var host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        len(fp8_reference)
    )
    var magnitudes = [Float32(0), 0.5, 1, 1.5, 2, 3, 4, 6]
    for block in range(pool_blocks):
        var sf_block = (
            block * scale_mult + 2
        ) % total_blocks if block < total_blocks else block
        for r in range(rows_per_block):
            var vr = block * rows_per_block + r
            var sr = sf_block * rows_per_block + r
            for d in range(V_DEPTH):
                var packed = Int(fp4_host[vr * PACKED_ROW + d // 2])
                var nibble = (packed >> (4 * (d % 2))) & 15
                var value = magnitudes[nibble & 7] * Float32(
                    -1 if nibble & 8 else 1
                )
                host[vr * QK_DEPTH + d] = (
                    value * sf_host[sr * SF_PER_ROW + d // 16].cast[.float32]()
                ).cast[.float8_e4m3fn]()
            for d in range(ROPE_DEPTH):
                host[vr * QK_DEPTH + V_DEPTH + d] = bitcast[
                    DType.float8_e4m3fn, 1
                ](
                    SIMD[DType.uint8, 1](
                        fp4_host[vr * PACKED_ROW + V_DEPTH // 2 + d]
                    )
                )[
                    0
                ]
    ctx.enqueue_copy(fp8_reference, host)
    var fp8_coll = FP8Collection(
        rebind[FP8Collection.blocks_tt_type](
            TileTensor(
                fp8_reference.unsafe_ptr(),
                row_major(
                    Int64(pool_blocks),
                    Idx[1],
                    Int64(NUM_LAYERS),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[QK_DEPTH],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(
            cache_lengths_dev, row_major(Int64(batch_size))
        ).as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(Int64(batch_size), Int64(lut_cols)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(max_seq),
        UInt32(max_context),
    )
    var fp8_kv = KVCacheMHAOperand(fp8_coll.get_key_cache(LAYER))
    var args = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size, max_context, max_seq, ctx
    )

    var q_tt = TileTensor(
        q_dev, row_major((total_q, Idx[num_heads], Idx[QK_DEPTH]))
    )
    var out_ref_tt = TileTensor(
        out_ref_dev, row_major((total_q, Idx[num_heads], Idx[V_DEPTH]))
    )
    var out_tt = TileTensor(
        out_dev, row_major((total_q, Idx[num_heads], Idx[V_DEPTH]))
    )
    var wide_tt = TileTensor(wide_dev, row_major(wide_size))
    var topk_lengths_tt = TileTensor(topk_lengths_dev, row_major(total_q))
    var row_offsets_tt = TileTensor(row_offsets_dev, row_major(batch_size + 1))
    var wide_ptr = (
        wide_dev.unsafe_ptr()
        .unsafe_bitcast[Int32]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )

    var attn_sink_ptr = Optional[ImmPointer[Float32, ImmutAnyOrigin]](None)
    if with_sink:
        attn_sink_ptr = Optional[ImmPointer[Float32, ImmutAnyOrigin]](
            sink_dev.unsafe_ptr()
            .unsafe_bitcast[Float32]()
            .as_unsafe_any_origin()
        )
    # The tiling the production dispatch picks: head<=64 runs the single-CTA
    # tile, head=128 the 2-CTA tile.
    comptime cta_group = 2 if num_heads == 128 else 1
    comptime config = MLASparseConfig[
        DType.bfloat16,
        b_topk_=128 if cta_group == 2 else 64,
        num_mbars_=2 if cta_group == 2 else 4,
        cta_group_=cta_group,
    ](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        input_qk_depth=QK_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=2 * topk,
        group=num_heads,
    )
    comptime scale = Float32(1.0) / 13.856406

    var ref_indices = ctx.enqueue_create_buffer[.int32](total_q * topk)
    var idx_host = ctx.enqueue_create_host_buffer[.int32](total_q * topk)
    for t in range(total_q):
        for j in range(topk):
            idx_host[t * topk + j] = Int32(wide_host[t * 2 * topk + j])
    ctx.enqueue_copy(ref_indices, idx_host)

    var ref_indices_ptr = ref_indices.unsafe_ptr().as_unsafe_any_origin()

    def _launch_ref(ctx: DeviceContext[]) raises {imm}:
        @inline(.always)
        def launch[SinkType: OptionalPointer](sink: SinkType) raises {imm}:
            mla_decode_sm100_dispatch[
                q_tt.dtype,
                type_of(fp8_kv),
                out_ref_tt.dtype,
                NullMask,
                config=MHAConfig[q_tt.dtype](num_heads, QK_DEPTH),
                depth=QK_DEPTH,
                num_heads=num_heads,
                group=num_heads,
                ragged=True,
                sparse=True,
                physical_indices=True,
                TopkLengthsPtrType=NonNullPointer[DType.int32],
                AttnSinkPtrType=SinkType,
                skip_masked_tiles=True,
            ](
                q_tt,
                fp8_kv,
                out_ref_tt,
                scale,
                row_offsets_tt,
                NullMask(),
                args.gpu_tile_tensor(),
                batch_size,
                max_seq,
                max_context,
                ctx,
                d_indices=ref_indices_ptr,
                indices_stride=topk,
                topk_lengths=NonNullPointer[DType.int32](
                    topk_lengths_tt.ptr.unsafe_bitcast[Int32]()
                    .as_imm()
                    .as_unsafe_any_origin()
                ),
                attn_sink_ptr=sink,
                num_partitions_in=Optional[Int](1),
            )

        if case .Some(sink) = attn_sink_ptr:
            launch(NonNullPointer[DType.float32](sink))
        else:
            launch(NullPointer[DType.float32]())

    def _launch_once(ctx: DeviceContext[]) raises {imm}:
        @inline(.always)
        def launch[SinkType: OptionalPointer](sink: SinkType) raises {imm}:
            mla_prefill_sparse_nvfp4_via_fp8[
                config=config,
                group=num_heads,
                q_depth=QK_DEPTH,
            ](
                out_tt,
                q_tt,
                kv,
                wide_ptr,
                2 * topk,
                topk_lengths_tt,
                row_offsets_tt,
                sink,
                scale,
                ctx,
                num_partitions_in=Optional[Int](1),
            )

        if case .Some(sink) = attn_sink_ptr:
            launch(NonNullPointer[DType.float32](sink))
        else:
            launch(NullPointer[DType.float32]())

    _launch_ref(ctx)
    _launch_once(ctx)
    ctx.synchronize()

    var out_ref = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var out = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_ref, out_ref_dev)
    ctx.enqueue_copy(out, out_dev)
    ctx.synchronize()

    var mismatches = 0
    var nonzero = 0
    var nonfinite = 0
    for i in range(out_size):
        var a = bitcast[DType.uint16, 1](SIMD[DType.bfloat16, 1](out_ref[i]))[0]
        var c = bitcast[DType.uint16, 1](SIMD[DType.bfloat16, 1](out[i]))[0]
        if a != c:
            if mismatches < 4:
                print("  mismatch at", i, ":", out_ref[i], out[i])
            mismatches += 1
        var v = out_ref[i].cast[.float32]()
        if abs(v) > 1e-6:
            nonzero += 1
        if v != v or abs(v) > 1e30:
            nonfinite += 1
    print(t"  mismatches: {mismatches}  of {out_size}  nonzero: {nonzero}")
    assert_equal(nonfinite, 0)
    if n_valid > 0:
        assert_true(nonzero > out_size // 2, "oracle output is mostly zero")
    assert_equal(mismatches, 0)

    comptime if perf_iters > 0:
        for _ in range(3):
            _launch_ref(ctx)
            _launch_once(ctx)
        ctx.synchronize()
        var ref_us = _us(
            ctx.execution_time(_launch_ref, perf_iters), perf_iters
        )
        var once_us = _us(
            ctx.execution_time(_launch_once, perf_iters), perf_iters
        )
        print(
            t"PERF {name}  fp8_ref_us: {ref_us}  convert_once_us: {once_us} "
            t" scratch_rows:"
            t" {nvfp4_prefill_scratch_rows[PAGE_SIZE](batch_size, max_context)}"
        )

    _ = fp4_dev
    _ = sf_dev
    _ = cache_lengths_dev
    _ = row_offsets_dev
    _ = lut_dev
    _ = scales_lut_dev
    _ = q_dev
    _ = wide_dev
    _ = topk_lengths_dev
    _ = sink_dev
    _ = out_ref_dev
    _ = out_dev


def test_noncausal_lengths[
    topk: Int = 4, invalid_slots: Bool = False, heads: Int = 8
](ctx: DeviceContext[]) raises:
    comptime tokens = 4
    var packed_host = ctx.enqueue_create_host_buffer[.uint8](
        2 * NUM_LAYERS * PAGE_SIZE * PACKED_ROW
    )
    for i in range(len(packed_host)):
        packed_host[i] = 0
    packed_host[(LAYER * PAGE_SIZE + 3) * PACKED_ROW] = 6
    var packed = ctx.enqueue_create_buffer[.uint8](len(packed_host))
    ctx.enqueue_copy(packed, packed_host)
    var sf = ctx.enqueue_create_buffer[.float8_e4m3fn](
        2 * NUM_LAYERS * PAGE_SIZE * SF_PER_ROW
    )
    sf.enqueue_fill(1)
    var prefix = ctx.enqueue_create_buffer[.uint32](1)
    prefix.enqueue_fill(0)
    var lut = ctx.enqueue_create_buffer[.uint32](1)
    lut.enqueue_fill(0)
    var offsets_host = ctx.enqueue_create_host_buffer[.uint32](2)
    offsets_host[0] = 0
    offsets_host[1] = tokens
    var offsets = ctx.enqueue_create_buffer[.uint32](2)
    ctx.enqueue_copy(offsets, offsets_host)
    var indices_host = ctx.enqueue_create_host_buffer[.int32](tokens * 2 * topk)
    for t in range(tokens):
        for j in range(topk):
            var row = j if topk == 4 else (j if j < 3 else (0 if j == 3 else 3))
            comptime if invalid_slots:
                row = -1 if j == 3 or j == 7 else (j if j < 3 else 3)
            indices_host[t * 2 * topk + j] = Int32(row)
            indices_host[t * 2 * topk + topk + j] = Int32(row)
    var indices = ctx.enqueue_create_buffer[.int32](len(indices_host))
    ctx.enqueue_copy(indices, indices_host)
    # A causal row's live count is the token's `min(top_k, causal count)` slots,
    # so those entries are packed and no slot below that count is `-1`. The
    # causal form's mask-free prefix takes that count from the bound alone, so
    # the causal launches below read this all-live row; the null-masked
    # launches keep the `-1` padding above, since they mask a slot by its index
    # wherever it sits.
    var causal_indices_host = ctx.enqueue_create_host_buffer[.int32](
        tokens * 2 * topk
    )
    for t in range(tokens):
        for j in range(topk):
            var row = j if topk == 4 else (j if j < 3 else (0 if j == 3 else 3))
            causal_indices_host[t * 2 * topk + j] = Int32(row)
            causal_indices_host[t * 2 * topk + topk + j] = Int32(row)
    var causal_indices = ctx.enqueue_create_buffer[.int32](
        len(causal_indices_host)
    )
    ctx.enqueue_copy(causal_indices, causal_indices_host)
    var lengths = ctx.enqueue_create_buffer[.uint32](tokens)
    lengths.enqueue_fill(UInt32(topk))
    var q = ctx.enqueue_create_buffer[.float8_e4m3fn](tokens * heads * QK_DEPTH)
    q.enqueue_fill(0)
    var out = ctx.enqueue_create_buffer[.bfloat16](tokens * heads * V_DEPTH)
    var collection = CollectionFP4(
        rebind[CollectionFP4.blocks_tt_type](
            TileTensor(
                packed,
                row_major(
                    Int64(2),
                    Idx[1],
                    Int64(NUM_LAYERS),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[PACKED_ROW],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(prefix, row_major(Int64(1))).as_unsafe_any_origin(),
        TileTensor(lut, row_major(Int64(1), Int64(1)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(tokens),
        UInt32(tokens),
        rebind[CollectionFP4.scales_tt_type](
            TileTensor(
                sf,
                row_major(
                    Int64(2),
                    Idx[1],
                    Int64(NUM_LAYERS),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[SF_PER_ROW],
                ),
            ).as_unsafe_any_origin()
        ),
    )
    comptime cta_group = 2 if heads == 128 else 1
    comptime config = MLASparseConfig[
        DType.bfloat16,
        b_topk_=128 if cta_group == 2 else 64,
        num_mbars_=2 if cta_group == 2 else 4,
        cta_group_=cta_group,
    ](
        num_q_heads=heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        input_qk_depth=QK_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=2 * topk,
        group=heads,
    )
    mla_prefill_sparse_nvfp4_via_fp8[
        config=config, group=heads, q_depth=QK_DEPTH
    ](
        TileTensor(out, row_major(tokens, Idx[heads], Idx[V_DEPTH])),
        TileTensor(q, row_major(tokens, Idx[heads], Idx[QK_DEPTH])),
        collection.get_key_cache(LAYER),
        indices.unsafe_ptr().as_unsafe_any_origin(),
        2 * topk,
        TileTensor(lengths, row_major(tokens)),
        TileTensor(offsets, row_major(2)),
        NullPointer[DType.float32](),
        Float32(1),
        ctx,
        num_partitions_in=Optional[Int](1),
    )
    var host = ctx.enqueue_create_host_buffer[.bfloat16](len(out))
    ctx.enqueue_copy(host, out)
    ctx.synchronize()
    for t in range(tokens):
        for h in range(heads):
            assert_equal(
                host[(t * heads + h) * V_DEPTH],
                Scalar[DType.bfloat16](1 if topk == 4 else 2),
            )
    # Deliberately overstate lengths to test the defensive causal cap and
    # ensure the causal specialization is forwarded to the shared launcher.
    mla_prefill_sparse_nvfp4_via_fp8[
        config=config,
        group=heads,
        q_depth=QK_DEPTH,
        causal=True,
    ](
        TileTensor(out, row_major(tokens, Idx[heads], Idx[V_DEPTH])),
        TileTensor(q, row_major(tokens, Idx[heads], Idx[QK_DEPTH])),
        collection.get_key_cache(LAYER),
        causal_indices.unsafe_ptr().as_unsafe_any_origin(),
        2 * topk,
        TileTensor(lengths, row_major(tokens)),
        TileTensor(offsets, row_major(2)),
        NullPointer[DType.float32](),
        Float32(1),
        ctx,
        num_partitions_in=Optional[Int](1),
    )
    ctx.enqueue_copy(host, out)
    ctx.synchronize()
    for t in range(tokens):
        for h in range(heads):
            assert_equal(
                host[(t * heads + h) * V_DEPTH],
                Scalar[DType.bfloat16](1 if topk == 4 and t == 3 else 0),
            )

    var q_tt = TileTensor(q, row_major(tokens, Idx[heads], Idx[QK_DEPTH]))
    var out_tt = TileTensor(out, row_major(tokens, Idx[heads], Idx[V_DEPTH]))
    var indices_ptr = indices.unsafe_ptr().as_unsafe_any_origin()
    var lengths_tt = TileTensor(lengths, row_major(tokens))
    var offsets_tt = TileTensor(offsets, row_major(2))
    var cache = collection.get_key_cache(LAYER)
    comptime if invalid_slots:
        mla_sparse_attention_flat[num_heads=heads, depth=QK_DEPTH](
            out_tt,
            q_tt,
            cache,
            indices_ptr,
            2 * topk,
            lengths_tt,
            Optional[ImmPointer[Float32, ImmutAnyOrigin]](None),
            Float32(1),
            ctx,
        )
        ctx.enqueue_copy(host, out)
        ctx.synchronize()
        for i in range(len(host)):
            assert_equal(
                host[i], Scalar[DType.bfloat16](2 if i % V_DEPTH == 0 else 0)
            )

    comptime if invalid_slots:
        prefix.enqueue_fill(8)
        var causal_cache = cache
        causal_cache.max_cache_length = UInt32(8 + tokens)
        mla_prefill_sparse_nvfp4_via_fp8[
            config=config,
            group=heads,
            q_depth=QK_DEPTH,
            causal=True,
        ](
            out_tt,
            q_tt,
            causal_cache,
            causal_indices.unsafe_ptr().as_unsafe_any_origin(),
            2 * topk,
            lengths_tt,
            offsets_tt,
            NullPointer[DType.float32](),
            Float32(1),
            ctx,
            num_partitions_in=Optional[Int](1),
        )
        ctx.enqueue_copy(host, out)
        ctx.synchronize()
        for i in range(len(host)):
            assert_equal(
                host[i], Scalar[DType.bfloat16](2 if i % V_DEPTH == 0 else 0)
            )
        var sink = ctx.enqueue_create_buffer[DType.float32](heads)
        sink.enqueue_fill(0)
        indices.enqueue_fill(-1)
        # No live entry: a row of `-1` slots with a length above zero states
        # more entries than it fills, which no producer does.
        lengths.enqueue_fill(0)
        for with_sink in [False, True]:

            @inline(.always)
            def run[SinkType: OptionalPointer](sink_arg: SinkType) raises {imm}:
                mla_prefill_sparse_nvfp4_via_fp8[
                    config=config,
                    group=heads,
                    q_depth=QK_DEPTH,
                    causal=True,
                ](
                    out_tt,
                    q_tt,
                    causal_cache,
                    indices_ptr,
                    2 * topk,
                    lengths_tt,
                    offsets_tt,
                    sink_arg,
                    Float32(1),
                    ctx,
                    num_partitions_in=Optional[Int](1),
                )

            if with_sink:
                run(
                    NonNullPointer[DType.float32](
                        sink.unsafe_ptr().as_imm().as_unsafe_any_origin()
                    )
                )
            else:
                run(NullPointer[DType.float32]())
            ctx.enqueue_copy(host, out)
            ctx.synchronize()
            for i in range(len(host)):
                assert_equal(host[i], Scalar[DType.bfloat16](0))
        ctx.enqueue_copy(indices, indices_host)
        lengths.enqueue_fill(UInt32(topk))
        prefix.enqueue_fill(0)

    comptime if invalid_slots:
        comptime split_slots = 256
        var split_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            tokens * 2 * split_slots
        )
        for t in range(tokens):
            for j in range(split_slots):
                var row = Int32(0 if j < split_slots // 2 else 3)
                split_indices_host[t * 2 * split_slots + j] = row
                split_indices_host[t * 2 * split_slots + split_slots + j] = row
        var split_indices = ctx.enqueue_create_buffer[DType.int32](
            len(split_indices_host)
        )
        ctx.enqueue_copy(split_indices, split_indices_host)
        var split_lengths = ctx.enqueue_create_buffer[DType.uint32](tokens)
        split_lengths.enqueue_fill(UInt32(split_slots))
        var split_lengths_tt = TileTensor(split_lengths, row_major(tokens))
        mla_sparse_attention[num_heads=heads, depth=QK_DEPTH](
            out_tt,
            q_tt,
            cache,
            split_indices.unsafe_ptr().as_unsafe_any_origin(),
            2 * split_slots,
            split_lengths_tt,
            offsets_tt,
            NullPointer[DType.float32](),
            Float32(1),
            8,
            12,
            ctx,
            num_partitions_in=Optional[Int](4),
        )
        ctx.enqueue_copy(host, out)
        ctx.synchronize()
        for i in range(len(host)):
            assert_equal(
                host[i], Scalar[DType.bfloat16](2 if i % V_DEPTH == 0 else 0)
            )
        _ = split_indices^
        _ = split_lengths^

    var scratch_owner = nvfp4_prefill_workspace[config=config](
        q_tt, cache, offsets_tt, ctx
    )
    var scratch_view = scratch_owner.view()
    var workspace = Optional[NVFP4PrefillWorkspaceView](scratch_view)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            mla_prefill_sparse_nvfp4_via_fp8[
                config=config, group=heads, q_depth=QK_DEPTH
            ](
                out_tt,
                q_tt,
                cache,
                indices_ptr,
                2 * topk,
                lengths_tt,
                offsets_tt,
                NullPointer[DType.float32](),
                Float32(1),
                rec,
                num_partitions_in=Optional[Int](1),
                workspace=workspace,
            )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.enqueue_copy(host, out)
    ctx.synchronize()
    for i in range(len(host)):
        assert_equal(
            host[i],
            Scalar[DType.bfloat16](
                (1 if topk == 4 else 2) if i % V_DEPTH == 0 else 0
            ),
        )
    var foreign_fp8 = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
        scratch_view.fp8_capacity
    )
    var foreign_pages = ctx.enqueue_create_buffer[DType.uint32](
        scratch_view.page_slots_capacity
    )
    var foreign_indices = ctx.enqueue_create_buffer[DType.int32](
        scratch_view.indices_capacity
    )
    foreign_fp8.enqueue_fill(3)
    foreign_pages.enqueue_fill(UInt32(0xA5A5A5A5))
    foreign_indices.enqueue_fill(Int32(0x55555555))
    # Reuse the scalar constructor's host stack with unrelated metadata.
    for i in range(32):
        var unrelated = MLADispatchScalarArgs[num_heads=heads, is_fp8_kv=True](
            7 + i, 31 + i, 19 + i, ctx
        )
        _ = unrelated^
    out.enqueue_fill(-1)
    ctx.synchronize()
    graph.replay()
    ctx.enqueue_copy(host, out)
    ctx.synchronize()
    for i in range(len(host)):
        assert_equal(
            host[i],
            Scalar[DType.bfloat16](
                (1 if topk == 4 else 2) if i % V_DEPTH == 0 else 0
            ),
        )

    var mixed = ctx.enqueue_create_host_buffer[.uint32](tokens)
    mixed[0] = 0
    mixed[1] = UInt32(topk)
    mixed[2] = 1
    mixed[3] = 3 if topk == 4 else 4
    ctx.enqueue_copy(lengths, mixed)
    out.enqueue_fill(-1)
    graph.replay()
    ctx.enqueue_copy(host, out)
    ctx.synchronize()
    for t in range(tokens):
        for h in range(heads):
            for d in range(V_DEPTH):
                var expected = Float32(0)
                if t == 1 and d == 0:
                    expected = Float32(1 if topk == 4 else 2)
                assert_equal(
                    host[(t * heads + h) * V_DEPTH + d],
                    expected.cast[.bfloat16](),
                )

    var foreign_fp8_host = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](
        len(foreign_fp8)
    )
    var foreign_pages_host = ctx.enqueue_create_host_buffer[DType.uint32](
        len(foreign_pages)
    )
    var foreign_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
        len(foreign_indices)
    )
    ctx.enqueue_copy(foreign_fp8_host, foreign_fp8)
    ctx.enqueue_copy(foreign_pages_host, foreign_pages)
    ctx.enqueue_copy(foreign_indices_host, foreign_indices)
    ctx.synchronize()
    for i in range(len(foreign_fp8_host)):
        assert_equal(
            bitcast[DType.uint8, 1](foreign_fp8_host[i]),
            bitcast[DType.uint8, 1](Float32(3).cast[DType.float8_e4m3fn]()),
        )
    for i in range(len(foreign_pages_host)):
        assert_equal(foreign_pages_host[i], UInt32(0xA5A5A5A5))
    for i in range(len(foreign_indices_host)):
        assert_equal(foreign_indices_host[i], Int32(0x55555555))
    _ = graph^
    ctx.synchronize()
    _ = scratch_owner^
    _ = foreign_fp8^
    _ = foreign_pages^
    _ = foreign_indices^
    _ = packed^
    _ = sf^
    _ = prefix^
    _ = lut^
    _ = offsets^
    _ = indices^
    _ = causal_indices^
    _ = lengths^
    _ = q^
    _ = out^


def main() raises:
    test_gate()
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            comptime if get_defined_int[
                "MLA_NVFP4_HELPER_CANARY_ONLY", 0
            ]() != 0:
                test_noncausal_lengths[8, invalid_slots=True](ctx)
                test_noncausal_lengths[4](ctx)
                test_noncausal_lengths[8](ctx)
                comptime if get_defined_int[
                    "MLA_SPARSE_SENTINEL_BALLOT", 1
                ]() == 1:
                    test_noncausal_lengths[8, invalid_slots=True, heads=40](ctx)
                    test_noncausal_lengths[8, invalid_slots=True, heads=48](ctx)
                    test_noncausal_lengths[8, invalid_slots=True, heads=56](ctx)
                    test_noncausal_lengths[8, invalid_slots=True, heads=64](ctx)
                    test_noncausal_lengths[8, invalid_slots=True, heads=128](
                        ctx
                    )
            else:
                test_noncausal_lengths[4](ctx)
                # Repeated physical rows make the selected count exceed context.
                test_noncausal_lengths[8](ctx)
                # Six live entries sum to 12; masked slots must not dilute the mean.
                test_noncausal_lengths[8, invalid_slots=True](ctx)
                seed(42)
                # A first chunk: no prefix.
                run_case[8]("h8_first_chunk", [0], [512], ctx)
                # Chunked prefill: a prefix from earlier chunks plus a new chunk.
                run_case[8]("h8_chunk", [3000], [96], ctx)
                # Ragged batch: prefixes and chunks of different lengths.
                run_case[8]("h8_b3_ragged", [4000, 0, 1500], [40, 300, 9], ctx)
                # Two entries share their first two blocks (a cached prefix).
                run_case[8](
                    "h8_b2_shared_prefix",
                    [700, 400],
                    [20, 33],
                    ctx,
                    shared_prefix_pages=2,
                )
                run_case[8](
                    "h8_masked_sink",
                    [2000, 900],
                    [12, 30],
                    ctx,
                    valid_topk=1000,
                    with_sink=True,
                )
                run_case[8, topk=128](
                    "h8_tiny", [60, 10], [4, 30], ctx, valid_topk=5
                )
                run_case[16]("h16_b2", [2500, 300], [8, 64], ctx)
                run_case[64](
                    "h64_b2_sink", [3000, 1200], [8, 16], ctx, with_sink=True
                )
                run_case[128]("h128_b2", [4000, 2000], [8, 24], ctx)
                run_case[128, topk=128](
                    "h128_tiny", [60, 10], [4, 30], ctx, valid_topk=5
                )

                # Top-k widths off the 64-key tile, permuted scale pages, and an
                # empty selection with a sink.
                run_case[8, topk=63](
                    "decode_h8_noncausal_tail",
                    [0, 17],
                    [4, 2],
                    ctx,
                    valid_topk=65,
                )
                run_case[64, topk=129](
                    "decode_h64_permuted_sf",
                    [256, 129],
                    [2, 3],
                    ctx,
                    valid_topk=129,
                    with_sink=True,
                )
                run_case[128, topk=257](
                    "decode_h128_tail", [256], [2], ctx, valid_topk=257
                )
                run_case[64, topk=128](
                    "decode_h64_empty_sink",
                    [128],
                    [2],
                    ctx,
                    valid_topk=0,
                    with_sink=True,
                )

                # Timing (prints PERF lines). GLM 5.3 runs 8 heads per GPU.
                run_case[8, perf_iters=20]("perf_h8_kv8k", [6144], [2048], ctx)
                run_case[8, perf_iters=10](
                    "perf_h8_8k_on_8k", [8192], [8192], ctx
                )
                run_case[8, perf_iters=10](
                    "perf_h8_8k_on_128k", [131072], [8192], ctx
                )
                print("convert-once: all cases match the FP8 oracle")
