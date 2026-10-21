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

The oracle is `mla_prefill_sparse_qkv_fp8` on the packed NVFP4 cache, which
expands every selected row in shared memory once per query. The convert-once
path expands the batch's whole context to FP8 once and runs the same kernel on
that, so it must match the oracle bit for bit.

The cache has several layers and the cases read a middle one, the page tables
are shuffled and the scales have their own, entries are ragged prefixes plus a
new chunk whose own rows are selected too, and two entries can share their
first pages, as a cached prefix does. `PERF` lines time the oracle and the
convert-once path.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import randn, seed
from std.testing import assert_equal, assert_false, assert_true

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.nvidia.sm100.mla_nvfp4_kv_to_fp8 import (
    mla_prefill_sparse_nvfp4_via_fp8,
    nvfp4_prefill_converts_once,
    nvfp4_prefill_scratch_rows,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
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
    ctx: DeviceContext,
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
        "test:",
        name,
        " batch:",
        batch_size,
        " heads:",
        num_heads,
        " q_tokens:",
        total_q,
        " max_context:",
        max_context,
        " topk:",
        topk,
        " valid:",
        n_valid,
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

    def _launch_ref(ctx: DeviceContext) raises {imm}:
        mla_prefill_sparse_qkv_fp8[
            config=config,
            group=num_heads,
            q_depth=QK_DEPTH,
            scale_block_size=0,
        ](
            out_ref_tt,
            q_tt,
            kv,
            wide_tt,
            topk_lengths_tt,
            attn_sink_ptr,
            scale,
            Int32(2 * topk),
            ctx,
        )

    def _launch_once(ctx: DeviceContext) raises {imm}:
        mla_prefill_sparse_nvfp4_via_fp8[
            config=config, group=num_heads, q_depth=QK_DEPTH
        ](
            out_tt,
            q_tt,
            kv,
            wide_ptr,
            2 * topk,
            topk_lengths_tt,
            row_offsets_tt,
            attn_sink_ptr,
            scale,
            ctx,
        )

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
    print("  mismatches:", mismatches, " of", out_size, " nonzero:", nonzero)
    assert_equal(nonfinite, 0)
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
            "PERF",
            name,
            " nvfp4_us:",
            ref_us,
            " convert_once_us:",
            once_us,
            " scratch_rows:",
            nvfp4_prefill_scratch_rows[PAGE_SIZE](batch_size, max_context),
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


def main() raises:
    test_gate()
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
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

            # Timing (prints PERF lines). GLM 5.3 runs 8 heads per GPU.
            run_case[8, perf_iters=20]("perf_h8_kv8k", [6144], [2048], ctx)
            run_case[8, perf_iters=10]("perf_h8_8k_on_8k", [8192], [8192], ctx)
            run_case[8, perf_iters=10](
                "perf_h8_8k_on_128k", [131072], [8192], ctx
            )
            print("convert-once: all cases match the NVFP4 oracle")
