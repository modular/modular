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
"""Tests for the MXFP4 lightning indexer, `mla_indexer_ragged_mxfp4_paged`.

Keys sit in `indexer_k_store`'s page layout and queries in vLLM's MXFP4
`fused_indexer_q_rope_quant` layout, quantized by their rule (or, for keys,
written by the store itself). A float64 host reference scores every live
(token, key): the scorer's scores are checked against it one by one, slots
past each row's bound must stay unwritten, and the op's top-k is checked for
validity and graded against the exact top-k. Building with
`-D FP8_INDEX_FOLD_DTYPE=float32` tightens the score check to the f32 fold.

Each poison corrupts one input after the reference is taken, the way one class
of kernel bug would, and the run asserts that some check catches it.
"""

from std.collections import Set
from std.math import align_up, ceil, ceildiv, cos, exp2, log2, sin
from std.memory import bitcast
from std.random import random_float64, random_ui64, seed
from std.testing import assert_true
from layout import TensorLayout, TileTensor, row_major
from layout.tile_tensor import ImmTileTensor, MutTileTensor
from max.gpu.host import DeviceContext, HostBuffer
from nn.attention.gpu.mla_index_fp8 import (
    _SCORES_BUDGET_BYTES,
    MXFP4IndexKCache,
    mla_indexer_ragged_mxfp4_paged,
    mxfp4_index_k_cache,
)
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    KTMATileT,
    mxfp4_index_score_sm100,
)
from nn.attention.gpu.sparse_index_fp8_sm100_prefill import (
    _FOLD_DTYPE,
    _index_swizzle,
    _k_box_rows,
)
from nn.attention.mha_mask import MaskName
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    KVCacheScalesMHAOperand,
)
from nn.indexer_k_store import (
    MXFP4_INDEX_RECORD_BYTES,
    indexer_k_norm_rope_store_layer,
)

from _index_select_grade import SelectionStats, grade_selection_row

comptime NH = 32
comptime DEPTH = 128
comptime ROW_BYTES = DEPTH // 2
comptime _E2M1 = SIMD[DType.float64, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
comptime _SENTINEL = Float32(-12345.0)

# Poisons. Each corrupts the device copy of one input after the reference is
# taken, mimicking one kernel bug class.
comptime P_NONE = 0
# K scale bytes 2-3 replaced by 0-1: the second MMA reads the first's scales.
comptime P_K_SF_ID = 1
# q block scales rotated by one head: a wrong B scale-factor permutation.
comptime P_Q_SF_HEAD = 2
# Key bytes 32-63 zeroed: the second K step of the contraction dropped.
comptime P_K_HALF = 3
# Key nibbles swapped: the wrong channel order within a byte.
comptime P_NIBBLES = 4
# Entry 0's first two logical pages swapped: a wrong page lookup.
comptime P_PAGE = 5
# Weights rotated by one head: the fold reading the wrong column's weight.
comptime P_WEIGHTS = 6
# Entry 0 one key row deeper than the reference: an off-by-one bound.
comptime P_BOUND = 7
comptime _NUM_POISONS = 8


def _fp4_code(value: Float32) -> UInt8:
    """Nearest-even E2M1 code, as `indexer_k_store` and vLLM round."""
    var magnitude = abs(value)
    var code: UInt8 = 0
    if magnitude > 5.0:
        code = 7
    elif magnitude >= 3.5:
        code = 6
    elif magnitude > 2.5:
        code = 5
    elif magnitude >= 1.75:
        code = 4
    elif magnitude > 1.25:
        code = 3
    elif magnitude >= 0.75:
        code = 2
    elif magnitude > 0.25:
        code = 1
    return code | UInt8((bitcast[.uint32](value) >> 28) & 8)


def _random_row(mut values: List[Float32]):
    """Fills 128 values whose four 32-channel groups differ in magnitude.

    Distinct group exponents are what tell a right scale byte from a wrong
    one; an occasional all-zero row exercises the smallest scale.
    """
    var zero_row = random_ui64(0, 63) == 0
    for g in range(4):
        var amp = exp2(Float64(Int(random_ui64(0, 6))) - 3.0)
        for c in range(32):
            values[g * 32 + c] = 0.0 if zero_row else Float32(
                random_float64(-1.0, 1.0) * amp
            )


def _quantize_row(
    values: List[Float32], mut codes: HostBuffer[.uint8], codes_off: Int
) -> SIMD[DType.uint8, 4]:
    """Packs 128 values as 64 E2M1 bytes; returns the four UE8M0 scales."""
    var scales = SIMD[DType.uint8, 4]()
    for g in range(4):
        var peak = Float32(6.0 * 2.0**-126)
        for c in range(32):
            peak = max(peak, abs(values[g * 32 + c]))
        var exponent = ceil(log2(peak * (1.0 / 6.0)))
        scales[g] = UInt8(exponent + 127)
        for pair in range(16):
            var lo = _fp4_code(values[g * 32 + 2 * pair] * exp2(-exponent))
            var hi = _fp4_code(values[g * 32 + 2 * pair + 1] * exp2(-exponent))
            codes[codes_off + g * 16 + pair] = lo | (hi << 4)
    return scales


def _decode_row(
    codes: HostBuffer[.uint8],
    codes_off: Int,
    scale_bytes: SIMD[DType.uint8, 4],
    mut out: List[Float64],
    out_off: Int,
):
    """Decodes one MXFP4 row; channel `2j` is byte `j`'s low nibble."""
    for c in range(DEPTH):
        var byte = codes[codes_off + c // 2]
        var nib = (byte >> 4) if c % 2 == 1 else (byte & 0xF)
        var v = _E2M1[Int(nib & 7)]
        if nib & 8:
            v = -v
        out[out_off + c] = v * exp2(Float64(Int(scale_bytes[c // 32])) - 127.0)


def _score_mxfp4[
    output_layout: TensorLayout,
    q_layout: TensorLayout,
    w_layout: TensorLayout,
    iro_layout: TensorLayout,
    //,
    kpool: Int,
    page_size: Int,
](
    output: MutTileTensor[.float32, output_layout, _],
    q: ImmTileTensor[.uint8, q_layout, _],
    weights: ImmTileTensor[.float32, w_layout, _],
    q_sf: ImmPointer[UInt32, ImmutAnyOrigin],
    input_row_offsets: ImmTileTensor[.uint32, iro_layout, _],
    k_cache: MXFP4IndexKCache[page_size],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    causal: Bool,
    ctx: DeviceContext,
) raises:
    """Runs the scorer alone, as the op does, into an f32 score matrix."""
    var k_op = KVCacheMHAOperand(k_cache)
    var ks_op = KVCacheScalesMHAOperand(k_cache)
    comptime k_box = _k_box_rows[_BM_KEY, page_size, 1]()
    var k_tma = rebind[KTMATileT[DType.uint8, k_box, ROW_BYTES]](
        k_cache.create_paged_tma_tile[
            _index_swizzle[DType.uint8, ROW_BYTES](),
            BN=k_box,
            BK=ROW_BYTES,
        ](ctx)
    )
    mxfp4_index_score_sm100[
        type_of(k_op),
        type_of(ks_op),
        NH,
        _is_cache_length_accurate=False,
        kpool=kpool,
        k_box_rows=k_box,
    ](
        output,
        q,
        q_sf,
        weights,
        k_op,
        ks_op,
        k_tma,
        input_row_offsets,
        batch_size,
        max_seq_len,
        max_num_keys,
        causal,
        ctx,
    )


def _run_case[
    kpool: Int,
    page_size: Int,
    top_k: Int,
    causal: Bool,
](
    ctx: DeviceContext,
    seq_lens: List[Int],
    cache_lens: List[Int],
    poison: Int = P_NONE,
    pitch_pad: Int = 16,
    scores_budget_bytes: Int = _SCORES_BUDGET_BYTES,
    use_store: Bool = False,
) raises -> Int:
    """Runs one case and returns how many checks failed.

    Parameters:
        kpool: Compression ratio; one key row per `kpool` tokens.
        page_size: Key rows per cache page.
        top_k: Keys selected per token.
        causal: Whether token `i` of an entry sees only keys up to its own
            position.

    Args:
        ctx: Device context.
        seq_lens: New tokens per entry.
        cache_lens: Cached tokens per entry before this call.
        poison: Which input to corrupt after the reference, or `P_NONE`.
        pitch_pad: Bytes past the rows and words in each page.
        scores_budget_bytes: The op's score budget; small values score the
            batch in several row windows.
        use_store: Whether `indexer_k_store` writes the keys, into the second
            layer of a two-layer leaf.

    Returns:
        The number of failed checks.
    """
    var batch_size = len(seq_lens)
    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    var max_rows = 0
    for b in range(batch_size):
        total_seq_len += seq_lens[b]
        max_seq_len = max(max_seq_len, seq_lens[b])
        max_cache_len = max(max_cache_len, cache_lens[b])
        max_rows = max(max_rows, (cache_lens[b] + seq_lens[b]) // kpool)

    # One extra row's page room so the bound poison has a row to read.
    var pages_per_seq = max(1, ceildiv(max_rows + 1, page_size))
    var num_pages = batch_size * pages_per_seq + 3
    # The store path writes and reads layer 1 of a two-layer leaf; otherwise
    # each block holds one page.
    var layers = 2 if use_store else 1
    var layer = layers - 1
    var layer_off = layer * page_size * MXFP4_INDEX_RECORD_BYTES
    var page_stride = layers * page_size * MXFP4_INDEX_RECORD_BYTES + pitch_pad

    var lut = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * pages_per_seq
    )
    var used = Set[Int]()
    for i in range(batch_size * pages_per_seq):
        var page = Int(random_ui64(0, UInt64(num_pages - 1)))
        while page in used:
            page = Int(random_ui64(0, UInt64(num_pages - 1)))
        used.add(page)
        lut[i] = UInt32(page)

    # Garbage everywhere first, so a read of an unwritten row or a stray scale
    # byte (0xFF is a NaN scale) is not silently zero.
    var cache = ctx.enqueue_create_host_buffer[.uint8](num_pages * page_stride)
    for i in range(num_pages * page_stride):
        cache[i] = UInt8(random_ui64(0, 255))

    # Where entry `b`'s key row `r` lives, through the unpoisoned table.
    var lut0 = List[Int](capacity=len(lut))
    for i in range(len(lut)):
        lut0.append(Int(lut[i]))

    @inline(.always)
    def code_off(b: Int, r: Int) {imm} -> Int:
        return (
            lut0[b * pages_per_seq + r // page_size] * page_stride
            + layer_off
            + (r % page_size) * ROW_BYTES
        )

    @inline(.always)
    def scale_off(b: Int, r: Int) {imm} -> Int:
        return (
            lut0[b * pages_per_seq + r // page_size] * page_stride
            + layer_off
            + page_size * ROW_BYTES
            + (r % page_size) * 4
        )

    var values = List[Float32](length=DEPTH, fill=0.0)
    var d_cache = ctx.enqueue_create_buffer[.uint8](num_pages * page_stride)
    if use_store:
        # Every token's key through the store, which keeps only the rows that
        # close a group: `kpool` consecutive tokens per row.
        comptime key_stride = DEPTH
        var count = 0
        var max_pos = 1
        for b in range(batch_size):
            count += cache_lens[b] + seq_lens[b]
            max_pos = max(max_pos, cache_lens[b] + seq_lens[b])
        var keys = ctx.enqueue_create_host_buffer[.bfloat16](
            max(count, 1) * key_stride
        )
        var positions = ctx.enqueue_create_host_buffer[.int64](max(count, 1))
        var slots = ctx.enqueue_create_host_buffer[.int64](max(count, 1))
        var norm_w = ctx.enqueue_create_host_buffer[.bfloat16](DEPTH)
        var freqs = ctx.enqueue_create_host_buffer[.float32](max_pos * 64)
        for d in range(DEPTH):
            norm_w[d] = Float32(random_float64(0.5, 1.5)).cast[.bfloat16]()
        for p in range(max_pos):
            for d in range(32):
                var angle = random_float64(0.0, 6.283185307179586)
                freqs[p * 64 + d] = Float32(cos(angle))
                freqs[p * 64 + 32 + d] = Float32(sin(angle))
        var k = 0
        for b in range(batch_size):
            for p in range(cache_lens[b] + seq_lens[b]):
                _random_row(values)
                for d in range(DEPTH):
                    keys[k * key_stride + d] = values[d].cast[.bfloat16]()
                positions[k] = Int64(p)
                var r = p // kpool
                slots[k] = Int64(
                    lut0[b * pages_per_seq + r // page_size] * page_size
                    + r % page_size
                )
                k += 1
        var d_keys = ctx.enqueue_create_buffer[.bfloat16](len(keys))
        var d_positions = ctx.enqueue_create_buffer[.int64](len(positions))
        var d_slots = ctx.enqueue_create_buffer[.int64](len(slots))
        var d_norm_w = ctx.enqueue_create_buffer[.bfloat16](DEPTH)
        var d_freqs = ctx.enqueue_create_buffer[.float32](len(freqs))
        ctx.enqueue_copy(dst_buf=d_keys, src_buf=keys)
        ctx.enqueue_copy(dst_buf=d_positions, src_buf=positions)
        ctx.enqueue_copy(dst_buf=d_slots, src_buf=slots)
        ctx.enqueue_copy(dst_buf=d_norm_w, src_buf=norm_w)
        ctx.enqueue_copy(dst_buf=d_freqs, src_buf=freqs)
        ctx.enqueue_copy(dst_buf=d_cache, src_buf=cache)
        indexer_k_norm_rope_store_layer[True, kpool](
            TileTensor(d_keys, row_major(count, key_stride)),
            TileTensor(d_positions, row_major(max(count, 1))),
            TileTensor(d_freqs, row_major(max_pos, 64)),
            TileTensor(d_norm_w, row_major(DEPTH)),
            TileTensor(d_slots, row_major(max(count, 1))),
            TileTensor(
                d_cache,
                row_major(
                    num_pages, 1, layers, page_size, 1, MXFP4_INDEX_RECORD_BYTES
                ),
            ),
            page_stride,
            layer,
            Float32(1e-6),
            ctx,
        )
        ctx.enqueue_copy(dst_buf=cache, src_buf=d_cache)
        ctx.synchronize()
    else:
        for b in range(batch_size):
            for r in range((cache_lens[b] + seq_lens[b]) // kpool):
                _random_row(values)
                var sf = _quantize_row(values, cache, code_off(b, r))
                comptime for g in range(4):
                    cache[scale_off(b, r) + g] = sf[g]

    # Queries, their block-scale words and the per-head weights.
    var q = ctx.enqueue_create_host_buffer[.uint8](
        max(total_seq_len, 1) * NH * ROW_BYTES
    )
    var q_sf = ctx.enqueue_create_host_buffer[.uint32](
        max(total_seq_len, 1) * NH
    )
    var weights = ctx.enqueue_create_host_buffer[.float32](
        max(total_seq_len, 1) * NH
    )
    for row in range(total_seq_len * NH):
        _random_row(values)
        var sf = _quantize_row(values, q, row * ROW_BYTES)
        q_sf[row] = bitcast[DType.uint32, 1](sf)
        weights[row] = Float32(random_float64(-1.0, 1.0))

    # The float64 reference, before any poison.
    var q_val = List[Float64](length=max(total_seq_len, 1) * NH * DEPTH, fill=0)
    for row in range(total_seq_len * NH):
        var w = q_sf[row]
        _decode_row(
            q,
            row * ROW_BYTES,
            SIMD[DType.uint8, 4](
                UInt8(w & 0xFF),
                UInt8((w >> 8) & 0xFF),
                UInt8((w >> 16) & 0xFF),
                UInt8(w >> 24),
            ),
            q_val,
            row * DEPTH,
        )
    var bounds = List[Int](capacity=total_seq_len)
    var ref_scores = List[List[Float32]](capacity=total_seq_len)
    var ref_mag = List[List[Float32]](capacity=total_seq_len)
    var ref_fold = List[List[Float32]](capacity=total_seq_len)
    var k_val = List[Float64](length=max(max_rows, 1) * DEPTH, fill=0)
    for b in range(batch_size):
        var num_keys = cache_lens[b] + seq_lens[b]
        for r in range(num_keys // kpool):
            var so = scale_off(b, r)
            _decode_row(
                cache,
                code_off(b, r),
                SIMD[DType.uint8, 4](
                    cache[so], cache[so + 1], cache[so + 2], cache[so + 3]
                ),
                k_val,
                r * DEPTH,
            )
        var q_start = 0
        for bb in range(b):
            q_start += seq_lens[bb]
        for i in range(seq_lens[b]):
            var t = q_start + i
            var bound = max(
                0, (num_keys - (seq_lens[b] - 1 - i) * Int(causal)) // kpool
            )
            bounds.append(bound)
            var scores = List[Float32](capacity=bound)
            var mags = List[Float32](capacity=bound)
            var folds = List[Float32](capacity=bound)
            var qp = q_val.unsafe_ptr()
            var kp = k_val.unsafe_ptr()
            for s in range(bound):
                var score = Float64(0)
                var mag = Float64(0)
                var fold = Float64(0)
                for h in range(NH):
                    var qo = (t * NH + h) * DEPTH
                    var dot = Float64(0)
                    var absdot = Float64(0)
                    comptime for c in range(0, DEPTH, 8):
                        var prod = qp.unsafe_load[width=8](
                            qo + c
                        ) * kp.unsafe_load[width=8](s * DEPTH + c)
                        dot += prod.reduce_add()
                        absdot += abs(prod).reduce_add()
                    var w = Float64(weights[t * NH + h])
                    score += max(dot, 0.0) * w
                    fold += abs(max(dot, 0.0) * w)
                    mag += absdot * abs(w)
                scores.append(Float32(score))
                mags.append(Float32(mag))
                folds.append(Float32(fold))
            ref_scores.append(scores^)
            ref_mag.append(mags^)
            ref_fold.append(folds^)

    # Poison the device-bound copies.
    var dev_lens = List[Int](capacity=batch_size)
    for b in range(batch_size):
        dev_lens.append(cache_lens[b])
    if poison == P_K_SF_ID:
        for b in range(batch_size):
            for r in range((cache_lens[b] + seq_lens[b]) // kpool):
                var so = scale_off(b, r)
                cache[so + 2] = cache[so]
                cache[so + 3] = cache[so + 1]
    elif poison == P_Q_SF_HEAD:
        for t in range(total_seq_len):
            var first = q_sf[t * NH]
            for h in range(NH - 1):
                q_sf[t * NH + h] = q_sf[t * NH + h + 1]
            q_sf[t * NH + NH - 1] = first
    elif poison == P_K_HALF:
        for b in range(batch_size):
            for r in range((cache_lens[b] + seq_lens[b]) // kpool):
                for j in range(ROW_BYTES // 2, ROW_BYTES):
                    cache[code_off(b, r) + j] = 0
    elif poison == P_NIBBLES:
        for b in range(batch_size):
            for r in range((cache_lens[b] + seq_lens[b]) // kpool):
                for j in range(ROW_BYTES):
                    var v = cache[code_off(b, r) + j]
                    cache[code_off(b, r) + j] = (v >> 4) | (v << 4)
    elif poison == P_PAGE:
        var tmp = lut[0]
        lut[0] = lut[1]
        lut[1] = tmp
    elif poison == P_WEIGHTS:
        for t in range(total_seq_len):
            var first = weights[t * NH]
            for h in range(NH - 1):
                weights[t * NH + h] = weights[t * NH + h + 1]
            weights[t * NH + NH - 1] = first
    elif poison == P_BOUND:
        dev_lens[0] += kpool

    var iro = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var lens = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    iro[0] = 0
    for b in range(batch_size):
        iro[b + 1] = iro[b] + UInt32(seq_lens[b])
        lens[b] = UInt32(dev_lens[b])

    var d_q = ctx.enqueue_create_buffer[.uint8](len(q))
    var d_q_sf = ctx.enqueue_create_buffer[.uint32](len(q_sf))
    var d_w = ctx.enqueue_create_buffer[.float32](len(weights))
    var d_iro = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var d_lens = ctx.enqueue_create_buffer[.uint32](batch_size)
    var d_lut = ctx.enqueue_create_buffer[.uint32](len(lut))
    ctx.enqueue_copy(dst_buf=d_q, src_buf=q)
    ctx.enqueue_copy(dst_buf=d_q_sf, src_buf=q_sf)
    ctx.enqueue_copy(dst_buf=d_w, src_buf=weights)
    ctx.enqueue_copy(dst_buf=d_iro, src_buf=iro)
    ctx.enqueue_copy(dst_buf=d_lens, src_buf=lens)
    ctx.enqueue_copy(dst_buf=d_lut, src_buf=lut)
    ctx.enqueue_copy(dst_buf=d_cache, src_buf=cache)

    var k_cache = mxfp4_index_k_cache[page_size](
        TileTensor(d_cache, row_major(num_pages, page_stride)),
        TileTensor(d_lut, row_major(batch_size, pages_per_seq)),
        TileTensor(d_lens, row_major(batch_size)),
        max_seq_len,
        max_cache_len + kpool,
    )
    if use_store:
        k_cache = mxfp4_index_k_cache[page_size](
            TileTensor(
                d_cache,
                row_major(
                    num_pages, 1, layers, page_size, 1, MXFP4_INDEX_RECORD_BYTES
                ),
            ),
            page_stride,
            layer,
            TileTensor(d_lut, row_major(batch_size, pages_per_seq)),
            TileTensor(d_lens, row_major(batch_size)),
            max_seq_len,
            max_cache_len + kpool,
        )
    var q_tt = TileTensor(d_q, row_major(total_seq_len, NH, ROW_BYTES))
    var q_sf_tt = TileTensor(d_q_sf, row_major(total_seq_len, NH))
    var w_tt = TileTensor(d_w, row_major(total_seq_len, NH))
    var iro_tt = TileTensor(d_iro, row_major(batch_size + 1))

    var failures = 0
    var label = String(
        "kpool=",
        kpool,
        " page=",
        page_size,
        " top_k=",
        top_k,
        " causal=",
        causal,
        " poison=",
        poison,
        " seq=",
        seq_lens[0],
        "... cache=",
        cache_lens[0],
        "... batch=",
        batch_size,
    )

    # Scores, one by one, and the slots past each bound left alone.
    var stride = align_up(max(max_rows + 1, 1), 8)
    var d_scores = ctx.enqueue_create_buffer[.float32](
        max(total_seq_len, 1) * stride
    )
    d_scores.enqueue_fill(_SENTINEL)
    if total_seq_len > 0:
        _score_mxfp4[kpool, page_size](
            TileTensor(d_scores, row_major(total_seq_len, stride)),
            q_tt,
            w_tt,
            d_q_sf.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            iro_tt,
            k_cache,
            batch_size,
            max_seq_len,
            stride,
            causal,
            ctx,
        )
    var scores = ctx.enqueue_create_host_buffer[.float32](len(d_scores))
    ctx.enqueue_copy(dst_buf=scores, src_buf=d_scores)
    ctx.synchronize()
    comptime fold_rtol: Float32 = 1e-6 if _FOLD_DTYPE == DType.float32 else 0.08
    var score_bad = 0
    var max_err = Float32(0)
    for t in range(total_seq_len):
        for s in range(stride):
            var got = scores[t * stride + s]
            if s < bounds[t]:
                var want = ref_scores[t][s]
                var tol = (
                    3e-5 * ref_mag[t][s] + fold_rtol * ref_fold[t][s] + 1e-30
                )
                var err = abs(got - want)
                if not (err <= tol):
                    if score_bad < 4:
                        print("  score", t, s, "got", got, "want", want)
                    score_bad += 1
                max_err = max(max_err, err / max(ref_mag[t][s], 1e-30))
            elif got != _SENTINEL:
                if score_bad < 4:
                    print("  wrote past bound", t, s, bounds[t], got)
                score_bad += 1
    if score_bad > 0:
        failures += 1

    # The op's selection: valid, complete, and graded against the exact top-k.
    var d_out = ctx.enqueue_create_buffer[.int32](max(total_seq_len, 1) * top_k)
    d_out.enqueue_fill(Int32(-7))
    comptime mask_name = MaskName.CAUSAL.name if causal else MaskName.NULL.name
    var out_tt = TileTensor(d_out, row_major(total_seq_len, top_k))
    mla_indexer_ragged_mxfp4_paged[
        NH, top_k, mask_name, page_size, kpool=kpool
    ](
        out_tt,
        q_tt,
        q_sf_tt,
        w_tt,
        iro_tt,
        k_cache,
        ctx,
        scores_budget_bytes,
    )
    var out = ctx.enqueue_create_host_buffer[.int32](len(d_out))
    ctx.enqueue_copy(dst_buf=out, src_buf=d_out)
    ctx.synchronize()

    var sel_bad = 0
    var stats = SelectionStats()
    for t in range(total_seq_len):
        var eff = min(top_k, bounds[t])
        var seen = Set[Int]()
        var valid = 0
        for k in range(top_k):
            var v = Int(out[t * top_k + k])
            if v == -1:
                continue
            if v < 0 or v >= bounds[t] or v in seen:
                if sel_bad < 4:
                    print("  bad pick", t, k, v, "bound", bounds[t])
                sel_bad += 1
            seen.add(v)
            valid += 1
        if valid != eff:
            if sel_bad < 4:
                print("  row", t, "picked", valid, "of", eff)
            sel_bad += 1
        grade_selection_row(ref_scores[t], out, t, top_k, stats)
    if sel_bad > 0 or not stats.passed():
        failures += 1

    print(
        "CASE",
        label,
        "rows=",
        total_seq_len,
        "max_rows=",
        max_rows,
        "score_bad=",
        score_bad,
        "max_rel_err=",
        max_err,
        "sel_bad=",
        sel_bad,
        "->",
        "FAIL" if failures > 0 else "ok",
    )
    stats.report(label)
    return failures


def _check[
    kpool: Int,
    page_size: Int,
    top_k: Int,
    causal: Bool,
](
    ctx: DeviceContext,
    seq_lens: List[Int],
    cache_lens: List[Int],
    pitch_pad: Int = 16,
    scores_budget_bytes: Int = _SCORES_BUDGET_BYTES,
    use_store: Bool = False,
) raises:
    var failures = _run_case[kpool, page_size, top_k, causal](
        ctx,
        seq_lens,
        cache_lens,
        P_NONE,
        pitch_pad,
        scores_budget_bytes,
        use_store,
    )
    assert_true(failures == 0, "MXFP4 indexer case failed")


def main() raises:
    seed(20261015)
    var ctx = DeviceContext()

    # Decode, ratio 1 (128-row pages): prime and ragged contexts, a row with
    # one key, and contexts deep enough to split the key range.
    _check[1, 128, 512, True](ctx, [1, 1, 1, 1], [126, 508, 1020, 4092])
    _check[1, 128, 512, True](ctx, [1, 1], [16380, 0])
    _check[1, 128, 512, True](ctx, [1, 1, 1], [131070, 65534, 8190])
    # Decode, ratio 2 (64-row pages: two pages per key tile). A one-token
    # context has no key row yet.
    _check[2, 64, 512, True](ctx, [1, 1, 1, 1], [0, 1, 1021, 4093])
    _check[2, 64, 512, True](ctx, [1, 1, 1], [32765, 131071, 2])
    # MTP: several tokens per entry against ragged contexts.
    _check[1, 128, 512, True](ctx, [2, 3, 1, 4], [4093, 1021, 127, 8191])
    _check[2, 64, 512, True](ctx, [6, 3, 2], [16381, 509, 0])
    # Prefill: prime chunks, fresh and over a cached prefix, an empty entry.
    _check[1, 128, 512, True](ctx, [509], [0])
    _check[1, 128, 512, True](ctx, [127, 1, 61], [1021, 509, 0])
    _check[2, 64, 512, True](ctx, [251, 0, 37], [2048, 7, 1023])
    _check[1, 128, 512, True](ctx, [1021], [3])
    # No mask; short rows that select all their keys.
    _check[1, 128, 64, False](ctx, [3, 5, 2], [61, 2, 127])
    _check[2, 128, 512, False](ctx, [17, 9], [253, 4093])
    # Page edges: contexts ending on and one past a page, a wide page pitch.
    _check[1, 128, 512, True](ctx, [1, 1, 1], [127, 128, 255], pitch_pad=48)
    _check[2, 64, 512, True](ctx, [2, 2, 1], [126, 128, 255], pitch_pad=96)
    # Several score row windows.
    _check[1, 128, 512, True](
        ctx, [97, 61, 13], [509, 1021, 3], scores_budget_bytes=96 * 1024
    )
    # Keys written by `indexer_k_store`, both ratios.
    _check[1, 128, 512, True](ctx, [5, 1], [1021, 300], use_store=True)
    _check[2, 64, 512, True](ctx, [4, 1, 33], [2047, 600, 0], use_store=True)

    # Every poison must fail some check.
    for p in range(1, _NUM_POISONS):
        var failures = _run_case[1, 128, 512, True](
            ctx, [3, 1, 2], [1021, 4093, 300], p
        )
        failures += _run_case[2, 64, 512, True](ctx, [1, 2], [8191, 509], p)
        assert_true(failures > 0, String("poison ", p, " went undetected"))
    print("PASS mxfp4 indexer")
