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
"""Tests the MXFP4 indexer's windowed kept-block hooks.

A host reference reads the scores the scorer stores, takes each row's maximum
over every block of keys, and applies the keep rule: the block of a row's
newest visible key is always kept, `-inf` blocks never are, `-0.0` ties with
`+0.0`, and every block above the k-th best value is kept; which blocks tied
with the k-th value are kept, and the order of the ids, are free. A row
considers only the blocks up to its pinned one. The ids a row-windowed
`BlockTopKScoreHook` writes must be a set the reference allows, with `-1` in
the unused slots, and a selection masked through `BlockTopKMaskScoreHook` must
be an exact top-k of the scores with every block outside that set at `-inf`.
Each poison is a bug the checks must catch.
"""

from std.builtin.sort import sort
from std.collections import Set
from std.math import align_up, ceildiv
from std.random import random_float64, random_ui64, seed
from std.testing import assert_true
from std.utils.numerics import inf, neg_inf
from layout import TensorLayout, TileTensor, row_major
from layout.tile_tensor import ImmTileTensor, MutTileTensor
from max.gpu.host import DeviceContext, HostBuffer
from kv_cache.paged_sparse_kv_index_remap import NoRemap
from nn.attention.gpu.mla_index_fp8 import (
    _SCORES_DTYPE,
    BlockTopKMaskScoreHook,
    BlockTopKScoreHook,
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
    _index_swizzle,
    _k_box_rows,
)
from nn.attention.mha_mask import MaskName
from nn.attention.mha_operand import KVCacheMHAOperand, KVCacheScalesMHAOperand
from nn.indexer_k_store import MXFP4_INDEX_RECORD_BYTES

comptime NH = 32
comptime ROW_BYTES = 64
comptime BLOCK = 8
comptime TOKEN_TOP_K = 512

# Poisons: the source hook pins the block after the newest one; the consumer
# hook reads the next row's ids.
comptime P_NONE = 0
comptime P_PIN = 1
comptime P_ROW = 2


def _check_kept(
    ids: HostBuffer[.int32],
    row: Int,
    top_k: Int,
    block_max: HostBuffer[.float32],
    num_blocks: Int,
    position: Int,
    ratio: Int,
) -> Bool:
    """Whether a row's ids are a set of blocks the keep rule allows.

    The row's candidates are its live blocks, up to the pinned block of its
    newest visible key, with `-inf` blocks dropped and the pinned block
    above every value. If at most `top_k` remain the row keeps them all;
    otherwise it keeps every candidate above the k-th best value and fills
    the rest with candidates tied with it. Slots past the kept ids hold `-1`.
    """
    var pinned = ((position + 1) // ratio - 1) // BLOCK
    var live = min(max(pinned + 1, 0), num_blocks)
    var vals = List[Float32](capacity=live)
    var finite = List[Float32]()
    for j in range(live):
        var v = (
            inf[DType.float32]() if j
            == pinned else block_max[row * num_blocks + j]
        )
        vals.append(v)
        if v > neg_inf[DType.float32]():
            finite.append(v)
    var want = min(top_k, len(finite))
    var kth = neg_inf[DType.float32]()
    if want > 0:
        sort(finite)
        kth = finite[len(finite) - want]
    var got = Set[Int]()
    var unused = 0
    for i in range(top_k):
        var b = Int(ids[row * top_k + i])
        if b == -1:
            unused += 1
            continue
        if b < 0 or b >= live or b in got:
            return False
        if not (vals[b] > neg_inf[DType.float32]()) or vals[b] < kth:
            return False
        got.add(b)
    if len(got) != want or unused != top_k - want:
        return False
    for j in range(live):
        if vals[j] > kth and j not in got:
            return False
    return want == 0 or pinned >= live or pinned in got


def _kept_ids(ids: HostBuffer[.int32], row: Int, top_k: Int) -> List[Int]:
    """Returns the block ids a row keeps, skipping unused slots."""
    var kept = List[Int]()
    for i in range(top_k):
        var b = Int(ids[row * top_k + i])
        if b >= 0:
            kept.append(b)
    return kept^


def _score[
    output_layout: TensorLayout,
    q_layout: TensorLayout,
    w_layout: TensorLayout,
    iro_layout: TensorLayout,
    //,
    kpool: Int,
    page_size: Int,
](
    output: MutTileTensor[_SCORES_DTYPE, output_layout, _],
    q: ImmTileTensor[.uint8, q_layout, _],
    weights: ImmTileTensor[.float32, w_layout, _],
    q_sf: ImmPointer[UInt32, ImmutAnyOrigin],
    input_row_offsets: ImmTileTensor[.uint32, iro_layout, _],
    k_cache: MXFP4IndexKCache[page_size],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    """Runs the causal scorer alone into a matrix of the op's score dtype."""
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
        True,
        ctx,
    )


def _selection_case[
    num_blocks: Int, top_k: Int
](
    ctx: DeviceContext, rows: Int, levels: Int, poison: Int = P_NONE
) raises -> Int:
    """Runs the selection alone on crafted maxima; returns the bad rows.

    Few distinct levels make ties at the k-th value common, and rows mix in
    `-inf`, `-0.0` and `+0.0` blocks.
    """
    var bmax = ctx.enqueue_create_host_buffer[.float32](rows * num_blocks)
    var pos = ctx.enqueue_create_host_buffer[.int32](rows)
    for r in range(rows):
        pos[r] = Int32(random_ui64(0, UInt64(num_blocks * BLOCK + 3)))
        for j in range(num_blocks):
            var u = random_ui64(0, 9)
            var v = Float32(Int(random_ui64(0, UInt64(levels))) - levels // 2)
            if u == 0:
                v = neg_inf[DType.float32]()
            elif u == 1:
                v = -0.0
            elif u == 2:
                v = 0.0
            bmax[r * num_blocks + j] = v
    var d_bmax = ctx.enqueue_create_buffer[.float32](len(bmax))
    var d_pos = ctx.enqueue_create_buffer[.int32](rows)
    var d_ids = ctx.enqueue_create_buffer[.int32](rows * top_k)
    var dev_pos = ctx.enqueue_create_host_buffer[.int32](rows)
    for r in range(rows):
        dev_pos[r] = pos[r] + Int32(BLOCK if poison == P_PIN else 0)
    ctx.enqueue_copy(dst_buf=d_bmax, src_buf=bmax)
    ctx.enqueue_copy(dst_buf=d_pos, src_buf=dev_pos)
    d_ids.enqueue_fill(Int32(-7))
    var d_live = ctx.enqueue_create_buffer[.int32](rows)
    var hook = BlockTopKScoreHook[BLOCK](
        d_bmax.unsafe_ptr().as_unsafe_any_origin(),
        num_blocks,
        rows,
        d_live.unsafe_ptr().as_unsafe_any_origin(),
        d_ids.unsafe_ptr().as_unsafe_any_origin(),
        top_k,
        d_pos.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        1,
    )
    hook.after_window(ctx, 0, rows)
    var ids = ctx.enqueue_create_host_buffer[.int32](rows * top_k)
    ctx.enqueue_copy(dst_buf=ids, src_buf=d_ids)
    ctx.synchronize()
    var bad = 0
    for r in range(rows):
        if not _check_kept(ids, r, top_k, bmax, num_blocks, Int(pos[r]), 1):
            if bad < 2:
                print(
                    "  selection row", r, "kept", len(_kept_ids(ids, r, top_k))
                )
            bad += 1
    print(
        "SELECT blocks=",
        num_blocks,
        "k=",
        top_k,
        "poison=",
        poison,
        "bad=",
        bad,
    )
    return bad


def _check_masked_selection(
    picks: HostBuffer[.int32],
    row: Int,
    scores: HostBuffer[_SCORES_DTYPE],
    stride: Int,
    bound: Int,
    keep: List[Bool],
    num_blocks: Int,
) -> Bool:
    """Whether a row's picks are an exact top-k of its host-masked scores.

    Keys below `bound` in a dropped block score `-inf`. Every key above the
    k-th best masked score must be picked and every pick must reach it, so
    only ties at the k-th value are free; a row keeping fewer keys than it
    picks may fill the rest with any dropped keys.
    """
    var eff = min(TOKEN_TOP_K, bound)
    var masked = List[Float32](capacity=bound)
    for s in range(bound):
        var kept = keep[row * num_blocks + s // BLOCK]
        masked.append(
            scores[row * stride + s].cast[.float32]() if kept else neg_inf[
                DType.float32
            ]()
        )
    var kth = inf[DType.float32]()
    if eff > 0:
        var sorted_masked = masked.copy()
        sort(sorted_masked)
        kth = sorted_masked[bound - eff]
    var picked = Set[Int]()
    for i in range(TOKEN_TOP_K):
        var v = Int(picks[row * TOKEN_TOP_K + i])
        if v == -1:
            continue
        if v < 0 or v >= bound or v in picked or masked[v] < kth:
            return False
        picked.add(v)
    for s in range(bound):
        if masked[s] > kth and s not in picked:
            return False
    return kth == neg_inf[DType.float32]() or len(picked) == eff


def _indexer_case[
    kpool: Int, page_size: Int, topk_blocks: Int
](
    ctx: DeviceContext,
    seq_lens: List[Int],
    cache_lens: List[Int],
    window_rows: Int,
    scores_budget_bytes: Int,
    poison: Int = P_NONE,
) raises -> Int:
    """Runs the hooks through the indexer against the host reference.

    Returns:
        The number of failed checks.
    """
    var batch_size = len(seq_lens)
    var total = 0
    var max_seq_len = 0
    var max_cache_len = 0
    var max_rows = 0
    for b in range(batch_size):
        total += seq_lens[b]
        max_seq_len = max(max_seq_len, seq_lens[b])
        max_cache_len = max(max_cache_len, cache_lens[b])
        max_rows = max(max_rows, (cache_lens[b] + seq_lens[b]) // kpool)
    var num_blocks = ceildiv(max_rows, BLOCK)

    # Random MXFP4 keys and queries with scales near 1; every E2M1 code is a
    # number, so random bytes are valid values.
    var pages_per_seq = max(1, ceildiv(max_rows + 1, page_size))
    var num_pages = batch_size * pages_per_seq
    var page_stride = page_size * MXFP4_INDEX_RECORD_BYTES + 16
    var cache = ctx.enqueue_create_host_buffer[.uint8](num_pages * page_stride)
    for p in range(num_pages):
        for i in range(page_size * ROW_BYTES):
            cache[p * page_stride + i] = UInt8(random_ui64(0, 255))
        for i in range(page_size * 4):
            cache[p * page_stride + page_size * ROW_BYTES + i] = UInt8(
                random_ui64(124, 130)
            )
    var lut = ctx.enqueue_create_host_buffer[.uint32](num_pages)
    for i in range(num_pages):
        lut[i] = UInt32(i)
    var q = ctx.enqueue_create_host_buffer[.uint8](total * NH * ROW_BYTES)
    for i in range(len(q)):
        q[i] = UInt8(random_ui64(0, 255))
    var q_sf = ctx.enqueue_create_host_buffer[.uint32](total * NH)
    var weights = ctx.enqueue_create_host_buffer[.float32](total * NH)
    for i in range(total * NH):
        var sf = UInt32(0)
        for byte in range(4):
            sf |= UInt32(random_ui64(124, 130)) << UInt32(8 * byte)
        q_sf[i] = sf
        weights[i] = Float32(random_float64(-1.0, 1.0))
    var iro = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var lens = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var pos = ctx.enqueue_create_host_buffer[.int32](total)
    iro[0] = 0
    for b in range(batch_size):
        iro[b + 1] = iro[b] + UInt32(seq_lens[b])
        lens[b] = UInt32(cache_lens[b])
        for i in range(seq_lens[b]):
            pos[Int(iro[b]) + i] = Int32(cache_lens[b] + i)

    var d_cache = ctx.enqueue_create_buffer[.uint8](len(cache))
    var d_lut = ctx.enqueue_create_buffer[.uint32](len(lut))
    var d_q = ctx.enqueue_create_buffer[.uint8](len(q))
    var d_q_sf = ctx.enqueue_create_buffer[.uint32](len(q_sf))
    var d_w = ctx.enqueue_create_buffer[.float32](len(weights))
    var d_iro = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var d_lens = ctx.enqueue_create_buffer[.uint32](batch_size)
    var d_pos = ctx.enqueue_create_buffer[.int32](total)
    ctx.enqueue_copy(dst_buf=d_cache, src_buf=cache)
    ctx.enqueue_copy(dst_buf=d_lut, src_buf=lut)
    ctx.enqueue_copy(dst_buf=d_q, src_buf=q)
    ctx.enqueue_copy(dst_buf=d_q_sf, src_buf=q_sf)
    ctx.enqueue_copy(dst_buf=d_w, src_buf=weights)
    ctx.enqueue_copy(dst_buf=d_iro, src_buf=iro)
    ctx.enqueue_copy(dst_buf=d_lens, src_buf=lens)
    var dev_pos = ctx.enqueue_create_host_buffer[.int32](total)
    for t in range(total):
        dev_pos[t] = pos[t] + Int32(kpool * BLOCK if poison == P_PIN else 0)
    ctx.enqueue_copy(dst_buf=d_pos, src_buf=dev_pos)

    var k_cache = mxfp4_index_k_cache[page_size](
        TileTensor(d_cache, row_major(num_pages, page_stride)),
        TileTensor(d_lut, row_major(batch_size, pages_per_seq)),
        TileTensor(d_lens, row_major(batch_size)),
        max_seq_len,
        max_cache_len + kpool,
    )
    var q_tt = TileTensor(d_q, row_major(total, NH, ROW_BYTES))
    var q_sf_tt = TileTensor(d_q_sf, row_major(total, NH))
    var w_tt = TileTensor(d_w, row_major(total, NH))
    var iro_tt = TileTensor(d_iro, row_major(batch_size + 1))
    comptime mask = MaskName.CAUSAL.name

    # The scores the op's top-k reads, scored alone.
    var stride = align_up(max_rows + 1, 8)
    var d_scores = ctx.enqueue_create_buffer[_SCORES_DTYPE](total * stride)
    _score[kpool, page_size](
        TileTensor(d_scores, row_major(total, stride)),
        q_tt,
        w_tt,
        d_q_sf.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        iro_tt,
        k_cache,
        batch_size,
        max_seq_len,
        stride,
        ctx,
    )
    var scores = ctx.enqueue_create_host_buffer[_SCORES_DTYPE](len(d_scores))
    ctx.enqueue_copy(dst_buf=scores, src_buf=d_scores)

    # Kept ids, a few rows per window.
    var d_out = ctx.enqueue_create_buffer[.int32](total * TOKEN_TOP_K)
    var d_scratch = ctx.enqueue_create_buffer[.float32](
        window_rows * num_blocks
    )
    var d_live = ctx.enqueue_create_buffer[.int32](window_rows)
    # One spare row so the row poison reads `-1`s past the last row.
    var d_ids = ctx.enqueue_create_buffer[.int32]((total + 1) * topk_blocks)
    d_ids.enqueue_fill(Int32(-1))
    mla_indexer_ragged_mxfp4_paged[
        NH, TOKEN_TOP_K, mask, page_size, kpool=kpool, write_logical=True
    ](
        TileTensor(d_out, row_major(total, TOKEN_TOP_K)),
        q_tt,
        q_sf_tt,
        w_tt,
        iro_tt,
        k_cache,
        ctx,
        NoRemap(),
        BlockTopKScoreHook[BLOCK](
            d_scratch.unsafe_ptr().as_unsafe_any_origin(),
            num_blocks,
            window_rows,
            d_live.unsafe_ptr().as_unsafe_any_origin(),
            d_ids.unsafe_ptr().as_unsafe_any_origin(),
            topk_blocks,
            d_pos.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            kpool,
        ),
        scores_budget_bytes,
    )
    var ids = ctx.enqueue_create_host_buffer[.int32](len(d_ids))
    ctx.enqueue_copy(dst_buf=ids, src_buf=d_ids)
    ctx.synchronize()

    # The reference's block maxima: `-inf` for a block with no key below the
    # row's bound.
    var bounds = List[Int](capacity=total)
    var bmax = ctx.enqueue_create_host_buffer[.float32](total * num_blocks)
    for t in range(total):
        var bound = (Int(pos[t]) + 1) // kpool
        bounds.append(bound)
        for j in range(num_blocks):
            bmax[t * num_blocks + j] = neg_inf[DType.float32]()
        for s in range(bound):
            var i = t * num_blocks + s // BLOCK
            bmax[i] = max(bmax[i], scores[t * stride + s].cast[.float32]())

    var failures = 0
    var id_bad = 0
    var keep = List[Bool](length=total * num_blocks, fill=False)
    var min_kept = topk_blocks
    for t in range(total):
        var got = _kept_ids(ids, t, topk_blocks)
        min_kept = min(min_kept, len(got))
        for j in got:
            if j < num_blocks:
                keep[t * num_blocks + j] = True
        if not _check_kept(
            ids, t, topk_blocks, bmax, num_blocks, Int(pos[t]), kpool
        ):
            if id_bad < 2:
                print("  ids row", t, "kept", len(got))
            id_bad += 1
    if id_bad > 0:
        failures += 1

    # The selection masked by the ids.
    var d_flags = ctx.enqueue_create_buffer[.uint8](window_rows * num_blocks)
    var row_shift = topk_blocks if poison == P_ROW else 0
    mla_indexer_ragged_mxfp4_paged[
        NH, TOKEN_TOP_K, mask, page_size, kpool=kpool, write_logical=True
    ](
        TileTensor(d_out, row_major(total, TOKEN_TOP_K)),
        q_tt,
        q_sf_tt,
        w_tt,
        iro_tt,
        k_cache,
        ctx,
        NoRemap(),
        BlockTopKMaskScoreHook[BLOCK](
            d_ids.unsafe_ptr()
            .unsafe_offset(row_shift)
            .as_imm()
            .as_unsafe_any_origin(),
            topk_blocks,
            d_flags.unsafe_ptr().as_unsafe_any_origin(),
            num_blocks,
            window_rows,
        ),
        scores_budget_bytes,
    )
    var out = ctx.enqueue_create_host_buffer[.int32](total * TOKEN_TOP_K)
    ctx.enqueue_copy(dst_buf=out, src_buf=d_out)
    ctx.synchronize()
    var sel_bad = 0
    for t in range(total):
        if not _check_masked_selection(
            out, t, scores, stride, bounds[t], keep, num_blocks
        ):
            if sel_bad < 2:
                print("  masked selection row", t)
            sel_bad += 1
    if sel_bad > 0:
        failures += 1
    print(
        "CASE kpool=",
        kpool,
        "blocks=",
        num_blocks,
        "k=",
        topk_blocks,
        "rows=",
        total,
        "window=",
        window_rows,
        "poison=",
        poison,
        "min_kept=",
        min_kept,
        "id_bad=",
        id_bad,
        "sel_bad=",
        sel_bad,
        "->",
        "FAIL" if failures > 0 else "ok",
    )
    return failures


def main() raises:
    seed(20261018)
    var ctx = DeviceContext()

    # The selection alone: heavy ties, fewer blocks than picks, one row a CTA.
    assert_true(_selection_case[700, 64](ctx, 64, 6) == 0)
    assert_true(_selection_case[700, 64](ctx, 64, 400) == 0)
    assert_true(_selection_case[5000, 2048](ctx, 16, 50) == 0)
    assert_true(_selection_case[300, 512](ctx, 16, 4) == 0)

    # Through the indexer over several windows, both ratios, at the model's
    # pick count and at one small enough to be selective.
    var budget = 256 * 1024
    assert_true(
        _indexer_case[1, 128, 2048](
            ctx, [37, 1, 20], [20000, 30000, 3000], 5, budget
        )
        == 0
    )
    assert_true(
        _indexer_case[2, 64, 64](ctx, [29, 3, 11], [6000, 9001, 0], 4, budget)
        == 0
    )
    assert_true(
        _indexer_case[1, 128, 64](ctx, [1, 1, 1], [16380, 4092, 300], 1, budget)
        == 0
    )

    # Every poison must fail some check.
    assert_true(_selection_case[700, 64](ctx, 64, 6, P_PIN) > 0)
    assert_true(
        _indexer_case[1, 128, 64](ctx, [9, 4], [5000, 3000], 3, budget, P_PIN)
        > 0
    )
    assert_true(
        _indexer_case[1, 128, 64](ctx, [9, 4], [5000, 3000], 3, budget, P_ROW)
        > 0
    )
    print("PASS windowed keep")
