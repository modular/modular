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
"""Tests the decode context-parallel indexer and `cp_merge_topk`.

One full random index-K cache is split into per-rank caches by
`ContextParallelLayout` (same lookup table, global cache lengths). Each rank
runs `mla_indexer_ragged_float8_paged_cp`, every rank merges the concatenated
scores with `cp_merge_topk`, and the union of the ranks' rows, mapped back to
global positions, must be a top-k of the full cache's scores: the same
multiset of scores as the true top-k, so membership may differ only by ties.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from kv_cache.context_parallel import ContextParallelLayout
from kv_cache.paged_sparse_kv_index_remap import PagedKVRemap
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from nn.attention.gpu.cp_merge_topk import cp_merge_topk
from nn.attention.gpu.mla_index_fp8 import (
    _SCORES_BUDGET_BYTES,
    mla_indexer_ragged_float8_paged_cp,
)
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    _INDEX_SWIZZLE,
    KTMATileT,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from nn.attention.mha_mask import MaskName
from nn.attention.mha_operand import KVCacheMHAOperand, KVCacheScalesMHAOperand
from layout import Idx, TensorLayout, TileTensor, row_major
from layout.tile_tensor import ImmTileTensor, MutTileTensor
from std.collections import Dict
from std.math import align_up, ceildiv
from std.random import rand, random_ui64, seed
from std.testing import assert_equal, assert_true
from std.utils.numerics import neg_inf


comptime NUM_HEADS = 32
comptime DEPTH = 128
comptime PAGE = 128
comptime KV_PARAMS = KVCacheStaticParams(
    num_heads=1, head_size=DEPTH, is_mla=True
)
comptime Collection = PagedKVCacheCollection[
    DType.float8_e4m3fn,
    KV_PARAMS,
    PAGE,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
    scale_dtype_=DType.float32,
    quantization_granularity_=128,
]


def _collection(
    k_dev: DeviceBuffer[DType.float8_e4m3fn],
    ks_dev: DeviceBuffer[DType.float32],
    cl_dev: DeviceBuffer[DType.uint32],
    lut_dev: DeviceBuffer[DType.uint32],
    batch_size: Int,
    pages_per_seq: Int,
    num_blocks: Int,
    max_seq_len: Int,
    max_cache_len: Int,
) -> Collection:
    return Collection(
        rebind[Collection.blocks_tt_type](
            TileTensor(
                k_dev,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(1),
                    Idx[PAGE],
                    Idx[1],
                    Idx[DEPTH],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(cl_dev, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(Int64(batch_size), Int64(pages_per_seq)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(max_seq_len),
        UInt32(max_cache_len),
        rebind[Collection.scales_tt_type](
            TileTensor(
                ks_dev,
                row_major(
                    Int64(num_blocks),
                    Idx[1],
                    Int64(1),
                    Idx[PAGE],
                    Idx[1],
                    Idx[1],
                ),
            ).as_unsafe_any_origin()
        ),
    )


def _score_reference[
    output_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    iro_layout: TensorLayout,
    //,
](
    output: MutTileTensor[DType.bfloat16, output_layout, _],
    q: ImmTileTensor[DType.float8_e4m3fn, q_layout, _],
    q_s: ImmTileTensor[DType.float32, qs_layout, _],
    input_row_offsets: ImmTileTensor[DType.uint32, iro_layout, _],
    k_collection: Collection,
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    """Scores every visible key of the full cache, as the op's scorer does."""
    var k_cache = k_collection.get_key_cache(0)
    var k_op = KVCacheMHAOperand(k_cache)
    var ks_op = KVCacheScalesMHAOperand(k_cache)
    var k_tma_tile = rebind[KTMATileT[DType.float8_e4m3fn, _BM_KEY, DEPTH]](
        k_cache.create_paged_tma_tile[_INDEX_SWIZZLE, BN=_BM_KEY, BK=DEPTH](ctx)
    )
    fp8_index_score_sm100[
        DType.float8_e4m3fn,
        type_of(k_op),
        type_of(ks_op),
        NUM_HEADS,
        DEPTH,
        _is_cache_length_accurate=False,
        N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
    ](
        output,
        q,
        q_s,
        k_op,
        ks_op,
        k_tma_tile,
        input_row_offsets,
        batch_size,
        max_seq_len,
        max_num_keys,
        True,
        ctx,
    )


def _run_rank_indexer[
    top_k: Int, cp_size: Int, cp_interleave: Int
](
    mut phys_dev: DeviceBuffer[DType.int32],
    mut scores_dev: DeviceBuffer[DType.bfloat16],
    q_dev: DeviceBuffer[DType.float8_e4m3fn],
    qs_dev: DeviceBuffer[DType.float32],
    iro_dev: DeviceBuffer[DType.uint32],
    lut_dev: DeviceBuffer[DType.uint32],
    rank_dev: DeviceBuffer[DType.int32],
    k_collection: Collection,
    total_seq_len: Int,
    batch_size: Int,
    pages_per_seq: Int,
    num_blocks: Int,
    budget: Int,
    ctx: DeviceContext,
) raises:
    mla_indexer_ragged_float8_paged_cp[
        DType.float8_e4m3fn,
        Collection,
        NUM_HEADS,
        DEPTH,
        top_k,
        MaskName.CAUSAL.name,
        cp_size,
        cp_interleave,
    ](
        TileTensor(phys_dev, row_major(0, top_k)),
        TileTensor(scores_dev, row_major(total_seq_len, top_k)),
        TileTensor(q_dev, row_major(total_seq_len, NUM_HEADS, DEPTH)).as_imm(),
        TileTensor(qs_dev, row_major(total_seq_len, NUM_HEADS)).as_imm(),
        TileTensor(iro_dev, row_major(batch_size + 1)).as_imm(),
        k_collection,
        UInt32(0),
        ctx,
        PagedKVRemap[PAGE](
            phys_dev.unsafe_ptr(),
            lut_dev.unsafe_ptr(),
            pages_per_seq,
            batch_size,
            UInt32(num_blocks),
        ),
        rank_dev.unsafe_ptr().as_imm(),
        budget,
    )


def _check_cp[
    top_k: Int, cp_size: Int, cp_interleave: Int
](
    seq_lens: List[Int],
    cache_lens: List[Int],
    lut_host: List[UInt32],
    lut_dev: DeviceBuffer[DType.uint32],
    pages_per_seq: Int,
    num_blocks: Int,
    k_full: HostBuffer[DType.float8_e4m3fn],
    ks_full: HostBuffer[DType.float32],
    q_dev: DeviceBuffer[DType.float8_e4m3fn],
    qs_dev: DeviceBuffer[DType.float32],
    iro_dev: DeviceBuffer[DType.uint32],
    cl_dev: DeviceBuffer[DType.uint32],
    ref_scores: HostBuffer[DType.bfloat16],
    ref_stride: Int,
    budget: Int,
    ctx: DeviceContext,
) raises:
    """Runs every rank of one group and checks the merged selection."""
    comptime Layout = ContextParallelLayout[cp_size, cp_interleave]
    var batch_size = len(seq_lens)
    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    for b in range(batch_size):
        total_seq_len += seq_lens[b]
        max_seq_len = max(max_seq_len, seq_lens[b])
        max_cache_len = max(max_cache_len, cache_lens[b])
    print(
        t"  cp_size={cp_size} interleave={cp_interleave} top_k={top_k}"
        t" tokens={total_seq_len} budget={budget}"
    )
    comptime n_cand = cp_size * top_k
    var n_rows = num_blocks * PAGE

    # Each token's batch entry and global key bound (causal).
    var tok_batch = List[Int]()
    var tok_bound = List[Int]()
    for b in range(batch_size):
        for t in range(seq_lens[b]):
            tok_batch.append(b)
            tok_bound.append(cache_lens[b] + t + 1)

    # Physical row -> page index of each batch entry, to map rows back.
    var page_of_block = List[Dict[Int, Int]]()
    for b in range(batch_size):
        var d = Dict[Int, Int]()
        for j in range(pages_per_seq):
            d[Int(lut_host[b * pages_per_seq + j])] = j
        page_of_block.append(d^)

    var all_scores_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
        total_seq_len * n_cand
    )
    var local_rows = List[List[Int32]]()

    for r in range(cp_size):
        var cp = Layout(r)
        # This rank's cache: its owned positions at their local slots.
        var k_rank = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
            n_rows * DEPTH
        )
        var ks_rank = ctx.enqueue_create_buffer[DType.float32](n_rows)
        # Slots the rank does not fill stay uninitialized: nothing reads them.
        with k_rank.map_to_host() as kh:
            with ks_rank.map_to_host() as ksh:
                for b in range(batch_size):
                    for p in range(cache_lens[b] + seq_lens[b]):
                        if not cp.owns(p):
                            continue
                        var s = cp.local_slot(p)
                        var src = (
                            Int(lut_host[b * pages_per_seq + p // PAGE]) * PAGE
                            + p % PAGE
                        )
                        var dst = (
                            Int(lut_host[b * pages_per_seq + s // PAGE]) * PAGE
                            + s % PAGE
                        )
                        for d in range(DEPTH):
                            kh[dst * DEPTH + d] = k_full[src * DEPTH + d]
                        ksh[dst] = ks_full[src]
        var rank_dev = ctx.enqueue_create_buffer[DType.int32](1)
        with rank_dev.map_to_host() as rh:
            rh[0] = Int32(r)
        var phys_dev = ctx.enqueue_create_buffer[DType.int32](
            total_seq_len * top_k
        )
        phys_dev.enqueue_fill(Int32(-2))
        var scores_dev = ctx.enqueue_create_buffer[DType.bfloat16](
            total_seq_len * top_k
        )
        var coll = _collection(
            k_rank,
            ks_rank,
            cl_dev,
            lut_dev,
            batch_size,
            pages_per_seq,
            num_blocks,
            max_seq_len,
            max_cache_len,
        )
        _run_rank_indexer[top_k, cp_size, cp_interleave](
            phys_dev,
            scores_dev,
            q_dev,
            qs_dev,
            iro_dev,
            lut_dev,
            rank_dev,
            coll,
            total_seq_len,
            batch_size,
            pages_per_seq,
            num_blocks,
            budget,
            ctx,
        )
        var phys_host = ctx.enqueue_create_host_buffer[DType.int32](
            total_seq_len * top_k
        )
        var sc_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
            total_seq_len * top_k
        )
        ctx.enqueue_copy(phys_host, phys_dev)
        ctx.enqueue_copy(sc_host, scores_dev)
        ctx.synchronize()

        # Each local selection is an owned, visible position, scored exactly
        # as the full cache scores it; a row takes min(top_k, local visible).
        var rows = List[Int32](capacity=total_seq_len * top_k)
        for t in range(total_seq_len):
            var b = tok_batch[t]
            var n_local = cp.local_len(tok_bound[t])
            var n_live = 0
            for j in range(top_k):
                var i = t * top_k + j
                var row = Int(phys_host[i])
                rows.append(phys_host[i])
                all_scores_host[t * n_cand + r * top_k + j] = sc_host[i]
                if row < 0:
                    assert_true(
                        sc_host[i] == neg_inf[DType.bfloat16](),
                        "a padding slot's score must be -inf",
                    )
                    continue
                n_live += 1
                var page = page_of_block[b][row // PAGE]
                var slot = page * PAGE + row % PAGE
                var pos = cp.global_pos(slot)
                assert_true(
                    slot < n_local and pos < tok_bound[t],
                    String(t"rank {r} token {t}: slot {slot} not visible"),
                )
                assert_true(
                    sc_host[i] == ref_scores[t * ref_stride + pos],
                    String(
                        t"rank {r} token {t} pos {pos}: score",
                        t" {sc_host[i]} != reference",
                        t" {ref_scores[t * ref_stride + pos]}",
                    ),
                )
            assert_equal(n_live, min(top_k, n_local))
        local_rows.append(rows^)
        _ = k_rank
        _ = ks_rank

    # Every rank merges the same gathered scores.
    var all_scores_dev = ctx.enqueue_create_buffer[DType.bfloat16](
        total_seq_len * n_cand
    )
    ctx.enqueue_copy(all_scores_dev, all_scores_host)
    var merged = List[List[Int]]()
    for _ in range(total_seq_len):
        merged.append(List[Int]())
    for r in range(cp_size):
        var cp = Layout(r)
        var rank_dev = ctx.enqueue_create_buffer[DType.int32](1)
        with rank_dev.map_to_host() as rh:
            rh[0] = Int32(r)
        var rows_dev = ctx.enqueue_create_buffer[DType.int32](
            total_seq_len * top_k
        )
        with rows_dev.map_to_host() as h:
            for i in range(total_seq_len * top_k):
                h[i] = local_rows[r][i]
        var out_dev = ctx.enqueue_create_buffer[DType.int32](
            total_seq_len * top_k
        )
        out_dev.enqueue_fill(Int32(-2))
        var len_dev = ctx.enqueue_create_buffer[DType.int32](total_seq_len)
        cp_merge_topk[cp_size, top_k](
            TileTensor(out_dev, row_major(total_seq_len, top_k)),
            TileTensor(len_dev, row_major(total_seq_len)),
            TileTensor(rows_dev, row_major(total_seq_len, top_k)).as_imm(),
            TileTensor(
                all_scores_dev, row_major(total_seq_len, n_cand)
            ).as_imm(),
            rank_dev.unsafe_ptr().as_imm(),
            ctx,
        )
        var out_host = ctx.enqueue_create_host_buffer[DType.int32](
            total_seq_len * top_k
        )
        var len_host = ctx.enqueue_create_host_buffer[DType.int32](
            total_seq_len
        )
        ctx.enqueue_copy(out_host, out_dev)
        ctx.enqueue_copy(len_host, len_dev)
        ctx.synchronize()
        for t in range(total_seq_len):
            var b = tok_batch[t]
            var n = Int(len_host[t])
            assert_true(0 <= n and n <= top_k, "length out of range")
            if cp.local_len(tok_bound[t]) == 0:
                assert_equal(n, 0, "a rank with no visible keys keeps none")
            for j in range(top_k):
                var row = Int(out_host[t * top_k + j])
                if j >= n:
                    assert_equal(row, -1, "slots past the length must be -1")
                    continue
                assert_true(row >= 0, "a selected slot must hold a row")
                var page = page_of_block[b][row // PAGE]
                var pos = cp.global_pos(page * PAGE + row % PAGE)
                assert_true(cp.owns(pos) and pos < tok_bound[t])
                merged[t].append(pos)

    # The union is a top-k of the full cache: distinct, `min(k, visible)` of
    # them, with the true top-k's multiset of scores.
    def _desc(lhs: Float32, rhs: Float32) -> Bool:
        return lhs > rhs

    var n_tied = 0
    for t in range(total_seq_len):
        var want = min(top_k, tok_bound[t])
        assert_equal(len(merged[t]), want, String(t"token {t}: merged size"))
        var seen = Dict[Int, Bool]()
        var got = List[Float32]()
        for pos in merged[t]:
            assert_true(pos not in seen, "a position was selected twice")
            seen[pos] = True
            got.append(ref_scores[t * ref_stride + pos].cast[DType.float32]())
        var all = List[Float32]()
        for p in range(tok_bound[t]):
            all.append(ref_scores[t * ref_stride + p].cast[DType.float32]())
        sort(got, _desc)
        sort(all, _desc)
        for i in range(want):
            assert_true(
                got[i] == all[i],
                String(
                    t"token {t}: rank {i} of the merged scores is {got[i]},"
                    t" the true top-k's is {all[i]}"
                ),
            )
        if want < tok_bound[t] and all[want - 1] == all[want]:
            n_tied += 1
    print(t"    ok ({n_tied} rows with a tie at the threshold)")
    _ = all_scores_dev


def test_cp_indexer[
    top_k: Int
](
    seq_lens: List[Int],
    cache_lens: List[Int],
    ctx: DeviceContext,
    budget: Int = _SCORES_BUDGET_BYTES,
) raises:
    """Builds one full cache and checks every group shape against it."""
    seed(1234)
    var batch_size = len(seq_lens)
    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    var max_total = 0
    for b in range(batch_size):
        total_seq_len += seq_lens[b]
        max_seq_len = max(max_seq_len, seq_lens[b])
        max_cache_len = max(max_cache_len, cache_lens[b])
        max_total = max(max_total, cache_lens[b] + seq_lens[b])
    print(
        t"test_cp_indexer top_k={top_k} batch={batch_size}"
        t" tokens={total_seq_len} max_keys={max_total}"
    )
    var pages_per_seq = ceildiv(max_total, PAGE)
    var num_blocks = batch_size * pages_per_seq + 3
    var n_rows = num_blocks * PAGE

    var q_dev = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
        total_seq_len * NUM_HEADS * DEPTH
    )
    with q_dev.map_to_host() as h:
        rand(h.as_span())
    var qs_dev = ctx.enqueue_create_buffer[DType.float32](
        total_seq_len * NUM_HEADS
    )
    with qs_dev.map_to_host() as h:
        rand(h.as_span())
    var iro_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    with iro_dev.map_to_host() as h:
        h[0] = 0
        for b in range(batch_size):
            h[b + 1] = h[b] + UInt32(seq_lens[b])
    var cl_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    with cl_dev.map_to_host() as h:
        for b in range(batch_size):
            h[b] = UInt32(cache_lens[b])

    # A random page table over distinct blocks, shared by every rank.
    var lut_host = List[UInt32]()
    var used = Dict[Int, Bool]()
    for _ in range(batch_size * pages_per_seq):
        var blk = Int(random_ui64(0, UInt64(num_blocks - 1)))
        while blk in used:
            blk = Int(random_ui64(0, UInt64(num_blocks - 1)))
        used[blk] = True
        lut_host.append(UInt32(blk))
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * pages_per_seq
    )
    with lut_dev.map_to_host() as h:
        for i in range(batch_size * pages_per_seq):
            h[i] = lut_host[i]

    var k_full = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](
        n_rows * DEPTH
    )
    var ks_full = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.synchronize()
    rand(k_full.as_span())
    rand(ks_full.as_span())
    var k_dev = ctx.enqueue_create_buffer[DType.float8_e4m3fn](n_rows * DEPTH)
    var ks_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(k_dev, k_full)
    ctx.enqueue_copy(ks_dev, ks_full)
    var full = _collection(
        k_dev,
        ks_dev,
        cl_dev,
        lut_dev,
        batch_size,
        pages_per_seq,
        num_blocks,
        max_seq_len,
        max_cache_len,
    )

    # Reference scores of every visible key, from the op's own scorer.
    var ref_stride = align_up(max_total, 8)
    var ref_dev = ctx.enqueue_create_buffer[DType.bfloat16](
        total_seq_len * ref_stride
    )
    _score_reference(
        TileTensor(ref_dev, row_major(total_seq_len, ref_stride)),
        TileTensor(q_dev, row_major(total_seq_len, NUM_HEADS, DEPTH)).as_imm(),
        TileTensor(qs_dev, row_major(total_seq_len, NUM_HEADS)).as_imm(),
        TileTensor(iro_dev, row_major(batch_size + 1)).as_imm(),
        full,
        batch_size,
        max_seq_len,
        ref_stride,
        ctx,
    )
    var ref_scores = ctx.enqueue_create_host_buffer[DType.bfloat16](
        total_seq_len * ref_stride
    )
    ctx.enqueue_copy(ref_scores, ref_dev)
    ctx.synchronize()

    _check_cp[top_k, 2, 1](
        seq_lens,
        cache_lens,
        lut_host,
        lut_dev,
        pages_per_seq,
        num_blocks,
        k_full,
        ks_full,
        q_dev,
        qs_dev,
        iro_dev,
        cl_dev,
        ref_scores,
        ref_stride,
        budget,
        ctx,
    )
    _check_cp[top_k, 4, 1](
        seq_lens,
        cache_lens,
        lut_host,
        lut_dev,
        pages_per_seq,
        num_blocks,
        k_full,
        ks_full,
        q_dev,
        qs_dev,
        iro_dev,
        cl_dev,
        ref_scores,
        ref_stride,
        budget,
        ctx,
    )
    _check_cp[top_k, 8, 1](
        seq_lens,
        cache_lens,
        lut_host,
        lut_dev,
        pages_per_seq,
        num_blocks,
        k_full,
        ks_full,
        q_dev,
        qs_dev,
        iro_dev,
        cl_dev,
        ref_scores,
        ref_stride,
        budget,
        ctx,
    )
    _check_cp[top_k, 2, PAGE](
        seq_lens,
        cache_lens,
        lut_host,
        lut_dev,
        pages_per_seq,
        num_blocks,
        k_full,
        ks_full,
        q_dev,
        qs_dev,
        iro_dev,
        cl_dev,
        ref_scores,
        ref_stride,
        budget,
        ctx,
    )
    _check_cp[top_k, 4, PAGE](
        seq_lens,
        cache_lens,
        lut_host,
        lut_dev,
        pages_per_seq,
        num_blocks,
        k_full,
        ks_full,
        q_dev,
        qs_dev,
        iro_dev,
        cl_dev,
        ref_scores,
        ref_stride,
        budget,
        ctx,
    )
    _check_cp[top_k, 8, PAGE](
        seq_lens,
        cache_lens,
        lut_host,
        lut_dev,
        pages_per_seq,
        num_blocks,
        k_full,
        ks_full,
        q_dev,
        qs_dev,
        iro_dev,
        cl_dev,
        ref_scores,
        ref_stride,
        budget,
        ctx,
    )
    # A group of one is the plain indexer plus its scores.
    _check_cp[top_k, 1, 1](
        seq_lens,
        cache_lens,
        lut_host,
        lut_dev,
        pages_per_seq,
        num_blocks,
        k_full,
        ks_full,
        q_dev,
        qs_dev,
        iro_dev,
        cl_dev,
        ref_scores,
        ref_stride,
        budget,
        ctx,
    )
    _ = k_dev
    _ = ks_dev
    _ = ref_dev


def main() raises:
    with DeviceContext() as ctx:
        # Ragged mix: a decode row at position 0 (every other rank sees no
        # key), deep decode rows, a 7-token chunk over a cached prefix, a
        # fresh 64-token prefill whose first rows see fewer keys than the
        # group has ranks, and a long prefill crossing k on every rank.
        var seq_lens: List[Int] = [1, 1, 7, 64, 1, 150]
        var cache_lens: List[Int] = [0, 2000, 300, 0, 900, 1100]
        test_cp_indexer[256](seq_lens, cache_lens, ctx)
        # The same batch in score windows of a few rows each, so the score
        # output is advanced past row 0 for every window but the first.
        test_cp_indexer[256](seq_lens, cache_lens, ctx, budget=40 * 2400 * 2)
        # The production k over deep decode and a 3-token MTP step: local
        # candidates above k at small groups and below it at 8.
        var dec_seq: List[Int] = [1, 1, 1, 3]
        var dec_cache: List[Int] = [20000, 5000, 1500, 9000]
        test_cp_indexer[2048](dec_seq, dec_cache, ctx)
