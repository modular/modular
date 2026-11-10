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

"""Decode context parallelism slice test for the SM100 sparse MLA decode.

Under decode context parallelism each rank attends over the selected rows it
owns, compacted to the front of each token's index row with a per-token valid
length (`topk_lengths` under `skip_masked_tiles`), and returns the attention LSE in
base 2 (`lse_out`) so the ranks can be merged.

Each case builds a random paged cache and a full selection of `topk` rows per
query token over ragged rows of 1 to 3 tokens (the decode kernel run on short
prefill rows). The reference is the existing sparse decode on the full
selection, plus a float64 host LSE. Then, for dcp in {2, 4, 8}, it splits
every token's selection into dcp disjoint, unevenly sized subsets (rank
`dcp - 1` owns nothing for the first token), compacts each to the front of a
row whose tail holds `-1` or rows the token must not see, runs the CP kernel
per rank, and merges on the host:

    m = max_r lse_r;  w_r = 2^(lse_r - m);  O = sum_r w_r O_r / sum_r w_r

The merged O must match the reference output and the merged LSE the host LSE.
Each split runs unsplit (`num_partitions == 1`, the kernel writes the LSE),
split-K (the combine writes it), and with the heuristic sized by
`split_len_bound`. An empty shard must give LSE `-inf` and O zero.

Covers the BF16 KV, BF16-Q FP8 KV, and FP8-Q FP8 KV sparse kernels.

`run_fused_cp_store` runs the fused FP8 branch the `.cp` op calls with a
decode context-parallel group: each rank must store exactly the new tokens it
owns, at `local_slot(pos)` and with the bytes a group of one stores, leave the
rest of its cache untouched, and return a finite LSE (`-inf` for a token with
no keys).

`run_bf16_cp_store` checks the same store through the BF16 branch the
`attention.mla.graph.decode.paged.sparse{,.cp}` ops call,
`run_bf16_cp_owned_rows` that branch on compacted owned rows with per-token
lengths below the top-k and an empty token (LSE `-inf`, O zero), merged
against a group of one, and
`run_bf16_branch_modes` the BF16 `.cp` op's two other modes against its
default: `latent_output` returns the attention's 512-wide latent output,
which `w_uv` projects to the default output with the same LSE, and
`q_absorbed` takes a pre-absorbed query `[q_nope @ w_uk, q_rope]` and attends
as the default does.

`run_nvfp4_cp_store` and `run_nvfp4_cp_owned_rows` run the same BF16 branch
over a packed NVFP4 latent cache, as the `.sparse.nvfp4.cp` op does: 256 bytes
of e2m1 pairs and 64 FP8 rope bytes per row, one E4M3 scale per 16 latent
elements in a scale pool paged apart from the values (its own lookup table, a
permutation of the values'). Each index row is the
selected value rows, then their scale-pool rows, each half compacted the same
way, with a per-token length counting either half. The store must write each
owned new token's packed row and its scales at the rank's local slot, both
exactly as a group of one writes them at the token's position, and leave every
other value and scale byte as it was; the ranks' outputs over their compacted
owned rows must merge to a group of one's over the whole selection.
"""

from std.math import ceildiv, exp, exp2, gcd, log, log2
from std.random import randn, seed

from max.gpu import *
from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from internal_utils.fp8_utils import cast_saturating
from kv_cache.context_parallel import ContextParallelLayout
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_utils import MHAConfig, NonNullPointer
from nn.attention.gpu.mla import SM100MLA, flare_mla_decoding
from nn.attention.gpu.mla_graph import (
    SM100MLAGraph,
    mla_decode_branch_bf16,
    mla_decode_branch_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.testing import assert_almost_equal, assert_true
from std.memory import bitcast
from std.utils.index import IndexList
from std.utils.numerics import isinf, isnan, min_or_neg_inf

comptime Q_DEPTH = 576  # 512 latent + 64 rope
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime NUM_HEADS = 64
comptime SCALE = Float32(0.125)
comptime LOG2E = 1.4426950408889634


struct _Lcg:
    """A deterministic generator so a failing case reproduces."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt32:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return UInt32(self.state >> 33)

    def unit(mut self) -> Float64:
        return Float64(self.next()) / 2147483648.0


def _coprime_multiplier(n: Int) -> Int:
    for m in [7, 11, 13, 17, 19, 23, 29, 31]:
        if gcd(m, n) == 1:
            return m
    return 1


def _phys_row(lut: List[Int], pages_per_batch: Int, b: Int, pos: Int) -> Int:
    """Returns the physical cache row of batch entry `b`'s position `pos`."""
    var page, off = divmod(pos, PAGE_SIZE)
    return lut[b * pages_per_batch + page] * PAGE_SIZE + off


def _owner(mut rng: _Lcg, token: Int, dcp: Int) -> Int:
    """Picks the rank that owns one selected row: skewed toward low ranks.

    The first token never picks rank `dcp - 1`, so that shard is empty.
    """
    var u = rng.unit()
    var ranks = dcp - 1 if token == 0 else dcp
    return min(Int(Float64(ranks) * u * u), ranks - 1)


def run_cp_case[
    q_type: DType, kv_type: DType
](
    name: StringLiteral,
    ctx: DeviceContext,
    seq_lens: List[Int],
    cache_len: Int,
    topk: Int,
) raises:
    comptime out_type = DType.bfloat16
    var batch_size = len(seq_lens)
    var q_max_seq_len = 0
    var total_tokens = 0
    for s in seq_lens:
        q_max_seq_len = max(q_max_seq_len, s)
        total_tokens += s
    print(
        t"case {name}: q={q_type} kv={kv_type} batch={batch_size}"
        t" tokens={total_tokens} cache_len={cache_len} topk={topk}"
    )

    # ---------------------------------------------------------------------
    # Paged cache with a shuffled page table.
    # ---------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    var max_keys = cache_len + q_max_seq_len
    var pages_per_batch = ceildiv(max_keys, PAGE_SIZE)
    var total_pages = batch_size * pages_per_batch
    var page_elems = PAGE_SIZE * Q_DEPTH
    var block_elems = total_pages * page_elems

    var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size * pages_per_batch
    )
    var mult_p = _coprime_multiplier(pages_per_batch)
    for b in range(batch_size):
        for p in range(pages_per_batch):
            lut_host[b * pages_per_batch + p] = UInt32(
                b * pages_per_batch + (p * mult_p + 1) % pages_per_batch
            )

    var lut = List[Int](length=batch_size * pages_per_batch, fill=0)
    for i in range(batch_size * pages_per_batch):
        lut[i] = Int(lut_host[i])

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    var k_src = ctx.enqueue_create_host_buffer[DType.bfloat16](block_elems)
    randn(k_src.as_span(), mean=0.0, standard_deviation=0.5)
    # The reference reads back the cache's own (possibly FP8) values.
    var k_ref = List[Float32](length=block_elems, fill=0)
    for i in range(block_elems):
        blocks_host[i] = k_src[i].cast[kv_type]()
        k_ref[i] = blocks_host[i].cast[DType.float32]()

    var q_elems = total_tokens * NUM_HEADS * Q_DEPTH
    var q_src = ctx.enqueue_create_host_buffer[DType.bfloat16](q_elems)
    randn(q_src.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_elems)
    var q_ref = List[Float32](length=q_elems, fill=0)
    for i in range(q_elems):
        q_host[i] = q_src[i].cast[q_type]()
        q_ref[i] = q_host[i].cast[DType.float32]()

    var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    var token_batch = List[Int](length=total_tokens, fill=0)
    var token_local = List[Int](length=total_tokens, fill=0)
    row_offsets_host[0] = 0
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(cache_len)
        var start = Int(row_offsets_host[b])
        for s in range(seq_lens[b]):
            token_batch[start + s] = b
            token_local[start + s] = s
        row_offsets_host[b + 1] = UInt32(start + seq_lens[b])

    # ---------------------------------------------------------------------
    # Full selection: `topk` distinct causally visible positions per token,
    # in a scrambled order (the kernel never relies on order).
    # ---------------------------------------------------------------------
    var sel = List[Int](length=total_tokens * topk, fill=0)
    for t in range(total_tokens):
        var visible = cache_len + token_local[t] + 1
        if topk > visible:
            raise Error("topk must not exceed the visible keys")
        var m = _coprime_multiplier(visible)
        for i in range(topk):
            sel[t * topk + i] = (i * m + 3 * t + 1) % visible

    # Host LSE of the full selection, base 2, in float64.
    var lse_ref = List[Float64](length=total_tokens * NUM_HEADS, fill=0)
    var scores = List[Float64](length=topk, fill=0)
    for t in range(total_tokens):
        var b = token_batch[t]
        for h in range(NUM_HEADS):
            var q_base = (t * NUM_HEADS + h) * Q_DEPTH
            var mx = Float64(-1e300)
            for i in range(topk):
                var k_base = (
                    _phys_row(lut, pages_per_batch, b, sel[t * topk + i])
                    * Q_DEPTH
                )
                var dot = Float32(0)
                for d in range(Q_DEPTH):
                    dot += q_ref[q_base + d] * k_ref[k_base + d]
                scores[i] = Float64(dot) * Float64(SCALE)
                mx = max(mx, scores[i])
            var acc = Float64(0)
            for i in range(topk):
                acc += exp(scores[i] - mx)
            lse_ref[t * NUM_HEADS + h] = (mx + log(acc)) * LOG2E

    # ---------------------------------------------------------------------
    # Device buffers.
    # ---------------------------------------------------------------------
    var blocks_dev = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_dev, blocks_host)
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](
        batch_size * pages_per_batch
    )
    ctx.enqueue_copy(lut_dev, lut_host)
    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    var row_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_dev, row_offsets_host)
    var q_dev = ctx.enqueue_create_buffer[q_type](q_elems)
    ctx.enqueue_copy(q_dev, q_host)

    comptime Collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](
            TileTensor(
                blocks_dev,
                row_major(
                    Int64(total_pages),
                    Idx[1],
                    Int64(NUM_LAYERS),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[Q_DEPTH],
                ),
            ).as_unsafe_any_origin()
        ),
        TileTensor(cache_lengths_dev, row_major(Int64(batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(
            lut_dev, row_major(Int64(batch_size), Int64(pages_per_batch))
        )
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)
    var row_offsets_tt = TileTensor(row_offsets_dev, row_major(batch_size + 1))
    var q_tt = TileTensor(
        q_dev, row_major(total_tokens, Idx[NUM_HEADS], Idx[Q_DEPTH])
    )
    var mla_args = MLADispatchScalarArgs[
        num_heads=NUM_HEADS,
        is_fp8_kv=(kv_type == DType.float8_e4m3fn),
    ](batch_size, cache_len, q_max_seq_len, ctx)

    var out_elems = total_tokens * NUM_HEADS * V_DEPTH
    var lse_elems = total_tokens * NUM_HEADS

    # ---------------------------------------------------------------------
    # Reference: the existing sparse decode over the full selection.
    # ---------------------------------------------------------------------
    var idx_full_host = ctx.enqueue_create_host_buffer[DType.int32](
        total_tokens * topk
    )
    for t in range(total_tokens):
        for i in range(topk):
            idx_full_host[t * topk + i] = Int32(
                _phys_row(
                    lut, pages_per_batch, token_batch[t], sel[t * topk + i]
                )
            )
    var idx_full_dev = ctx.enqueue_create_buffer[DType.int32](
        total_tokens * topk
    )
    ctx.enqueue_copy(idx_full_dev, idx_full_host)
    var out_ref_dev = ctx.enqueue_create_buffer[out_type](out_elems)
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](NUM_HEADS, Q_DEPTH),
        ragged=True,
        sparse=True,
        kernels=SM100MLA,
    ](
        TileTensor(
            out_ref_dev, row_major(total_tokens, Idx[NUM_HEADS], Idx[V_DEPTH])
        ),
        q_tt,
        kv_cache,
        CausalMask(),
        row_offsets_tt,
        SCALE,
        ctx,
        mla_args.gpu_tile_tensor(),
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            idx_full_dev.unsafe_ptr()
        ),
        indices_stride=topk,
    )
    var out_ref = ctx.enqueue_create_host_buffer[out_type](out_elems)
    ctx.enqueue_copy(out_ref, out_ref_dev)
    ctx.synchronize()

    # Rank buffers, reused across every split.
    var idx_rank_host = ctx.enqueue_create_host_buffer[DType.int32](
        total_tokens * topk
    )
    var len_rank_host = ctx.enqueue_create_host_buffer[DType.int32](
        total_tokens
    )
    var idx_rank_dev = ctx.enqueue_create_buffer[DType.int32](
        total_tokens * topk
    )
    var len_rank_dev = ctx.enqueue_create_buffer[DType.int32](total_tokens)
    var out_rank_dev = ctx.enqueue_create_buffer[out_type](out_elems)
    var lse_rank_dev = ctx.enqueue_create_buffer[DType.float32](lse_elems)
    var out_rank = ctx.enqueue_create_host_buffer[out_type](out_elems)
    var lse_rank = ctx.enqueue_create_host_buffer[DType.float32](lse_elems)

    var rng = _Lcg(UInt64(topk * 131 + cache_len))
    var owner = List[Int](length=total_tokens * topk, fill=0)
    var merged_o = List[Float64](length=out_elems, fill=0)
    var merged_m = List[Float64](length=lse_elems, fill=0)
    var merged_w = List[Float64](length=lse_elems, fill=0)

    for dcp in [1, 2, 4, 8]:
        for t in range(total_tokens):
            for i in range(topk):
                owner[t * topk + i] = 0 if dcp == 1 else _owner(rng, t, dcp)
        var bound = ceildiv(topk, dcp) + 16
        # (num_partitions, split_len_bound): unsplit, split-K, heuristic.
        for mode in range(3):
            var num_partitions = 1 if mode == 0 else (4 if mode == 1 else 0)
            var split_len_bound = bound if mode == 2 else 0
            for i in range(lse_elems):
                merged_m[i] = Float64(-1e300)
                merged_w[i] = 0
            for i in range(out_elems):
                merged_o[i] = 0
            var empty_shards = 0
            for r in range(dcp):
                # Compact this rank's rows; pad with -1 or rows it must not
                # read (another rank's).
                for t in range(total_tokens):
                    var n = 0
                    for i in range(topk):
                        if owner[t * topk + i] == r:
                            idx_rank_host[t * topk + n] = idx_full_host[
                                t * topk + i
                            ]
                            n += 1
                    len_rank_host[t] = Int32(n)
                    for j in range(n, topk):
                        idx_rank_host[t * topk + j] = (
                            Int32(-1) if j % 2
                            == 0 else idx_full_host[t * topk + (j * 5) % topk]
                        )
                ctx.enqueue_copy(idx_rank_dev, idx_rank_host)
                ctx.enqueue_copy(len_rank_dev, len_rank_host)
                # Garbage in O and the LSE, so a slot the kernel skips
                # shows up.
                ctx.enqueue_memset(out_rank_dev, Scalar[out_type](7.0))
                ctx.enqueue_memset(lse_rank_dev, Float32(123.0))
                var np_in = Optional[Int](
                    num_partitions
                ) if num_partitions > 0 else Optional[Int](None)
                flare_mla_decoding[
                    rank=3,
                    config=MHAConfig[q_type](NUM_HEADS, Q_DEPTH),
                    ragged=True,
                    sparse=True,
                    skip_masked_tiles=True,
                    kernels=SM100MLA,
                ](
                    TileTensor(
                        out_rank_dev,
                        row_major(total_tokens, Idx[NUM_HEADS], Idx[V_DEPTH]),
                    ),
                    q_tt,
                    kv_cache,
                    CausalMask(),
                    row_offsets_tt,
                    SCALE,
                    ctx,
                    mla_args.gpu_tile_tensor(),
                    d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                        idx_rank_dev.unsafe_ptr()
                    ),
                    indices_stride=topk,
                    topk_lengths=NonNullPointer[DType.int32](
                        len_rank_dev.unsafe_ptr()
                        .as_imm()
                        .as_unsafe_any_origin()
                    ),
                    num_partitions_in=np_in,
                    lse_out=NonNullPointer[DType.float32](lse_rank_dev),
                    split_len_bound=split_len_bound,
                )
                ctx.enqueue_copy(out_rank, out_rank_dev)
                ctx.enqueue_copy(lse_rank, lse_rank_dev)
                ctx.synchronize()

                for t in range(total_tokens):
                    for h in range(NUM_HEADS):
                        var lse = Float64(lse_rank[t * NUM_HEADS + h])
                        var o_base = (t * NUM_HEADS + h) * V_DEPTH
                        if len_rank_host[t] == 0:
                            # An empty shard: -inf LSE and zero O.
                            assert_true(
                                isinf(lse) and lse < 0,
                                String(t"empty shard LSE {lse}"),
                            )
                            for d in range(V_DEPTH):
                                assert_true(
                                    out_rank[o_base + d].cast[DType.float32]()
                                    == 0,
                                    "empty shard O is not zero",
                                )
                            if h == 0:
                                empty_shards += 1
                            continue
                        assert_true(not isnan(lse) and not isinf(lse))
                        # Online base-2 merge of this rank into the running
                        # (max, weight, weighted O).
                        var i = t * NUM_HEADS + h
                        var m_new = max(merged_m[i], lse)
                        var c_old = exp2(merged_m[i] - m_new)
                        var c_new = exp2(lse - m_new)
                        merged_w[i] = merged_w[i] * c_old + c_new
                        for d in range(V_DEPTH):
                            merged_o[o_base + d] = merged_o[
                                o_base + d
                            ] * c_old + c_new * Float64(
                                out_rank[o_base + d].cast[DType.float32]()
                            )
                        merged_m[i] = m_new

            if dcp > 1:
                assert_true(empty_shards > 0, "the split has no empty shard")
            var max_o_err = Float64(0)
            var max_lse_err = Float64(0)
            for i in range(lse_elems):
                var lse = merged_m[i] + log2(merged_w[i])
                max_lse_err = max(max_lse_err, abs(lse - lse_ref[i]))
                assert_almost_equal(lse, lse_ref[i], atol=1e-4, rtol=0)
                for d in range(V_DEPTH):
                    var got = merged_o[i * V_DEPTH + d] / merged_w[i]
                    var want = Float64(
                        out_ref[i * V_DEPTH + d].cast[DType.float32]()
                    )
                    max_o_err = max(max_o_err, abs(got - want))
                    assert_almost_equal(got, want, atol=5e-2, rtol=1e-2)
            print(
                t"  dcp={dcp} np={num_partitions} bound={split_len_bound}"
                t" empty_shards={empty_shards} max_o_err={max_o_err}"
                t" max_lse_err={max_lse_err}"
            )

    _ = blocks_dev^
    _ = lut_dev^
    _ = cache_lengths_dev^
    _ = row_offsets_dev^
    _ = q_dev^
    _ = idx_full_dev^
    _ = out_ref_dev^
    _ = idx_rank_dev^
    _ = len_rank_dev^
    _ = out_rank_dev^
    _ = lse_rank_dev^
    _ = mla_args^


# ===-----------------------------------------------------------------------===#
# The fused `.cp` branch: CP latent store + per-token lengths + LSE output.
# ===-----------------------------------------------------------------------===#

comptime F_HEADS = 8
comptime F_NOPE = 128
comptime F_Q_DEPTH = F_NOPE + 64
comptime F_M_GRAN = 1
comptime F_N_GRAN = 128
comptime F_K_GRAN = 128


def _cache_pattern(i: Int) -> Scalar[DType.float8_e4m3fn]:
    return cast_saturating[DType.float8_e4m3fn](Float32((i % 37) - 18) * 0.125)


def _run_fused_cp[
    dcp_size: Int, dcp_interleave: Int
](
    ctx: DeviceContext,
    seq_lens: List[Int],
    cache_len: Int,
    rank: Int,
    mut blocks_out: List[UInt8],
    mut lse_out: List[Float32],
    mut lengths_out: List[Int],
) raises:
    """Runs `mla_decode_branch_fp8`, as the `.cp` op does, for one rank.

    The cache starts as `_cache_pattern`. Each token attends over the first
    rows its rank holds (none for the first token), and the branch stores the
    new tokens the rank owns. Returns the cache bytes, the LSE and the
    lengths.
    """
    comptime fp8 = DType.float8_e4m3fn
    comptime bf16 = DType.bfloat16
    comptime topk = 16
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    var layout = ContextParallelLayout[dcp_size, dcp_interleave](rank)
    var batch_size = len(seq_lens)
    var q_max = 0
    var total = 0
    for s in seq_lens:
        q_max = max(q_max, s)
        total += s
    var num_keys = cache_len + q_max
    var pages = ceildiv(num_keys, PAGE_SIZE)
    var block_elems = batch_size * pages * PAGE_SIZE * Q_DEPTH

    var blocks_host = ctx.enqueue_create_host_buffer[fp8](block_elems)
    for i in range(block_elems):
        blocks_host[i] = _cache_pattern(i)
    var blocks_dev = ctx.enqueue_create_buffer[fp8](block_elems)
    ctx.enqueue_copy(blocks_dev, blocks_host)
    var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size * pages
    )
    for i in range(batch_size * pages):
        lut_host[i] = UInt32(i)
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size * pages)
    ctx.enqueue_copy(lut_dev, lut_host)
    var cl_host = ctx.enqueue_create_host_buffer[DType.uint32](batch_size)
    var ro_host = ctx.enqueue_create_host_buffer[DType.uint32](batch_size + 1)
    ro_host[0] = 0
    for b in range(batch_size):
        cl_host[b] = UInt32(cache_len)
        ro_host[b + 1] = ro_host[b] + UInt32(seq_lens[b])
    var cl_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    ctx.enqueue_copy(cl_dev, cl_host)
    var ro_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    ctx.enqueue_copy(ro_dev, ro_host)

    # Rows this rank already holds, compacted: its first local slots.
    var held = layout.local_len(cache_len)
    var idx_host = ctx.enqueue_create_host_buffer[DType.int32](total * topk)
    var len_host = ctx.enqueue_create_host_buffer[DType.int32](total)
    for b in range(batch_size):
        for s in range(seq_lens[b]):
            var t = Int(ro_host[b]) + s
            var n = 0 if t == 0 else min(topk, held)
            len_host[t] = Int32(n)
            for i in range(topk):
                idx_host[t * topk + i] = Int32(
                    (b * pages * PAGE_SIZE + i) if i < n else -1
                )
    var idx_dev = ctx.enqueue_create_buffer[DType.int32](total * topk)
    ctx.enqueue_copy(idx_dev, idx_host)
    var len_dev = ctx.enqueue_create_buffer[DType.int32](total)
    ctx.enqueue_copy(len_dev, len_host)

    # Deterministic activations and weights, identical on every rank.
    var q_n = total * F_HEADS * F_Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[bf16](q_n)
    for i in range(q_n):
        q_host[i] = Scalar[bf16](Float32((i % 97) - 48) * 0.05)
    var q_dev = ctx.enqueue_create_buffer[bf16](q_n)
    ctx.enqueue_copy(q_dev, q_host)
    var kv_n = total * Q_DEPTH
    var kv_host = ctx.enqueue_create_host_buffer[bf16](kv_n)
    for i in range(kv_n):
        kv_host[i] = Scalar[bf16](Float32((i % 61) - 30) * 0.03)
    var kv_dev = ctx.enqueue_create_buffer[bf16](kv_n)
    ctx.enqueue_copy(kv_dev, kv_host)
    var wuk_n = F_HEADS * V_DEPTH * F_NOPE
    var wuk_host = ctx.enqueue_create_host_buffer[fp8](wuk_n)
    for i in range(wuk_n):
        wuk_host[i] = cast_saturating[fp8](Float32((i % 29) - 14) * 0.25)
    var wuk_dev = ctx.enqueue_create_buffer[fp8](wuk_n)
    ctx.enqueue_copy(wuk_dev, wuk_host)
    var wuv_n = F_HEADS * V_DEPTH * V_DEPTH
    var wuv_host = ctx.enqueue_create_host_buffer[fp8](wuv_n)
    for i in range(wuv_n):
        wuv_host[i] = cast_saturating[fp8](Float32((i % 23) - 11) * 0.25)
    var wuv_dev = ctx.enqueue_create_buffer[fp8](wuv_n)
    ctx.enqueue_copy(wuv_dev, wuv_host)
    comptime uk_sn = V_DEPTH // F_N_GRAN
    comptime uk_sk = ceildiv(F_NOPE, F_K_GRAN)
    comptime uv_sn = V_DEPTH // F_N_GRAN
    comptime uv_sk = V_DEPTH // F_K_GRAN
    var uks_dev = ctx.enqueue_create_buffer[DType.float32](
        F_HEADS * uk_sn * uk_sk
    )
    ctx.enqueue_memset(uks_dev, Float32(0.0125))
    var uvs_dev = ctx.enqueue_create_buffer[DType.float32](
        F_HEADS * uv_sn * uv_sk
    )
    ctx.enqueue_memset(uvs_dev, Float32(0.0125))
    var freqs_n = num_keys * 64
    var freqs_host = ctx.enqueue_create_host_buffer[DType.float32](freqs_n)
    for i in range(freqs_n):
        freqs_host[i] = Float32(0.5) if i % 2 == 0 else Float32(0.25)
    var freqs_dev = ctx.enqueue_create_buffer[DType.float32](freqs_n)
    ctx.enqueue_copy(freqs_dev, freqs_host)
    var gamma_dev = ctx.enqueue_create_buffer[bf16](V_DEPTH)
    ctx.enqueue_memset(gamma_dev, Scalar[bf16](1.0))
    var out_dev = ctx.enqueue_create_buffer[bf16](total * F_HEADS * V_DEPTH)
    var lse_dev = ctx.enqueue_create_buffer[DType.float32](total * F_HEADS)
    ctx.enqueue_memset(lse_dev, Float32(123.0))
    ctx.synchronize()

    var kv_collection = PagedKVCacheCollection[
        fp8, kv_params, PAGE_SIZE, scales_origin=MutAnyOrigin
    ](
        TileTensor(
            blocks_dev,
            row_major(
                Int64(batch_size * pages),
                Idx[1],
                Int64(NUM_LAYERS),
                Idx[PAGE_SIZE],
                Idx[1],
                Idx[Q_DEPTH],
            ),
        ).as_unsafe_any_origin(),
        TileTensor(cl_dev, row_major(batch_size))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(batch_size, pages))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_max),
        UInt32(cache_len),
    )
    var kv_in_tt = TileTensor(
        kv_dev.unsafe_ptr(), row_major((total, Idx[Q_DEPTH]))
    )

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[bf16, width]:
        return kv_in_tt.load[width=width](Coord(coords))

    var mla_args = MLADispatchScalarArgs[num_heads=F_HEADS, is_fp8_kv=True](
        batch_size, cache_len, q_max, ctx
    )
    mla_decode_branch_fp8[
        m_scale_granularity=F_M_GRAN,
        n_scale_granularity=F_N_GRAN,
        k_scale_granularity=F_K_GRAN,
        mask_str="causal",
        sparse_mla=True,
        skip_masked_tiles=True,
        dcp_size=dcp_size,
        dcp_interleave=dcp_interleave,
        kernels=SM100MLAGraph,
    ](
        TileTensor(
            out_dev.unsafe_ptr(),
            row_major((total, Idx[F_HEADS], Idx[V_DEPTH])),
        ),
        TileTensor(
            q_dev.unsafe_ptr(), row_major((total, Idx[F_HEADS], Idx[F_Q_DEPTH]))
        ),
        TileTensor(ro_dev.unsafe_ptr(), row_major(batch_size + 1)),
        TileTensor(freqs_dev.unsafe_ptr(), row_major((num_keys, Idx[64]))),
        TileTensor(gamma_dev.unsafe_ptr(), row_major(Idx[V_DEPTH])),
        kv_collection,
        UInt32(0),
        SCALE,
        Float32(1e-6),
        TileTensor(
            wuk_dev.unsafe_ptr(),
            row_major((Idx[F_HEADS], Idx[V_DEPTH], Idx[F_NOPE])),
        ),
        TileTensor(
            uks_dev.unsafe_ptr(),
            row_major((Idx[F_HEADS], Idx[uk_sn], Idx[uk_sk])),
        ),
        TileTensor(
            wuv_dev.unsafe_ptr(),
            row_major((Idx[F_HEADS], Idx[V_DEPTH], Idx[V_DEPTH])),
        ),
        TileTensor(
            uvs_dev.unsafe_ptr(),
            row_major((Idx[F_HEADS], Idx[uv_sn], Idx[uv_sk])),
        ),
        mla_args.gpu_tile_tensor(),
        kv_input_fn,
        ctx,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](idx_dev.unsafe_ptr()),
        indices_stride=topk,
        topk_lengths=NonNullPointer[DType.int32](
            len_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        lse_out=NonNullPointer[DType.float32](lse_dev),
        split_len_bound=topk,
        dcp_rank=rank,
    )
    ctx.enqueue_copy(blocks_host, blocks_dev)
    var lse_host = ctx.enqueue_create_host_buffer[DType.float32](
        total * F_HEADS
    )
    ctx.enqueue_copy(lse_host, lse_dev)
    ctx.synchronize()

    blocks_out = List[UInt8](length=block_elems, fill=0)
    for i in range(block_elems):
        blocks_out[i] = bitcast[DType.uint8](blocks_host[i])
    lse_out = List[Float32](length=total * F_HEADS, fill=0)
    for i in range(total * F_HEADS):
        lse_out[i] = lse_host[i]
    lengths_out = List[Int](length=total, fill=0)
    for t in range(total):
        lengths_out[t] = Int(len_host[t])

    _ = blocks_dev^
    _ = lut_dev^
    _ = cl_dev^
    _ = ro_dev^
    _ = idx_dev^
    _ = len_dev^
    _ = q_dev^
    _ = kv_dev^
    _ = wuk_dev^
    _ = wuv_dev^
    _ = uks_dev^
    _ = uvs_dev^
    _ = freqs_dev^
    _ = gamma_dev^
    _ = out_dev^
    _ = lse_dev^
    _ = mla_args^


def run_fused_cp_store[
    dcp_size: Int, dcp_interleave: Int
](ctx: DeviceContext, seq_lens: List[Int], cache_len: Int) raises:
    """Checks the CP latent store and the LSE of the fused `.cp` branch.

    The group of one stores every new token at its position. Each rank of a
    larger group must store exactly the new tokens it owns, at
    `local_slot(pos)`, with the same bytes, and leave every other byte of its
    cache as it was. Tokens with keys get a finite LSE and the empty one
    `-inf`.
    """
    print(
        t"fused_cp_store: dcp={dcp_size} interleave={dcp_interleave}"
        t" cache_len={cache_len}"
    )
    var ref_bytes = List[UInt8]()
    var lse = List[Float32]()
    var lengths = List[Int]()
    _run_fused_cp[1, 1](ctx, seq_lens, cache_len, 0, ref_bytes, lse, lengths)
    var num_keys = cache_len
    var q_max = 0
    for s in seq_lens:
        q_max = max(q_max, s)
    num_keys += q_max
    var pages = ceildiv(num_keys, PAGE_SIZE)
    for r in range(dcp_size):
        var got = List[UInt8]()
        _run_fused_cp[dcp_size, dcp_interleave](
            ctx, seq_lens, cache_len, r, got, lse, lengths
        )
        var layout = ContextParallelLayout[dcp_size, dcp_interleave](r)
        var want = List[UInt8](length=len(ref_bytes), fill=0)
        for i in range(len(want)):
            want[i] = bitcast[DType.uint8](_cache_pattern(i))
        var stored = 0
        for b in range(len(seq_lens)):
            for s in range(seq_lens[b]):
                var pos = cache_len + s
                if not layout.owns(pos):
                    continue
                var slot = layout.local_slot(pos)
                var src = (b * pages * PAGE_SIZE + pos) * Q_DEPTH
                var dst = (b * pages * PAGE_SIZE + slot) * Q_DEPTH
                for d in range(Q_DEPTH):
                    want[dst + d] = ref_bytes[src + d]
                stored += 1
        var bad = 0
        for i in range(len(want)):
            if got[i] != want[i]:
                bad += 1
        print(t"  rank={r} stored={stored} mismatched_bytes={bad}")
        assert_true(bad == 0, "the CP store differs from the expected image")
        for t in range(len(lengths)):
            for h in range(F_HEADS):
                var v = lse[t * F_HEADS + h]
                if lengths[t] == 0:
                    assert_true(isinf(v) and v < 0, "empty token LSE")
                else:
                    assert_true(
                        not isnan(v) and not isinf(v), "token LSE not finite"
                    )


# ===-----------------------------------------------------------------------===#
# The BF16 branch: CP latent store, latent output and absorbed query.
# ===-----------------------------------------------------------------------===#

comptime B_HEADS = 64
comptime B_NOPE = 128
comptime B_ROPE = Q_DEPTH - V_DEPTH
comptime B_Q_DEPTH = B_NOPE + B_ROPE
comptime B_V = 128
comptime B_TOPK = 32


def _bf16_cache_pattern(i: Int) -> Scalar[DType.bfloat16]:
    return Scalar[DType.bfloat16](Float32((i % 37) - 18) * 0.0625)


def _bf16_q(i: Int) -> Scalar[DType.bfloat16]:
    return Scalar[DType.bfloat16](Float32((i % 97) - 48) * 0.02)


def _bf16_w_uk(i: Int) -> Scalar[DType.bfloat16]:
    """`w_uk`, `[B_HEADS, V_DEPTH, B_NOPE]`."""
    return Scalar[DType.bfloat16](Float32((i % 29) - 14) * 0.01)


def _bf16_w_uv(i: Int) -> Scalar[DType.bfloat16]:
    """`w_uv`, `[B_HEADS, B_V, V_DEPTH]`."""
    return Scalar[DType.bfloat16](Float32((i % 23) - 11) * 0.01)


def _bf16_owned_selection[
    dcp_size: Int, dcp_interleave: Int
](rank: Int, t: Int, cache_len: Int) -> List[Int]:
    """Token `t`'s selected cached positions that `rank` owns, in order.

    Token 0 selects nothing. Every other token selects `B_TOPK` distinct
    cached positions in a scrambled order, which the group's ranks split by
    ownership.
    """
    var owned = List[Int]()
    if t == 0:
        return owned^
    var layout = ContextParallelLayout[dcp_size, dcp_interleave](rank)
    var m = _coprime_multiplier(cache_len)
    for i in range(B_TOPK):
        var pos = (i * m + 3 * t + 1) % cache_len
        if layout.owns(pos):
            owned.append(pos)
    return owned^


def _run_bf16_branch[
    dcp_size: Int,
    dcp_interleave: Int,
    q_absorbed: Bool = False,
    latent_output: Bool = False,
    owned_rows: Bool = False,
    heads: Int = B_HEADS,
](
    ctx: DeviceContext,
    seq_lens: List[Int],
    cache_len: Int,
    rank: Int,
    q_in: List[Scalar[DType.bfloat16]],
    mut blocks_out: List[UInt16],
    mut out_out: List[Float32],
    mut lse_out: List[Float32],
) raises:
    """Runs `mla_decode_branch_bf16`, as the BF16 `.cp` op does, for one rank.

    The BF16 cache starts as `_bf16_cache_pattern`. `q_in` is
    `[tokens, heads, B_Q_DEPTH]`, or `[tokens, heads, Q_DEPTH]` already
    absorbed with `q_absorbed`. Every token attends over the same `B_TOPK`
    causally visible rows. With `owned_rows`, each token instead attends over
    the rows of `_bf16_owned_selection` this rank owns, compacted to the front
    of its index row as local slots with `-1` after them and a per-token
    length below `B_TOPK` (0 for token 0), and the rank's cache holds each
    cached position's `_bf16_cache_pattern` row at its local slot, so the
    ranks' results merge to a group of one's. Returns the cache as raw 16-bit
    words, the output (`[tokens, heads, B_V]`, or `V_DEPTH` wide with
    `latent_output`) and the LSE.
    """
    comptime bf16 = DType.bfloat16
    comptime q_depth = Q_DEPTH if q_absorbed else B_Q_DEPTH
    comptime out_depth = V_DEPTH if latent_output else B_V
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    var batch_size = len(seq_lens)
    var q_max = 0
    var total = 0
    for s in seq_lens:
        q_max = max(q_max, s)
        total += s
    var num_keys = cache_len + q_max
    var pages = ceildiv(num_keys, PAGE_SIZE)
    var block_elems = batch_size * pages * PAGE_SIZE * Q_DEPTH

    var blocks_host = ctx.enqueue_create_host_buffer[bf16](block_elems)
    for i in range(block_elems):
        blocks_host[i] = _bf16_cache_pattern(i)
    var layout = ContextParallelLayout[dcp_size, dcp_interleave](rank)
    comptime if owned_rows:
        # Each cached position this rank owns, at its local slot.
        for b in range(batch_size):
            for pos in range(cache_len):
                if not layout.owns(pos):
                    continue
                var src = (b * pages * PAGE_SIZE + pos) * Q_DEPTH
                var dst = (
                    b * pages * PAGE_SIZE + layout.local_slot(pos)
                ) * Q_DEPTH
                for d in range(Q_DEPTH):
                    blocks_host[dst + d] = _bf16_cache_pattern(src + d)
    var blocks_dev = ctx.enqueue_create_buffer[bf16](block_elems)
    ctx.enqueue_copy(blocks_dev, blocks_host)
    var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size * pages
    )
    for i in range(batch_size * pages):
        lut_host[i] = UInt32(i)
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size * pages)
    ctx.enqueue_copy(lut_dev, lut_host)
    var cl_host = ctx.enqueue_create_host_buffer[DType.uint32](batch_size)
    var ro_host = ctx.enqueue_create_host_buffer[DType.uint32](batch_size + 1)
    ro_host[0] = 0
    for b in range(batch_size):
        cl_host[b] = UInt32(cache_len)
        ro_host[b + 1] = ro_host[b] + UInt32(seq_lens[b])
    var cl_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    ctx.enqueue_copy(cl_dev, cl_host)
    var ro_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    ctx.enqueue_copy(ro_dev, ro_host)

    # `B_TOPK` distinct causally visible rows per token, in a scrambled order.
    var idx_host = ctx.enqueue_create_host_buffer[DType.int32](total * B_TOPK)
    var len_host = ctx.enqueue_create_host_buffer[DType.int32](total)
    for b in range(batch_size):
        for s in range(seq_lens[b]):
            var t = Int(ro_host[b]) + s
            comptime if owned_rows:
                if B_TOPK > cache_len:
                    raise Error("B_TOPK must not exceed the cached keys")
                var owned = _bf16_owned_selection[dcp_size, dcp_interleave](
                    rank, t, cache_len
                )
                len_host[t] = Int32(len(owned))
                for i in range(B_TOPK):
                    idx_host[t * B_TOPK + i] = Int32(
                        b * pages * PAGE_SIZE + layout.local_slot(owned[i])
                    ) if i < len(owned) else Int32(-1)
                continue
            var visible = cache_len + s + 1
            if B_TOPK > visible:
                raise Error("B_TOPK must not exceed the visible keys")
            var m = _coprime_multiplier(visible)
            len_host[t] = Int32(B_TOPK)
            for i in range(B_TOPK):
                var pos = (i * m + 3 * t + 1) % visible
                idx_host[t * B_TOPK + i] = Int32(b * pages * PAGE_SIZE + pos)
    var idx_dev = ctx.enqueue_create_buffer[DType.int32](total * B_TOPK)
    ctx.enqueue_copy(idx_dev, idx_host)
    var len_dev = ctx.enqueue_create_buffer[DType.int32](total)
    ctx.enqueue_copy(len_dev, len_host)

    var q_n = total * heads * q_depth
    if len(q_in) != q_n:
        raise Error("q_in has the wrong size")
    var q_host = ctx.enqueue_create_host_buffer[bf16](q_n)
    for i in range(q_n):
        q_host[i] = q_in[i]
    var q_dev = ctx.enqueue_create_buffer[bf16](q_n)
    ctx.enqueue_copy(q_dev, q_host)
    var kv_n = total * Q_DEPTH
    var kv_host = ctx.enqueue_create_host_buffer[bf16](kv_n)
    for i in range(kv_n):
        kv_host[i] = Scalar[bf16](Float32((i % 61) - 30) * 0.03)
    var kv_dev = ctx.enqueue_create_buffer[bf16](kv_n)
    ctx.enqueue_copy(kv_dev, kv_host)
    var wuk_n = heads * V_DEPTH * B_NOPE
    var wuk_host = ctx.enqueue_create_host_buffer[bf16](wuk_n)
    for i in range(wuk_n):
        wuk_host[i] = _bf16_w_uk(i)
    var wuk_dev = ctx.enqueue_create_buffer[bf16](wuk_n)
    ctx.enqueue_copy(wuk_dev, wuk_host)
    var wuv_n = heads * B_V * V_DEPTH
    var wuv_host = ctx.enqueue_create_host_buffer[bf16](wuv_n)
    for i in range(wuv_n):
        wuv_host[i] = _bf16_w_uv(i)
    var wuv_dev = ctx.enqueue_create_buffer[bf16](wuv_n)
    ctx.enqueue_copy(wuv_dev, wuv_host)
    var freqs_n = num_keys * B_ROPE
    var freqs_host = ctx.enqueue_create_host_buffer[DType.float32](freqs_n)
    for i in range(freqs_n):
        freqs_host[i] = Float32(0.5) if i % 2 == 0 else Float32(0.25)
    var freqs_dev = ctx.enqueue_create_buffer[DType.float32](freqs_n)
    ctx.enqueue_copy(freqs_dev, freqs_host)
    var gamma_dev = ctx.enqueue_create_buffer[bf16](V_DEPTH)
    ctx.enqueue_memset(gamma_dev, Scalar[bf16](1.0))
    var out_n = total * heads * out_depth
    var out_dev = ctx.enqueue_create_buffer[bf16](out_n)
    ctx.enqueue_memset(out_dev, Scalar[bf16](7.0))
    var lse_dev = ctx.enqueue_create_buffer[DType.float32](total * heads)
    ctx.enqueue_memset(lse_dev, Float32(123.0))
    ctx.synchronize()

    var kv_collection = PagedKVCacheCollection[
        bf16, kv_params, PAGE_SIZE, scales_origin=MutAnyOrigin
    ](
        TileTensor(
            blocks_dev,
            row_major(
                Int64(batch_size * pages),
                Idx[1],
                Int64(NUM_LAYERS),
                Idx[PAGE_SIZE],
                Idx[1],
                Idx[Q_DEPTH],
            ),
        ).as_unsafe_any_origin(),
        TileTensor(cl_dev, row_major(batch_size))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(batch_size, pages))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_max),
        UInt32(cache_len),
    )
    var kv_in_tt = TileTensor(
        kv_dev.unsafe_ptr(), row_major((total, Idx[Q_DEPTH]))
    )

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[bf16, width]:
        return kv_in_tt.load[width=width](Coord(coords))

    var mla_args = MLADispatchScalarArgs[num_heads=heads, is_fp8_kv=False](
        batch_size, cache_len, q_max, ctx
    )
    mla_decode_branch_bf16[
        mask_str="causal",
        sparse_mla=True,
        skip_masked_tiles=True,
        dcp_size=dcp_size,
        dcp_interleave=dcp_interleave,
        q_absorbed=q_absorbed,
        latent_output=latent_output,
        kernels=SM100MLAGraph,
    ](
        TileTensor(
            out_dev.unsafe_ptr(),
            row_major((total, Idx[heads], Idx[out_depth])),
        ),
        TileTensor(
            q_dev.unsafe_ptr(), row_major((total, Idx[heads], Idx[q_depth]))
        ),
        TileTensor(ro_dev.unsafe_ptr(), row_major(batch_size + 1)),
        TileTensor(freqs_dev.unsafe_ptr(), row_major((num_keys, Idx[B_ROPE]))),
        TileTensor(gamma_dev.unsafe_ptr(), row_major(Idx[V_DEPTH])),
        kv_collection,
        UInt32(0),
        SCALE,
        Float32(1e-6),
        TileTensor(
            wuk_dev.unsafe_ptr(),
            row_major((Idx[heads], Idx[V_DEPTH], Idx[B_NOPE])),
        ),
        TileTensor(
            wuv_dev.unsafe_ptr(),
            row_major((Idx[heads], Idx[B_V], Idx[V_DEPTH])),
        ),
        mla_args.gpu_tile_tensor(),
        kv_input_fn,
        ctx,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](idx_dev.unsafe_ptr()),
        indices_stride=B_TOPK,
        topk_lengths=NonNullPointer[DType.int32](
            len_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        lse_out=NonNullPointer[DType.float32](lse_dev),
        split_len_bound=B_TOPK,
        dcp_rank=rank,
    )
    ctx.enqueue_copy(blocks_host, blocks_dev)
    var out_host = ctx.enqueue_create_host_buffer[bf16](out_n)
    ctx.enqueue_copy(out_host, out_dev)
    var lse_host = ctx.enqueue_create_host_buffer[DType.float32](total * heads)
    ctx.enqueue_copy(lse_host, lse_dev)
    ctx.synchronize()

    blocks_out = List[UInt16](length=block_elems, fill=0)
    for i in range(block_elems):
        blocks_out[i] = bitcast[DType.uint16](blocks_host[i])
    out_out = List[Float32](length=out_n, fill=0)
    for i in range(out_n):
        out_out[i] = out_host[i].cast[DType.float32]()
    lse_out = List[Float32](length=total * heads, fill=0)
    for i in range(total * heads):
        lse_out[i] = lse_host[i]

    _ = blocks_dev^
    _ = lut_dev^
    _ = cl_dev^
    _ = ro_dev^
    _ = idx_dev^
    _ = len_dev^
    _ = q_dev^
    _ = kv_dev^
    _ = wuk_dev^
    _ = wuv_dev^
    _ = freqs_dev^
    _ = gamma_dev^
    _ = out_dev^
    _ = lse_dev^
    _ = mla_args^


def _bf16_plain_q[
    heads: Int = B_HEADS
](total: Int) -> List[Scalar[DType.bfloat16]]:
    var q = List[Scalar[DType.bfloat16]](capacity=total * heads * B_Q_DEPTH)
    for i in range(total * heads * B_Q_DEPTH):
        q.append(_bf16_q(i))
    return q^


def run_bf16_cp_store[
    dcp_size: Int, dcp_interleave: Int
](ctx: DeviceContext, seq_lens: List[Int], cache_len: Int) raises:
    """Checks the CP latent store and the LSE of the BF16 branch.

    Like `run_fused_cp_store`: each rank stores exactly the new tokens it
    owns, at `local_slot(pos)`, with the words a group of one stores there,
    and leaves every other word of its cache as it was.
    """
    print(
        t"bf16_cp_store: dcp={dcp_size} interleave={dcp_interleave}"
        t" cache_len={cache_len}"
    )
    var total = 0
    var q_max = 0
    for s in seq_lens:
        total += s
        q_max = max(q_max, s)
    var q = _bf16_plain_q(total)
    var ref_words = List[UInt16]()
    var out = List[Float32]()
    var lse = List[Float32]()
    _run_bf16_branch[1, 1](ctx, seq_lens, cache_len, 0, q, ref_words, out, lse)
    var pages = ceildiv(cache_len + q_max, PAGE_SIZE)
    for r in range(dcp_size):
        var got = List[UInt16]()
        _run_bf16_branch[dcp_size, dcp_interleave](
            ctx, seq_lens, cache_len, r, q, got, out, lse
        )
        var layout = ContextParallelLayout[dcp_size, dcp_interleave](r)
        var want = List[UInt16](length=len(ref_words), fill=0)
        for i in range(len(want)):
            want[i] = bitcast[DType.uint16](_bf16_cache_pattern(i))
        var stored = 0
        for b in range(len(seq_lens)):
            for s in range(seq_lens[b]):
                var pos = cache_len + s
                if not layout.owns(pos):
                    continue
                var slot = layout.local_slot(pos)
                var src = (b * pages * PAGE_SIZE + pos) * Q_DEPTH
                var dst = (b * pages * PAGE_SIZE + slot) * Q_DEPTH
                var changed = False
                for d in range(Q_DEPTH):
                    want[dst + d] = ref_words[src + d]
                    changed = changed or ref_words[src + d] != bitcast[
                        DType.uint16
                    ](_bf16_cache_pattern(src + d))
                assert_true(changed, "the plain store left a new row as is")
                stored += 1
        var bad = 0
        for i in range(len(want)):
            if got[i] != want[i]:
                bad += 1
        print(t"  rank={r} stored={stored} mismatched_words={bad}")
        assert_true(bad == 0, "the CP store differs from the expected image")
        for v in lse:
            assert_true(not isnan(v) and not isinf(v), "token LSE not finite")


def run_bf16_cp_owned_rows[
    dcp_size: Int, dcp_interleave: Int, heads: Int = B_HEADS
](ctx: DeviceContext, seq_lens: List[Int], cache_len: Int) raises:
    """Checks the BF16 `.cp` branch on compacted owned rows and empty tokens.

    Each rank attends over the selected rows it owns, compacted as local
    slots with a per-token length below `B_TOPK` (`_run_bf16_branch` with
    `owned_rows`). A token with no rows, token 0 on every rank, must return
    LSE `-inf` and a zero output; every other row a finite LSE. The ranks'
    base-2 LSE merge,

        m = max_r lse_r;  w_r = 2^(lse_r - m);  O = sum_r w_r O_r / sum_r w_r

    must match a group of one attending over the whole selection.
    """
    print(
        t"bf16_cp_owned_rows: dcp={dcp_size} interleave={dcp_interleave}"
        t" heads={heads}"
        t" cache_len={cache_len}"
    )
    var total = 0
    for s in seq_lens:
        total += s
    var q = _bf16_plain_q[heads](total)
    var words = List[UInt16]()
    var ref_out = List[Float32]()
    var ref_lse = List[Float32]()
    _run_bf16_branch[1, 1, owned_rows=True, heads=heads](
        ctx, seq_lens, cache_len, 0, q, words, ref_out, ref_lse
    )
    var n = total * heads
    var merged_m = List[Float64](length=n, fill=min_or_neg_inf[DType.float64]())
    var merged_w = List[Float64](length=n, fill=0)
    var merged_o = List[Float64](length=n * B_V, fill=0)
    var empty_rows = 0
    var short_rows = 0
    for r in range(dcp_size):
        var out = List[Float32]()
        var lse = List[Float32]()
        _run_bf16_branch[
            dcp_size, dcp_interleave, owned_rows=True, heads=heads
        ](ctx, seq_lens, cache_len, r, q, words, out, lse)
        for t in range(total):
            var length = len(
                _bf16_owned_selection[dcp_size, dcp_interleave](r, t, cache_len)
            )
            if length == 0:
                empty_rows += 1
            elif length < B_TOPK:
                short_rows += 1
            for h in range(heads):
                var i = t * heads + h
                var v = Float64(lse[i])
                if length == 0:
                    assert_true(
                        isinf(v) and v < 0, String(t"empty token LSE {v}")
                    )
                    for d in range(B_V):
                        assert_true(
                            out[i * B_V + d] == 0, "empty token O is not zero"
                        )
                    continue
                assert_true(
                    not isnan(v) and not isinf(v), "token LSE not finite"
                )
                var m_new = max(merged_m[i], v)
                var c_old = exp2(merged_m[i] - m_new)
                var c_new = exp2(v - m_new)
                merged_w[i] = merged_w[i] * c_old + c_new
                for d in range(B_V):
                    var o_old = merged_o[i * B_V + d]
                    merged_o[i * B_V + d] = o_old * c_old + c_new * Float64(
                        out[i * B_V + d]
                    )
                merged_m[i] = m_new
    assert_true(empty_rows >= dcp_size, "no rank had an empty token")
    assert_true(short_rows > 0, "no rank had a partial selection")

    var max_out_err = Float64(0)
    var max_lse_err = Float64(0)
    for t in range(total):
        for h in range(heads):
            var i = t * heads + h
            if t == 0:
                # The group of one has no rows for token 0 either.
                assert_true(
                    isinf(ref_lse[i]) and ref_lse[i] < 0, "reference LSE"
                )
                assert_true(merged_w[i] == 0, "a rank had rows for token 0")
                continue
            var lse = merged_m[i] + log2(merged_w[i])
            max_lse_err = max(max_lse_err, abs(lse - Float64(ref_lse[i])))
            assert_almost_equal(lse, Float64(ref_lse[i]), atol=1e-2, rtol=1e-3)
            for d in range(B_V):
                var o = merged_o[i * B_V + d] / merged_w[i]
                var want = Float64(ref_out[i * B_V + d])
                max_out_err = max(max_out_err, abs(o - want))
                assert_almost_equal(o, want, atol=2e-2, rtol=2e-2)
    print(
        t"  empty_rows={empty_rows} short_rows={short_rows}"
        t" max_out_err={max_out_err} max_lse_err={max_lse_err}"
    )


def run_bf16_branch_modes(
    ctx: DeviceContext, seq_lens: List[Int], cache_len: Int
) raises:
    """Checks the BF16 `.cp` op's `latent_output` and `q_absorbed` modes.

    `latent_output` must return the latent the default projects through
    `w_uv`, with the same LSE bit for bit; `q_absorbed`, given the default's
    query absorbed on the host, `[q_nope @ w_uk, q_rope]`, must attend as the
    default does, to BF16 rounding of the absorbed query.
    """
    print(t"bf16_branch_modes: cache_len={cache_len}")
    var total = 0
    for s in seq_lens:
        total += s
    var q = _bf16_plain_q(total)
    var words = List[UInt16]()
    var out = List[Float32]()
    var lse = List[Float32]()
    _run_bf16_branch[1, 1](ctx, seq_lens, cache_len, 0, q, words, out, lse)
    var max_abs = Float32(0)
    for v in out:
        assert_true(not isnan(v) and not isinf(v), "default output not finite")
        max_abs = max(max_abs, abs(v))
    assert_true(max_abs > 0, "the default output is all zero")

    # `latent_output`: the latent, projected by `w_uv` on the host.
    var latent = List[Float32]()
    var latent_lse = List[Float32]()
    _run_bf16_branch[1, 1, latent_output=True](
        ctx, seq_lens, cache_len, 0, q, words, latent, latent_lse
    )
    var max_proj_err = Float64(0)
    for t in range(total):
        for h in range(B_HEADS):
            var i = t * B_HEADS + h
            assert_true(
                latent_lse[i] == lse[i], "latent_output changed the LSE"
            )
            for v in range(B_V):
                var acc = Float64(0)
                for l in range(V_DEPTH):
                    acc += Float64(latent[i * V_DEPTH + l]) * Float64(
                        _bf16_w_uv((h * B_V + v) * V_DEPTH + l).cast[
                            DType.float32
                        ]()
                    )
                var want = Float64(out[i * B_V + v])
                max_proj_err = max(max_proj_err, abs(acc - want))
                assert_almost_equal(acc, want, atol=1e-2, rtol=1e-2)

    # `q_absorbed`: the default's query, absorbed on the host.
    var q_abs = List[Scalar[DType.bfloat16]](
        length=total * B_HEADS * Q_DEPTH, fill=0
    )
    for t in range(total):
        for h in range(B_HEADS):
            var q_base = (t * B_HEADS + h) * B_Q_DEPTH
            var a_base = (t * B_HEADS + h) * Q_DEPTH
            for l in range(V_DEPTH):
                var acc = Float32(0)
                for n in range(B_NOPE):
                    acc += (
                        q[q_base + n].cast[DType.float32]()
                        * _bf16_w_uk((h * V_DEPTH + l) * B_NOPE + n).cast[
                            DType.float32
                        ]()
                    )
                q_abs[a_base + l] = acc.cast[DType.bfloat16]()
            for j in range(B_ROPE):
                q_abs[a_base + V_DEPTH + j] = q[q_base + B_NOPE + j]
    var absorbed = List[Float32]()
    var absorbed_lse = List[Float32]()
    _run_bf16_branch[1, 1, q_absorbed=True](
        ctx, seq_lens, cache_len, 0, q_abs, words, absorbed, absorbed_lse
    )
    var max_abs_err = Float64(0)
    var max_lse_err = Float64(0)
    for i in range(total * B_HEADS):
        max_lse_err = max(
            max_lse_err, abs(Float64(absorbed_lse[i]) - Float64(lse[i]))
        )
        assert_almost_equal(absorbed_lse[i], lse[i], atol=1e-2, rtol=1e-2)
    for i in range(len(out)):
        max_abs_err = max(max_abs_err, abs(Float64(absorbed[i] - out[i])))
        assert_almost_equal(absorbed[i], out[i], atol=1e-2, rtol=2e-2)
    print(
        t"  max_out={max_abs} max_proj_err={max_proj_err}"
        t" max_absorbed_err={max_abs_err} max_absorbed_lse_err={max_lse_err}"
    )


# ===-----------------------------------------------------------------------===#
# NVFP4 latent cache
# ===-----------------------------------------------------------------------===#

# A packed row: the 512-element latent as e2m1 pairs, then the 64 rope dims as
# FP8 bytes.
comptime NV_PACKED_ROW = V_DEPTH // 2 + B_ROPE
# One E4M3 scale per 16 latent elements.
comptime NV_SF_PER_ROW = V_DEPTH // 16
# `quantization_granularity` counts packed bytes: 16 elements are 8 bytes.
comptime NV_QUANT_GRANULARITY = 8


def _nvfp4_value_byte(i: Int) -> UInt8:
    """Byte `i` of the initial packed cache.

    Latent bytes hold any two e2m1 codes; rope bytes stay small finite FP8
    (never the E4M3 NaN encodings 0x7F and 0xFF).
    """
    if i % NV_PACKED_ROW < V_DEPTH // 2:
        return UInt8((i * 37 + 11) % 256)
    var mag = UInt8((i * 13 + 5) % 0x38)
    return mag | (UInt8(0x80) if i % 3 == 0 else UInt8(0))


def _nvfp4_scale_byte(i: Int) -> UInt8:
    """Byte `i` of the initial scale pool: an E4M3 scale in about [0.1, 1]."""
    return UInt8(0x28 + (i * 5) % 24)


def _nvfp4_scale_row(
    scales_lut: List[Int], pages: Int, b: Int, slot: Int
) -> Int:
    """Batch entry `b`'s slot `slot` in the scale pool.

    The pool's blocks are `PAGE_SIZE` rows apart (one layer).
    """
    return scales_lut[b * pages + slot // PAGE_SIZE] * PAGE_SIZE + (
        slot % PAGE_SIZE
    )


def _run_nvfp4_branch[
    dcp_size: Int,
    dcp_interleave: Int,
    sel_dcp: Int = dcp_size,
    sel_interleave: Int = dcp_interleave,
    heads: Int = B_HEADS,
](
    ctx: DeviceContext,
    seq_lens: List[Int],
    cache_len: Int,
    rank: Int,
    q_in: List[Scalar[DType.bfloat16]],
    mut values_out: List[UInt8],
    mut scales_out: List[UInt8],
    mut scales_lut_out: List[Int],
    mut out_out: List[Float32],
    mut lse_out: List[Float32],
    sel_rank: Int = -1,
) raises:
    """Runs `mla_decode_branch_bf16` over an NVFP4 cache, as the
    `.sparse.nvfp4.cp` op does, for one rank.

    The packed values start as `_nvfp4_value_byte` and the scale pool as
    `_nvfp4_scale_byte`; then each cached position this rank owns gets the
    initial value row and scale row of its global position at its local slot,
    so the ranks hold one cache between them. Each token attends over the
    rows of `_bf16_owned_selection` this rank owns, compacted as local slots:
    value rows first, their scale-pool rows `B_TOPK` columns later, `-1`
    after both and a per-token length. With `sel_dcp` and `sel_rank` the
    selection is instead the rows rank `sel_rank` of a `sel_dcp` group owns
    (`sel_interleave` positions in turn), so a group of one can attend over
    exactly one rank's share at the positions' own slots. Returns the value
    and scale bytes, the scale lookup table, the output
    `[tokens, heads, B_V]` and the LSE.
    """
    comptime bf16 = DType.bfloat16
    comptime fp8 = DType.float8_e4m3fn
    comptime kv_params = KVCacheStaticParams(
        num_heads=1,
        head_size=NV_PACKED_ROW,
        is_mla=True,
        fp8_tail_size=B_ROPE,
    )
    comptime stride = 2 * B_TOPK
    var batch_size = len(seq_lens)
    var q_max = 0
    var total = 0
    for s in seq_lens:
        q_max = max(q_max, s)
        total += s
    var num_keys = cache_len + q_max
    var pages = ceildiv(num_keys, PAGE_SIZE)
    var n_pages = batch_size * pages
    var value_elems = n_pages * PAGE_SIZE * NV_PACKED_ROW
    var scale_elems = n_pages * PAGE_SIZE * NV_SF_PER_ROW
    if B_TOPK > cache_len:
        raise Error("B_TOPK must not exceed the cached keys")

    # The values' table is the identity; the scales' a permutation of it.
    var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](n_pages)
    var slut_host = ctx.enqueue_create_host_buffer[DType.uint32](n_pages)
    var mult = _coprime_multiplier(n_pages)
    scales_lut_out = List[Int](length=n_pages, fill=0)
    for i in range(n_pages):
        lut_host[i] = UInt32(i)
        var sb = (i * mult + 1) % n_pages
        slut_host[i] = UInt32(sb)
        scales_lut_out[i] = sb

    var values_host = ctx.enqueue_create_host_buffer[DType.uint8](value_elems)
    for i in range(value_elems):
        values_host[i] = _nvfp4_value_byte(i)
    var scales_host = ctx.enqueue_create_host_buffer[fp8](scale_elems)
    for i in range(scale_elems):
        scales_host[i] = bitcast[fp8, 1](
            SIMD[DType.uint8, 1](_nvfp4_scale_byte(i))
        )[0]
    var layout = ContextParallelLayout[dcp_size, dcp_interleave](rank)
    for b in range(batch_size):
        for pos in range(cache_len):
            if not layout.owns(pos):
                continue
            var slot = layout.local_slot(pos)
            var src = b * pages * PAGE_SIZE + pos
            var dst = b * pages * PAGE_SIZE + slot
            for d in range(NV_PACKED_ROW):
                values_host[dst * NV_PACKED_ROW + d] = _nvfp4_value_byte(
                    src * NV_PACKED_ROW + d
                )
            var src_s = _nvfp4_scale_row(scales_lut_out, pages, b, pos)
            var dst_s = _nvfp4_scale_row(scales_lut_out, pages, b, slot)
            for g in range(NV_SF_PER_ROW):
                scales_host[dst_s * NV_SF_PER_ROW + g] = bitcast[fp8, 1](
                    SIMD[DType.uint8, 1](
                        _nvfp4_scale_byte(src_s * NV_SF_PER_ROW + g)
                    )
                )[0]
    var values_dev = ctx.enqueue_create_buffer[DType.uint8](value_elems)
    ctx.enqueue_copy(values_dev, values_host)
    var scales_dev = ctx.enqueue_create_buffer[fp8](scale_elems)
    ctx.enqueue_copy(scales_dev, scales_host)
    var lut_dev = ctx.enqueue_create_buffer[DType.uint32](n_pages)
    ctx.enqueue_copy(lut_dev, lut_host)
    var slut_dev = ctx.enqueue_create_buffer[DType.uint32](n_pages)
    ctx.enqueue_copy(slut_dev, slut_host)
    var cl_host = ctx.enqueue_create_host_buffer[DType.uint32](batch_size)
    var ro_host = ctx.enqueue_create_host_buffer[DType.uint32](batch_size + 1)
    ro_host[0] = 0
    for b in range(batch_size):
        cl_host[b] = UInt32(cache_len)
        ro_host[b + 1] = ro_host[b] + UInt32(seq_lens[b])
    var cl_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    ctx.enqueue_copy(cl_dev, cl_host)
    var ro_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    ctx.enqueue_copy(ro_dev, ro_host)

    # Value rows, then their scale rows, both compacted the same way.
    var idx_host = ctx.enqueue_create_host_buffer[DType.int32](total * stride)
    var len_host = ctx.enqueue_create_host_buffer[DType.int32](total)
    for b in range(batch_size):
        for s in range(seq_lens[b]):
            var t = Int(ro_host[b]) + s
            var owned = _bf16_owned_selection[sel_dcp, sel_interleave](
                sel_rank if sel_rank >= 0 else rank, t, cache_len
            )
            len_host[t] = Int32(len(owned))
            for i in range(B_TOPK):
                var value_row = Int32(-1)
                var scale_row = Int32(-1)
                if i < len(owned):
                    var slot = layout.local_slot(owned[i])
                    value_row = Int32(b * pages * PAGE_SIZE + slot)
                    scale_row = Int32(
                        _nvfp4_scale_row(scales_lut_out, pages, b, slot)
                    )
                idx_host[t * stride + i] = value_row
                idx_host[t * stride + B_TOPK + i] = scale_row
    var idx_dev = ctx.enqueue_create_buffer[DType.int32](total * stride)
    ctx.enqueue_copy(idx_dev, idx_host)
    var len_dev = ctx.enqueue_create_buffer[DType.int32](total)
    ctx.enqueue_copy(len_dev, len_host)

    var q_n = total * heads * B_Q_DEPTH
    if len(q_in) != q_n:
        raise Error("q_in has the wrong size")
    var q_host = ctx.enqueue_create_host_buffer[bf16](q_n)
    for i in range(q_n):
        q_host[i] = q_in[i]
    var q_dev = ctx.enqueue_create_buffer[bf16](q_n)
    ctx.enqueue_copy(q_dev, q_host)
    var kv_n = total * Q_DEPTH
    var kv_host = ctx.enqueue_create_host_buffer[bf16](kv_n)
    for i in range(kv_n):
        kv_host[i] = Scalar[bf16](Float32((i % 61) - 30) * 0.03)
    var kv_dev = ctx.enqueue_create_buffer[bf16](kv_n)
    ctx.enqueue_copy(kv_dev, kv_host)
    var wuk_n = heads * V_DEPTH * B_NOPE
    var wuk_host = ctx.enqueue_create_host_buffer[bf16](wuk_n)
    for i in range(wuk_n):
        wuk_host[i] = _bf16_w_uk(i)
    var wuk_dev = ctx.enqueue_create_buffer[bf16](wuk_n)
    ctx.enqueue_copy(wuk_dev, wuk_host)
    var wuv_n = heads * B_V * V_DEPTH
    var wuv_host = ctx.enqueue_create_host_buffer[bf16](wuv_n)
    for i in range(wuv_n):
        wuv_host[i] = _bf16_w_uv(i)
    var wuv_dev = ctx.enqueue_create_buffer[bf16](wuv_n)
    ctx.enqueue_copy(wuv_dev, wuv_host)
    var freqs_n = num_keys * B_ROPE
    var freqs_host = ctx.enqueue_create_host_buffer[DType.float32](freqs_n)
    for i in range(freqs_n):
        freqs_host[i] = Float32(0.5) if i % 2 == 0 else Float32(0.25)
    var freqs_dev = ctx.enqueue_create_buffer[DType.float32](freqs_n)
    ctx.enqueue_copy(freqs_dev, freqs_host)
    var gamma_dev = ctx.enqueue_create_buffer[bf16](V_DEPTH)
    ctx.enqueue_memset(gamma_dev, Scalar[bf16](1.0))
    var out_n = total * heads * B_V
    var out_dev = ctx.enqueue_create_buffer[bf16](out_n)
    ctx.enqueue_memset(out_dev, Scalar[bf16](7.0))
    var lse_dev = ctx.enqueue_create_buffer[DType.float32](total * heads)
    ctx.enqueue_memset(lse_dev, Float32(123.0))
    ctx.synchronize()

    comptime Collection = PagedKVCacheCollection[
        DType.uint8,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=fp8,
        quantization_granularity_=NV_QUANT_GRANULARITY,
    ]
    var values_tt = TileTensor(
        values_dev,
        row_major(
            Int64(n_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[1],
            Idx[NV_PACKED_ROW],
        ),
    )
    var scales_tt = TileTensor(
        scales_dev,
        row_major(
            Int64(n_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[1],
            Idx[NV_SF_PER_ROW],
        ),
    )
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](values_tt.as_unsafe_any_origin()),
        TileTensor(cl_dev, row_major(batch_size))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut_dev, row_major(batch_size, pages))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(q_max),
        UInt32(cache_len),
        rebind[Collection.scales_tt_type](scales_tt.as_unsafe_any_origin()),
        TileTensor(slut_dev, row_major(batch_size, pages))
        .as_imm()
        .as_unsafe_any_origin(),
    )
    var kv_in_tt = TileTensor(
        kv_dev.unsafe_ptr(), row_major((total, Idx[Q_DEPTH]))
    )

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[bf16, width]:
        return kv_in_tt.load[width=width](Coord(coords))

    var mla_args = MLADispatchScalarArgs[num_heads=heads, is_fp8_kv=True](
        batch_size, cache_len, q_max, ctx
    )
    mla_decode_branch_bf16[
        mask_str="causal",
        sparse_mla=True,
        skip_masked_tiles=True,
        dcp_size=dcp_size,
        dcp_interleave=dcp_interleave,
        kernels=SM100MLAGraph,
    ](
        TileTensor(
            out_dev.unsafe_ptr(),
            row_major((total, Idx[heads], Idx[B_V])),
        ),
        TileTensor(
            q_dev.unsafe_ptr(),
            row_major((total, Idx[heads], Idx[B_Q_DEPTH])),
        ),
        TileTensor(ro_dev.unsafe_ptr(), row_major(batch_size + 1)),
        TileTensor(freqs_dev.unsafe_ptr(), row_major((num_keys, Idx[B_ROPE]))),
        TileTensor(gamma_dev.unsafe_ptr(), row_major(Idx[V_DEPTH])),
        kv_collection,
        UInt32(0),
        SCALE,
        Float32(1e-6),
        TileTensor(
            wuk_dev.unsafe_ptr(),
            row_major((Idx[heads], Idx[V_DEPTH], Idx[B_NOPE])),
        ),
        TileTensor(
            wuv_dev.unsafe_ptr(),
            row_major((Idx[heads], Idx[B_V], Idx[V_DEPTH])),
        ),
        mla_args.gpu_tile_tensor(),
        kv_input_fn,
        ctx,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](idx_dev.unsafe_ptr()),
        indices_stride=stride,
        topk_lengths=NonNullPointer[DType.int32](
            len_dev.unsafe_ptr().as_imm().as_unsafe_any_origin()
        ),
        lse_out=NonNullPointer[DType.float32](lse_dev),
        split_len_bound=B_TOPK,
        dcp_rank=rank,
    )
    ctx.enqueue_copy(values_host, values_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    var out_host = ctx.enqueue_create_host_buffer[bf16](out_n)
    ctx.enqueue_copy(out_host, out_dev)
    var lse_host = ctx.enqueue_create_host_buffer[DType.float32](total * heads)
    ctx.enqueue_copy(lse_host, lse_dev)
    ctx.synchronize()

    values_out = List[UInt8](length=value_elems, fill=0)
    for i in range(value_elems):
        values_out[i] = values_host[i]
    scales_out = List[UInt8](length=scale_elems, fill=0)
    for i in range(scale_elems):
        scales_out[i] = bitcast[DType.uint8, 1](SIMD[fp8, 1](scales_host[i]))[0]
    out_out = List[Float32](length=out_n, fill=0)
    for i in range(out_n):
        out_out[i] = out_host[i].cast[DType.float32]()
    lse_out = List[Float32](length=total * heads, fill=0)
    for i in range(total * heads):
        lse_out[i] = lse_host[i]

    _ = values_dev^
    _ = scales_dev^
    _ = lut_dev^
    _ = slut_dev^
    _ = cl_dev^
    _ = ro_dev^
    _ = idx_dev^
    _ = len_dev^
    _ = q_dev^
    _ = kv_dev^
    _ = wuk_dev^
    _ = wuv_dev^
    _ = freqs_dev^
    _ = gamma_dev^
    _ = out_dev^
    _ = lse_dev^
    _ = mla_args^


def run_nvfp4_cp_store[
    dcp_size: Int, dcp_interleave: Int
](ctx: DeviceContext, seq_lens: List[Int], cache_len: Int) raises:
    """Checks the CP latent store of the NVFP4 branch: values and scales.

    Each rank must store exactly the new tokens it owns: the packed row at
    `local_slot(pos)` with the bytes a group of one stores at `pos`, and its
    32 scales at the local slot's scale-pool row with the scales a group of
    one stores at `pos`'s. Every other value and scale byte keeps the image
    the rank started from.
    """
    print(
        t"nvfp4_cp_store: dcp={dcp_size} interleave={dcp_interleave}"
        t" cache_len={cache_len}"
    )
    var total = 0
    var q_max = 0
    for s in seq_lens:
        total += s
        q_max = max(q_max, s)
    var q = _bf16_plain_q(total)
    var ref_values = List[UInt8]()
    var ref_scales = List[UInt8]()
    var slut = List[Int]()
    var out = List[Float32]()
    var lse = List[Float32]()
    _run_nvfp4_branch[1, 1](
        ctx, seq_lens, cache_len, 0, q, ref_values, ref_scales, slut, out, lse
    )
    var pages = ceildiv(cache_len + q_max, PAGE_SIZE)
    for r in range(dcp_size):
        var got_values = List[UInt8]()
        var got_scales = List[UInt8]()
        _run_nvfp4_branch[dcp_size, dcp_interleave](
            ctx,
            seq_lens,
            cache_len,
            r,
            q,
            got_values,
            got_scales,
            slut,
            out,
            lse,
        )
        var layout = ContextParallelLayout[dcp_size, dcp_interleave](r)
        # The image the rank started from: the initial pattern, with its
        # owned cached positions moved to their local slots.
        var want_values = List[UInt8](length=len(ref_values), fill=0)
        for i in range(len(want_values)):
            want_values[i] = _nvfp4_value_byte(i)
        var want_scales = List[UInt8](length=len(ref_scales), fill=0)
        for i in range(len(want_scales)):
            want_scales[i] = _nvfp4_scale_byte(i)
        for b in range(len(seq_lens)):
            for pos in range(cache_len):
                if not layout.owns(pos):
                    continue
                var slot = layout.local_slot(pos)
                var src = b * pages * PAGE_SIZE + pos
                var dst = b * pages * PAGE_SIZE + slot
                for d in range(NV_PACKED_ROW):
                    want_values[dst * NV_PACKED_ROW + d] = _nvfp4_value_byte(
                        src * NV_PACKED_ROW + d
                    )
                var src_s = _nvfp4_scale_row(slut, pages, b, pos)
                var dst_s = _nvfp4_scale_row(slut, pages, b, slot)
                for g in range(NV_SF_PER_ROW):
                    want_scales[dst_s * NV_SF_PER_ROW + g] = _nvfp4_scale_byte(
                        src_s * NV_SF_PER_ROW + g
                    )
        # Then the new tokens it owns, as the group of one stored them.
        var stored = 0
        for b in range(len(seq_lens)):
            for s in range(seq_lens[b]):
                var pos = cache_len + s
                if not layout.owns(pos):
                    continue
                var slot = layout.local_slot(pos)
                var src = b * pages * PAGE_SIZE + pos
                var dst = b * pages * PAGE_SIZE + slot
                var changed = False
                for d in range(NV_PACKED_ROW):
                    var v = ref_values[src * NV_PACKED_ROW + d]
                    want_values[dst * NV_PACKED_ROW + d] = v
                    changed = changed or v != _nvfp4_value_byte(
                        src * NV_PACKED_ROW + d
                    )
                var src_s = _nvfp4_scale_row(slut, pages, b, pos)
                var dst_s = _nvfp4_scale_row(slut, pages, b, slot)
                var scales_changed = False
                for g in range(NV_SF_PER_ROW):
                    var v = ref_scales[src_s * NV_SF_PER_ROW + g]
                    want_scales[dst_s * NV_SF_PER_ROW + g] = v
                    scales_changed = scales_changed or v != _nvfp4_scale_byte(
                        src_s * NV_SF_PER_ROW + g
                    )
                assert_true(changed, "the plain store left a new row as is")
                assert_true(
                    scales_changed, "the plain store left a new row's scales"
                )
                stored += 1
        var bad_values = 0
        for i in range(len(want_values)):
            if got_values[i] != want_values[i]:
                bad_values += 1
        var bad_scales = 0
        for i in range(len(want_scales)):
            if got_scales[i] != want_scales[i]:
                bad_scales += 1
        print(
            t"  rank={r} stored={stored} mismatched_value_bytes={bad_values}"
            t" mismatched_scale_bytes={bad_scales}"
        )
        assert_true(
            bad_values == 0, "the CP store's values differ from the expected"
        )
        assert_true(
            bad_scales == 0, "the CP store's scales differ from the expected"
        )


def run_nvfp4_cp_owned_rows[
    dcp_size: Int, dcp_interleave: Int, heads: Int = B_HEADS
](ctx: DeviceContext, seq_lens: List[Int], cache_len: Int) raises:
    """Checks the NVFP4 `.cp` branch on compacted owned value and scale rows.

    Each rank, reading its share at local slots through the value and scale
    halves of its index rows, must return exactly, bit for bit, what a group
    of one returns for the same selected positions at their own slots. As in
    `run_bf16_cp_owned_rows`, a token with no rows (token 0) must return LSE
    `-inf` and a zero output, every other a finite LSE, and the ranks' base-2
    LSE merge must match a group of one over the whole selection. That merge
    is exact in the LSE; the NVFP4 kernel multiplies FP8 probabilities by the
    FP8 values, so each rank's output rounds its own normalized probabilities
    and the merged output matches to FP8 precision, within 5% of the output's
    largest magnitude.
    """
    print(
        t"nvfp4_cp_owned_rows: dcp={dcp_size} interleave={dcp_interleave}"
        t" heads={heads}"
        t" cache_len={cache_len}"
    )
    var total = 0
    for s in seq_lens:
        total += s
    var q = _bf16_plain_q[heads](total)
    var values = List[UInt8]()
    var scales = List[UInt8]()
    var slut = List[Int]()
    var ref_out = List[Float32]()
    var ref_lse = List[Float32]()
    _run_nvfp4_branch[1, 1, heads=heads](
        ctx, seq_lens, cache_len, 0, q, values, scales, slut, ref_out, ref_lse
    )
    var n = total * heads
    var merged_m = List[Float64](length=n, fill=min_or_neg_inf[DType.float64]())
    var merged_w = List[Float64](length=n, fill=0)
    var merged_o = List[Float64](length=n * B_V, fill=0)
    var empty_rows = 0
    var short_rows = 0
    var max_abs = Float64(0)
    for r in range(dcp_size):
        var out = List[Float32]()
        var lse = List[Float32]()
        _run_nvfp4_branch[dcp_size, dcp_interleave, heads=heads](
            ctx, seq_lens, cache_len, r, q, values, scales, slut, out, lse
        )
        # The same selected positions, read by a group of one at their own
        # slots: the rank must match it exactly.
        var one_out = List[Float32]()
        var one_lse = List[Float32]()
        _run_nvfp4_branch[
            1,
            1,
            sel_dcp=dcp_size,
            sel_interleave=dcp_interleave,
            heads=heads,
        ](
            ctx,
            seq_lens,
            cache_len,
            0,
            q,
            values,
            scales,
            slut,
            one_out,
            one_lse,
            sel_rank=r,
        )
        var bad = 0
        for i in range(n):
            var same = (
                lse[i] == one_lse[i] or isinf(lse[i]) and isinf(one_lse[i])
            )
            if not same:
                bad += 1
            for d in range(B_V):
                if out[i * B_V + d] != one_out[i * B_V + d]:
                    bad += 1
        assert_true(
            bad == 0,
            String(t"rank {r} differs from a group of one in {bad} values"),
        )
        for t in range(total):
            var length = len(
                _bf16_owned_selection[dcp_size, dcp_interleave](r, t, cache_len)
            )
            if length == 0:
                empty_rows += 1
            elif length < B_TOPK:
                short_rows += 1
            for h in range(heads):
                var i = t * heads + h
                var v = Float64(lse[i])
                if length == 0:
                    assert_true(
                        isinf(v) and v < 0, String(t"empty token LSE {v}")
                    )
                    for d in range(B_V):
                        assert_true(
                            out[i * B_V + d] == 0, "empty token O is not zero"
                        )
                    continue
                assert_true(
                    not isnan(v) and not isinf(v), "token LSE not finite"
                )
                var m_new = max(merged_m[i], v)
                var c_old = exp2(merged_m[i] - m_new)
                var c_new = exp2(v - m_new)
                merged_w[i] = merged_w[i] * c_old + c_new
                for d in range(B_V):
                    var o_old = merged_o[i * B_V + d]
                    merged_o[i * B_V + d] = o_old * c_old + c_new * Float64(
                        out[i * B_V + d]
                    )
                merged_m[i] = m_new
    assert_true(empty_rows >= dcp_size, "no rank had an empty token")
    assert_true(short_rows > 0, "no rank had a partial selection")

    for v in ref_out:
        max_abs = max(max_abs, abs(Float64(v)))
    assert_true(max_abs > 0, "the reference output is all zero")
    var max_out_err = Float64(0)
    var max_lse_err = Float64(0)
    for t in range(total):
        for h in range(heads):
            var i = t * heads + h
            if t == 0:
                assert_true(
                    isinf(ref_lse[i]) and ref_lse[i] < 0, "reference LSE"
                )
                assert_true(merged_w[i] == 0, "a rank had rows for token 0")
                continue
            var lse = merged_m[i] + log2(merged_w[i])
            max_lse_err = max(max_lse_err, abs(lse - Float64(ref_lse[i])))
            assert_almost_equal(lse, Float64(ref_lse[i]), atol=1e-3, rtol=1e-4)
            for d in range(B_V):
                var o = merged_o[i * B_V + d] / merged_w[i]
                var want = Float64(ref_out[i * B_V + d])
                max_out_err = max(max_out_err, abs(o - want))
                assert_almost_equal(o, want, atol=0.05 * max_abs, rtol=0)
    print(
        t"  empty_rows={empty_rows} short_rows={short_rows} max_out={max_abs}"
        t" max_out_err={max_out_err} max_lse_err={max_lse_err}"
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(7)
            # Small top-k: decode rows, then ragged 1-3 token prefill rows.
            run_cp_case[DType.float8_e4m3fn, DType.float8_e4m3fn](
                "fp8_k64_decode", ctx, [1, 1, 1], cache_len=200, topk=64
            )
            run_cp_case[DType.float8_e4m3fn, DType.float8_e4m3fn](
                "fp8_k64_ragged", ctx, [1, 3, 2], cache_len=300, topk=64
            )
            run_cp_case[DType.bfloat16, DType.bfloat16](
                "bf16_k64_ragged", ctx, [2, 1, 3], cache_len=130, topk=64
            )
            run_cp_case[DType.bfloat16, DType.float8_e4m3fn](
                "bf16q_fp8kv_k64_ragged", ctx, [3, 1], cache_len=257, topk=64
            )
            # GLM's top-k.
            run_cp_case[DType.float8_e4m3fn, DType.float8_e4m3fn](
                "fp8_k2048_ragged", ctx, [1, 2, 3], cache_len=2100, topk=2048
            )
            run_cp_case[DType.bfloat16, DType.bfloat16](
                "bf16_k2048_ragged", ctx, [2, 1], cache_len=2300, topk=2048
            )
            # The fused branch: CP store + per-token lengths + LSE.
            run_fused_cp_store[4, 1](ctx, [1, 3, 2], cache_len=300)
            run_fused_cp_store[2, PAGE_SIZE](ctx, [2, 3], cache_len=250)
            # The BF16 branch: CP store, latent output and absorbed query.
            run_bf16_cp_store[2, 1](ctx, [1, 3, 2], cache_len=101)
            run_bf16_cp_store[4, PAGE_SIZE](ctx, [2, 3], cache_len=250)
            run_bf16_branch_modes(ctx, [1, 3, 2], cache_len=101)
            # Compacted owned rows, short lengths and an empty token.
            run_bf16_cp_owned_rows[4, 1](ctx, [1, 3, 2], cache_len=101)
            run_bf16_cp_owned_rows[2, PAGE_SIZE](ctx, [2, 3], cache_len=250)
            # The NVFP4 branch: values and scales stored at the local slot,
            # and compacted owned value/scale rows merged across ranks.
            run_nvfp4_cp_store[2, 1](ctx, [1, 3, 2], cache_len=101)
            run_nvfp4_cp_store[8, 1](ctx, [2, 3, 1], cache_len=300)
            run_nvfp4_cp_store[4, PAGE_SIZE](ctx, [2, 3], cache_len=250)
            run_nvfp4_cp_owned_rows[4, 1](ctx, [1, 3, 2], cache_len=101)
            run_nvfp4_cp_owned_rows[8, 1](ctx, [2, 3, 1], cache_len=300)
            run_nvfp4_cp_owned_rows[2, PAGE_SIZE](ctx, [2, 3], cache_len=250)
            # A DCP group smaller than TP attends its share of the heads:
            # TP8 x DCP4 runs 32 of GLM's 64 per device, TP8 x DCP2 16.
            run_bf16_cp_owned_rows[4, 1, heads=32](
                ctx, [1, 3, 2], cache_len=101
            )
            run_bf16_cp_owned_rows[2, PAGE_SIZE, heads=16](
                ctx, [2, 3], cache_len=250
            )
            run_nvfp4_cp_owned_rows[4, 1, heads=32](
                ctx, [2, 3, 1], cache_len=300
            )
            run_nvfp4_cp_owned_rows[2, PAGE_SIZE, heads=16](
                ctx, [2, 3], cache_len=250
            )
