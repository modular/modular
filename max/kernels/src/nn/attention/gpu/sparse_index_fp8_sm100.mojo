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
"""SM100 (B200) tensor-core FP8 MLA lightning-indexer score kernel.

Computes the same per-(query token, key) logit as the scalar
`nn.index_fp8.fp8_index_kernel`, but runs the depth-128 dot product on the
tcgen05 tensor cores instead of a serial FMA loop:

    score[token, key] = k_scale[key]
                        * Σ_head relu(q[token, head] · k[key]) * q_scale[token, head]

Q is `[total_seq, num_heads, depth]` fp8-e4m3, K is paged `[keys, depth]` fp8-e4m3
with a per-token `k_scale`, and the head reduction is a **sum** over `num_heads`.

Layout (crux), cloning the shipped MSA prefill scorer
(`Kernels/lib/msa/sparse_indexer_prefill.mojo`) with the operand roles inverted:

- **MMA_M = key tile** (`BM_key`): A operand = this CTA's K tile `[BM_key, depth]`,
  loaded once and reused across the MTP query tokens.
- **MMA_N = 128 = (query-token × head)**: B operand = a pair of query tokens'
  `[N_TOKENS * num_heads, depth]`, `transpose_b=True` -> `S^T = K @ Q^T =
  [key, (token, head)]`. This is DeepGEMM's `sm100_mqa_logits` packing
  (`BLOCK_Q = 128 / num_heads`, so `N_TOKENS = 2` at `num_heads = 64`).
- **MMA_K = depth = 128** contraction, fp8 in / f32 TMEM accumulation.

Each lane owns a set of `(key_row, (token, head)_col)` fragment elements
(identical ownership to the MSA epilogue at `BM = 64`). The epilogue applies the
branchless relu `(x + |x|) * 0.5`, multiplies by `q_scale[token, head]`, **sums**
over the head columns of each token (columns of the same token in the N-tile),
cross-lane reduces over the 4 lanes sharing a key row (`lane_group_sum`, changed
from the MSA's `max`), multiplies by `k_scale[key]`, and writes one f32 per
(token, key).

Grid `(batch, ceil(num_keys / BM_key), seq_slices)`: key tiles and token tiles
are independent outputs, so there is no split-K and no cross-CTA reduction.
`BM_key = 64` (256 CTAs at batch=8, num_keys=2048) intentionally oversubscribes
the grid for the launch-starved decode regime and reuses the MSA `BM = 64`
fragment map verbatim. grid.z splits a sequence's token tiles across CTAs when
the key-tile grid alone underfills the machine (low-key prefill); decode always
launches one slice.

Token tiles are software-pipelined: Q and its scales are double-buffered so
tile nt+1's TMA and q_scale loads fly under tile nt's MMA and epilogue. The
TMEM accumulator stays single-stage (the drain is a TMEM->register copy that
precedes the epilogue math, so the next MMA already overlaps the math; a
second stage measured as a pure loss by halving TMEM-limited CTAs/SM on large
grids). Decode launches allocate only the SMEM prefix (Q buffer 1 is last in
the layout and unreachable at a single token tile), keeping decode occupancy
unchanged.

Prefill / causal masking and the `-inf` tail-fill fusion are Slice 2/3 (not here);
this kernel is a drop-in for the score buffer the top-k stage consumes.

NVIDIA SM100 only (SS-UMMA / TMA / tcgen05). Verified against
`nn.index_fp8.fp8_index_naive` via `test_index_fp8` and end-to-end top-k set
match via `test_mla_index_fp8`.
"""

from std.gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.memory import AddressSpace, external_memory
from std.gpu.sync import named_barrier
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from std.math import align_up, ceildiv
from std.sys import has_nvidia_gpu_accelerator, size_of
from std.utils.index import Index
import std.gpu.primitives.warp as warp

from layout import TensorLayout, TileTensor, UNKNOWN_VALUE
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import (
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    create_split_tma,
)

from nn.attention.gpu.nvidia.sm100.mha_1q import SM100TensorAccumulatorSS
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.sparse_index_fp8_sm100_prefill import (
    _PREFILL_MIN_TOKEN_TILES,
    fp8_index_score_sm100_prefill,
)


comptime _INDEX_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B

# Baked into the epilogue fragment map (rows w*16 + i*8 + l//4); changing it
# means rewriting the epilogue.
comptime _BM_KEY = 64

# Token-block count above which nh=32 pure-prefill routes to the K-streaming
# prefill kernel (see the prefill route in `fp8_index_score_sm100`). Much higher
# than nh=64's `_PREFILL_MIN_TOKEN_TILES` (16): measured B200 causal cache=0
# crossover reaches a safe >=12% win only at 448 tiles (seq ~1792).
comptime _PREFILL_MIN_TOKEN_TILES_NH32 = 448

# Q buffer 1 sits at the END of the SMEM layout so a decode launch (every
# batch entry a single token tile, so buffer 1 is never touched) can allocate
# only the prefix and keep the un-pipelined kernel's CTAs/SM. 128B alignment
# (not the 1KB swizzle atom) suffices: MMASmemDescriptor.create preserves the
# full byte address, so the TMA write and the MMA read share the swizzle
# phase off the same base.
comptime _Q1SmemOffset[
    dtype: DType, BM_key: Int, MMA_N: Int, depth: Int
] = align_up(
    (BM_key + MMA_N) * depth * size_of[Scalar[dtype]]()
    + (2 * MMA_N + BM_key) * size_of[Float32]()
    + 6 * size_of[SharedMemBarrier](),
    128,
)


comptime QTMATileT[
    dtype: DType, MMA_N: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(MMA_N, 1, depth), _INDEX_SWIZZLE]
comptime KTMATileT[
    dtype: DType, BM_key: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(BM_key, 1, depth), _INDEX_SWIZZLE]


@always_inline
def _fp8_index_body[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[DType.uint32, VLLT, ImmutAnyOrigin],
    q_s: TileTensor[DType.float32, QSLT, ImmutAnyOrigin],
    output: TileTensor[DType.float32, OutLT, MutAnyOrigin],
    max_num_keys: Int,
    causal: Int,
    nt_start: Int,
    n_local: Int,
):
    comptime assert valid_length.flat_rank == 1
    comptime MMA_N = N_TOKENS * num_heads
    comptime assert (
        MMA_N == 128
    ), "MMA_N must pack to 128 (N_TOKENS * num_heads); got " + String(MMA_N)
    comptime AT = DType.float32
    comptime SW = _INDEX_SWIZZLE
    comptime NTHREADS = 128

    var tid = thread_idx.x
    var b = block_idx.x
    var key_start = Int(block_idx.y) * BM_key

    var start_of_seq = Int(valid_length[b])
    var end_of_seq = Int(valid_length[b + 1])
    var seq_len = end_of_seq - start_of_seq

    var num_keys = Int(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len

    comptime UMMAType = SM100TensorAccumulatorSS[
        dtype,
        AT,
        MMA_M=BM_key,
        MMA_N=MMA_N,
        BM=BM_key,
        BN=MMA_N,
        BK=depth,
        compute_BK=align_up(depth, 16),
        num_softmax_threads=NTHREADS,
        swizzle_a=SW,
        swizzle_b=SW,
        transpose_b=True,
        pipeline_stages=1,
    ]

    comptime k_elems = BM_key * depth
    comptime q_elems = MMA_N * depth
    var smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
        name="fp8_index_sm100_smem",
    ]()
    var k_smem = smem
    var q_smem = smem + k_elems
    var qs_smem = (smem + k_elems + q_elems).bitcast[Float32]()
    var ks_smem = qs_smem + 2 * MMA_N
    # mbar[0] = K staging-done (one-shot); mbar[1..2] = Q staging-done, one per
    # Q double-buffer; mbar[3..4] = accumulator handshake (2 for
    # pipeline_stages=1); mbar[5] = tcgen05 TMEM base slot.
    var mbar = (ks_smem + BM_key).bitcast[SharedMemBarrier]()
    var k_mbar = mbar
    var q_mbar = mbar + 1
    var acc_mbar = mbar + 3
    var ptr_tmem = (mbar + 5).bitcast[UInt32]()
    var q1_smem = smem + _Q1SmemOffset[dtype, BM_key, MMA_N, depth]

    comptime k_flat_layout = tt_row_major[k_elems]()
    comptime q_flat_layout = tt_row_major[q_elems]()

    var umma_p = UMMAType(acc_mbar.as_unsafe_any_origin())
    var umma_c = UMMAType(acc_mbar.as_unsafe_any_origin())

    if tid == 0:
        k_mbar[0].init()
        q_mbar[0].init()
        q_mbar[1].init()
        umma_p.init()
    named_barrier[Int32(NTHREADS)]()

    # tcgen05 alloc is warp-collective (.sync.aligned): exactly one warp.
    # Release the lock right after so other CTAs sharing the SM can allocate.
    comptime TMEM_COLS = UInt32(align_up(MMA_N, 32))
    if warp_id() == 0:
        tcgen05_alloc[1](ptr_tmem, TMEM_COLS)
        tcgen05_release_allocation_lock[1]()
    named_barrier[Int32(NTHREADS)]()
    var tmem_addr: UInt32 = ptr_tmem[0]

    # --- Stage the K tile once (A operand). A full BM_key-row tile is always
    # staged; rows past this batch's num_keys hold unrelated pool data (the
    # paged descriptor's bound is the whole pool, not num_keys, so it does not
    # zero them). Correctness comes from the epilogue's `key_local < num_keys`
    # guard, which drops those rows' scores before any write to `output`.
    comptime k_bytes = k_elems * size_of[dtype]()
    comptime q_bytes = q_elems * size_of[dtype]()
    if tid == 0:
        var k_row0 = Int(k_operand.row_idx(UInt32(b), UInt32(key_start)))
        var k_dst = TileTensor[
            dtype, type_of(k_flat_layout), address_space=AddressSpace.SHARED
        ](k_smem, k_flat_layout)
        k_mbar[0].expect_bytes(Int32(k_bytes))
        k_tma.async_copy_3d(k_dst, k_mbar[0], (0, 0, k_row0))
        # The first token tile's Q TMA is independent of K, so it rides
        # alongside the K TMA instead of stalling behind the k_mbar wait. This
        # prologue arm is the sole producer of q_mbar[0]'s first completion;
        # the loop only ever issues tile nt+1 into the other buffer.
        var q_dst = TileTensor[
            dtype, type_of(q_flat_layout), address_space=AddressSpace.SHARED
        ](q_smem, q_flat_layout)
        q_mbar[0].expect_bytes(Int32(q_bytes))
        q_tma.async_copy_3d(
            q_dst,
            q_mbar[0],
            (0, 0, (start_of_seq + nt_start * N_TOKENS) * num_heads),
        )

    # k_scale depends only on this CTA's resident K rows, so stage all BM_key
    # scales once while the TMAs are in flight; the epilogue would otherwise
    # re-load them from global memory in every token tile's critical path.
    if Int(tid) < BM_key:
        var key_local = key_start + Int(tid)
        if key_local < num_keys:
            var ks_ptr = ks_operand.block_paged_ptr[1](
                UInt32(b), UInt32(key_local), UInt32(0), UInt32(0)
            )
            ks_smem[tid] = ks_ptr[0].cast[DType.float32]()
        else:
            ks_smem[tid] = 0.0

    # Tile 0's q_scale staging: the loop below only stages tile nt+1 during
    # iteration nt, so the first tile's buffer must be filled here (published
    # by the pre-loop named_barrier).
    if nt_start * N_TOKENS + Int(tid) // num_heads < seq_len:
        qs_smem[tid] = q_s[
            start_of_seq + nt_start * N_TOKENS + Int(tid) // num_heads,
            Int(tid) % num_heads,
        ][0]
    else:
        qs_smem[tid] = 0.0
    k_mbar[0].wait(0)

    var w = Int(warp_id())
    var l = Int(lane_id())
    comptime frag_simdwidth = 2

    umma_c.tmem_arrive_init()
    named_barrier[Int32(NTHREADS)]()

    # Software pipeline over token tiles: everything tile nt+1 needs that does
    # not depend on tile nt's results (its Q TMA into the other Q buffer, the
    # q_scale global loads) is issued before tile nt's MMA, so it is in flight
    # underneath the MMA wait and the epilogue. The q_scale SMEM store is
    # deferred to after the epilogue (load-early / store-late) so the thread
    # never stalls on the dependent store. The MMA <-> epilogue ordering rides
    # entirely on the accumulator mbars; the one named_barrier per iteration
    # publishes the cross-thread q_scale staging and keeps its buffers exactly
    # one tile deep.
    for it in range(n_local):
        var tok0 = (nt_start + it) * N_TOKENS
        var q_buf = it & 1
        var q_next = 1 - q_buf
        var has_next = it + 1 < n_local

        if has_next and tid == 0:
            var q_row0 = (start_of_seq + tok0 + N_TOKENS) * num_heads
            var q_dst = TileTensor[
                dtype, type_of(q_flat_layout), address_space=AddressSpace.SHARED
            ](q1_smem if q_next == 1 else q_smem, q_flat_layout)
            q_mbar[q_next].expect_bytes(Int32(q_bytes))
            q_tma.async_copy_3d(q_dst, q_mbar[q_next], (0, 0, q_row0))

        var qs_tok = Int(tid) // num_heads
        var qs_head = Int(tid) % num_heads
        var qs_next: Float32 = 0.0
        if has_next:
            var qs_local = tok0 + N_TOKENS + qs_tok
            if qs_local < seq_len:
                qs_next = q_s[start_of_seq + qs_local, qs_head][0]

        # Buffer q_buf completes once per round trip of the ring: local
        # iteration it is its (it // 2)-th completion, hence the wait parity.
        q_mbar[q_buf].wait(UInt32((it >> 1) & 1))

        var qk_desc = UMMAType.mma_descriptors(
            k_smem.as_unsafe_any_origin(),
            (q1_smem if q_buf == 1 else q_smem).as_unsafe_any_origin(),
        )
        var s_acc_p = UMMAType.c_t(tmem_addr)
        var s_acc_c = UMMAType.c_t(tmem_addr)
        if tid == 0:
            umma_p.wait_for_tmem()
            umma_p.mma(
                rebind[UMMAType.a_t](qk_desc.get_a()),
                rebind[UMMAType.b_t](qk_desc.get_b()),
                s_acc_p,
                0,
            )

        var c = umma_c.wait_for_mma(s_acc_c)
        var reg = UMMAType.c_t.allocate_register_tile()
        c.copy_to(reg)
        umma_c.tmem_arrive()

        comptime for i in range(2):
            var row = w * 16 + i * 8 + l // 4
            var key_local = key_start + row

            # Accumulate per j-group with comptime indices only: one lane's two
            # fragment elements of a j-group always share a token, and a
            # runtime token index into the SIMD accumulator spills it to local
            # memory (measured 18x slower at num_heads == 4 / N_TOKENS == 32).
            # The reduction below stays straight-line (guards only at the
            # store): a per-token skip branch measured ~25% slower on prefill
            # across all head counts by breaking the shuffle-chain pipelining.
            var jgroup_sum = SIMD[AT, MMA_N // 8](0)
            comptime for j in range(MMA_N // 8):
                var v = rebind[SIMD[AT, frag_simdwidth]](reg[i, 0, j, 0])
                comptime for cc in range(frag_simdwidth):
                    var col = (l % 4) * 2 + j * 8 + cc
                    var raw = v[cc]
                    # branchless relu
                    var s = (
                        (raw + abs(raw)) * 0.5 * qs_smem[q_buf * MMA_N + col][0]
                    )
                    jgroup_sum[j] += s

            comptime for t in range(N_TOKENS):
                var tok_sum = Scalar[AT](0)
                comptime if num_heads >= 8:
                    # An 8-column j-group never straddles a token boundary, so
                    # token t owns j-groups [t * jpt, (t + 1) * jpt).
                    comptime jpt = num_heads // 8
                    comptime for jj in range(jpt):
                        tok_sum += jgroup_sum[t * jpt + jj]
                else:
                    # num_heads == 4: j-group t // 2 spans tokens 2j and
                    # 2j + 1; which half this lane's elements belong to
                    # depends only on the lane.
                    var own = ((l % 4) >= 2) == (t % 2 == 1)
                    tok_sum = jgroup_sum[t // 2] if own else Scalar[AT](0)
                var row_sum = warp.lane_group_sum[num_lanes=4](
                    SIMD[AT, 1](tok_sum)
                )[0]
                var tok_local = tok0 + t
                # Fused causal mask: token tok_local sees keys up to
                # cache_len + tok_local (cache_len = num_keys - seq_len), so
                # forbidden slots keep the caller's -inf fill and the separate
                # mask pass over the whole score buffer is skipped. Branchless
                # (causal is 0 or 1): a branch here in the unrolled token loop
                # measured +4-9% on the non-causal path from codegen alone.
                var key_bound = num_keys - (seq_len - 1 - tok_local) * causal
                if (
                    (l % 4) == 0
                    and row < BM_key
                    and key_local < key_bound
                    and tok_local < seq_len
                ):
                    var k_scale = ks_smem[row]
                    var global_token = start_of_seq + tok_local
                    output.raw_store(
                        global_token * max_num_keys + key_local,
                        k_scale * row_sum,
                    )

        if has_next:
            qs_smem[q_next * MMA_N + Int(tid)] = qs_next

        named_barrier[Int32(NTHREADS)]()

    if warp_id() == 0:
        tcgen05_dealloc[1](tmem_addr, TMEM_COLS)


@__name(t"fp8_index_score_sm100_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
def _fp8_index_score_kernel_sm100[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[DType.uint32, VLLT, ImmutAnyOrigin],
    q_s: TileTensor[DType.float32, QSLT, ImmutAnyOrigin],
    output: TileTensor[DType.float32, OutLT, MutAnyOrigin],
    max_num_keys: Int,
    causal: Int,
):
    var b = block_idx.x
    var key_start = Int(block_idx.y) * BM_key
    var start_of_seq = Int(valid_length[b])
    var seq_len = Int(valid_length[b + 1]) - start_of_seq
    var num_keys = Int(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len

    # Bail uniformly (every thread) before the helper's first collective op
    # (TMA mbar / tcgen05 alloc); a divergent early return would deadlock them.
    # OOB keys keep the caller's `-inf` fill.
    if key_start >= num_keys or seq_len <= 0:
        return

    # Flat launch covers every token tile of this sequence (grid.z == 1).
    _fp8_index_body[
        dtype,
        KOperand,
        KSOperand,
        VLLT,
        QSLT,
        OutLT,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
    ](
        q_tma,
        k_tma,
        k_operand,
        ks_operand,
        valid_length,
        q_s,
        output,
        max_num_keys,
        causal,
        0,
        ceildiv(seq_len, N_TOKENS),
    )


@__name(t"fp8_index_score_sm100_split_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
def _fp8_index_score_kernel_sm100_split[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[DType.uint32, VLLT, ImmutAnyOrigin],
    q_s: TileTensor[DType.float32, QSLT, ImmutAnyOrigin],
    output: TileTensor[DType.float32, OutLT, MutAnyOrigin],
    max_num_keys: Int,
    causal: Int,
):
    var b = block_idx.x
    var key_start = Int(block_idx.y) * BM_key
    var start_of_seq = Int(valid_length[b])
    var seq_len = Int(valid_length[b + 1]) - start_of_seq
    var num_keys = Int(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len

    # grid.z splits this sequence's token tiles across CTAs; bounds are
    # uniform per CTA.
    var n_token_tiles = ceildiv(seq_len, N_TOKENS)
    var tiles_per_slice = ceildiv(n_token_tiles, Int(grid_dim.z))
    var nt_start = Int(block_idx.z) * tiles_per_slice

    # Bail uniformly (every thread) before the helper's first collective op
    # (TMA mbar / tcgen05 alloc); a divergent early return would deadlock them.
    # OOB keys keep the caller's `-inf` fill.
    if key_start >= num_keys or seq_len <= 0 or nt_start >= n_token_tiles:
        return
    var n_local = min(tiles_per_slice, n_token_tiles - nt_start)

    _fp8_index_body[
        dtype,
        KOperand,
        KSOperand,
        VLLT,
        QSLT,
        OutLT,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
    ](
        q_tma,
        k_tma,
        k_operand,
        ks_operand,
        valid_length,
        q_s,
        output,
        max_num_keys,
        causal,
        nt_start,
        n_local,
    )


@always_inline
def fp8_index_score_sm100[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    num_heads: Int,
    depth: Int,
    _is_cache_length_accurate: Bool,
](
    output: TileTensor[DType.float32, ...],
    q: TileTensor[mut=False, dtype, ...],
    q_s: TileTensor[mut=False, DType.float32, ...],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[mut=False, DType.uint32, ...],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    causal: Bool,
    ctx: DeviceContext,
) raises:
    """Launch the SM100 tensor-core FP8 indexer scorer into `output`.

    NVIDIA SM100 only: uses SS-UMMA, tcgen05 TMEM, and TMA staging. Writes the
    same `[total_seq, max_num_keys]` score buffer as the scalar
    `nn.index_fp8.fp8_index_kernel`. Out-of-range keys are left untouched (the
    caller's `-inf` fill covers them).

    Parameters:
        dtype: FP8 element type of Q and K (float8_e4m3fn).
        KOperand: `MHAOperand` type for the K values.
        KSOperand: `MHAOperand` type for the per-token K scales.
        num_heads: Query index heads (must be 64 so N_TOKENS * num_heads == 128).
        depth: Head dimension (contraction, must be 128).
        _is_cache_length_accurate: When False, `num_keys = cache_length + seq_len`;
            when True, `cache_length` already includes the new tokens.

    Args:
        output: Score buffer `[total_seq, max_num_keys]`, f32.
        q: Query tensor `[total_seq, num_heads, depth]`, fp8.
        q_s: Query scales `[total_seq, num_heads]`, f32.
        k_operand: K values as an `MHAOperand`.
        ks_operand: K scales as an `MHAOperand`.
        valid_length: Ragged query-token offsets `[batch + 1]`.
        batch_size: Batch size.
        max_seq_len: Upper bound on any batch entry's query-token count; when
            it fits a single token tile, the launch allocates only the SMEM
            prefix (Q buffer 1 is unreachable).
        max_num_keys: Row stride of `output` (>= every per-batch key count).
        causal: Apply the causal mask in the epilogue store guard (token t
            sees keys up to cache_len + t); forbidden slots keep the caller's
            `-inf` fill, replacing a separate mask pass over the buffer.
        ctx: Device context.
    """
    # The N-tile packs N_TOKENS = 128 // num_heads whole query tokens, so any
    # divisor of 128 is structurally admissible; the gate lists the validated
    # counts (64 = DeepSeek V3.2 replicated; 32 = GLM 5.x replicated; 8 / 4 =
    # TP-head-sharded indexers, e.g. GLM's 32 heads over 4 or 8 ranks).
    comptime assert num_heads in (64, 32, 8, 4), (
        "SM100 FP8 indexer scorer requires num_heads in {4, 8, 32, 64} (N-tile"
        " of 128)"
    )
    comptime assert (
        depth == 128
    ), "SM100 FP8 indexer scorer requires depth == 128"
    comptime BM_key = _BM_KEY
    # The K tile is staged with ONE TMA copy of BM_key contiguous physical rows
    # at row_idx(b, key_start), which is only correct when BM_key virtual key
    # rows never straddle a page boundary: page_size must be 0 (contiguous /
    # ragged) or a multiple of BM_key. The dispatch sites (index_fp8 /
    # mla_index_fp8) route any other page_size to the scalar kernel; this is the
    # backstop so a future caller cannot reintroduce the wrong-page read.
    comptime assert (
        KOperand.page_size == 0 or KOperand.page_size % BM_key == 0
    ), (
        "SM100 FP8 indexer scorer requires K-cache page_size == 0 or a multiple"
        " of BM_key"
    )
    comptime MMA_N = 128
    comptime N_TOKENS = MMA_N // num_heads

    var total_q_rows = Int(q.dim[0]()) * num_heads
    var q_tma_tile = create_split_tma[
        Index(MMA_N, 1, depth),
        Index(UNKNOWN_VALUE, 1, depth),
        _INDEX_SWIZZLE,
    ](
        ctx,
        rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](q.ptr),
        total_q_rows,
    )
    var k_tma_tile = k_operand.create_tma_tile[
        _INDEX_SWIZZLE,
        BN=BM_key,
        depth=depth,
        BK=depth,
    ](ctx)

    # Prefill route: the warp-specialized K-streaming kernel (one CTA per token
    # block, causal-triangle trim) beats the K-resident scorer only when the
    # scorer's full seq x keys rectangle work overtakes prefill's 1-CTA/SM
    # occupancy deficit (232 regs, consumer-bound, ~2x lower issue utilization
    # than the scorer's 4-5 CTA/SM). nh=64 (N_TOKENS=2) wins broadly (measured
    # +7-44%, growing with seq) so it routes at >= _PREFILL_MIN_TOKEN_TILES (16).
    # nh=32 (N_TOKENS=4) wins only for long PURE-prefill under a causal mask:
    # measured cache=0 crossover is >=384 token tiles and a safe >=12% by 448
    # (seq ~1792); a cached prefix (max_num_keys > max_seq_len) or a NULL mask
    # regress on prefill (cache=2048 +8%, NULL +43%), so gate nh=32 on causal +
    # cache-free. nh in {4, 8} never route. (The prior "too few token blocks
    # underfill" rationale was wrong: at seq >= 1024 the nh=32 grid fills the
    # machine yet still loses -- the penalty is per-CTA occupancy, not underfill.)
    comptime if num_heads == 64 or num_heads == 32:
        comptime min_tiles = (
            _PREFILL_MIN_TOKEN_TILES if num_heads
            == 64 else _PREFILL_MIN_TOKEN_TILES_NH32
        )
        var to_prefill = ceildiv(max_seq_len, N_TOKENS) >= min_tiles
        comptime if num_heads == 32:
            to_prefill = to_prefill and causal and max_num_keys <= max_seq_len
        if to_prefill:
            fp8_index_score_sm100_prefill[
                dtype,
                KOperand,
                KSOperand,
                num_heads,
                depth,
                BM_key,
                N_TOKENS,
                _is_cache_length_accurate,
            ](
                rebind[QTMATileT[dtype, N_TOKENS * num_heads, depth]](
                    q_tma_tile
                ),
                rebind[KTMATileT[dtype, BM_key, depth]](k_tma_tile),
                k_operand,
                ks_operand,
                valid_length,
                q_s,
                output,
                batch_size,
                max_seq_len,
                max_num_keys,
                Int(causal),
                ctx,
            )
            return

    comptime kernel_flat = _fp8_index_score_kernel_sm100[
        dtype,
        KOperand,
        KSOperand,
        type_of(valid_length.as_immut()).LayoutType,
        type_of(q_s).LayoutType,
        type_of(output).LayoutType,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
    ]
    comptime kernel_split = _fp8_index_score_kernel_sm100_split[
        dtype,
        KOperand,
        KSOperand,
        type_of(valid_length.as_immut()).LayoutType,
        type_of(q_s).LayoutType,
        type_of(output).LayoutType,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
    ]

    comptime q1_offset = _Q1SmemOffset[dtype, BM_key, MMA_N, depth]
    comptime smem_bytes = q1_offset + MMA_N * depth * size_of[Scalar[dtype]]()
    var smem_bytes_rt = q1_offset if max_seq_len <= N_TOKENS else smem_bytes

    # Split each sequence's token tiles over grid.z only when the key-tile
    # grid leaves SMs idle (batch-1 prefill at 2K keys is 32 CTAs on ~148
    # SMs), targeting ~2 waves. Slices re-stage the (L2-hot) K tile and halve
    # the per-slice pipeline depth, so a grid already at one wave never
    # splits (a 256-CTA GLM MTP-decode shape measured -29% when split), and a
    # slice never gets fewer than 2 token tiles.
    comptime sm_count = ctx.default_device_info.sm_count
    var base_ctas = batch_size * ceildiv(max_num_keys, BM_key)
    var num_slices = 1
    if base_ctas < sm_count:
        num_slices = max(
            1,
            min(
                ceildiv(2 * sm_count, base_ctas),
                ceildiv(ceildiv(max_seq_len, N_TOKENS), 2),
            ),
        )

    if num_slices > 1:
        ctx.enqueue_function[kernel_split](
            rebind[QTMATileT[dtype, MMA_N, depth]](q_tma_tile),
            rebind[KTMATileT[dtype, BM_key, depth]](k_tma_tile),
            k_operand,
            ks_operand,
            valid_length.as_immut(),
            q_s,
            output,
            max_num_keys,
            Int(causal),
            grid_dim=(batch_size, ceildiv(max_num_keys, BM_key), num_slices),
            block_dim=128,
            shared_mem_bytes=smem_bytes_rt,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_bytes)
            ),
        )
    else:
        ctx.enqueue_function[kernel_flat](
            rebind[QTMATileT[dtype, MMA_N, depth]](q_tma_tile),
            rebind[KTMATileT[dtype, BM_key, depth]](k_tma_tile),
            k_operand,
            ks_operand,
            valid_length.as_immut(),
            q_s,
            output,
            max_num_keys,
            Int(causal),
            grid_dim=(batch_size, ceildiv(max_num_keys, BM_key), 1),
            block_dim=128,
            shared_mem_bytes=smem_bytes_rt,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_bytes)
            ),
        )
