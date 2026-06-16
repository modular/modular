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
"""Prefill-path MiniMax-M3 sparse-attention (MSA) indexer.

For each (ragged) query token and index head, this selects the top-k key
*blocks* to attend to. It runs as two launches:

1. `_prefill_block_score_kernel` -- SM100 (B200) tensor-core (SS-UMMA) scorer.
   One CTA scores a QTILE(=64)-row query tile of one batch against every
   causally-reachable key block. Q (one head's tile) and the key block are
   TMA-staged into 128B-swizzled SMEM and multiplied on tensor cores as
   `S = Q @ K_block^T * sm_scale` (bf16 inputs, f32 TMEM accumulation). Each of
   the 4 index heads runs its own MMA against the key block. Each query row has
   its own causal key count `prefix_len + local_index + 1`; the per-query mask is
   applied to the score fragment after the MMA (a key column >= that query's count
   is excluded from its max), so one key-block read serves every query row in the
   tile. Init/local forcing is applied before writing one f32 per (head, query,
   block) into a caller-owned score buffer.
2. `_prefill_topk_kernel` -- one CTA per (query token, index head). Selects the
   top-k blocks from the score row via `block_select_topk`.

Queries are ragged: `input_row_offsets[b]` gives the start of batch `b`'s tokens.
Selection-only (M3 disables the index value/output on every sparse layer); score
type is `max`.
"""

from std.gpu import (
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.memory import AddressSpace, external_memory
from std.gpu.primitives import warp
from std.gpu.sync import named_barrier
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from std.math import align_up, ceildiv, clamp, max, min
from std.sys import size_of
from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf

from layout import TensorLayout, TileTensor, UNKNOWN_VALUE
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import (
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    create_split_tma,
)

from nn.attention.gpu.sparse_indexer_common import block_select_topk
from nn.attention.gpu.nvidia.sm100.mha_1q import SM100TensorAccumulatorSS
from nn.attention.mha_operand import MHAOperand


@always_inline
def _token_batch(
    t: Int,
    batch: Int,
    input_row_offsets: UnsafePointer[Scalar[DType.uint32], ImmutAnyOrigin],
) -> Int:
    """Return the batch index owning ragged query token `t`."""
    var b = 0
    for bi in range(batch):
        if t < Int(input_row_offsets[bi + 1]):
            b = bi
            break
    return b


comptime _SCORE_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B

# K SMEM ring depth, shared with the launcher's SMEM sizing: prefetch this many
# K blocks ahead of the MMA/epilogue.
#
# Occupancy coupling (load-bearing): SMEM/CTA = 4-head Q (64 KB) + NSTAGE*K
# (NSTAGE*32 KB) + barriers, against ~227 KB SMEM/SM on B200. NSTAGE=1 -> ~96 KB
# -> 2 CTAs/SM resident; NSTAGE>=2 -> >=128 KB -> 1 CTA/SM. At 1 CTA/SM only the
# CTA's 4 warps run, so the per-block tcgen05 MMA `wait` + per-(block,head)
# `named_barrier` stalls have nothing to hide behind (DRAM is far from
# saturated). With 2 CTAs/SM the scheduler interleaves one CTA's compute under
# the other's wait/barrier, which more than replaces the intra-CTA K prefetch
# lost at NSTAGE=1.
comptime _K_PREFETCH_STAGES = 1

comptime QTMATileT[dtype: DType, BM: Int, BK: Int] = SplitLastDimTMATensorTile[
    dtype, Index(BM, 1, BK), _SCORE_SWIZZLE
]
comptime KTMATileT[dtype: DType, BN: Int, BK: Int] = SplitLastDimTMATensorTile[
    dtype, Index(BN, 1, BK), _SCORE_SWIZZLE
]


@__name(t"sparse_indexer_prefill_score_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
def _prefill_block_score_kernel[
    dtype: DType,
    KOperand: MHAOperand,
    IROLT: TensorLayout,
    PLLT: TensorLayout,
    ScoreLT: TensorLayout,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
    QTILE: Int,
](
    q_tma: QTMATileT[dtype, QTILE, idx_head_dim],
    k_tma: KTMATileT[dtype, block_size, idx_head_dim],
    k_operand: KOperand,  # bf16 index-K cache, single head (for ragged row math)
    input_row_offsets: TileTensor[
        DType.uint32, IROLT, ImmutAnyOrigin
    ],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, PLLT, ImmutAnyOrigin],  # [batch]
    score: TileTensor[
        DType.float32, ScoreLT, MutAnyOrigin
    ],  # [num_index_heads, total_q, max_num_blocks]
    batch: Int,
    max_num_blocks: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    num_chunks: Int,
):
    """SM100 (B200) BF16 tensor-core (SS-UMMA) MSA prefill block scorer.

    One CTA scores a QTILE(=BM=64)-row query tile of one batch against every
    causally-reachable key block. Q (one head's tile) is the MMA A operand
    [BM, idx_head_dim]; each key block is the B operand [block_size,
    idx_head_dim], transpose_b=True so S = Q @ K_block^T. bf16 in, f32 TMEM
    accumulation.

    The block loop is split-K over `block_idx.z`: each of `num_chunks` CTAs owns
    a disjoint, contiguous slice of this tile's K blocks, so every block is
    scored by exactly one chunk and the per-block writes need no cross-CTA
    reduction.

    Four gotchas, all load-bearing:
    1. SS-UMMA SMEM must be staged via TMA, because the hardware applies the 128B
       swizzle XOR on the copy, and the UMMA descriptor reads it back with the
       matching XOR. Manual swizzle-less stores silently compute a wrong dot.
    2. tcgen05 alloc/dealloc/release are `.sync.aligned` WARP collectives. Call
       them from exactly ONE warp (warp 0), because calling on all warps hangs.
    3. One MMA scores the whole query tile against a key block, but each query row
       has its OWN causal cutoff. The per-query mask is applied to the fragment
       AFTER the MMA (a column key >= that query's num_keys is excluded from its
       max), not to the K read, so one key block serves every query row in the
       tile.
    4. Split-K disjoint-block invariant: each chunk handles blocks
       `[chunk_start, chunk_end)` (non-overlapping), so every block is written by
       exactly one CTA -- no reduction, no race. An empty chunk
       (`chunk_start >= max_blocks_tile`) must early-return uniformly before any
       collective op (TMA mbar / tcgen05 alloc), same as an empty query tile, or
       a straggler in a collective hangs the block. The K ring is re-indexed by
       the chunk-LOCAL iteration `it = blk - chunk_start` for phase/stage math,
       while absolute `blk` drives K coords and score offsets.
    """
    comptime assert (
        input_row_offsets.flat_rank == 1 and prefix_lens.flat_rank == 1
    )
    comptime INIT_SCORE = Float32(1.0e30)
    comptime LOCAL_SCORE = Float32(1.0e29)

    comptime BM = QTILE
    comptime BN = block_size
    comptime BK = idx_head_dim
    comptime AT = DType.float32
    comptime SW = _SCORE_SWIZZLE
    comptime NTHREADS = 128

    var tid = thread_idx.x
    var qt = block_idx.x  # query-tile index within the batch
    var b = block_idx.y

    var iro_b = Int(input_row_offsets[b])
    var extend_b = Int(input_row_offsets[b + 1]) - iro_b
    var q_tile_start = Int(qt) * BM
    # Over-provisioned grid (sized for the longest batch): empty tiles bail
    # before any collective op (TMA mbar / tcgen05 alloc). A straggler thread
    # in a collective would hang the block.
    if q_tile_start >= extend_b:
        return
    var n_tile_q = min(BM, extend_b - q_tile_start)

    # Tile-wide causal extent: the last query's key count bounds the block loop.
    var num_keys_tile_max = (
        Int(prefix_lens[b]) + (q_tile_start + n_tile_q - 1) + 1
    )
    var max_blocks_tile = ceildiv(num_keys_tile_max, block_size)

    # Split-K slice for this CTA (block_idx.z); the slice is CTA-uniform, so the
    # empty-chunk early-return below stays uniform -- a straggler in a
    # collective would hang the block.
    var chunk_id = Int(block_idx.z)
    var chunk_blocks = ceildiv(max_blocks_tile, num_chunks)
    var chunk_start = chunk_id * chunk_blocks
    var chunk_end = min(chunk_start + chunk_blocks, max_blocks_tile)
    if chunk_start >= max_blocks_tile:
        return
    var chunk_num_blocks = chunk_end - chunk_start

    comptime UMMAType = SM100TensorAccumulatorSS[
        dtype,
        AT,
        MMA_M=BM,
        MMA_N=BN,
        BM=BM,
        BN=BN,
        BK=BK,
        compute_BK=align_up(BK, 16),
        num_softmax_threads=NTHREADS,
        swizzle_a=SW,
        swizzle_b=SW,
        transpose_b=True,
        pipeline_stages=1,
    ]

    # --- SMEM layout (K is software-pipelined): all H query heads staged ONCE
    # [H][BM,BK] | K ring buffer [NSTAGE][BN,BK] | mbars. Q is constant across
    # the block loop, so it is loaded once up front and the per-block loop only
    # streams K -- prefetching K[blk+NSTAGE-1] while MMA+epilogue consume K[blk].
    comptime NSTAGE = _K_PREFETCH_STAGES
    comptime q_elems = BM * BK
    comptime k_elems = BN * BK
    var smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
        name="msa_prefill_mma_smem",
    ]()
    var q_smem = smem
    var k_smem = smem + num_index_heads * q_elems
    # TMA only uses .ptr from the destination (swizzle is in the descriptor), so
    # a flat row-major SMEM tile of the right byte size is sufficient.
    comptime q_flat_layout = tt_row_major[q_elems]()
    comptime k_flat_layout = tt_row_major[k_elems]()

    var mbar = (smem + num_index_heads * q_elems + NSTAGE * k_elems).bitcast[
        SharedMemBarrier
    ]()
    # mbar[0] = Q-prologue staging-done (used exactly once, never reused, so its
    # phase flip never collides with K-stage parity); mbar[1..NSTAGE] = per-stage
    # K-staging-done; mbar[NSTAGE+1..NSTAGE+2] = accumulator handshake (count from
    # the SS type); mbar[NSTAGE+3..] = tcgen05 TMEM base slot.
    var q_mbar = mbar
    var k_mbar = mbar + 1
    var acc_mbar = mbar + 1 + NSTAGE
    var ptr_tmem = (mbar + 1 + NSTAGE + 2).bitcast[UInt32]()

    var umma_p = UMMAType(acc_mbar.as_unsafe_any_origin())
    var umma_c = UMMAType(acc_mbar.as_unsafe_any_origin())

    if tid == 0:
        q_mbar[0].init()
        comptime for s in range(NSTAGE):
            k_mbar[s].init()
        umma_p.init()
    named_barrier[Int32(NTHREADS)]()

    # tcgen05 alloc on ONE warp (warp-collective .sync.aligned).
    #
    # Occupancy coupling (load-bearing): TMEM is 512 cols per SM, shared across
    # resident CTAs. The S=Q@K^T accumulator is [BM, MMA_N=BN] f32, so it needs
    # only BN cols. Allocating the full 512 would let just one CTA hold TMEM,
    # re-serializing the two CTAs that SMEM alone would allow. Allocating only
    # TMEM_COLS (= BN, padded to the 32-col tcgen05 granularity) lets up to
    # 512//TMEM_COLS CTAs hold TMEM at once. The alloc permit (a per-SM
    # serialization token, not the columns) must also be relinquished right
    # after alloc, so the second CTA can grab the permit and alloc its own cols
    # while the first computes; releasing it only at dealloc (the mha_1q
    # 1-CTA/SM idiom) would re-serialize the two CTAs.
    comptime TMEM_COLS = UInt32(align_up(BN, 32))
    if warp_id() == 0:
        tcgen05_alloc[1](ptr_tmem, TMEM_COLS)
        tcgen05_release_allocation_lock[1]()
    named_barrier[Int32(NTHREADS)]()
    var tmem_addr: UInt32 = ptr_tmem[0]

    var q_row0 = iro_b + q_tile_start
    var w = Int(warp_id())
    var l = Int(lane_id())
    comptime frag_simdwidth = 2

    comptime q_bytes = q_elems * size_of[dtype]()
    comptime k_bytes = k_elems * size_of[dtype]()

    if tid == 0:
        q_mbar[0].expect_bytes(Int32(num_index_heads * q_bytes))
        comptime for h in range(num_index_heads):
            var q_dst = TileTensor[
                dtype, type_of(q_flat_layout), address_space=AddressSpace.SHARED
            ](q_smem + h * q_elems, q_flat_layout)
            q_tma.async_copy_3d(q_dst, q_mbar[0], (0, h, q_row0))
    q_mbar[0].wait(0)

    # Ring indexed by chunk-LOCAL iter `it = blk - chunk_start`: stage
    # `s = it % NSTAGE` toggles phase every NSTAGE iters; consumer waits with
    # phase = (it // NSTAGE) & 1. No Q-prologue collision (Q uses its own mbar).
    # `issue_k` takes absolute `blk` for K coords, chunk-local `it` for the stage.
    @parameter
    @always_inline
    def issue_k(blk: Int, it: Int):
        if tid == 0:
            var key_start = blk * block_size
            var k_row0 = Int(k_operand.row_idx(UInt32(b), UInt32(key_start)))
            var s = it % NSTAGE
            var k_dst = TileTensor[
                dtype, type_of(k_flat_layout), address_space=AddressSpace.SHARED
            ](k_smem + s * k_elems, k_flat_layout)
            k_mbar[s].expect_bytes(Int32(k_bytes))
            k_tma.async_copy_3d(k_dst, k_mbar[s], (0, 0, k_row0))

    var n_prefetch = min(NSTAGE, chunk_num_blocks)
    for it in range(n_prefetch):
        issue_k(chunk_start + it, it)

    umma_c.tmem_arrive_init()
    named_barrier[Int32(NTHREADS)]()

    for it in range(chunk_num_blocks):
        var blk = chunk_start + it
        var key_start = blk * block_size
        var s = it % NSTAGE
        # k_mbar[s] phase = number of full ring cycles completed (chunk-local).
        var k_phase = UInt32((it // NSTAGE) & 1)
        k_mbar[s].wait(k_phase)

        # B descriptor for this K ring stage; A descriptor rebuilt per head.
        comptime for h in range(num_index_heads):
            var qk_desc = UMMAType.mma_descriptors(
                (q_smem + h * q_elems).as_unsafe_any_origin(),
                (k_smem + s * k_elems).as_unsafe_any_origin(),
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
            named_barrier[Int32(NTHREADS)]()

            var c = umma_c.wait_for_mma(s_acc_c)
            var reg = UMMAType.c_t.allocate_register_tile()
            c.copy_to(reg)
            umma_c.tmem_arrive()

            # Each lane owns rows {w*16 + i*8 + l//4 : i in 0,1} and columns
            # {(l%4)*2 + j*8 + cc : j in 0..BN//8, cc in 0,1}. Reduce each owned
            # row's columns into a per-(query,head) max with the per-query causal
            # mask, then lane-group-reduce across the 4 lanes sharing a row.
            comptime for i in range(2):
                var row = w * 16 + i * 8 + l // 4
                var q_local = q_tile_start + row
                var num_keys_q = Int(prefix_lens[b]) + q_local + 1
                var lane_row_max = min_or_neg_inf[AT]()
                comptime for j in range(BN // 8):
                    var v = rebind[SIMD[AT, frag_simdwidth]](reg[i, 0, j, 0])
                    comptime for cc in range(frag_simdwidth):
                        var col = (l % 4) * 2 + j * 8 + cc
                        var key = key_start + col
                        # Per-query causal mask: exclude keys >= this query's own
                        # causal count from its max (shared K tile, per-query cut).
                        if key < num_keys_q:
                            var s_v = v[cc] * sm_scale
                            if s_v > lane_row_max:
                                lane_row_max = s_v
                # Reduce across the 4 lanes (l%4) that share this row.
                var row_max = warp.lane_group_max[num_lanes=4](
                    SIMD[AT, 1](lane_row_max)
                )[0]
                if (l % 4) == 0 and row < n_tile_q:
                    var num_blocks_q = ceildiv(num_keys_q, block_size)
                    if blk < num_blocks_q:
                        var local_start_q = max(0, num_blocks_q - local_blocks)
                        var val = row_max
                        if blk < init_blocks:
                            val = INIT_SCORE
                        if blk >= local_start_q:
                            val = LOCAL_SCORE
                        score.to_layout_tensor().ptr_at_offset(
                            Index(h, iro_b + q_local, 0)
                        )[blk] = val

        # Prefetch the block that will land in this freed ring stage. All threads
        # have finished reading K[s] for every head's MMA (the accumulator
        # handshake serialized MMA-consume per head), so reissuing into stage s
        # is safe. One elected thread issues; the wait happens NSTAGE iters on.
        named_barrier[Int32(NTHREADS)]()
        var next_it = it + NSTAGE
        if next_it < chunk_num_blocks:
            issue_k(chunk_start + next_it, next_it)

    # Permit already relinquished right after alloc (see occupancy note above);
    # only the columns are freed here.
    if warp_id() == 0:
        tcgen05_dealloc[1](tmem_addr, TMEM_COLS)


@__name(t"sparse_indexer_prefill_topk")
def _prefill_topk_kernel[
    IROLT: TensorLayout,
    PLLT: TensorLayout,
    ScoreLT: TensorLayout,
    OutLT: TensorLayout,
    block_size: Int,
](
    input_row_offsets: TileTensor[
        DType.uint32, IROLT, ImmutAnyOrigin
    ],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, PLLT, ImmutAnyOrigin],  # [batch]
    score: TileTensor[
        DType.float32, ScoreLT, MutAnyOrigin
    ],  # [num_index_heads, total_q, max_num_blocks]
    out_idxs: TileTensor[
        DType.int32, OutLT, MutAnyOrigin
    ],  # [num_index_heads, total_q, topk]
    batch: Int,
    max_num_blocks: Int,
    topk: Int,
):
    comptime assert (
        input_row_offsets.flat_rank == 1 and prefix_lens.flat_rank == 1
    )
    var t = block_idx.x
    var h = block_idx.y

    var iro_ptr = input_row_offsets.to_layout_tensor().ptr_at_offset(Index(0))
    var b = _token_batch(
        t,
        batch,
        rebind[UnsafePointer[Scalar[DType.uint32], ImmutAnyOrigin]](iro_ptr),
    )
    var local_idx = t - Int(input_row_offsets[b])
    var num_keys = Int(prefix_lens[b]) + local_idx + 1
    var num_blocks = ceildiv(num_keys, block_size)

    var score_row = score.to_layout_tensor().ptr_at_offset(Index(h, t, 0))
    var out_row = out_idxs.to_layout_tensor().ptr_at_offset(Index(h, t, 0))
    block_select_topk[DType.float32, DType.int32](
        rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](score_row),
        num_blocks,
        topk,
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](out_row),
    )


@always_inline
def sparse_indexer_prefill_score[
    dtype: DType,
    KOperand: MHAOperand,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    q: TileTensor[dtype, ...],  # [total_q, num_index_heads, idx_head_dim]
    k_operand: KOperand,  # bf16 index-K cache, single head
    input_row_offsets: TileTensor[DType.uint32, ...],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, total_q, max_num_blocks]
    batch: Int,
    total_q: Int,
    max_seqlen_q: Int,
    max_num_blocks: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    ctx: DeviceContext,
) raises:
    """Launch the prefill block-scoring kernel into `score`.

    See `sparse_indexer_prefill` for the argument contract. Exposed separately so
    tests can drive scoring and selection independently.
    """
    comptime BLOCK_DIM = 128
    comptime assert (
        BLOCK_DIM % WARP_SIZE == 0
    ), "block_dim must be a multiple of the warp size"
    # SM100 SS-UMMA tile geometry: BM=64 query rows, MMA_N=block_size keys,
    # BK=idx_head_dim contraction. The accumulator asserts BM%MMA_M, BN%MMA_N,
    # compute_BK%16; idx_head_dim and block_size feed BK/BN directly.
    comptime QTILE = 64

    # Split-K so low-batch / short-extend launches fill the GPU. Base grid is
    # only `q_tiles * batch` CTAs (a 256-token chunked-prefill against a long
    # prefix has q_tiles=4 -> 4 CTAs on a 148-SM B200, with the K-block loop
    # serialized inside each). Splitting the K-block range over `block_idx.z`
    # raises in-flight CTAs (each resides 1/SM, SMEM-bound) so the scheduler can
    # hide the per-iteration MMA/barrier latency that dominates at 1 CTA/SM.
    #
    # CAPTURE-SAFETY: `num_chunks` depends ONLY on graph-constant quantities
    # (`max_seqlen_q`, `batch`), never on per-call `prefix_lens`/`num_keys`
    # (which vary inside a CUDA-graph capture). Chunks past a tile's live block
    # count early-return on-device, so any chunk count is valid -- each block is
    # written by exactly one chunk, no reduction. Capped so well-subscribed
    # shapes (large q_tiles*batch) get chunks=1 and do not regress.
    var q_tiles = ceildiv(max_seqlen_q, QTILE)
    # TARGET_GRID is several SM-counts' worth of CTAs: at 1 CTA/SM (SMEM-bound)
    # the kernel is barrier/MMA-wait latency-bound, so it wants several waves of
    # short CTAs to hide that latency. Too small under-fills bulk shapes; too
    # large over-fragments short tiles (per-chunk TMA/tcgen05 prologue cost).
    comptime TARGET_GRID = 1024
    comptime MAX_CHUNKS = 512
    var chunks_for_grid = TARGET_GRID // max(1, q_tiles * batch)
    # Depth-balanced split-K (load-bearing for deep fresh-prefill shapes):
    # `chunks_for_grid` alone under-splits a grid-full launch whose per-tile K
    # loops are long and uneven -- causal means tile qt does ~qt blocks, so the
    # deepest tile (~q_tiles blocks) runs long after the short tiles finish and
    # the scheduler runs dry of warps to hide the per-(block,head) MMA/barrier
    # latency. So also split so the deepest tile's chunk holds
    # ~TARGET_BLOCKS_PER_CHUNK blocks, breaking the long CTAs into more uniform
    # ones: too small over-fragments (per-chunk tcgen05/Q-prologue cost), too
    # large under-splits the deep tiles.
    # CAPTURE-SAFE: `max_tile_blocks` uses `max_seqlen_q` (graph constant) only,
    # never `prefix_lens`; QTILE < block_size so multiply before the divide.
    comptime TARGET_BLOCKS_PER_CHUNK = 8
    var max_tile_blocks = ceildiv(q_tiles * QTILE, block_size)
    var chunks_for_depth = ceildiv(max_tile_blocks, TARGET_BLOCKS_PER_CHUNK)
    var num_chunks = clamp(
        max(chunks_for_grid, chunks_for_depth), 1, MAX_CHUNKS
    )

    comptime score_kernel = _prefill_block_score_kernel[
        dtype,
        KOperand,
        type_of(input_row_offsets).LayoutType,
        type_of(prefix_lens).LayoutType,
        type_of(score).LayoutType,
        num_index_heads,
        idx_head_dim,
        block_size,
        QTILE,
    ]

    # 3D TMA over Q [total_q, num_index_heads, idx_head_dim]: one async_copy_3d
    # loads a [QTILE, 1, idx_head_dim] tile at (depth=0, head=h, row=q_row0).
    # Hardware applies the 128B swizzle the UMMA descriptor reads back with.
    var q_ptr = q.to_layout_tensor().ptr_at_offset(Index(0, 0, 0))
    var q_tma = create_split_tma[
        Index(QTILE, 1, idx_head_dim),
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, idx_head_dim),
        _SCORE_SWIZZLE,
    ](
        ctx,
        rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
            q_ptr.as_immutable().as_unsafe_any_origin()
        ),
        total_q,
        num_index_heads,
    )
    var k_tma = k_operand.create_tma_tile[
        _SCORE_SWIZZLE,
        BN=block_size,
        depth=idx_head_dim,
        BK=idx_head_dim,
    ](ctx)

    # SMEM: H query heads staged once + _K_PREFETCH_STAGES K ring buffers, both
    # bf16; plus barriers: q-prologue(1) + K stages(_K_PREFETCH_STAGES) +
    # accumulator handshake(2) + tcgen05 TMEM slot(rounded up to 1 barrier).
    comptime _N_BARRIERS = 1 + _K_PREFETCH_STAGES + 2 + 1
    comptime smem_bytes = (
        (num_index_heads * QTILE + _K_PREFETCH_STAGES * block_size)
        * idx_head_dim
        * size_of[Scalar[dtype]]()
        + _N_BARRIERS * size_of[SharedMemBarrier]()
    )
    ctx.enqueue_function[score_kernel](
        rebind[QTMATileT[dtype, QTILE, idx_head_dim]](q_tma),
        rebind[KTMATileT[dtype, block_size, idx_head_dim]](k_tma),
        k_operand,
        input_row_offsets.as_immut(),
        prefix_lens.as_immut(),
        score,
        batch,
        max_num_blocks,
        init_blocks,
        local_blocks,
        sm_scale,
        num_chunks,
        grid_dim=(q_tiles, batch, num_chunks),
        block_dim=BLOCK_DIM,
        shared_mem_bytes=smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_bytes)
        ),
    )


@always_inline
def sparse_indexer_prefill_topk[
    num_index_heads: Int,
    block_size: Int,
](
    input_row_offsets: TileTensor[DType.uint32, ...],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, total_q, max_num_blocks]
    out_idxs: TileTensor[DType.int32, ...],  # [num_index_heads, total_q, topk]
    batch: Int,
    total_q: Int,
    max_num_blocks: Int,
    topk: Int,
    ctx: DeviceContext,
) raises:
    """Launch the prefill top-k selection kernel from `score` into `out_idxs`.

    See `sparse_indexer_prefill` for the argument contract. Exposed separately so
    tests can drive scoring and selection independently.
    """
    # `block_select_topk` extracts the k winners serially: each iteration runs a
    # `block_dim`-wide reduction for the next winner, then evicts it (an
    # O(k * num_blocks) chain). The serial dependency makes the kernel
    # latency-bound on the per-iteration reduction, not throughput-bound; more
    # warps (higher occupancy) don't help -- the lever is the reduction width.
    # Size block_dim to the work, not a fixed 128:
    #   - small num_blocks (bulk prefill): a one-warp block (32) makes each
    #     reduction a single warp shuffle + a trivial 1-warp barrier (shortest
    #     path); a wider block's extra threads just idle.
    #   - large num_blocks (long prefix): the per-thread scan
    #     (num_blocks/block_dim) dominates, so a wider block parallelizes it.
    # Target ~16 blocks/thread, clamped to [WARP_SIZE, 128], warp-multiple.
    # block_select_topk reads `block_dim.x` at runtime, so a runtime value is
    # valid (no comptime dependence).
    var topk_block_dim = clamp(
        align_up(ceildiv(max_num_blocks, 16), WARP_SIZE), WARP_SIZE, 128
    )
    # block_select_topk requires a warp-multiple block_dim; the formula gives
    # one, but assert to guard a future regression.
    debug_assert(
        topk_block_dim % WARP_SIZE == 0,
        "topk block_dim must be a multiple of the warp size",
    )
    comptime topk_kernel = _prefill_topk_kernel[
        type_of(input_row_offsets).LayoutType,
        type_of(prefix_lens).LayoutType,
        type_of(score).LayoutType,
        type_of(out_idxs).LayoutType,
        block_size,
    ]
    ctx.enqueue_function[topk_kernel](
        input_row_offsets.as_immut(),
        prefix_lens.as_immut(),
        score,
        out_idxs,
        batch,
        max_num_blocks,
        topk,
        grid_dim=(total_q, num_index_heads),
        block_dim=topk_block_dim,
    )


@always_inline
def sparse_indexer_prefill[
    dtype: DType,
    KOperand: MHAOperand,
    num_index_heads: Int,
    idx_head_dim: Int,
    block_size: Int,
](
    q: TileTensor[dtype, ...],  # [total_q, num_index_heads, idx_head_dim]
    k_operand: KOperand,  # bf16 index-K cache, single head
    input_row_offsets: TileTensor[DType.uint32, ...],  # [batch + 1]
    prefix_lens: TileTensor[DType.uint32, ...],  # [batch]
    score: TileTensor[
        DType.float32, ...
    ],  # [num_index_heads, total_q, max_num_blocks]
    out_idxs: TileTensor[DType.int32, ...],  # [num_index_heads, total_q, topk]
    batch: Int,
    total_q: Int,
    max_seqlen_q: Int,
    max_num_blocks: Int,
    topk: Int,
    init_blocks: Int,
    local_blocks: Int,
    sm_scale: Float32,
    ctx: DeviceContext,
) raises:
    """Compute MSA top-k block indices for a prefill step (selection only).

    Args:
        q: Query tensor `[total_q, num_index_heads, idx_head_dim]` (normed +
            roped by the caller), ragged across the batch.
        k_operand: Index-K cache as an `MHAOperand` (single head).
        input_row_offsets: Ragged query-token offsets `[batch + 1]`.
        prefix_lens: Per-batch cached-key count preceding the query tokens
            `[batch]` (0 for a fresh prefill).
        score: Caller-owned scratch `[num_index_heads, total_q, max_num_blocks]`;
            written then consumed (and mutated) by the two launches.
        out_idxs: Output block indices `[num_index_heads, total_q, topk]`, int32,
            `-1`-padded.
        batch: Batch size.
        total_q: Total ragged query tokens (`input_row_offsets[batch]`).
        max_seqlen_q: Max per-batch query-token count (`max(extend_b)`); sizes the
            query-tile grid dimension of the SM100 tensor-core scoring kernel.
        max_num_blocks: Row stride of `score` (>= every per-token block count).
        topk: Number of blocks to select.
        init_blocks: Always-keep leading blocks (forced score 1e30).
        local_blocks: Always-keep trailing/local blocks (forced score 1e29).
        sm_scale: QK scale.
        ctx: Device context.
    """
    sparse_indexer_prefill_score[
        dtype, KOperand, num_index_heads, idx_head_dim, block_size
    ](
        q,
        k_operand,
        input_row_offsets,
        prefix_lens,
        score,
        batch,
        total_q,
        max_seqlen_q,
        max_num_blocks,
        init_blocks,
        local_blocks,
        sm_scale,
        ctx,
    )
    sparse_indexer_prefill_topk[num_index_heads, block_size](
        input_row_offsets,
        prefix_lens,
        score,
        out_idxs,
        batch,
        total_q,
        max_num_blocks,
        topk,
        ctx,
    )
