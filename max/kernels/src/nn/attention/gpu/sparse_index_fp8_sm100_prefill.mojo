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
"""SM100 (B200) warp-specialized PREFILL variant of the FP8 MLA indexer scorer.

Computes the identical per-(query token, key) logit as the shipped scorer
(`sparse_index_fp8_sm100.fp8_index_score_sm100`) and the scalar
`nn.index_fp8.fp8_index_kernel`:

    score[token, key] = k_scale[key]
                        * Σ_head relu(q[token, head] · k[key]) * q_scale[token, head]

The shipped kernel is K-resident / Q-streaming: one CTA holds a 64-key tile and
streams every query token past it. That maps a batch-1 prefill onto only
`num_keys / 64` CTAs (128 at 8192 keys < 148 SMs) and runs one serial
warpgroup, so it is latency-bound (measured ~6% achieved occupancy: one active
warp per scheduler, no MMA↔epilogue overlap).

This kernel INVERTS which operand persists (the S^T orientation and the BM=64
head-sum fragment map are kept verbatim -- only the streamed operand changes):

- **Q resident** as the B operand `[MMA_N = N_TOKENS * num_heads, depth]`: a CTA
  owns one N_TOKENS-token block, staged once.
- **K streams** as the A operand `[BM_key = 64, depth]` through a deep SMEM
  prefetch ring, `S^T = K @ Q^T = [key, (token, head)]`, so the epilogue reduces
  over the (token, head) COLUMNS exactly like the shipped kernel (heads stay
  columns; all head counts in {4, 8, 32, 64} work uniformly, no cross-warp
  reduction).
- Grid `(batch, ceil(seq_len / N_TOKENS), 1)`: one CTA per query-token block.
  A batch-1 GLM prefill (1024 tokens, num_heads=32, N_TOKENS=4) is 256 CTAs --
  it fills the machine without a seq-split -- and each CTA streams every key.

Warp specialization (256 threads = two warpgroups), mirroring the MSA prefill
scorer (`Kernels/lib/msa/sparse_indexer_prefill.mojo`, PR #91938):
- WG0 (warps 0-3, threads 0-127) = score/epilogue consumer. Reads each S^T
  stage out of TMEM, applies the branchless relu, sums over the head columns of
  each token, cross-lane reduces the 4 lanes sharing a key row, scales by
  k_scale, and writes one f32 per (token, key) under the fused causal guard.
- WG1 (warps 4-7, threads 128-255) = producer: warp 4 = MMA (TMEM owner +
  `K @ Q^T` per K tile), warp 5 = TMA (deep K-ring producer), warps 6-7 idle
  (register-dealloc to the floor). Role-to-role mbars (k_full/k_empty for the K
  ring, the accumulator's own MMA↔TMEM handshake for the multi-stage S^T)
  replace the shipped kernel's per-iteration whole-CTA `named_barrier`.

Token blocks are independent outputs, so there is no split-K and no cross-CTA
reduction. Because a CTA sees EVERY key for its resident token block, this
layout is also the prerequisite for the planned fused score+top-k-select
variant that never materializes the `[total_seq, max_num_keys]` score buffer
(see KERN-3139).

Standalone routing (measured): the token-block grid is `ceil(seq / N_TOKENS)`
with `N_TOKENS = 128 // num_heads`, so it only fills the machine -- and only wins
over the K-resident scorer -- at `num_heads == 64` (`N_TOKENS = 2`; measured
+7-17% at nh64 prefill). At `num_heads` in {4, 8, 32} too few token blocks
underfill the grid and the heavier per-key-tile head-sum makes the kernel
consumer-bound (a net loss vs K-resident), so `fp8_index_score_sm100` routes
only `num_heads == 64` prefill here; every other shape (and decode) stays on the
shipped K-resident path byte-identically. The kernel body itself supports all
head counts uniformly, kept general for the fused score+top-k-select follow-up.

NVIDIA SM100 only (SS-UMMA / TMA / tcgen05). Verified against
`nn.index_fp8.fp8_index_naive` via `test_index_fp8` and end-to-end top-k set
match via `test_mla_index_fp8`.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
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
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from std.math import align_up, ceildiv
from std.sys import size_of
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
import std.gpu.primitives.warp as warp

from layout import TensorLayout, TileTensor
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
)

from std.gpu.compute.arch.mma_nvidia_sm100 import mma_arrive

from nn.attention.gpu.nvidia.sm100.mha_1q import SM100TensorAccumulatorSS
from nn.attention.mha_operand import MHAOperand


# The K-resident scorer's fragment map bakes in BM_key = 64 (rows w*16+i*8+l//4),
# and the head-sum epilogue reuses it verbatim. Defined locally (not imported
# from `sparse_index_fp8_sm100`) so that file can route to this one without an
# import cycle.
comptime _INDEX_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime QTMATileT[
    dtype: DType, MMA_N: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(MMA_N, 1, depth), _INDEX_SWIZZLE]
comptime KTMATileT[
    dtype: DType, BM_key: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(BM_key, 1, depth), _INDEX_SWIZZLE]


# Two warpgroups: WG0 (warps 0-3) = 128 epilogue/score consumers (the BM=64
# head-sum fragment map needs exactly 128 threads), WG1 (warps 4-7) = producer
# (MMA warp 4, TMA warp 5, idle warps 6-7). Single source of truth for the
# reqntid/minctasm launch cap, the kernel body, and the launcher.
comptime _PREFILL_NTHREADS = 256
comptime _NUM_SOFTMAX_THREADS = 128
# Producer register floor (MMA + TMA warps); the consumer claims the rest so its
# TMEM->register fragment does not spill. Mirrors the MSA scorer's split.
comptime _NUM_REG_PRODUCER = 40
comptime _NUM_REG_CONSUMER = 232
# K-ring prefetch depth (SMEM) and S^T TMEM stages. The K ring hides the TMA
# latency (MMA runs behind the load warp); the S^T stages hide the MMA->epilogue
# latency (2 -> the consumer reads stage `it` while the MMA writes `it+1`).
# 2 S stages * MMA_N=128 cols = 256 of 512 TMEM cols. K stage = 64*128 fp8 = 8KB;
# 8 stages = 64KB, comfortably one CTA/SM against the ~227KB SM100 SMEM budget.
comptime _K_RING_STAGES = 8
comptime _S_TMEM_STAGES = 2


@__name(t"fp8_index_score_prefill_sm100_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
# Cap the launch register count so `_PREFILL_NTHREADS` threads fit at 1 CTA/SM
# (reqntid + minctasm), mirroring the MSA scorer and FA4: without it the
# 256-thread launch requests > 65536/256 regs/thread and the warpgroup
# reg-alloc/dealloc below has no reserved pool to redistribute.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_PREFILL_NTHREADS)
    )
)
@__llvm_metadata(`nvvm.minctasm`=SIMDSize(1))
def _fp8_index_score_prefill_kernel_sm100[
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
    comptime assert valid_length.flat_rank == 1
    comptime MMA_N = N_TOKENS * num_heads
    comptime assert (
        MMA_N == 128
    ), "MMA_N must pack to 128 (N_TOKENS * num_heads); got " + String(MMA_N)
    comptime AT = DType.float32
    comptime SW = _INDEX_SWIZZLE
    comptime NTHREADS = _PREFILL_NTHREADS
    comptime NSTAGE = _K_RING_STAGES
    comptime N_S = _S_TMEM_STAGES
    # WG1 (producer) thread roles.
    comptime MMA_WARP = 4
    comptime TMA_WARP = 5
    comptime MMA_LANE = MMA_WARP * WARP_SIZE  # lane 0 of the MMA warp issues MMAs

    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var token_block = Int(block_idx.y)
    var tok0 = token_block * N_TOKENS

    var start_of_seq = Int(valid_length[b])
    var end_of_seq = Int(valid_length[b + 1])
    var seq_len = end_of_seq - start_of_seq

    var num_keys = Int(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len

    # Bail uniformly (every thread) before any collective op (TMA mbar / tcgen05
    # alloc); a divergent early return deadlocks them. A token block past the
    # sequence produces no output (the caller's -inf fill covers those rows).
    if tok0 >= seq_len or seq_len <= 0:
        return

    # Keys this CTA must stream: bounded by the deepest live token of the block
    # under the causal mask (each token still gets its own per-key guard in the
    # epilogue). Non-causal streams every key; causal trims the triangle a
    # zero-prefix fresh prefill leaves off the end.
    var last_tok = min(tok0 + N_TOKENS, seq_len) - 1
    var block_key_bound = num_keys - (seq_len - 1 - last_tok) * causal
    var n_key_tiles = ceildiv(block_key_bound, BM_key)

    comptime UMMAType = SM100TensorAccumulatorSS[
        dtype,
        AT,
        MMA_M=BM_key,
        MMA_N=MMA_N,
        BM=BM_key,
        BN=MMA_N,
        BK=depth,
        compute_BK=align_up(depth, 16),
        num_softmax_threads=_NUM_SOFTMAX_THREADS,
        swizzle_a=SW,
        swizzle_b=SW,
        transpose_b=True,
        pipeline_stages=N_S,
    ]

    comptime k_elems = BM_key * depth
    comptime q_elems = MMA_N * depth
    var smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
        name="fp8_index_sm100_prefill_smem",
    ]()
    # Q resident (B operand, one token block) | K ring (A operand) | q_scale
    # resident | mbars: q(1) + k_full(NSTAGE) + k_empty(NSTAGE) +
    # accumulator(2*N_S) | tcgen05 TMEM base slot.
    var q_smem = smem
    var k_smem = smem + q_elems
    var qs_smem = (smem + q_elems + NSTAGE * k_elems).bitcast[Float32]()
    var mbar = (qs_smem + MMA_N).bitcast[SharedMemBarrier]()
    var q_mbar = mbar
    var k_full = mbar + 1
    var k_empty = mbar + 1 + NSTAGE
    var acc_mbar = mbar + 1 + 2 * NSTAGE
    var ptr_tmem = (acc_mbar + 2 * N_S).bitcast[UInt32]()

    comptime q_flat_layout = tt_row_major[q_elems]()
    comptime k_flat_layout = tt_row_major[k_elems]()

    var umma_p = UMMAType(acc_mbar.as_unsafe_any_origin())
    var umma_c = UMMAType(acc_mbar.as_unsafe_any_origin())

    if tid == 0:
        q_mbar[0].init()
        comptime for s in range(NSTAGE):
            k_full[s].init()
            k_empty[s].init()
        umma_p.init()

    # tcgen05 alloc is warp-collective (.sync.aligned): exactly one warp (the MMA
    # warp). Release the lock right after so co-resident CTAs can allocate. All
    # N_S * MMA_N cols are one block; the accumulator strides stage * MMA_N.
    comptime TMEM_COLS = UInt32(N_S * align_up(MMA_N, 32))
    if warp_id() == MMA_WARP:
        tcgen05_alloc[1](ptr_tmem, TMEM_COLS)
        tcgen05_release_allocation_lock[1]()
    named_barrier[Int32(NTHREADS)]()
    var tmem_addr: UInt32 = ptr_tmem[0]

    comptime k_bytes = k_elems * size_of[dtype]()
    comptime q_bytes = q_elems * size_of[dtype]()

    # Resident Q staging (one token block, published to the MMA warp by q_mbar)
    # + q_scale staging (one f32 per (token, head) column, published to the
    # consumer by the barrier below) + TMEM-free seeding for the accumulator.
    if tid == MMA_LANE:
        var q_dst = TileTensor[
            dtype, type_of(q_flat_layout), address_space=AddressSpace.SHARED
        ](q_smem, q_flat_layout)
        q_mbar[0].expect_bytes(Int32(q_bytes))
        q_tma.async_copy_3d(
            q_dst, q_mbar[0], (0, 0, (start_of_seq + tok0) * num_heads)
        )
    if tid < MMA_N:
        var qs_tok = tid // num_heads
        if tok0 + qs_tok < seq_len:
            qs_smem[tid] = q_s[start_of_seq + tok0 + qs_tok, tid % num_heads][0]
        else:
            qs_smem[tid] = 0.0
    if warp_id() < 4:
        umma_c.tmem_arrive_init()
    named_barrier[Int32(NTHREADS)]()

    if warp_id() < 4:
        # ---- score/epilogue consumer warpgroup (the shipped head-sum path) ----
        warpgroup_reg_alloc[_NUM_REG_CONSUMER]()
        var w = Int(warp_id())
        var l = Int(lane_id())
        comptime frag_simdwidth = 2

        for it in range(n_key_tiles):
            var key_tile_base = it * BM_key
            var c = umma_c.wait_for_mma(UMMAType.c_t(tmem_addr))
            var reg = UMMAType.c_t.allocate_register_tile()
            c.copy_to(reg)
            umma_c.tmem_arrive()

            comptime for i in range(2):
                var row = w * 16 + i * 8 + l // 4
                var key_local = key_tile_base + row

                # k_scale for this key row (load-early, reused across the block's
                # tokens; the streamed keys mean it can no longer be staged once
                # per CTA the way the resident-key kernel does). Guarded by
                # key_local < num_keys so an OOB pool row is never dereferenced.
                var k_scale: Float32 = 0.0
                if (l % 4) == 0 and row < BM_key and key_local < num_keys:
                    k_scale = ks_operand.block_paged_ptr[1](
                        UInt32(b), UInt32(key_local), UInt32(0), UInt32(0)
                    )[0].cast[DType.float32]()

                # Per j-group accumulate with comptime indices only (a runtime
                # token index into the SIMD accumulator spills to local memory).
                var jgroup_sum = SIMD[AT, MMA_N // 8](0)
                comptime for j in range(MMA_N // 8):
                    var v = rebind[SIMD[AT, frag_simdwidth]](reg[i, 0, j, 0])
                    comptime for cc in range(frag_simdwidth):
                        var col = (l % 4) * 2 + j * 8 + cc
                        var raw = v[cc]
                        # branchless relu
                        var s = (raw + abs(raw)) * 0.5 * qs_smem[col][0]
                        jgroup_sum[j] += s

                comptime for t in range(N_TOKENS):
                    var tok_sum = Scalar[AT](0)
                    comptime if num_heads >= 8:
                        # An 8-column j-group never straddles a token boundary.
                        comptime jpt = num_heads // 8
                        comptime for jj in range(jpt):
                            tok_sum += jgroup_sum[t * jpt + jj]
                    else:
                        # num_heads == 4: j-group t // 2 spans tokens 2j and
                        # 2j + 1; which half this lane owns depends on the lane.
                        var own = ((l % 4) >= 2) == (t % 2 == 1)
                        tok_sum = jgroup_sum[t // 2] if own else Scalar[AT](0)
                    var row_sum = warp.lane_group_sum[num_lanes=4](
                        SIMD[AT, 1](tok_sum)
                    )[0]
                    var tok_local = tok0 + t
                    # Fused causal mask (branchless): token tok_local sees keys up
                    # to cache_len + tok_local, so forbidden slots keep the
                    # caller's -inf fill and the separate mask pass is skipped.
                    var key_bound = (
                        num_keys - (seq_len - 1 - tok_local) * causal
                    )
                    if (
                        (l % 4) == 0
                        and row < BM_key
                        and key_local < key_bound
                        and tok_local < seq_len
                    ):
                        var global_token = start_of_seq + tok_local
                        output.raw_store(
                            global_token * max_num_keys + key_local,
                            k_scale * row_sum,
                        )
    else:
        if warp_id() == MMA_WARP:
            # ---- MMA warp: TMEM owner + K @ Q^T per K tile ----
            # The whole role runs on one lane (the MMA lane): the tcgen05 MMA and
            # its commits are single-thread async ops, so no warp-collective elect
            # is involved (issuing the k_empty commit from a warp-collective
            # `elect_mma_arrive` here misaligned under nh4's register pressure).
            warpgroup_reg_dealloc[_NUM_REG_PRODUCER]()
            if tid == MMA_LANE:
                var kc_state = PipelineState[NSTAGE]()
                q_mbar[0].wait(0)  # resident Q staged (MMA B operand)
                for it in range(n_key_tiles):
                    var s = Int(kc_state.index())
                    k_full[s].wait(kc_state.phase())
                    umma_p.wait_for_tmem()
                    var qk_desc = UMMAType.mma_descriptors(
                        (k_smem + s * k_elems).as_unsafe_any_origin(),
                        q_smem.as_unsafe_any_origin(),
                    )
                    umma_p.mma(
                        rebind[UMMAType.a_t](qk_desc.get_a()),
                        rebind[UMMAType.b_t](qk_desc.get_b()),
                        UMMAType.c_t(tmem_addr),
                        0,
                    )
                    # Release K stage s only after the MMA has drained it
                    # (tcgen05.commit tracks the async MMA); a plain mbar arrive
                    # would let the load warp overwrite K mid-read.
                    mma_arrive(k_empty + s)
                    kc_state.step()
        elif warp_id() == TMA_WARP:
            # ---- load/TMA warp: deep K-ring producer ----
            warpgroup_reg_dealloc[_NUM_REG_PRODUCER]()
            var kp_state = PipelineState[NSTAGE](0, 1, 0)
            var n_prefetch = min(NSTAGE, n_key_tiles)

            @parameter
            @always_inline
            def issue_k(it: Int, state: PipelineState[NSTAGE]):
                var s = Int(state.index())
                var k_row0 = Int(
                    k_operand.row_idx(UInt32(b), UInt32(it * BM_key))
                )
                var k_dst = TileTensor[
                    dtype,
                    type_of(k_flat_layout),
                    address_space=AddressSpace.SHARED,
                ](k_smem + s * k_elems, k_flat_layout)
                if tid == TMA_WARP * WARP_SIZE:
                    k_full[s].expect_bytes(Int32(k_bytes))
                    k_tma.async_copy_3d(k_dst, k_full[s], (0, 0, k_row0))

            # Prologue: fill the first NSTAGE stages (fresh, no k_empty wait).
            for it in range(n_prefetch):
                issue_k(it, kp_state)
                kp_state.step()
            # Refills: wait k_empty at that stage/phase (MMA done with the prior
            # occupant, tile it-NSTAGE) before reissuing.
            for it in range(n_prefetch, n_key_tiles):
                k_empty[Int(kp_state.index())].wait(kp_state.phase())
                issue_k(it, kp_state)
                kp_state.step()
        else:
            # Idle warps 6-7: drop to the setmaxnreg floor so the consumer +
            # MMA/TMA warps can claim this warpgroup's register-file share.
            warpgroup_reg_dealloc[24]()

    # Single whole-CTA drain: the consumer's last `tmem_arrive` happens-before
    # this barrier, so no S^T stage is live when the MMA warp frees TMEM.
    named_barrier[Int32(NTHREADS)]()
    if warp_id() == MMA_WARP:
        tcgen05_dealloc[1](tmem_addr, TMEM_COLS)


@always_inline
def _prefill_smem_bytes[dtype: DType, depth: Int, BM_key: Int]() -> Int:
    """SMEM byte size for the prefill kernel (shared by launcher + kernel)."""
    comptime MMA_N = 128
    comptime k_elems = BM_key * depth
    comptime q_elems = MMA_N * depth
    comptime n_mbars = 1 + 2 * _K_RING_STAGES + 2 * _S_TMEM_STAGES
    return (
        (q_elems + _K_RING_STAGES * k_elems) * size_of[Scalar[dtype]]()
        + MMA_N * size_of[Float32]()
        + n_mbars * size_of[SharedMemBarrier]()
        + size_of[UInt32]()
    )


# Route a shape to the K-streaming prefill kernel when a sequence spans at least
# this many token blocks. Below it (decode / small extend), one token block is a
# handful of CTAs that badly underfill the machine, so the shipped K-resident
# kernel (one CTA per key tile, oversubscribed for decode) stays faster -- those
# shapes keep the shipped path byte-identically.
comptime _PREFILL_MIN_TOKEN_TILES = 16


@always_inline
def fp8_index_score_sm100_prefill[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
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
    valid_length: TileTensor[mut=False, DType.uint32, ...],
    q_s: TileTensor[mut=False, DType.float32, ...],
    output: TileTensor[DType.float32, ...],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    causal: Int,
    ctx: DeviceContext,
) raises:
    """Enqueue the warp-specialized K-streaming prefill scorer into `output`.

    One CTA per (batch, token block); each streams every causally-reachable key
    for its resident `N_TOKENS`-token block. Called only from
    `fp8_index_score_sm100` on shapes with many token blocks per sequence.
    """
    comptime kernel = _fp8_index_score_prefill_kernel_sm100[
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
    comptime smem_bytes = _prefill_smem_bytes[dtype, depth, BM_key]()
    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        k_operand,
        ks_operand,
        valid_length.as_immut(),
        q_s,
        output,
        max_num_keys,
        causal,
        grid_dim=(batch_size, ceildiv(max_seq_len, N_TOKENS), 1),
        block_dim=_PREFILL_NTHREADS,
        shared_mem_bytes=smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_bytes)
        ),
    )
