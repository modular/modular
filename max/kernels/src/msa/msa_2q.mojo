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
"""KV-block-major sparse MHA (MSA) forward kernel for SM100 (B200), BF16, D=128.

The inverse of the query-major `msa_1q.mojo`: one CTA owns ONE 128-token KV
block (a CSR row) and gathers the queries that selected it.  The reverse-CSR
lists, per (batch, kv-block), its attending query tokens; one work-item == one
(head_kv, non-empty CSR row).  The CTA bulk-TMAs the block once, then loops
`ceil(q_count/BM)` Q-tiles (QK -> softmax -> PV per tile) against the resident
block, gather4-loading each tile's queries and scattering results into
`O_partial`/`LSE_partial` (combine LSE-merges them into O).  Softmax is a single
full tile (no online correction); the epilogue scatters per (query, split_slot).

Scope: flat KV (`page_size == 0`) or whole-block paged KV (`page_size == BN`),
BF16, grouped-query qheadperkv in {1, 2, 4, 8, 16} (the group's query heads pack
into the M-tile, sharing one KV load).  Diagonal-block causal via per-query
logical position.
"""

from std.math import ceildiv, recip, align_up, log, log2, isfinite
from std.sys import size_of
from std.sys._assembly import inlined_assembly
from std.math.constants import log2e

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import (
    DeviceContext,
    FuncAttribute,
    DeviceBuffer,
    DeviceAttribute,
)
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import (
    AddressSpace,
    CacheEviction,
    cp_async_bulk_tensor_2d_gather4,
    external_memory,
    fence_async_view_proxy,
)
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_release_allocation_lock,
)
from std.gpu.compute.arch.mma_nvidia_sm100 import mma_arrive
from std.gpu.sync import named_barrier
from std.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
)
import std.gpu.primitives.warp as warp

from layout import IntTuple, Layout, LayoutTensor
from layout.layout_tensor import copy_local_to_shared
from layout.swizzle import make_swizzle
from layout.tma_async import PipelineState, SharedMemBarrier
from layout._utils import IndexList
from layout.tma_async import (
    TMATensorTile,
    _gather4_box_width,
    create_tma_tile_gather4,
)

from std.logger import Logger
from std.collections import OptionalReg

from nn.attention.gpu.nvidia.sm100.attention_utils import mul_ftz
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_operand import kv_sub_tile_rows as _kv_sub_tile_rows
from nn.attention.gpu.nvidia.sm90.attention import (
    elect,
    KVTMATile,
)
from nn.attention.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
)
from nn.softmax import _rowmax_online_softmax, _rowsum

# MSA-private copy of the SM100 accumulator/descriptor machinery (carries the
# num_m_mmas==2 ping-pong without touching dense MHA / MLA decode).
from msa.msa_sm100_accum import (
    MSASM100TensorAccumulatorSS as SM100TensorAccumulatorSS,
    MSASM100TensorAccumulatorTS as SM100TensorAccumulatorTS,
)

# Host CSR builder + combine for the end-to-end dispatch.
from msa.k2q_csr import build_k2q_csr, balanced_target_q_per_cta
from msa.k2q_csr_device import build_k2q_csr_device, k2q_csr_sizes
from msa.msa_combine import (
    combine_max_splits,
    msa_combine_dispatch,
    real_to_fake,
)

from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

comptime logger = Logger()


# ===-----------------------------------------------------------------------===#
# Q gather4 tile (BF16 D=128 SWIZZLE_128B: box_w 64, 2 col groups).  tile_height
# is BM -- one row per gathered query (qheadperkv == 1); the SMEM destination
# ordering matches the bulk-TMA Q layout so the QK A-descriptor is unchanged.
# ===-----------------------------------------------------------------------===#


comptime Gather4QTile[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    depth: Int,
] = TMATensorTile[
    dtype,
    2,
    tile_shape=IndexList[2](
        BM, _gather4_box_width[dtype, depth, swizzle_mode]()
    ),
    desc_shape=IndexList[2](
        1, _gather4_box_width[dtype, depth, swizzle_mode]()
    ),
]


# ===-----------------------------------------------------------------------===#
# Q-row index staging.  One Q-LOAD WG (4 warps = 128 threads) cooperatively
# stages the BM tile rows.  GQA head-fast packing: tile row `slot` is token
# `slot // group`, q-head `head_kv_idx*group + slot % group`, flat row
# `((q_abs)*head_kv + head_kv_idx)*group + h_in_group` (for group == 1 the
# identity tok==slot, h==0).  A slot past the tile's valid tokens writes -1 so
# gather4 zero-fills it (a zero Q row -> 0 logit; never scattered out).
#
# The packed `qsplit` (q | split<<24) is staged once into `qsplit_smem` so the
# epilogue reuses it without a second GMEM read; `qloc` is decoded from the same
# word (qsplit & 0xFFFFFF).  All head-rows of a token share its qsplit/qpos.
# ===-----------------------------------------------------------------------===#


@always_inline
def _stage_q_rows_to_smem[
    BM: Int, group: Int, causal: Bool, coop_threads: Int
](
    coop_tid: Int,
    qsplit_indices: UnsafePointer[
        mut=False, Int32, _
    ],  # packed q|split<<24, indexed [row_start + tok]
    idx_smem: UnsafePointer[
        mut=True, Int32, _, address_space=AddressSpace.SHARED
    ],
    qsplit_smem: UnsafePointer[
        mut=True, Int32, _, address_space=AddressSpace.SHARED
    ],
    qpos_smem: UnsafePointer[
        mut=True, Int32, _, address_space=AddressSpace.SHARED
    ],
    causal_q_offset: Int32,  # seqlen_k - seqlen_q; qpos = qloc + this (causal)
    row_start: Int,  # CSR token base for this tile (token units)
    count: Int,  # valid tokens in this tile (token units, <= BM // group)
    batch_q_base: Int,  # cu_seqlens_q[batch] (CSR q is batch-local)
    head_q: Int,
    head_kv_idx: Int,  # this CTA's kv head; q head = head_kv_idx*group + h
):
    comptime num_passes = ceildiv(BM, coop_threads)
    comptime for row_pass in range(num_passes):
        var slot = coop_tid + row_pass * coop_threads
        if slot < BM:
            # Head-fast packing: tile row `slot` -> token `slot // group`,
            # head-in-group `slot % group`.  For group == 1 this is the identity
            # (tok == slot, h == 0) so the staged row/qsplit/qpos are unchanged.
            var tok = slot // group if group > 1 else slot
            var h_in_group = slot % group if group > 1 else 0
            if tok < count:
                # CSR holds batch-local q; the global query row is
                # `batch_q_base + qloc`.  Flat Q row packs head fast:
                # (q_abs*head_kv + head_kv_idx)*group + h.  head_q == head_kv*group
                # so for group == 1 this == q_abs*head_q + head_kv_idx.
                var packed = qsplit_indices[row_start + tok]
                var qloc = Int(packed) & 0xFFFFFF
                var q_abs = batch_q_base + qloc
                var head_kv = head_q // group
                idx_smem[slot] = Int32(
                    (q_abs * head_kv + head_kv_idx) * group + h_in_group
                )
                qsplit_smem[slot] = packed
                comptime if causal:
                    # Derive the logical q pos in-register (= qloc + (seqlen_k -
                    # seqlen_q)) instead of an uncoalesced q_positions[q_abs]
                    # scatter; bit-identical to the GMEM value.  All head-rows of
                    # a token share its position.
                    qpos_smem[slot] = Int32(qloc) + causal_q_offset
            else:
                idx_smem[slot] = Int32(-1)
                qsplit_smem[slot] = Int32(-1)
                comptime if causal:
                    qpos_smem[slot] = Int32(-1)


# ===-----------------------------------------------------------------------===#
# Forward kernel
# ===-----------------------------------------------------------------------===#


@__llvm_arg_metadata(q_gather4, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads[True]() + 4 * WARP_SIZE)
    )
)
@__llvm_metadata(`nvvm.minctasm`=SIMDSize(1))
@__name(
    t"sm100_msa_2q_depth{config.depth}_{KVLUTType.dtype}_{output_type}_nqh{config.num_heads}_nkvh{config.num_heads // group}",
)
def _msa_sm100_block_major[
    KVLUTType: MHAOperand,
    output_type: DType,
    config: MHAConfig,
    group: Int,
    swizzle_mode: TensorMapSwizzle,
    causal: Bool,
    has_seqused: Bool,
](
    q_gather4: Gather4QTile[
        KVLUTType.dtype, swizzle_mode, BM=config.block_m(), depth=config.depth
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    o_partial_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    lse_partial_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    kv_lut: KVLUTType,
    # Reverse-CSR + schedule (host-built, uploaded).
    scheduler_metadata: UnsafePointer[Int32, MutAnyOrigin],  # [work, 6]
    # Device work-count (single Int32): grid is launched at a host-known
    # capacity, so CTAs with `block_idx.x >= work_count[0]` are idle and
    # PDL-early-exit (signal dependents, write nothing).
    work_count_ptr: UnsafePointer[Int32, MutAnyOrigin],
    k2q_row_ptr: UnsafePointer[Int32, MutAnyOrigin],  # [head_kv, rows+1]
    qsplit_indices: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [head_kv, total_q*topk]
    cu_seqlens_k: UnsafePointer[Int32, MutAnyOrigin],  # [batch+1]
    cu_seqlens_q: UnsafePointer[Int32, MutAnyOrigin],  # [batch+1]
    # Per-batch effective KV length, read only when `has_seqused`.  Columns
    # >= seqused_k[batch] poison to -inf (mirrors decode's valid_key).
    # Contract: seqused_k[b] <= seqlen_k[b] (used <= allocated).
    seqused_k: UnsafePointer[Int32, MutAnyOrigin],  # [batch]
    total_q: Int,
    total_rows: Int,  # CSR rows per head (k2q_row_ptr stride - 1)
    nnz: Int,  # total_q * topk (qsplit_indices head stride)
    head_q: Int,
    scale: Float32,
):
    """Block-major sparse MHA forward.  One CTA == one (head_kv, CSR row) work
    item from `scheduler_metadata[work_idx]`.  Loads the row's 128-token KV block
    once, then loops `ceil(q_count/BM)` Q-tiles (QK/softmax/PV per tile) against
    it, scattering `O_partial` (+ `LSE_partial`) per (query, split_slot).
    """
    comptime kv_type = KVLUTType.dtype
    comptime assert kv_type == config.dtype
    comptime assert kv_type.is_half_float(), "BF16 only"
    comptime assert (
        KVLUTType.page_size == 0 or KVLUTType.page_size == config.block_n()
    ), "page_size must be 0 or == BN (no intra-block page split)"
    comptime assert group in (1, 2, 4, 8, 16), "qheadperkv in {1,2,4,8,16}"
    comptime assert config.block_m() % group == 0, "BM must be a group multiple"

    comptime num_softmax_threads: Int = config.num_consumer_threads()
    # Per-WG consumer width: the softmax/epilogue tile-work is owned by a group
    # of `consumer_group_size` threads (== `num_softmax_threads` while a single
    # 256-thread WG runs each tile; the ping-pong flip drops it to 128 so two
    # 128-thread WGs alternate Q-tiles).  Per-WG layout/frag/ring/accumulator
    # sizing derives from this knob; cross-WG rendezvous + scatter stay sized by
    # `num_softmax_threads`.
    comptime consumer_group_size = num_softmax_threads // 2
    comptime num_consumer_warps = consumer_group_size // 32

    comptime cta_group = 1
    comptime BM: Int = config.block_m()
    comptime BN: Int = config.block_n()
    comptime BK: Int = config.padded_depth
    comptime depth: Int = config.depth
    comptime padded_depth: Int = config.padded_depth
    comptime MMA_M: Int = 128 if (BM % 128) == 0 else 64
    comptime MMA_N0: Int = BN
    comptime MMA_N1: Int = config.padded_depth
    comptime MMA_K: Int = 16 if kv_type.is_half_float() else 32
    comptime num_row_fragments = consumer_group_size // 128
    comptime row_fragment_size = min(32 // num_row_fragments, BM // 4)
    comptime WM = row_fragment_size
    comptime num_m_blocks_per_warp = BM // (16 * num_consumer_warps)
    comptime assert num_m_blocks_per_warp * 16 == WM
    comptime assert WM * num_consumer_warps == BM
    # One 4-warp WG covers the full BM=128 tile = num_m_blocks_per_warp m-blocks
    # of 16 rows each (1 at the 256-thread seam, 2 in the 128-thread ping-pong).
    # VecP/VecO frag layouts track c_t via this count.
    comptime num_m_mmas = num_m_blocks_per_warp
    comptime num_n_mmas = 1
    comptime pipeline_stages = config.num_pipeline_stages
    var tid = UInt32(thread_idx.x)
    var warp_idx = UInt32(warp_id[broadcast=True]())
    # warp_idx-range dispatch: softmax WG0 = warps 0-3 (tid 0-127), softmax
    # WG1 = warps 4-7 (tid 128-255), producer = warps 8-11 (tid 256-383,
    # combined idx+K+V+Q-gather4), QK MMA = warp 12, K/V load = warps 13/14,
    # PV MMA = warp 15.  QK and PV run on separate warps so the tcgen05 pipe
    # sees back-to-back issues from two PCs (decouples the MMA-output wait).
    comptime softmax_threads = 2 * consumer_group_size  # warps 0-7
    comptime producer_warp0 = softmax_threads // WARP_SIZE  # warp 8
    comptime mma_warp = producer_warp0 + 4  # warp 12
    # Dedicated one-shot K/V load warps: K=warp13, V=warp14; warp15 is the PV
    # MMA warp.
    comptime kload_warp = mma_warp + 1  # warp 13
    comptime vload_warp = mma_warp + 2  # warp 14
    comptime pv_warp = mma_warp + 3  # warp 15 (PV MMA, split off the QK warp)
    # Softmax-WG index (0 or 1) within the two ping-pong WGs; softmax base = tid 0.
    var sm_wg_idx: UInt32 = warp.broadcast(tid // UInt32(consumer_group_size))
    comptime accum_type = get_accum_type[kv_type]()
    comptime max_tmem_cols = 512
    comptime use_p_smem = padded_depth + MMA_N0 // 2 > max_tmem_cols
    comptime num_s = (
        max_tmem_cols
        // MMA_N0 if use_p_smem else (max_tmem_cols - (MMA_N0 // 2) - MMA_N1)
        // MMA_N0
    )
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        kv_type,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=MMA_N0,
        BM=BM,
        BN=BN,
        BK=BK,
        compute_BK=align_up(depth, MMA_K),
        num_softmax_threads=consumer_group_size,
        swizzle_a=swizzle_mode,
        swizzle_b=swizzle_mode,
        transpose_b=True,
        pipeline_stages=num_s,
    ]
    comptime UMMA1_MMA_N = min(MMA_N1, 256) if use_p_smem else MMA_N1
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        kv_type,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=UMMA1_MMA_N,
        BM=BM,
        BN=MMA_N1,
        BK=BN,
        num_softmax_threads=consumer_group_size,
        swizzle_b=swizzle_mode,
        transpose_b=False,
    ]

    # Q rides a `q_stage`-slot ring so the next group's gather4 overlaps the
    # current group's QK/softmax/PV; K/V are loaded ONCE and pinned (the CTA
    # loops num_q_groups Q-tiles against the one resident block).
    comptime q_stage = 2
    # O rides a 2-stage TMEM ring so PV(N)+write_output(N) overlap
    # softmax(N+1)+QK(N+2).
    comptime o_stage = 2
    comptime q_or_out_kv_elems = 1
    comptime q_tile_elems = BM * padded_depth * q_or_out_kv_elems
    comptime q_smem_size = q_tile_elems * q_stage
    q_smem = external_memory[
        Scalar[kv_type],
        address_space=AddressSpace.SHARED,
        alignment=128,
        name="msa_2q_dynamic_shared_memory",
    ]()

    comptime p_smem_elems = BM * MMA_N0 if use_p_smem else 0
    p_smem = q_smem + q_smem_size

    comptime kv_smem_size = config.kv_smem_size(True)
    kv_smem = q_smem + q_smem_size + p_smem_elems

    comptime p_frag_size = BM * MMA_N0 // (
        consumer_group_size * num_m_blocks_per_warp
    )
    comptime o_frag_size = BM * MMA_N1 // (
        consumer_group_size * num_m_blocks_per_warp
    )
    comptime frag_simdwidth = 2

    comptime num_row_blocks_per_mma = 2
    comptime element_layout = Layout.row_major(1, frag_simdwidth)
    comptime vec_output_row_shape = IntTuple(num_row_blocks_per_mma, num_m_mmas)
    comptime p_vec_output_layout = Layout(
        IntTuple(
            vec_output_row_shape,
            IntTuple(
                p_frag_size // (num_row_blocks_per_mma * frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(frag_simdwidth, p_frag_size),
            IntTuple(
                num_row_blocks_per_mma * frag_simdwidth,
                num_m_mmas * p_frag_size,
            ),
        ),
    )
    comptime o_vec_output_layout = Layout(
        IntTuple(
            vec_output_row_shape,
            IntTuple(
                o_frag_size // (num_row_blocks_per_mma * frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(frag_simdwidth, o_frag_size),
            IntTuple(
                num_row_blocks_per_mma * frag_simdwidth,
                num_m_mmas * o_frag_size,
            ),
        ),
    )
    comptime num_rows_per_warp = p_vec_output_layout[0].size()
    comptime num_cols_output = o_vec_output_layout[1].size()

    comptime mma_thread_layout = Layout.row_major(8, 4)

    # --- pipeline mbars (KV portion identical to dense `_mha_sm100`) ---
    produced_mbar_kv = (kv_smem + kv_smem_size).bitcast[SharedMemBarrier]()
    producer_mbar_kv = produced_mbar_kv + pipeline_stages
    mma_mbar = producer_mbar_kv + pipeline_stages
    umma_0 = UMMA0Type(mma_mbar.as_unsafe_any_origin())
    umma_1_ts = UMMA1Type(mma_mbar.as_unsafe_any_origin() + 2 * num_s)
    ptr_tmem_addr = (mma_mbar + 2 * num_s + 2).bitcast[UInt32]()
    # Q ring (after the tmem-addr slot): per slot a `q_produced` (load -> QK) and
    # a `q_consumed` (softmax-epilogue -> load) barrier, then the per-slot idx /
    # qpos staging.  The idx-ready handshake is an intra-WG named_barrier(9): the
    # 4-warp Q-LOAD WG stages idx then gathers, so a block-local rendezvous works.
    q_produced_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    q_consumed_mbar = q_produced_mbar + q_stage
    idx_smem = (q_consumed_mbar + q_stage).bitcast[Int32]()
    qpos_smem = idx_smem + BM * q_stage
    # Packed qsplit (q|split<<24) staged once per slot in the Q-load and reused by
    # the epilogue -- no second GMEM read of the index.
    qsplit_smem = qpos_smem + BM * q_stage
    # 2-stage P ring (overlays the upper halves of S0/S1 in TMEM): p_full =
    # softmax wrote P[s] -> PV may consume; p_empty = PV done with P[s] -> softmax
    # may overwrite.  p_full arrived by every softmax thread (one per group);
    # p_empty arrived by the PV MMA's tcgen05.commit (one async arrival).
    p_full_mbar = (qsplit_smem + BM * q_stage).bitcast[SharedMemBarrier]()
    p_empty_mbar = p_full_mbar + num_s
    # 2-stage O ring (mirror of the P ring, roles swapped): o_full = PV wrote
    # O[s] -> softmax may read it; o_empty = softmax done reading O[s] -> PV may
    # overwrite.  o_full arrived by the PV MMA's tcgen05.commit (one async
    # arrival); o_empty arrived by every softmax thread after copy_to(O).
    o_full_mbar = p_empty_mbar + num_s
    o_empty_mbar = o_full_mbar + o_stage
    # Causal qpos crosses Q-LOAD WG -> softmax WG with no dedicated mbar: the
    # write->read is ordered transitively (q_produced -> QK -> umma_0 -> softmax),
    # and q_consumed_mbar covers the WAR before the slot is re-staged.

    # Asymmetric reg identity (B200, minctasm=1, 512 thr): 256*softmax +
    # 128*producer + 128*spare = 65536.  256*176 + 128*112 + 128*48 = 65536.
    comptime num_producer_regs = 112
    comptime num_softmax_regs = 176
    comptime num_spare_regs = 48

    # PDL-safe idle-CTA early exit (mirror MLA `pdl_early_exit`).  The grid is
    # launched at a host-known capacity >= the on-device work_count, so trailing
    # CTAs have no work item.  They must still signal dependents or combine
    # hangs (CUDA_ERROR_ILLEGAL_ADDRESS).  Read the device work_count BEFORE the
    # first barrier / any scheduler_metadata access; idle CTAs write nothing to
    # O_partial/LSE_partial (combine reads only the valid slots -> bit-same).
    if Int(block_idx.x) >= Int(work_count_ptr[0]):
        barrier()
        launch_dependent_grids()
        return

    if tid == 0:
        comptime for i in range(pipeline_stages):
            produced_mbar_kv[i].init(1)
            producer_mbar_kv[i].init(Int32(num_softmax_threads))
        comptime for s in range(q_stage):
            # q_produced: Q-LOAD WG's gather4 TMAs (tx-count) -> QK warp.
            q_produced_mbar[s].init(1)
            # q_consumed: one softmax ping-pong WG releases the slot -> Q-LOAD
            # WG warp-0 (the single waiter now) before its Q is overwritten.
            q_consumed_mbar[s].init(Int32(consumer_group_size))
        comptime for s in range(num_s):
            # p_full: all softmax threads write their P fragment then arrive.
            p_full_mbar[s].init(Int32(consumer_group_size))
            # p_empty: single async arrival from the PV MMA's tcgen05.commit.
            p_empty_mbar[s].init(1)
        comptime for s in range(o_stage):
            # o_full: single async arrival from the PV MMA's tcgen05.commit.
            o_full_mbar[s].init(1)
            # o_empty: all softmax threads arrive after reading O out of TMEM.
            o_empty_mbar[s].init(Int32(consumer_group_size))
        umma_0.init()
        umma_1_ts.init()

    # ---- Work item (head_kv, CSR row) from scheduler_metadata[grid.x] -------
    var work_idx = Int(block_idx.x)
    var meta = work_idx * 6
    var head_kv_idx = Int(scheduler_metadata[meta + 0])
    var row_linear = Int(scheduler_metadata[meta + 1])
    var q_begin = Int(scheduler_metadata[meta + 2])
    var q_count = Int(scheduler_metadata[meta + 3])
    var batch = Int(scheduler_metadata[meta + 4])
    var kv_block_idx = Int(scheduler_metadata[meta + 5])

    # CSR row slice for this (head, row): the work item owns q_count TOKENS
    # (<= q_per_cta, load-balanced) starting at q_begin.  The CTA loads its KV
    # block ONCE and loops num_q_groups Q-tiles against it.  GQA packs `group`
    # q-heads per token (head fast), so one BM-row tile holds BM // group tokens
    # and num_q_groups = ceil(q_count / (BM // group)) (== ceil(q_count / BM) for
    # group == 1).
    comptime tokens_per_group = BM // group
    var rp_base = head_kv_idx * (total_rows + 1)
    var row_base = Int(k2q_row_ptr[rp_base + row_linear]) + q_begin
    var num_q_groups = ceildiv(q_count, tokens_per_group)
    var q_idx_base = head_kv_idx * nnz  # qsplit_indices head stride
    var batch_k_base = Int(cu_seqlens_k[batch])
    # KV block base row.  Flat/batch-packed operand folds the batch into the row
    # (cu_seqlens_k[b] + blk*BN, batch_idx 0); the paged operand passes the
    # batch-local row and lets the page table resolve it via
    # lookup_table[batch, blk] (mirrors msa_1q decode).
    var base_kv_row: UInt32 = UInt32(batch_k_base + kv_block_idx * BN)
    var batch_q_base = Int(cu_seqlens_q[batch])
    # This block's logical KV start position (= kv_block_idx*BN, batch-local).
    var kv_block_start: Int = kv_block_idx * BN
    # Tokens of this block actually present in the batch's K (tail clamp).
    var seqlen_k = Int(cu_seqlens_k[batch + 1]) - batch_k_base
    # Paged operand: batch-local row + real batch index (page table resolves it).
    var populate_batch: UInt32 = UInt32(0)
    comptime if KVLUTType.page_size != 0:
        base_kv_row = UInt32(kv_block_start)
        populate_batch = UInt32(batch)
    # Causal shift seqlen_k - seqlen_q (per-CTA scalar); used to derive
    # qpos = qloc + this without the per-row q_positions GMEM scatter.
    var causal_q_offset: Int32 = 0
    comptime if causal:
        # Anchor the diagonal on the logical KV length: raw seqused when present,
        # else dense (no min(dense, seqused)).  Without seqused this is the dense
        # length, so the offset is byte-identical to HEAD.
        var eff_seqlen_k = seqlen_k
        comptime if has_seqused:
            eff_seqlen_k = Int(seqused_k[batch])
        causal_q_offset = Int32(eff_seqlen_k) - (
            Int32(cu_seqlens_q[batch + 1]) - Int32(batch_q_base)
        )

    comptime stage_sz = BN * padded_depth
    comptime kv_bytes = BN * padded_depth * size_of[kv_type]()
    comptime q_bytes = BM * padded_depth * size_of[kv_type]()
    # Mask alignment for the contiguous KV bulk-TMA `populate` (whole-block).
    comptime base_alignment: Int = BN

    barrier()

    # PDL: this barrier is the last divergence-free point before the warp-spec
    # split (warps early-return below), so signal dependents (combine) here --
    # mirrors dense `_mha_sm100`.  A producer CTA that skipped this would hang
    # the waiting combine kernel.  No-op on non-SM90+.
    launch_dependent_grids()

    # Dedicated one-shot K/V load warps 13/14 (tid 416-479): K ->
    # produced_mbar_kv[0] (stage 0), V -> produced_mbar_kv[1] (stage 1),
    # both phase 0 (never reloaded).  Both fall through the block barrier + PDL
    # above, drop to the 48-reg floor (a one-shot TMA issue needs no extra regs),
    # and never join the alloc rendezvous or the rings.  Warp 15 (PV MMA, below)
    # DOES join.  expect_bytes is a single-thread mbar mutation -> gate it (+ the
    # TMA) on the elected lane.
    if warp_idx == UInt32(kload_warp):
        warpgroup_reg_dealloc[num_spare_regs]()
        var paged_rows = kv_lut.populate[BN, base_alignment](
            populate_batch, base_kv_row
        )
        var e = elect()
        if e != 0:
            produced_mbar_kv[0].expect_bytes(Int32(kv_bytes))
            paged_rows.tma_copy_k[needs_partial=False](
                k_tma_op,
                kv_smem,
                produced_mbar_kv[0],
                kv_head_idx=UInt32(head_kv_idx),
                elect=e,
            )
        return
    if warp_idx == UInt32(vload_warp):
        warpgroup_reg_dealloc[num_spare_regs]()
        var paged_rows = kv_lut.populate[BN, base_alignment](
            populate_batch, base_kv_row
        )
        var e = elect()
        if e != 0:
            produced_mbar_kv[1].expect_bytes(Int32(kv_bytes))
            paged_rows.tma_copy_v[needs_partial=False](
                v_tma_op,
                kv_smem + UInt32(stage_sz),
                produced_mbar_kv[1],
                kv_head_idx=UInt32(head_kv_idx),
                elect=e,
            )
        return

    # Per-group VALID ROW count for group qg (last group may be partial).  A tile
    # of BM rows holds tokens_per_group tokens, each expanded to `group` head-rows;
    # the valid rows are min(BM, valid_tokens * group).  For group == 1 this is
    # min(BM, q_count - qg*BM), the prior single-head form.
    @parameter
    @always_inline
    def group_count(qg: Int) -> Int:
        return min(BM, (q_count - qg * tokens_per_group) * group)

    # ====================================================================== #
    # Q-LOAD WG: warps 8-11.  K/V are one-shot bulk-TMA'd into stages 0/1 by the
    # dedicated load warps 13/14, so this WG does ONLY Q.  Per Q-group all 4
    # warps cooperatively stage the group's q-row indices, then ROUND-ROBIN
    # gather4 the group's Q into the ring slot -- each warp owns BM/(4*4) chunks,
    # so no single-thread gather4 serializer.  An intra-WG named_barrier(9, 128
    # thr) carries the idx-ready handshake: warp-0 acquires the slot, rendezvous,
    # all 4 stage idx, rendezvous, all 4 gather, rendezvous before slot reuse.
    # ====================================================================== #
    if warp_idx >= UInt32(producer_warp0) and warp_idx < UInt32(mma_warp):
        warpgroup_reg_dealloc[num_producer_regs]()
        comptime num_q_load_warps = 4
        comptime num_q_load_threads = num_q_load_warps * WARP_SIZE
        var warp_in_wg = Int(warp_idx) - producer_warp0  # 0..3
        var coop_tid = Int(tid) - producer_warp0 * WARP_SIZE  # 0..127

        var s = 0
        var q_phase: UInt32 = 0  # phase of the slot's consumed barrier
        for qg in range(num_q_groups):
            # warp-0 acquires slot `s`: after the first wrap, wait for the softmax
            # WG to release the slot's prior O-staging (one waiter now).  The
            # release is the (qg//q_stage)-th parity flip, so wait to LEAVE the
            # prior parity q_phase ^ 1.
            if warp_in_wg == 0 and qg >= q_stage:
                q_consumed_mbar[s].wait(q_phase ^ 1)
            # Rendezvous: the other 3 warps must not stage slot `s` until warp-0
            # has confirmed it free.
            named_barrier[Int32(num_q_load_threads)](Int32(9))

            # All 128 threads cooperatively stage this group's q rows (+ qpos).
            # row_start / count are in TOKEN units (BM rows == tokens_per_group
            # tokens * group head-rows); the helper expands head fast.
            _stage_q_rows_to_smem[
                BM, group, causal, coop_threads=num_q_load_threads
            ](
                coop_tid,
                qsplit_indices + q_idx_base,
                idx_smem + s * BM,
                qsplit_smem + s * BM,
                qpos_smem + s * BM,
                causal_q_offset,
                row_base + qg * tokens_per_group,
                min(tokens_per_group, q_count - qg * tokens_per_group),
                batch_q_base,
                head_q,
                head_kv_idx,
            )
            # q_produced arrive (single-thread mbar mutation): warp-0 lane-0 sets
            # the gather4 tx-count.  Issued AFTER staging so this arrive RELEASES
            # the qpos write to the QK -> softmax chain (softmax apply_mask reads
            # qpos after waiting umma_0); setting it before staging (as a slot-
            # acquire) would leave qpos unreleased -> a cross-WG read hazard.
            if warp_in_wg == 0 and elect() != 0:
                q_produced_mbar[s].expect_bytes(Int32(q_bytes))
            # Rendezvous: idx staged + expect_bytes set before any gather4 reads
            # idx / arrives bytes to q_produced.
            named_barrier[Int32(num_q_load_threads)](Int32(9))

            # Round-robin gather4: each warp's lane-0 issues its strided 4-row
            # chunks (c = g*4 + warp_in_wg, elem_off = cg*BM*box_w + c*4*box_w so
            # the QK A-descriptor sees the same SMEM layout) -- each warp owns
            # BM/(4*4)=8 chunks instead of one thread serializing all 32.
            # q_gather4 must be the grid-constant arg here (a by-value copy would
            # stage the TMA descriptor off the constant bank), so the loop is
            # inline.
            var e = elect()
            if e != 0:
                comptime g4_box_w = _gather4_box_width[
                    kv_type, depth, swizzle_mode
                ]()
                comptime g4_col_groups = ceildiv(depth, g4_box_w)
                comptime g4_per_warp = (BM // 4) // num_q_load_warps
                comptime assert g4_per_warp * num_q_load_warps * 4 == BM
                var g4_desc = UnsafePointer(to=q_gather4.descriptor).bitcast[
                    NoneType
                ]()
                var g4_mbar = q_produced_mbar[s].unsafe_ptr()
                var g4_idx = idx_smem + s * BM
                var g4_smem = q_smem + s * q_tile_elems
                comptime for g in range(g4_per_warp):
                    var c = g * num_q_load_warps + warp_in_wg
                    var g4_row = c * 4
                    comptime for cg in range(g4_col_groups):
                        var elem_off = cg * BM * g4_box_w + c * 4 * g4_box_w
                        # Fire-and-forget L2 prefetch of the NEXT col group's same
                        # 4 rows.  No SMEM dst / no mbar -> warms L2 only, output
                        # unchanged.
                        comptime if cg + 1 < g4_col_groups:
                            inlined_assembly[
                                (
                                    "cp.async.bulk.prefetch.tensor.2d.L2.global"
                                    ".tile::gather4.L2::cache_hint"
                                    " [$0, {$1, $2, $3, $4, $5}], $6;"
                                ),
                                NoneType,
                                constraints="l,r,r,r,r,r,l",
                                has_side_effect=True,
                            ](
                                g4_desc,
                                Int32((cg + 1) * g4_box_w),
                                g4_idx[g4_row + 0],
                                g4_idx[g4_row + 1],
                                g4_idx[g4_row + 2],
                                g4_idx[g4_row + 3],
                                Int64(CacheEviction.EVICT_LAST._value),
                            )
                        cp_async_bulk_tensor_2d_gather4[
                            cta_group=1,
                            # EVICT_LAST: Q rows are re-fetched across this CTA's
                            # q-groups, so keep them L2-resident.  The next col
                            # group is L2-prefetched just above.
                            eviction_policy=CacheEviction.EVICT_LAST,
                        ](
                            (g4_smem + elem_off).mut_cast[True](),
                            g4_desc,
                            g4_mbar,
                            Int32(cg * g4_box_w),
                            g4_idx[g4_row + 0],
                            g4_idx[g4_row + 1],
                            g4_idx[g4_row + 2],
                            g4_idx[g4_row + 3],
                        )
            # Rendezvous: hold the slot until every warp issued its gather4
            # (none may re-stage idx for the next reuse of `s` before all read).
            named_barrier[Int32(num_q_load_threads)](Int32(9))
            s += 1
            if s == q_stage:
                s = 0
                q_phase ^= 1

    elif warp_idx == UInt32(mma_warp):  # QK MMA (warp 12, owns TMEM alloc)
        # MMA warps sit in the 48-reg floor pool (= ref: MMA/K-load/V-load = 48
        # each).  Identity: 256*176 (softmax) + 128*112 (producer warps 8-11) +
        # 128*48 (QK warp 12 + K/V-load 13/14 + PV warp 15) = 65536.  Giving an
        # MMA warp 112 would shift it out of the 48 pool and overflow the 65536
        # block budget by 2048 -> setmaxnreg.inc stall/hang.
        warpgroup_reg_dealloc[num_spare_regs]()
        tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)
        # tcgen05-alloc rendezvous: 256 softmax + 32 (QK) + 32 (PV) = 320.
        # Dedicated barrier id 8 (not the default 0) so the counted barrier
        # never aliases the full-block `barrier()` (id 0) or the scatter.
        named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)](Int32(8))
        if tid != UInt32(mma_warp * WARP_SIZE):
            return
        var tmem_addr: UInt32 = ptr_tmem_addr[0]
        var s_tmem: UInt32
        comptime if use_p_smem:
            s_tmem = tmem_addr + UInt32(1 << 20)  # bank 1
        else:
            s_tmem = tmem_addr
        s_accumulator = UMMA0Type.c_t(s_tmem)
        # K resident in stage 0 (phase 0, never reloaded).  Wait K before the QK
        # prologue (V is the PV warp's concern).
        produced_mbar_kv[0].wait(UInt32(0))

        # QK(qg): Q[qg%q_stage] x K -> S[qg%q_stage].  P overlays S[s]'s upper
        # half, so QK(qg) reusing S[s] would clobber P[s] from group qg-num_s
        # while PV may still read it -> gate the S-write on PV releasing the
        # slot (p_empty is a cross-warp edge: PV arrives, QK waits).
        @parameter
        @always_inline
        def issue_qk(qg: Int):
            var s = qg % q_stage
            var q_phase: UInt32 = UInt32((qg // q_stage) & 1)
            qk_desc = UMMA0Type.mma_descriptors(
                q_smem.as_unsafe_any_origin() + s * q_tile_elems,
                kv_smem.as_unsafe_any_origin(),
            )
            var q_desc = qk_desc.get_a()
            var k_desc = qk_desc.get_b()
            umma_0.wait_for_tmem()
            q_produced_mbar[s].wait(q_phase)
            if qg >= num_s:
                p_empty_mbar[s].wait(UInt32((qg // num_s - 1) & 1))
            umma_0.mma(
                rebind[UMMA0Type.a_t](q_desc),
                rebind[UMMA0Type.b_t](k_desc),
                s_accumulator,
                0,
            )

        # QK runs its own loop over all Q-groups; the p_empty mbar (arrived by
        # the PV warp) paces slot reuse -- no shared interleaved schedule.
        for qg in range(num_q_groups):
            issue_qk(qg)
        # PV warp owns the dealloc (it is the last warp to touch TMEM).

    elif warp_idx == UInt32(pv_warp):  # PV MMA (warp 15, reads shared TMEM)
        warpgroup_reg_dealloc[num_spare_regs]()
        # Join the same alloc rendezvous as QK + softmax so TMEM is allocated
        # before reading the broadcast base.  PV does NOT alloc.
        named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)](Int32(8))
        if tid != UInt32(pv_warp * WARP_SIZE):
            return
        var tmem_addr: UInt32 = ptr_tmem_addr[0]
        comptime offset_bytes_per = BN * config.padded_depth * size_of[
            kv_type
        ]()

        # PV(pv_qi): P[pv_qi%num_s] x V -> O[pv_qi%o_stage].
        @parameter
        @always_inline
        def issue_pv(pv_qi: Int):
            var pv_s = pv_qi % num_s
            var pv_phase: UInt32 = UInt32((pv_qi // num_s) & 1)
            var pv_o = pv_qi % o_stage
            var pv_o_phase: UInt32 = UInt32((pv_qi // o_stage) & 1)
            var v_desc = UMMA1Type.b_mma_descriptor(
                kv_smem.as_unsafe_any_origin()
            )
            var v = v_desc + Int(UInt32(offset_bytes_per) * UInt32(1))
            var p_tmem = tmem_addr + UInt32(pv_s * MMA_N0 + MMA_N0 // 2)
            p_desc = UMMA1Type.a_mma_descriptor(p_tmem)
            var o_tmem = tmem_addr + UInt32(MMA_N0 * num_s + pv_o * MMA_N1)
            output_accumulator = UMMA1Type.c_t(o_tmem)
            # Wait for softmax to release O[pv_o] (read prior group's O out of
            # TMEM) before reusing the slot.
            if pv_qi >= o_stage:
                o_empty_mbar[pv_o].wait(pv_o_phase ^ 1)
            # Wait for softmax to write P[pv_s] (P ring full).
            p_full_mbar[pv_s].wait(pv_phase)
            umma_1_ts.mma(
                rebind[UMMA1Type.a_t](p_desc),
                rebind[UMMA1Type.b_t](v),
                output_accumulator,
                UInt32(0),
            )
            # Release P[pv_s] AFTER the MMA's TMEM read completes; O[pv_o] is
            # produced -> softmax may read it.  Both tcgen05.commits are async-
            # ordered behind this warp's MMA.
            mma_arrive(p_empty_mbar + pv_s)
            mma_arrive(o_full_mbar + pv_o)

        # V resident in stage 1 (phase 0, never reloaded).  Wait V before the
        # first PV.
        produced_mbar_kv[1].wait(UInt32(0))
        # PV runs its own loop over all Q-groups; the p_full / o_empty mbars
        # (arrived by softmax) pace it -- QK and PV issue from two PCs so the
        # tcgen05 pipe sees back-to-back issues, decoupling the MMA-output wait.
        for pv_qi in range(num_q_groups):
            issue_pv(pv_qi)

        # PV is the last warp to read/write TMEM (PV(qg) trails QK(qg) through
        # softmax), so it owns the dealloc -- no extra cross-warp fence needed.
        tcgen05_release_allocation_lock[cta_group]()
        tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)

    else:  # softmax warp-group (loop over Q-groups, single tile each)
        warpgroup_reg_alloc[num_softmax_regs]()

        # Ping-pong stage: softmax WG0 (warps 0-3) -> stage 0 (even qg), WG1
        # (warps 4-7) -> stage 1 (odd qg).  Each WG pins its umma_0 ring slot ==
        # stage and only ever drives that slot's tmem-empty / mma-full barriers.
        var stage: Int = Int(sm_wg_idx)
        # Pre-arrive this WG's tmem-empty slot so the first QK mma for `stage`
        # can proceed.
        _ = umma_0.mbar[UInt32(num_s + stage)].arrive()

        # WG-local warp id: each ping-pong WG (consumer_group_size threads) owns
        # warps 0..num_consumer_warps-1 of its own tile.  At the 256-thread seam
        # this is the plain tid//32 (one 8-warp WG).
        var sm_warp_id = UInt32(warp_id[broadcast=True]()) % UInt32(
            num_consumer_warps
        )
        var lane = UInt32(lane_id())
        var warp_y: UInt32 = sm_warp_id
        # The datapath-half warp_y remap only applies when a single >4-warp WG
        # splits one M=128 accumulator by datapath half (the seam); each
        # ping-pong WG owns the full tile via num_m_mmas blocks, so disable it.
        comptime if num_consumer_warps > 4:
            warp_y = 2 * (warp_y % 4) + (warp_y // 4)
        comptime assert num_consumer_warps == 4 or num_consumer_warps == 8
        mask_warp_row = warp_y * UInt32(WM)
        var scale_log2e: Scalar[accum_type] = scale.cast[accum_type]() * log2e

        rowmax = LayoutTensor[
            UMMA0Type.accum_t,
            Layout.row_major(num_rows_per_warp),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()
        rowsum = LayoutTensor[
            UMMA0Type.accum_t,
            Layout.row_major(num_rows_per_warp),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()
        comptime VecPType = LayoutTensor[
            accum_type,
            p_vec_output_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
            element_layout=element_layout,
        ]
        comptime VecOType = LayoutTensor[
            accum_type,
            o_vec_output_layout,
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
            element_layout=element_layout,
        ]

        p_reg_tile = UMMA0Type.c_t.allocate_register_tile()
        output_reg_tile = UMMA1Type.c_t.allocate_register_tile()

        @parameter
        @always_inline
        def vectorize_p_reg_tile(out result: VecPType):
            result = {p_reg_tile.ptr}

        @parameter
        @always_inline
        def vectorize_o_reg_tile(out result: VecOType):
            result = {output_reg_tile.ptr}

        # ---- Mask: poison tail (block tokens past seqlen_k / past `count`
        # rows) + diagonal causal to -inf on RAW S (scale folds into softmax).
        # Fragment->row map mirrors the verified dense prefill `_apply_mask`:
        # register (i, m_mma) of (warp_y, lane) owns M-row
        # `warp_y*WM + m_mma*16 + i*8 + lane//4` and column `frag_col + j*8 + c`.
        # One 4-warp WG covers BM=128 = num_m_mmas 16-row m-blocks (1 at the seam,
        # 2 in the ping-pong); m_mma steps the 16-row block, i the 8-row sub-block.
        @parameter
        @always_inline
        def apply_mask(
            q_pos_row: UnsafePointer[
                mut=True, Int32, _, address_space=AddressSpace.SHARED
            ],
            count_g: Int,
        ):
            var p_vec = vectorize_p_reg_tile()
            var frag_col: UInt32 = (
                lane * UInt32(frag_simdwidth) % UInt32(MMA_N0)
            ) % 8
            var fragment_row: UInt32 = lane // UInt32(4)
            var neg_inf_vec = SIMD[accum_type, frag_simdwidth](
                min_or_neg_inf[accum_type]()
            )
            # Lane offsets for the frag_simdwidth-wide column group (0,1,...).
            var c_lanes = SIMD[DType.int32, frag_simdwidth](0)
            comptime for c in range(frag_simdwidth):
                c_lanes[c] = Int32(c)
            comptime for m_mma in range(num_m_mmas):
                comptime for i in range(num_row_blocks_per_mma):
                    var m_row = (
                        mask_warp_row
                        + UInt32(m_mma * 16)
                        + UInt32(i * 8)
                        + fragment_row
                    )
                    var q_valid = Int(m_row) < count_g
                    # Collapse the per-element poison compares into one per-row
                    # column limit: poison iff slot >= col_limit (matches ref
                    # r2p_bitmask_below).  Tail mask: slot >= seqlen_k -
                    # kv_block_start.  Invalid row: every slot poisoned
                    # (col_limit 0).  Causal: slot > qp - kv_block_start.
                    # The tail is the logical KV length minus block start, raw
                    # seqused when present (no min(dense, seqused)).  The loaded
                    # tile is BN wide, so the poison loop caps cols at the tile.
                    var col_limit = Int32(seqlen_k - kv_block_start)
                    comptime if has_seqused:
                        # Blocks past seqused get col_limit <= 0 (whole tile
                        # poisoned).
                        col_limit = Int32(
                            Int(seqused_k[batch]) - kv_block_start
                        )
                    if not q_valid:
                        col_limit = 0
                    comptime if causal:
                        if q_valid:
                            var qp = q_pos_row[Int(m_row)]
                            var causal_limit = qp - Int32(kv_block_start) + 1
                            col_limit = min(col_limit, causal_limit)
                    comptime for n_mma in range(num_n_mmas):
                        comptime for j in range(MMA_N0 // 8):
                            var col0 = (
                                frag_col
                                + UInt32(n_mma * MMA_N0)
                                + UInt32(j * 8)
                            )
                            # Raw S (no *scale): the scale folds into the
                            # softmax FFMA2 post-rowmax.
                            var v = rebind[SIMD[accum_type, frag_simdwidth]](
                                p_vec[i, m_mma, j, n_mma]
                            )
                            # Vector select over the frag_simdwidth lanes: one
                            # FSEL on the packed group, no scalar per-element
                            # branch.
                            var slots = (
                                SIMD[DType.int32, frag_simdwidth](Int32(col0))
                                + c_lanes
                            )
                            var poison = slots.ge(
                                SIMD[DType.int32, frag_simdwidth](col_limit)
                            )
                            p_vec[i, m_mma, j, n_mma] = rebind[
                                p_vec.element_type
                            ](poison.select(neg_inf_vec, v))

        # ---- Scatter epilogue: O_partial[flat_row, :], LSE_partial[flat] ----
        @parameter
        @always_inline
        def write_output(
            stage: Int,
            rowsum_inv: type_of(rowsum),
            rowmax_v: type_of(rowmax),
            vout: VecOType,
            count_g: Int,
            qsplit_ring: UnsafePointer[
                mut=False, Int32, _, address_space=AddressSpace.SHARED
            ],
            o_smem_base: UnsafePointer[
                mut=True, Scalar[kv_type], _, address_space=AddressSpace.SHARED
            ],
        ):
            # Rescale O by 1/rowsum, packed (one FMUL2 per col-pair) instead of a
            # scalar mul per element.
            comptime for row in range(num_rows_per_warp):
                var rs_inv = SIMD[DType.float32, 2](
                    rowsum_inv[row][0].cast[DType.float32]()
                )
                comptime for col in range(num_cols_output):
                    vout[row, col] = rebind[vout.element_type](
                        mul_ftz(
                            rebind[SIMD[DType.float32, 2]](vout[row, col]),
                            rs_inv,
                        )
                    )

            # No SMEM round-trip: scale is in-register above, write the O frags
            # straight to GMEM (regs->GMEM).  Each thread owns the tcgen05 C-frag
            # element (i,m_mma) at M-row warp_y*WM + m_mma*16 + i*8 + lane//4 and a
            # set of 2-wide depth groups at cols frag_col + col*8 (frag_col=
            # (lane*2)%8) -- same map apply_mask uses.  These cols are strided, so
            # a natural store is a scatter (L1TEX-bound, ~8 sectors/request).
            # Instead pack 4 adjacent pairs (8 BF16 = 16B) and STG.128 at the FAKE
            # column real_to_fake(natural): a lane-group's 32 owned cols map to 32
            # CONTIGUOUS fake cols, so 4 coalesced STG.128 per row.  combine reads
            # back at the same fake position (real_to_fake), keeping O natural.
            comptime assert (
                num_cols_output % 4 == 0
            ), "STG.128 packs 4 col-pairs"
            var frag_col = Int((lane * UInt32(frag_simdwidth)) % UInt32(8))
            # Decode the row's (qloc, split) from the packed `qsplit` the Q-load
            # already staged into this group's ring slot -- no second GMEM read.
            # `r` is the in-tile row == ring slot.  `flat` is the LSE_partial row;
            # O_partial row_base = flat*depth.
            comptime for row in range(num_rows_per_warp):
                comptime i = row % num_row_blocks_per_mma
                comptime m_mma = row // num_row_blocks_per_mma
                var r = Int(
                    mask_warp_row
                    + UInt32(m_mma * 16)
                    + UInt32(i * 8)
                    + (lane // UInt32(4))
                )
                if r < count_g:
                    var packed = qsplit_ring[r]
                    var qloc = Int(packed) & 0xFFFFFF
                    var split_slot = (Int(packed) >> 24) & 0xFF
                    var q_abs = batch_q_base + qloc
                    # Head-fast GQA: tile row r -> q-head head_kv_idx*group +
                    # r % group (== head_kv_idx for group == 1).  Flat O/LSE row
                    # is q_abs*head_q + that head (head_q == head_kv*group).
                    var head_idx = (
                        head_kv_idx * group + r % group if group
                        > 1 else head_kv_idx
                    )
                    var flat = (
                        split_slot * total_q * head_q
                        + q_abs * head_q
                        + head_idx
                    )
                    var row_base = flat * depth
                    comptime for g in range(num_cols_output // 4):
                        comptime c0 = 4 * g
                        var p0 = vout[row, c0 + 0].cast[output_type]()
                        var p1 = vout[row, c0 + 1].cast[output_type]()
                        var p2 = vout[row, c0 + 2].cast[output_type]()
                        var p3 = vout[row, c0 + 3].cast[output_type]()
                        var packed16 = (
                            rebind[SIMD[output_type, 2]](p0)
                            .join(rebind[SIMD[output_type, 2]](p1))
                            .join(
                                rebind[SIMD[output_type, 2]](p2).join(
                                    rebind[SIMD[output_type, 2]](p3)
                                )
                            )
                        )
                        var fake = real_to_fake(frag_col + c0 * 8)
                        (o_partial_ptr + row_base + fake).store[alignment=16](
                            packed16
                        )

                    # LSE_partial[flat] in the NATURAL log domain (what a softmax
                    # combine consumes).  rowmax is RAW S; the kernel does base-2
                    # softmax exp2(S*scale_log2e - m*scale_log2e), so the natural
                    # LSE = (m*scale_log2e + log2(sum2)) * LN2 (log2, not log).
                    # Bad sum (fully-masked row) -> -inf so combine skips this
                    # slot.  Only lane%4==0 writes.
                    if lane % UInt32(4) == 0:
                        var sum2 = recip(rowsum_inv[row])[0].cast[
                            DType.float32
                        ]()
                        var m_raw = rowmax_v[row][0].cast[DType.float32]()
                        var lse = (
                            m_raw * Float32(scale_log2e) + log2(sum2)
                        ) * (Float32(1) / Float32(log2e))
                        lse_partial_ptr[flat] = (
                            lse if (sum2 > Float32(0))
                            and isfinite(sum2) else min_or_neg_inf[
                                DType.float32
                            ]()
                        )

        # tcgen05-alloc rendezvous: 256 softmax + 32 (QK) + 32 (PV) = 320
        # (dedicated barrier id 8, see the MMA-warp side).
        named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)](Int32(8))
        var tmem_addr = ptr_tmem_addr[0]
        var s_tmem: UInt32
        var o_tmem: UInt32
        comptime if use_p_smem:
            o_tmem = tmem_addr  # bank 0
            s_tmem = tmem_addr + UInt32(1 << 20)  # bank 1
        else:
            # The datapath-half bank shift only applies when a single >4-warp WG
            # splits one accumulator; each ping-pong WG reads the full M=128
            # tile from bank 0 (no shift).
            comptime if num_consumer_warps > 4:
                if sm_wg_idx != 0:
                    tmem_addr += 1 << 20
            s_tmem = tmem_addr
            # O base; the else path indexes the 2-stage O ring per group.
            o_tmem = tmem_addr + UInt32(MMA_N0 * num_s)
        p_accumulator = UMMA0Type.c_t(s_tmem)
        output_accumulator = UMMA1Type.c_t(o_tmem)
        # P overlays the upper half of S[stage] -- per-group p_desc computed in
        # the loop (stage == q-ring slot s == qg % num_s).

        # rowmax saved for LSE (re-used per group, single-tile softmax has no
        # cross-group state).
        rowmax_for_lse = LayoutTensor[
            UMMA0Type.accum_t,
            Layout.row_major(num_rows_per_warp),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()

        # Ping-pong: this WG handles only its alternate Q-tiles qg = qi*2 + stage.
        # Its ring slot s == o_s == stage is CONSTANT (the producer advances the
        # ring once per qg, so slot qg%2 == stage); the per-slot phase toggles
        # each of THIS WG's iterations (ph = qi&1).  umma_0's mbar[stage] /
        # mbar[num_s+stage] are driven directly so the held slot's QK-full /
        # tmem-empty handshake tracks the QK warp (which naturally steps qg%2).
        var s = stage
        var o_s = stage
        for qi in range((num_q_groups + (1 - stage)) // 2):
            var qg = qi * 2 + stage
            var ph: UInt32 = UInt32(qi & 1)
            var count_g = group_count(qg)

            # Wait for QK on THIS slot, copy S to registers, mask, softmax.
            umma_0.mbar[UInt32(stage)].wait(ph)
            p_acc = p_accumulator[UInt32(stage)]
            comptime if use_p_smem:
                tcgen05_fence_after()
            p_acc.copy_to(p_reg_tile)
            # Slot read out of TMEM -> QK may overwrite it (tmem-empty).
            _ = umma_0.mbar[UInt32(num_s + stage)].arrive()

            apply_mask(qpos_smem + s * BM, count_g)

            var attention_rowmax = _rowmax_online_softmax[
                1, mma_thread_layout, use_exp2=True, fold_scale_fma=True
            ](
                vectorize_p_reg_tile(),
                rowmax,
                init_rowmax=True,
                scale_log2e=scale_log2e,
            )
            rowmax.copy_from(attention_rowmax)
            var attention_rowsum = _rowsum[
                mma_thread_layout, packed_reduce=True
            ](vectorize_p_reg_tile())
            rowsum.copy_from(attention_rowsum)

            # P -> SMEM/TMEM for the PV MMA.
            comptime if use_p_smem:
                comptime p_swizzle = make_swizzle[
                    num_rows=WM // 2, row_size=MMA_N0, access_size=8
                ]()
                p_smem_tile = LayoutTensor[
                    kv_type,
                    Layout.row_major(BM, MMA_N0),
                    address_space=AddressSpace.SHARED,
                ](p_smem.bitcast[Scalar[kv_type]]())
                p_smem_warp_tile = p_smem_tile.tile[WM, MMA_N0](Int(warp_y), 0)
                copy_local_to_shared[
                    thread_layout=mma_thread_layout, swizzle=p_swizzle
                ](
                    p_smem_warp_tile.vectorize[1, 2](),
                    UMMA0Type.c_t.rows_of_frags(p_reg_tile)
                    .vectorize[1, 2]()
                    .transpose(),
                )
                fence_async_view_proxy()
                named_barrier[Int32(consumer_group_size)](Int32(6 + stage))
                umma_1_ts.tmem_arrive()
                _ = p_full_mbar[s].arrive()
            else:
                var p_tmem = s_tmem + UInt32(s * MMA_N0 + MMA_N0 // 2)
                p_desc = UMMA1Type.a_mma_descriptor(p_tmem)
                p_desc.copy_from(UMMA0Type.c_t.rows_of_frags(p_reg_tile))
                # P[s] is written -> PV may consume this stage.
                _ = p_full_mbar[s].arrive()

            # Save rowmax for LSE before reciprocating rowsum.
            rowmax_for_lse.copy_from(rowmax)

            # Guard the rcp: a fully-masked row has sum 0/NaN -> rcp would be
            # inf/NaN and poison O.  Fold bad sum to 1.0; the LSE is forced -inf
            # below so combine skips the row's O_partial.
            comptime for row in range(num_rows_per_warp):
                var rs = rowsum[row][0]
                var safe = rs if (rs > Scalar[accum_type](0)) and isfinite(
                    rs
                ) else Scalar[accum_type](1)
                rowsum[row] = recip(safe)[0]

            comptime if use_p_smem:
                umma_1_ts.wait_for_mma()
                tcgen05_fence_after()
                output_accumulator.copy_to(output_reg_tile)
            else:
                # 2-stage O ring: read this group's O from slot o_s == stage.
                # Wait for PV to produce O[o_s] (o_full), copy it out, then
                # release the slot (o_empty) so PV(qg+o_stage) may overwrite it.
                var o_acc = UMMA1Type.c_t(o_tmem + UInt32(o_s * MMA_N1))
                o_full_mbar[o_s].wait(ph)
                o_acc.copy_to(output_reg_tile)
                _ = o_empty_mbar[o_s].arrive()

            # Scatter into the group's Q-ring slot (its Q already consumed).
            write_output(
                stage,
                rowsum,
                rowmax_for_lse,
                vectorize_o_reg_tile(),
                count_g,
                qsplit_smem + s * BM,
                q_smem + s * q_tile_elems,
            )
            # Release the slot back to the load thread (all this WG's threads
            # arrive; the scatter above has read the staged O out of SMEM).
            _ = q_consumed_mbar[s].arrive()


# ===-----------------------------------------------------------------------===#
# Dispatch
# ===-----------------------------------------------------------------------===#


@always_inline
def msa_sm100_block_major_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    output_type: DType,
    //,
    config: MHAConfig,
    group: Int,
](
    o_partial: DeviceBuffer[output_type],
    lse_partial: DeviceBuffer[DType.float32],
    q_arg: DeviceBuffer[q_type],
    k: KVType,
    v: KVType,
    scheduler_metadata: UnsafePointer[Int32, MutAnyOrigin],
    grid_work: Int,
    work_count_ptr: UnsafePointer[Int32, MutAnyOrigin],
    k2q_row_ptr: UnsafePointer[Int32, MutAnyOrigin],
    qsplit_indices: UnsafePointer[Int32, MutAnyOrigin],
    cu_seqlens_k: UnsafePointer[Int32, MutAnyOrigin],
    cu_seqlens_q: UnsafePointer[Int32, MutAnyOrigin],
    total_q: Int,
    total_rows: Int,
    nnz: Int,
    head_q: Int,
    scale: Float32,
    ctx: DeviceContext,
    q_positions: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    seqused_k: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    """Dispatch for the block-major MSA forward.  Grid is `(grid_work, 1, 1)` --
    a host-known capacity >= the on-device `work_count`, so no host readback /
    sync is needed.  CTAs past `work_count_ptr[0]` are idle and PDL-early-exit.
    Produces `O_partial`/`LSE_partial`; PDL (`OVERLAP_AT_END`) chains this fwd
    with the combine kernel that merges them into O.
    """
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match."
    comptime assert config.dtype.is_half_float(), "BF16 only"
    comptime assert group in (1, 2, 4, 8, 16), "qheadperkv in {1,2,4,8,16}"

    comptime swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    # BM == BN == 128: one M-tile holds a whole block's queries (up to BN), one
    # N-tile is the block.  MMA_M == 128 (BM % 128 == 0).
    comptime new_config = MHAConfig[config.dtype](
        config.num_heads,
        config.depth,
        num_queries_per_block=128,
        num_keys_per_block=128,
        BK=config.BK,
        num_pipeline_stages=2 if config.padded_depth >= 512 else 4,
    )
    comptime BM = new_config.block_m()
    comptime BN = new_config.block_n()
    comptime BK = new_config.padded_depth
    comptime assert BM % 64 == 0, "SM90 requires BM%64==0"
    comptime assert BK % 64 == 0, "B200 requires BK%64 (128B swizzle)"
    # +4 spare load warps over the 12 working warps (idle this step; they get
    # the dedicated Q/K/V load roles next).  Only the block_dim grows -- every
    # SMEM/ring size stays driven by num_threads[True]() (the 12 working warps).
    comptime num_threads = new_config.num_threads[True]() + 4 * WARP_SIZE
    comptime assert new_config.algorithm == FlashAttentionAlgorithm(3)

    var q = rebind[UnsafePointer[Scalar[KVType.dtype], MutAnyOrigin]](q_arg)

    # Q gather4 descriptor: tile_height == BM rows, viewed flat as
    # [total_q*head_q, depth]; gathered rows are the queries that selected this
    # block.
    q_gather4 = create_tma_tile_gather4[
        KVType.dtype,
        tile_height=BM,
        tile_width=config.depth,
        tile_stride=config.depth,
        swizzle_mode=swizzle_mode,
    ](ctx, q, total_q * head_q)

    comptime kv_sub_BN = _kv_sub_tile_rows(BN, KVType.page_size)
    k_tma_op = k.create_tma_tile[
        swizzle_mode,
        BN=kv_sub_BN,
        depth=config.depth,
        BK=new_config.padded_depth,
    ](ctx)
    v_tma_op = v.create_tma_tile[
        swizzle_mode,
        BN=kv_sub_BN,
        depth=config.depth,
        BK=new_config.padded_depth,
    ](ctx)

    comptime KVPtrT = UnsafePointer[Int32, MutAnyOrigin]

    comptime use_p_smem = config.padded_depth + BN // 2 > 512
    comptime num_s = (
        512
        // BN if use_p_smem else (512 - (BN // 2) - config.padded_depth)
        // BN
    )
    comptime q_stage = 2
    comptime p_smem_bytes = BM * BN * size_of[
        config.dtype
    ]() if use_p_smem else 0
    # The Q ring is q_stage tiles; shared_mem_bytes only counts ONE Q tile, so
    # add the extra (q_stage-1).
    comptime extra_q_bytes = (q_stage - 1) * BM * config.padded_depth * size_of[
        config.dtype
    ]()
    # mbar region (2*num_s+3 SharedMemBarriers @ 8B) + ptr_tmem(2 u32 -> in the
    # +3); then the Q ring: q_produced + q_consumed (q_stage each) and the
    # per-slot idx (BM) + qpos (BM) + qsplit (BM) Int32 (no idx-ready mbar -- the
    # intra-WG named_barrier(9) carries it).  qsplit holds the packed
    # (q|split<<24) the epilogue reuses (no 2nd GMEM index read).
    comptime q_ring_mbar_bytes = 2 * q_stage * 8
    comptime idx_smem_bytes = 3 * BM * q_stage * size_of[DType.int32]()
    # p_full[num_s] + p_empty[num_s] for the 2-stage P ring.
    comptime p_ring_mbar_bytes = 2 * num_s * 8
    # o_full[o_stage] + o_empty[o_stage] for the 2-stage O ring.
    comptime o_stage = 2
    comptime o_ring_mbar_bytes = 2 * o_stage * 8
    comptime extra_smem = (
        (2 * num_s + 3) * 8
        + p_smem_bytes
        + extra_q_bytes
        + q_ring_mbar_bytes
        + idx_smem_bytes
        + p_ring_mbar_bytes
        + o_ring_mbar_bytes
    )
    comptime smem_use = new_config.shared_mem_bytes[
        True, sm_90=True
    ]() + extra_smem

    logger.info("------ Dispatching to SM100 Model-B block-major MSA ------")

    @parameter
    @always_inline
    def launch[causal: Bool, has_seqused: Bool](sk: KVPtrT) raises:
        ctx.enqueue_function[
            _msa_sm100_block_major[
                KVLUTType=KVType,
                output_type=output_type,
                config=new_config,
                group=group,
                swizzle_mode=swizzle_mode,
                causal=causal,
                has_seqused=has_seqused,
            ]
        ](
            q_gather4,
            k_tma_op,
            v_tma_op,
            o_partial.unsafe_ptr(),
            lse_partial.unsafe_ptr(),
            k,
            scheduler_metadata,
            work_count_ptr,
            k2q_row_ptr,
            qsplit_indices,
            cu_seqlens_k,
            cu_seqlens_q,
            sk,
            total_q,
            total_rows,
            nnz,
            head_q,
            scale,
            grid_dim=(grid_work, 1, 1),
            block_dim=(num_threads, 1, 1),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_END),
        )

    # `q_positions` is the external causal toggle: present => in-kernel causal
    # (the kernel derives the per-CTA causal_q_offset from cu_seqlens itself).
    var sk_ptr = seqused_k.value() if seqused_k else KVPtrT.unsafe_dangling()
    if q_positions:
        if seqused_k:
            launch[True, True](sk_ptr)
        else:
            launch[True, False](KVPtrT.unsafe_dangling())
    else:
        if seqused_k:
            launch[False, True](sk_ptr)
        else:
            launch[False, False](KVPtrT.unsafe_dangling())


# ===-----------------------------------------------------------------------===#
# End-to-end prefill dispatch: host CSR -> fwd -> combine -> O
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct PrebuiltSchedule(ImplicitlyCopyable, Movable):
    """The reusable scheduler overlay, already on the device.

    Lets a caller build the schedule once (host or device builder) and reuse
    its overlay across prefill calls.  Carries only the four tensors the fwd +
    combine read off the reusable object -- `meta` (scheduler_metadata),
    `work_count`, `qsplit` (the packed q-indices), `split_counts`.  The CSR
    proper (`row_ptr`) is NOT stored: reuse rebuilds it fresh each call and
    substitutes this overlay, matching the reference (CSR fresh, overlay
    reused).  Sizes (`total_rows`, `work_capacity`) come from the fresh build,
    not from here -- `work_capacity` is `len(meta) // 6`.

    Reuse trusts the overlay: the caller MUST build it from the SAME `q2k`
    selection, `cu_seqlens`, and shapes as the call it's passed to.  The reuse
    branch re-checks buffer lengths only, never the selection contents.
    """

    var meta: DeviceBuffer[DType.int32]  # [work_capacity, 6]
    var work_count: DeviceBuffer[DType.int32]  # [1] = valid rows
    var qsplit: DeviceBuffer[DType.int32]  # [head_kv, total_q * topk]
    var split_counts: DeviceBuffer[DType.int32]  # [batch, max_sq, head_kv]


@always_inline
def msa_sm100_prefill_b_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    output_type: DType,
    //,
    config: MHAConfig,
    group: Int,
](
    o: DeviceBuffer[output_type],  # [total_q, head_q, depth]
    lse: DeviceBuffer[DType.float32],  # [total_q, head_q]
    q_arg: DeviceBuffer[q_type],  # [total_q, head_q, depth]
    k: KVType,
    v: KVType,
    q2k: List[Int32],  # [head_kv, total_q, topk] batch-local block ids (host)
    cu_seqlens_q: List[Int32],  # [batch+1] (host)
    cu_seqlens_k: List[Int32],  # [batch+1] (host)
    topk: Int,
    scale: Float32,
    ctx: DeviceContext,
    q_positions: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    seqused_k: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    prebuilt: Optional[PrebuiltSchedule] = None,
) raises:
    """End-to-end KV-block-major sparse MHA prefill for SM100.

    Inverts the query-major selection `q2k` into a reverse-CSR on the host
    (`build_k2q_csr`), uploads it, launches the block-major forward
    (`O_partial`/`LSE_partial` per (query, split-slot)), then the combine
    (`msa_combine`) that LSE-merges each query's slots into the final O.  The
    external contract is the query-major one: `q2k` + Q/K/V + `cu_seqlens` +
    `scale` (+ optional `q_positions` for in-kernel causal) -> O.

    The CSR builder caps each work item at a load-balanced `q_per_cta`
    (`balanced_target_q_per_cta`, ~`num_sms*2` items); the fwd CTA loops
    `ceil(q_count/BM)` Q-tiles against its resident KV block (no query dropped).
    `group == 1`, BF16, non-paged KV.
    """
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match."
    comptime assert config.dtype.is_half_float(), "BF16 only"
    comptime assert group in (1, 2, 4, 8, 16), "qheadperkv in {1,2,4,8,16}"
    comptime depth = config.depth

    var batch = len(cu_seqlens_q) - 1
    var head_q = config.num_heads
    var head_kv = head_q // group
    var total_q = Int(cu_seqlens_q[batch])
    var blk_kv = config.num_keys_per_block

    var max_sq = 0
    var max_sk = 0
    for b in range(batch):
        max_sq = max(max_sq, Int(cu_seqlens_q[b + 1] - cu_seqlens_q[b]))
        max_sk = max(max_sk, Int(cu_seqlens_k[b + 1] - cu_seqlens_k[b]))

    var nnz = total_q * topk
    var sc_len = batch * max_sq * head_kv

    # cu_seqlens upload (the fwd kernel reads them on-device) -- always built;
    # not part of the reusable schedule.
    var cuq_d = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var cuk_d = ctx.enqueue_create_buffer[DType.int32](batch + 1)

    # ---- Host: invert the query-major selection into the reverse-CSR ----
    # Load-balance the q-chunk cap (~num_sms*2 work items) so one CTA loads
    # its KV block once and loops ceil(q_count/BM) Q-groups, instead of one
    # CTA per BM queries (avoids ~10x CTA over-subscription).  The CSR proper
    # is rebuilt fresh every call (matching the reference); a prebuilt only
    # substitutes the scheduler overlay.
    var num_sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var q_per_cta = balanced_target_q_per_cta(
        total_q, topk, blk_kv, head_kv, num_sms, config.block_m()
    )
    var csr = build_k2q_csr(
        q2k,
        cu_seqlens_q,
        cu_seqlens_k,
        head_kv,
        total_q,
        topk,
        blk_kv,
        max_sq,
        max_sk,
        q_per_cta,
    )

    var work = csr.work_count
    var total_rows = csr.total_rows
    var work_cap = csr.work_capacity
    var rp_len = head_kv * (csr.total_rows + 1)
    var qidx_len = head_kv * nnz

    # row_ptr (CSR proper) + cu_seqlens: uploaded fresh on both branches.
    var rp_d = ctx.enqueue_create_buffer[DType.int32](rp_len)
    var rp_h = ctx.enqueue_create_host_buffer[DType.int32](rp_len)
    var cuq_h = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
    var cuk_h = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
    ctx.synchronize()
    for i in range(rp_len):
        rp_h[i] = csr.k2q_row_ptr[i]
    for i in range(batch + 1):
        cuq_h[i] = cu_seqlens_q[i]
        cuk_h[i] = cu_seqlens_k[i]
    ctx.enqueue_copy(rp_d, rp_h)
    ctx.enqueue_copy(cuq_d, cuq_h)
    ctx.enqueue_copy(cuk_d, cuk_h)

    # Scheduler overlay the fwd/combine read: prebuilt substitutes it, else
    # built fresh from this call's CSR.
    var meta_d: DeviceBuffer[DType.int32]
    var qsplit_d: DeviceBuffer[DType.int32]
    var sc_d: DeviceBuffer[DType.int32]
    var wc_d: DeviceBuffer[DType.int32]

    if prebuilt:
        var sched = prebuilt.value()
        # qsplit vs the fresh CSR's q-index shape (the reference cross-check).
        if len(sched.qsplit) != qidx_len:
            raise Error("prebuilt qsplit len mismatch")
        if len(sched.split_counts) != max(sc_len, 1):
            raise Error("prebuilt split_counts len mismatch")
        if len(sched.meta) != max(work_cap * 6, 1):
            raise Error("prebuilt meta len mismatch")
        if len(sched.work_count) != 1:
            raise Error("prebuilt work_count len mismatch")
        meta_d = sched.meta
        qsplit_d = sched.qsplit
        sc_d = sched.split_counts
        wc_d = sched.work_count
    else:
        var meta_h = ctx.enqueue_create_host_buffer[DType.int32](
            max(work * 6, 1)
        )
        var qsplit_h = ctx.enqueue_create_host_buffer[DType.int32](qidx_len)
        var sc_h = ctx.enqueue_create_host_buffer[DType.int32](sc_len)
        var wc_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.synchronize()
        wc_h[0] = Int32(work)
        for i in range(work * 6):
            meta_h[i] = csr.scheduler_metadata[i]
        for i in range(qidx_len):
            qsplit_h[i] = csr.qsplit_indices[i]
        for i in range(sc_len):
            sc_h[i] = csr.split_counts[i]

        meta_d = ctx.enqueue_create_buffer[DType.int32](max(work * 6, 1))
        qsplit_d = ctx.enqueue_create_buffer[DType.int32](qidx_len)
        sc_d = ctx.enqueue_create_buffer[DType.int32](sc_len)
        wc_d = ctx.enqueue_create_buffer[DType.int32](1)
        ctx.enqueue_copy(meta_d, meta_h)
        ctx.enqueue_copy(qsplit_d, qsplit_h)
        ctx.enqueue_copy(sc_d, sc_h)
        ctx.enqueue_copy(wc_d, wc_h)
    _ = csr^

    # ---- Intermediates: O_partial [topk,total_q,head_q,D], LSE [topk,..] ----
    var o_part_d = ctx.enqueue_create_buffer[output_type](
        topk * total_q * head_q * depth
    )
    var lse_part_d = ctx.enqueue_create_buffer[DType.float32](
        topk * total_q * head_q
    )

    # ---- Forward: block-major fwd -> O_partial / LSE_partial ----------------
    msa_sm100_block_major_dispatch[config=config, group=group](
        o_part_d,
        lse_part_d,
        q_arg,
        k,
        v,
        meta_d.unsafe_ptr().as_unsafe_any_origin(),
        work,
        wc_d.unsafe_ptr().as_unsafe_any_origin(),
        rp_d.unsafe_ptr().as_unsafe_any_origin(),
        qsplit_d.unsafe_ptr().as_unsafe_any_origin(),
        cuk_d.unsafe_ptr().as_unsafe_any_origin(),
        cuq_d.unsafe_ptr().as_unsafe_any_origin(),
        total_q,
        total_rows,
        nnz,
        head_q,
        scale,
        ctx,
        q_positions,
        seqused_k,
    )

    # ---- Combine: LSE-merge each query's split slots -> final O / LSE -------
    msa_combine_dispatch[depth=depth](
        o,
        lse,
        o_part_d,
        lse_part_d,
        sc_d.unsafe_ptr().as_unsafe_any_origin(),
        cuq_d.unsafe_ptr().as_unsafe_any_origin(),
        batch,
        head_q,
        head_kv,
        max_sq,
        total_q,
        topk,
        ctx,
    )


def msa_sm100_prefill_b_device_csr_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    output_type: DType,
    //,
    config: MHAConfig,
    group: Int,
    topk: Int,
](
    o: DeviceBuffer[output_type],  # [total_q, head_q, depth]
    lse: DeviceBuffer[DType.float32],  # [total_q, head_q]
    q_arg: DeviceBuffer[q_type],  # [total_q, head_q, depth]
    k: KVType,
    v: KVType,
    q2k: DeviceBuffer[
        DType.int32
    ],  # [head_kv, total_q, topk] block ids (device)
    cu_seqlens_q: List[Int32],  # [batch+1] (host, sets sizing + cu upload)
    cu_seqlens_k: List[Int32],  # [batch+1] (host)
    scale: Float32,
    ctx: DeviceContext,
    q_positions: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    seqused_k: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    prebuilt: Optional[PrebuiltSchedule] = None,
) raises:
    """End-to-end sparse MHA prefill with the DEVICE CSR builder.

    Same external contract as `msa_sm100_prefill_b_dispatch`, except the
    query-major selection `q2k` is already on the device and the reverse-CSR is
    built on-device (`build_k2q_csr_device`) instead of host + upload.  The
    forward + combine are byte-for-byte the host-CSR path (they consume the same
    contract tensors).  `topk` is a comptime parameter (the device builder is
    templated on it).
    """
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match."
    comptime assert config.dtype.is_half_float(), "BF16 only"
    comptime assert group in (1, 2, 4, 8, 16), "qheadperkv in {1,2,4,8,16}"
    comptime depth = config.depth

    var batch = len(cu_seqlens_q) - 1
    var head_q = config.num_heads
    var head_kv = head_q // group
    var total_q = Int(cu_seqlens_q[batch])
    var blk_kv = config.num_keys_per_block

    var max_sq = 0
    var max_sk = 0
    for b in range(batch):
        max_sq = max(max_sq, Int(cu_seqlens_q[b + 1] - cu_seqlens_q[b]))
        max_sk = max(max_sk, Int(cu_seqlens_k[b + 1] - cu_seqlens_k[b]))

    var nnz = total_q * topk
    var sc_len = batch * max_sq * head_kv

    # cu_seqlens upload (the fwd kernel reads them on-device) -- always built;
    # not part of the reusable schedule.
    var cuq_h = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
    var cuk_h = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
    ctx.synchronize()
    for i in range(batch + 1):
        cuq_h[i] = cu_seqlens_q[i]
        cuk_h[i] = cu_seqlens_k[i]
    var cuq_d = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var cuk_d = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    ctx.enqueue_copy(cuq_d, cuq_h)
    ctx.enqueue_copy(cuk_d, cuk_h)

    # The CSR proper is rebuilt fresh every call (matching the reference); a
    # prebuilt only substitutes the scheduler overlay.  Size + build always.
    var num_sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    # Load-balanced q-chunk cap (see msa_sm100_prefill_b_dispatch): the fwd
    # CTA loops ceil(q_count/BM) Q-groups against one resident KV block.
    var q_per_cta = balanced_target_q_per_cta(
        total_q, topk, blk_kv, head_kv, num_sms, config.block_m()
    )
    var sizes = k2q_csr_sizes(
        cu_seqlens_k,
        head_kv,
        blk_kv,
        max_sk,
        total_q,
        topk,
        num_sms,
        q_per_cta,
    )
    var total_rows = sizes.total_rows
    var work_cap = sizes.work_capacity
    var rp_len = head_kv * (total_rows + 1)
    var qidx_len = head_kv * nnz

    # The fwd reads `qsplit` (overlay) from the prebuilt when present; on reuse
    # the builder still writes the four overlay tensors, so route them to fresh
    # throwaway buffers and bind the object's into the fwd.  Default: the
    # builder fills the buffers the fwd reads (byte-identical to before).
    var bld_qsplit_d = ctx.enqueue_create_buffer[DType.int32](qidx_len)
    var bld_meta_d = ctx.enqueue_create_buffer[DType.int32](
        max(work_cap * 6, 1)
    )
    var bld_wc_d = ctx.enqueue_create_buffer[DType.int32](1)
    var bld_sc_d = ctx.enqueue_create_buffer[DType.int32](max(sc_len, 1))

    # CSR proper + scratch (device).  Pre-zero / -1-fill where the builder
    # expects it (mirrors the host build's memset/0xFF init).
    var rp_d = ctx.enqueue_create_buffer[DType.int32](rp_len)
    var rmap_d = ctx.enqueue_create_buffer[DType.int32](
        max(batch * sizes.max_kv_blocks, 1)
    )
    var rcoords_d = ctx.enqueue_create_buffer[DType.int32](
        max(total_rows * 2, 1)
    )
    var rcounts_d = ctx.enqueue_create_buffer[DType.int32](
        max(head_kv * total_rows, 1)
    )
    var tcounts_d = ctx.enqueue_create_buffer[DType.int32](
        sizes.tile_counts_len(head_kv)
    )
    rp_d.enqueue_fill(Int32(0))
    bld_qsplit_d.enqueue_fill(Int32(-1))
    bld_wc_d.enqueue_fill(Int32(0))
    bld_sc_d.enqueue_fill(Int32(0))
    rcounts_d.enqueue_fill(Int32(0))

    build_k2q_csr_device[topk=topk](
        q2k.unsafe_ptr().as_unsafe_any_origin(),
        cuq_d.unsafe_ptr().as_unsafe_any_origin(),
        cuk_d.unsafe_ptr().as_unsafe_any_origin(),
        rp_d.unsafe_ptr().as_unsafe_any_origin(),
        bld_qsplit_d.unsafe_ptr().as_unsafe_any_origin(),
        bld_meta_d.unsafe_ptr().as_unsafe_any_origin(),
        bld_wc_d.unsafe_ptr().as_unsafe_any_origin(),
        bld_sc_d.unsafe_ptr().as_unsafe_any_origin(),
        rmap_d.unsafe_ptr().as_unsafe_any_origin(),
        rcoords_d.unsafe_ptr().as_unsafe_any_origin(),
        rcounts_d.unsafe_ptr().as_unsafe_any_origin(),
        tcounts_d.unsafe_ptr().as_unsafe_any_origin(),
        head_kv,
        total_q,
        blk_kv,
        max_sq,
        sizes,
        ctx,
        q_per_cta,
    )

    # Overlay the fwd/combine read: prebuilt substitutes it, else the freshly
    # built one.
    var meta_d: DeviceBuffer[DType.int32]
    var qsplit_d: DeviceBuffer[DType.int32]
    var sc_d: DeviceBuffer[DType.int32]
    var wc_d: DeviceBuffer[DType.int32]

    if prebuilt:
        var sched = prebuilt.value()
        # qsplit vs the fresh CSR's q-index shape (the reference cross-check).
        if len(sched.qsplit) != qidx_len:
            raise Error("prebuilt qsplit len mismatch")
        if len(sched.split_counts) != max(sc_len, 1):
            raise Error("prebuilt split_counts len mismatch")
        if len(sched.meta) != max(work_cap * 6, 1):
            raise Error("prebuilt meta len mismatch")
        if len(sched.work_count) != 1:
            raise Error("prebuilt work_count len mismatch")
        meta_d = sched.meta
        qsplit_d = sched.qsplit
        sc_d = sched.split_counts
        wc_d = sched.work_count
    else:
        meta_d = bld_meta_d
        qsplit_d = bld_qsplit_d
        sc_d = bld_sc_d
        wc_d = bld_wc_d

    # work_count is produced on-device; the fwd-kernel grid is sized at the
    # host-known capacity `work_cap` (>= work_count), and the kernel reads
    # `wc_d` to early-exit idle CTAs -- no host readback / sync (that drains the
    # pipeline between CSR and fwd and blocks PDL).

    # Intermediates: O_partial [topk,total_q,head_q,D], LSE [topk,..].
    var o_part_d = ctx.enqueue_create_buffer[output_type](
        topk * total_q * head_q * depth
    )
    var lse_part_d = ctx.enqueue_create_buffer[DType.float32](
        topk * total_q * head_q
    )

    msa_sm100_block_major_dispatch[config=config, group=group](
        o_part_d,
        lse_part_d,
        q_arg,
        k,
        v,
        meta_d.unsafe_ptr().as_unsafe_any_origin(),
        work_cap,
        wc_d.unsafe_ptr().as_unsafe_any_origin(),
        rp_d.unsafe_ptr().as_unsafe_any_origin(),
        qsplit_d.unsafe_ptr().as_unsafe_any_origin(),
        cuk_d.unsafe_ptr().as_unsafe_any_origin(),
        cuq_d.unsafe_ptr().as_unsafe_any_origin(),
        total_q,
        total_rows,
        nnz,
        head_q,
        scale,
        ctx,
        q_positions,
        seqused_k,
    )

    msa_combine_dispatch[depth=depth, max_splits=combine_max_splits[topk]()](
        o,
        lse,
        o_part_d,
        lse_part_d,
        sc_d.unsafe_ptr().as_unsafe_any_origin(),
        cuq_d.unsafe_ptr().as_unsafe_any_origin(),
        batch,
        head_q,
        head_kv,
        max_sq,
        total_q,
        topk,
        ctx,
    )
