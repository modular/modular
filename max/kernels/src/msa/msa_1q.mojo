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
"""Unified per-token BLOCK-sparse MHA (MSA) prefill + decode kernel for SM100
(B200), BF16 Q/K/V, D=128.

Each KV tile bulk-TMAs the 128-token block named by `d_indices` (`topk` counts
BLOCKS) instead of marching contiguously: one block id -> BN contiguous tokens
at `block_id * BN` via the same `populate` + `tma_copy_k`/`tma_copy_v`.

One CTA == one (query token, kv-head).  Decode (`msa_sm100_dispatch`) is the
seqlen-1 case (one CTA per (batch, kv-head)); prefill
(`msa_sm100_prefill_dispatch`) enumerates B*S query tokens through the same
body, one per CTA, the token axis on grid.x so a long prompt skips the 65535
grid.z cap.  Decode split-K on; prefill split-K unsupported.  Single GQA group,
fixed topk-blocks, precomputed block indices.

KV is flat (`page_size == 0`) or whole-block paged (`page_size == BN`; the page
table resolves `block_id * BN`).  A `-1` (unselected) block is skipped:
`mask_unselected` redirects its load to block 0 (no OOB page lookup) and the
softmax poisons its columns.
"""

from std.math import ceildiv, exp2, recip, align_up
from std.math.uutils import umod
from std.math.constants import log2e

from std.sys import align_of, simd_width_of, size_of

import std.gpu.primitives.warp as warp
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import AddressSpace, external_memory, fence_async_view_proxy
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_release_allocation_lock,
)
from layout import IntTuple, Layout, LayoutTensor
from layout.layout_tensor import copy_local_to_shared, copy_sram_to_dram
from layout.swizzle import make_swizzle
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
)
from std.logger import Logger
from std.collections import OptionalReg

from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_operand import kv_sub_tile_rows as _kv_sub_tile_rows
from nn.attention.gpu.nvidia.sm90.attention import (
    _apply_mask,
    _get_position,
    elect,
    get_q_head_idx,
    ImmutTileTensor1D,
    KVTMATile,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    Pack,
    q_tma,
    QTMATile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    TransientScheduler,
    WorkInfo,
)
from nn.attention.mha_utils import (
    FlashAttentionAlgorithm,
    get_start_and_end_for_partitions,
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    StaticInt,
    _is_decoding,
)
from nn.softmax import (
    _online_softmax_correction,
    _rowmax_online_softmax,
    _rowsum,
)

# Reuse the dense decode kernel's accumulator/descriptor machinery UNCHANGED.
from nn.attention.gpu.nvidia.sm100.mha_1q import (
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
)

from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple
from std.gpu.sync import named_barrier
from layout.tensor_core_async import tile_layout_k_major

comptime logger = Logger()

# ===-----------------------------------------------------------------------===#
# Sparse decode kernel
# ===-----------------------------------------------------------------------===#


@__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads[True]())
    )
)
@__llvm_metadata(`nvvm.minctasm`=SIMDSize(1))
@__name(
    t"sm100_msa_1q_depth{config.depth}_{KVLUTType.dtype}_{output_type}_nqh{config.num_heads}_nkvh{config.num_heads // group}",
)
def _msa_sm100[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
    # --- sparse (MSA) parameters ---
    split_k: Bool,
    sink: Bool,
    extra_kv: Bool,
    variable_topk: Bool,
    fp8: Bool,
    # Per-token index keying.  False: decode -- one CTA per (batch, kv-head),
    # index row keyed on `prompt_idx` (batch).  True: per-token prefill -- one
    # CTA per (query token, kv-head), index keyed on the global query row;
    # `d_indices` is `[head_kv, total_q, topk_blocks]`.
    per_token_index: Bool,
    # In-kernel causal masking.  False: no masking, bit-identical to the
    # precomputed-causality path (the `kv_logical_pos`/`q_positions` args go
    # unread).  True: the softmax warp poisons slots with `kv_logical_pos >
    # q_pos` to -inf before the rowmax.
    causal: Bool,
    # Unselected (`-1`) block handling.  False (default): every block id is a
    # real in-band block, bit-identical to HEAD.  True: a `-1` slot loads block
    # 0 instead (so a paged page-table lookup never sees a negative block) and
    # the softmax poisons its columns to -inf.  Needed once an indexer can emit
    # `-1` padding into `d_indices`.
    mask_unselected: Bool,
](
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM=config.block_m(),
        depth=config.depth,
        group=group,
        decoding=_is_decoding[MaxSeqLenType](),
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
    o_ptr_arg: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    kv_lut: KVLUTType,
    d_indices: UnsafePointer[Int32, MutAnyOrigin],
    indices_stride: Int,  # topk in BLOCKS
    scale: Float32,
    batch_size: UInt32,
    total_q: UInt32,  # per_token_index: rows in [head_kv, total_q, topk_blocks]
    num_keys_arg: UInt32,  # key axis in TOKENS (topk_blocks * BN)
    # Causal inputs, read only when `causal`.  `kv_logical_pos` is
    # `[head_kv, total_q, topk_blocks]` Int32: the logical START position of each
    # block (`-1` for unselected, same as `d_indices`).  `q_positions` is
    # `[total_q]`
    # Int32: the query token's logical position in its sequence, indexed by
    # `prompt_idx`.  Dangling when `causal` is False.
    kv_logical_pos: UnsafePointer[Int32, MutAnyOrigin],
    q_positions: UnsafePointer[Int32, MutAnyOrigin],
    pack: Pack[
        MaskType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxSeqLenType,
        PartitionType,
    ],
):
    """Block-sparse per-token MHA (MSA) for SM100.  One CTA == one (query token,
    kv-head); the GQA group's heads are the M-tile.  `per_token_index` False keys
    the selection on the batch (decode); True keys it on the global query row
    (prefill, token axis on grid.x).  The position is built from the block
    indices via the non-ragged math -- it never consults cu_seqlens, so
    ragged-packed Q is correct (each query row owns its own index row).

    split-K on (decode); `sink`/`extra_kv`/`variable_topk`/`fp8` False.  Single
    GQA group, fixed topk-blocks, BF16.  In-kernel causal: when
    `kv_logical_pos`/`q_positions` are non-null the softmax warp poisons slots
    with `kv_logical_pos > q_pos`.  Null => the precomputed-causality path.
    """
    comptime kv_type = KVLUTType.dtype
    comptime assert kv_type == config.dtype
    # Flat (page_size==0) or whole-block paged (page_size==BN).  Either gives
    # needs_partial=False, so the inlined dense produce_kv body is exactly right;
    # a page that splits a block (0 < page_size < BN) needs produce_kv_partial.
    # Same scope as the prefill block-major arm (msa_2q).
    comptime assert (
        KVLUTType.page_size == 0 or KVLUTType.page_size == config.block_n()
    ), "page_size must be 0 or == BN (no intra-block page split)"
    # Per-token prefill runs through the decode body: B*S tokens enumerated as
    # decode work-units (max_seq_len == 1), so `_is_decoding` holds either way.
    comptime assert _is_decoding[
        MaxSeqLenType
    ](), "msa runs decode-shaped tiles"
    comptime assert not (
        per_token_index and split_k
    ), "prefill (per-token) split-K unsupported"

    # split_k mirrors PartitionType.do_partition (enqueue derives it, so they
    # can't disagree).
    comptime assert (
        split_k == PartitionType.do_partition
    ), "split_k must match PartitionType.do_partition"
    comptime assert not sink, "attention sink unsupported"
    comptime assert not extra_kv, "extra-KV unsupported"
    comptime assert not variable_topk, "variable topk unsupported"
    comptime assert not fp8, "FP8 unsupported (BF16 only)"
    comptime assert SinkType.is_null, "sink pointer must be null"
    comptime assert kv_type.is_half_float(), "BF16 only"

    comptime num_softmax_threads: Int = config.num_consumer_threads()
    comptime num_softmax_warps = num_softmax_threads // 32

    comptime cta_group = 1
    comptime BM: Int = config.block_m()
    comptime BN: Int = config.block_n()
    comptime BK: Int = config.padded_depth
    comptime depth: Int = config.depth
    comptime padded_depth: Int = config.padded_depth
    comptime MMA_M: Int = 128 if (BM % 128) == 0 else 64
    # The GQA group is the M-tile; decode BM=64 caps it at MMA_M (Kimi's 64 is
    # exactly at the cap).  A larger group would silently overflow the tile.
    comptime assert group <= MMA_M, "GQA group exceeds the M-tile (MMA_M)"
    comptime MMA_N0: Int = BN
    comptime MMA_N1: Int = config.padded_depth
    comptime MMA_K: Int = 16 if kv_type.is_half_float() else 32
    comptime num_row_fragments = num_softmax_threads // 128
    comptime assert (32 % num_row_fragments) == 0
    comptime row_fragment_size = min(32 // num_row_fragments, BM // 4)
    comptime assert num_row_fragments * row_fragment_size <= 32
    comptime WM = row_fragment_size
    comptime num_m_blocks_per_warp = BM // (16 * num_softmax_warps)
    comptime assert num_m_blocks_per_warp * 16 == WM
    comptime assert WM * num_softmax_warps == BM
    comptime num_m_mmas = 1
    comptime num_n_mmas = 1
    comptime num_k_mmas = BK // MMA_K
    comptime num_heads: Int = config.num_heads
    comptime pipeline_stages = config.num_pipeline_stages
    var tid = UInt32(thread_idx.x)
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)
    comptime accum_type = get_accum_type[kv_type]()
    comptime assert accum_type.is_floating_point()
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
        num_softmax_threads=num_softmax_threads,
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
        num_softmax_threads=num_softmax_threads,
        swizzle_b=swizzle_mode,
        transpose_b=False,
    ]
    comptime UMMA1TypeSS = SM100TensorAccumulatorSS[
        kv_type,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=UMMA1_MMA_N,
        BM=BM,
        BN=MMA_N1,
        BK=BN,
        compute_BK=align_up(BN, MMA_K),
        num_softmax_threads=num_softmax_threads,
        swizzle_a=swizzle_mode,
        swizzle_b=swizzle_mode,
        transpose_b=False,
        pipeline_stages=1,
    ]
    mask = pack.mask
    scheduler = pack.scheduler
    valid_length = pack.valid_length
    kv_input_row_offsets = pack.kv_input_row_offsets
    max_seq_len = pack.max_seq_len
    partition = pack.partition

    comptime assert size_of[output_type]() >= size_of[kv_type]()
    comptime q_or_out_kv_elems = size_of[output_type]() // size_of[kv_type]()
    comptime q_smem_size = BM * padded_depth * q_or_out_kv_elems
    q_smem = external_memory[
        Scalar[kv_type],
        address_space=AddressSpace.SHARED,
        alignment=128,
        name="msa_dynamic_shared_memory",
    ]()

    comptime p_smem_elems = BM * MMA_N0 if use_p_smem else 0
    p_smem = q_smem + q_smem_size

    comptime kv_smem_size = config.kv_smem_size(True)
    kv_smem = q_smem + q_smem_size + p_smem_elems

    comptime p_frag_size = BM * MMA_N0 // (
        num_softmax_threads * num_m_blocks_per_warp
    )
    comptime o_frag_size = BM * MMA_N1 // (
        num_softmax_threads * num_m_blocks_per_warp
    )
    comptime assert p_frag_size == 2 * (WM // 8) * (MMA_N0 // 8)
    comptime assert o_frag_size == 2 * (WM // 8) * (MMA_N1 // 8)
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
    comptime num_cols_p = p_vec_output_layout[1].size()
    comptime num_cols_output = o_vec_output_layout[1].size()

    comptime accum_simd_width = simd_width_of[accum_type]()
    comptime row_alignment = align_of[SIMD[accum_type, accum_simd_width]]()
    comptime kv_num_heads = num_heads // group

    comptime mma_thread_layout = Layout.row_major(8, 4)
    comptime ragged = not ValidLengthType.is_null

    # Sink disabled -> never dereferenced; kept (as in dense `_mha_sm100`) so
    # the comptime-dead sink branches in the softmax warp still parse.
    var sink_weights_ptr = UnsafePointer[
        Scalar[kv_type], ImmutAnyOrigin
    ].unsafe_dangling()

    # --- pipeline mbars (identical layout to dense `_mha_sm100`) ---
    produced_mbar_kv = (kv_smem + kv_smem_size).bitcast[SharedMemBarrier]()
    producer_mbar_kv = produced_mbar_kv + pipeline_stages
    mma_mbar = producer_mbar_kv + pipeline_stages
    umma_0 = UMMA0Type(mma_mbar.as_unsafe_any_origin())
    umma_1_ts = UMMA1Type(mma_mbar.as_unsafe_any_origin() + 2 * num_s)
    umma_1_ss = UMMA1TypeSS(mma_mbar.as_unsafe_any_origin() + 2 * num_s)
    ptr_tmem_addr = (mma_mbar + 2 * num_s + 2).bitcast[UInt32]()

    comptime num_producer_regs = 56
    comptime num_softmax_regs = 224

    var tile_summary = MHATileSummary[ValidLengthType](
        batch_size,
        ceildiv(max_seq_len.as_uint32(), UInt32(BM))
        * partition.num_partitions(),
        valid_length,
        max_seq_len.as_uint32(),
    )
    var state: MHATileState = scheduler.initial_state(
        ptr_tmem_addr.as_unsafe_any_origin() + 2, tile_summary
    )

    initial_seq_info = scheduler.unsafe_seq_info(tile_summary, state)
    comptime assert not SchedulerType.may_advance

    if tid == 0:
        comptime for i in range(pipeline_stages):
            produced_mbar_kv[i].init(1)
            producer_mbar_kv[i].init(Int32(num_softmax_threads))
        umma_0.init()
        umma_1_ts.init()

    comptime PositionType = MHAPosition[
        BM,
        BN,
        depth,
        padded_depth,
        num_heads,
        group,
        _is_decoding[MaxSeqLenType](),
    ]

    @parameter
    @always_inline
    def get_position[pos_ragged: Bool](seq_info: SeqInfo) -> PositionType:
        return _get_position[
            BM,
            BN,
            depth,
            padded_depth,
            num_heads,
            group,
            pos_ragged,
            _is_cache_length_accurate,
        ](
            seq_info,
            kv_lut,
            max_seq_len,
            num_keys_arg,
            kv_input_row_offsets,
        )

    var position: PositionType

    comptime if per_token_index:
        # Per-token prefill: the token axis is on grid.x, so `block_idx.x` is the
        # global query row and `block_idx.y` the kv-head (grid (total_q,
        # num_kv_heads, 1)).  Each global query row
        # owns its own index row, Q is packed [total_q, ...], so the position is
        # the non-ragged decode math regardless of host `ragged` -- it NEVER
        # consults cu_seqlens.  This isolates the per-token mapping so the decode
        # path (else) is byte-identical: `seq_len == 1` (StaticInt[1]),
        # `start_of_seq == 0`, NoPartition (`prompt_offset == 0`).  `_get_position`
        # with ragged=False then yields q_row == prompt_idx, num_keys == topk.
        var pt_work = WorkInfo(
            UInt32(0),  # prompt_offset (NoPartition)
            UInt32(block_idx.y),  # head_idx == kv-head
            UInt32(block_idx.x),  # prompt_idx == global query row
            True,
        )
        var pt_seq_info = SeqInfo(UInt32(1), UInt32(0), pt_work)
        position = get_position[False](pt_seq_info)
    else:
        position = get_position[ragged](initial_seq_info)

    # `indices_stride` counts topk BLOCKS; the key (column) axis is in TOKENS =
    # topk_blocks * BN.  One BN tile == one block.  Consumers iterate this
    # partition's TOKEN slice [kv_start, kv_end) in steps of BN; `_apply_mask`
    # masks columns >= num_keys (== topk tokens).
    var topk_blocks: UInt32 = UInt32(indices_stride)
    var topk: UInt32 = topk_blocks * UInt32(BN)

    # Split-K over the TOKEN axis: this CTA (block_idx.x == prompt_offset ==
    # partition_idx) takes a BN-aligned (whole-block) slice.  `[BN]` over
    # topk tokens yields block-aligned partitions directly (no *BN rescaling --
    # kv_start/kv_end are already token columns).  Partition sizes are BN-aligned
    # (no tail); empty partitions get kv_start == kv_end == topk and early-exit.
    var kv_start: UInt32 = 0
    var kv_end: UInt32 = topk

    comptime if PartitionType.do_partition:
        var s_e = get_start_and_end_for_partitions[BN](
            Int(topk),
            Int(partition.num_partitions()),
            Int(position.prompt_offset),
        )
        kv_start = UInt32(s_e[0])
        kv_end = UInt32(s_e[1])

    var kv_tile_start_row: UInt32 = kv_start
    var end: UInt32 = kv_end

    comptime assert num_s > 0

    barrier()
    comptime stage_sz = BN * padded_depth
    comptime kv_bytes = BN * padded_depth * size_of[kv_type]()
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[kv_type]()
    comptime q_smem_layout_consumer = tile_layout_k_major[
        kv_type, BM, padded_depth, swizzle_mode=swizzle_mode
    ]()
    comptime q_copy_rows = max(group, 8)
    comptime q_bytes = q_copy_rows * padded_depth * size_of[kv_type]()
    # Number of BN tiles in THIS partition's slice (0 if the partition is empty).
    var n_tiles: Int = ceildiv(Int(kv_end - kv_start), BN)
    # Index row for this CTA.  Decode: keyed on the batch (`prompt_idx`).
    # Per-token: `d_indices` is `[head_kv, total_q, topk]`, the global query row
    # is `prompt_idx` (B*S tokens enumerated as work-units), the kv-head is
    # `position.kv_head_idx()`.
    var idx_row: UInt32
    comptime if per_token_index:
        idx_row = position.kv_head_idx() * total_q + position.prompt_idx
    else:
        idx_row = position.prompt_idx
    var d_idx_base = d_indices + Int(idx_row) * indices_stride

    # Causal: this query token's logical position, read once per CTA
    # (the M-tile is one token's GQA-group heads, so q_pos is shared).  Indexed
    # by `prompt_idx` (global query row in per-token prefill, batch in decode).
    # The slot logical positions share `d_indices`' `idx_row` keying.
    var q_pos: Int32 = 0
    comptime if causal:
        q_pos = q_positions[Int(position.prompt_idx)]
    # Per-block logical-pos array `[head_kv, total_q, topk_blocks]`; one entry per
    # block, same idx_row keying as `d_indices` (`indices_stride == topk_blocks`).
    var kv_logical_base: Int = Int(idx_row) * indices_stride

    # Block id -> token-row base.  Local block index in this partition's slice is
    # `kv_start / BN + t` (kv_start is BN-aligned).
    comptime base_alignment: Int = MaskType.start_column_alignment[
        BM, BN, KVLUTType.page_size
    ]()

    @parameter
    @always_inline
    def block_base_row(t: Int) -> UInt32:
        var blk = d_idx_base[Int(kv_start) // BN + t]
        # A `-1` (unselected) slot must NOT reach `populate`: as UInt32 the row
        # `-1*BN` wraps huge and a paged page-table lookup divmods it into an OOB
        # LUT column.  Redirect to block 0 (real in-band bytes) -- `apply_mask`
        # poisons its columns, so the loaded values never enter the softmax.
        comptime if mask_unselected:
            if Int(blk) < 0:
                return UInt32(0)
        # Block id must land in this batch's physical KV band; an OOB id reads
        # garbage or another batch's rows.  Release-gated (no perf cost).
        var max_kv_blocks = (
            Int(kv_lut.cache_length(Int(position.prompt_idx))) // BN
        )
        debug_assert(
            0 <= Int(blk) and Int(blk) < max_kv_blocks,
            "MSA block id out of band: blk=",
            Int(blk),
            " max_kv_blocks=",
            max_kv_blocks,
        )
        return UInt32(blk) * UInt32(BN)

    # ====================================================================== #
    # Warp-group 0: producer.  tid 96 = load thread (Q 4d + block-contiguous K/V
    # bulk-TMA), mirroring dense `produce`'s interleaving: Preheader Q0,K0; Body
    # K_t,V_{t-1}; Exit V{-1}.  Each tile bulk-loads one selected block (the
    # dense `produce_kv` populate+tma_copy body inlined, base row from the index).
    # ====================================================================== #
    if warp_group_idx == 0:
        warpgroup_reg_dealloc[num_producer_regs]()

        if tid == 96:
            # Load thread.
            @parameter
            @always_inline
            def q_producer(
                offset: UInt32,
                out tile: LayoutTensor[
                    kv_type,
                    Layout.row_major(BM, padded_depth),
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ],
            ):
                tile = {q_smem.as_unsafe_any_origin() + offset}

            # Inlined dense `produce_kv` body: the producer is single-threaded so
            # `elect()` elects this lone thread.  K and V each occupy a fresh
            # pipeline stage; `base_row` = the selected block's token base.
            # page_size==0 => needs_partial=False.
            @parameter
            @always_inline
            def emit_k(
                base_row: UInt32,
                mut state: PipelineState[pipeline_stages],
                do_wait: Bool,
            ):
                var write_idx = state.index()
                var write_phase = state.phase()
                ref p_mbar = produced_mbar_kv[write_idx]
                if do_wait:
                    producer_mbar_kv[write_idx].wait(write_phase)
                    p_mbar.expect_bytes(Int32(kv_bytes))
                var paged_rows = kv_lut.populate[BN, base_alignment](
                    position.prompt_idx, base_row
                )
                var e = elect()
                paged_rows.tma_copy_k[needs_partial=False](
                    k_tma_op,
                    kv_smem + UInt32(stage_sz) * write_idx,
                    p_mbar,
                    kv_head_idx=position.kv_head_idx(),
                    elect=e,
                )
                state.step()

            @parameter
            @always_inline
            def emit_v(
                base_row: UInt32, mut state: PipelineState[pipeline_stages]
            ):
                var write_idx = state.index()
                var write_phase = state.phase()
                ref p_mbar = produced_mbar_kv[write_idx]
                producer_mbar_kv[write_idx].wait(write_phase)
                p_mbar.expect_bytes(Int32(kv_bytes))
                var paged_rows = kv_lut.populate[BN, base_alignment](
                    position.prompt_idx, base_row
                )
                var e = elect()
                paged_rows.tma_copy_v[needs_partial=False](
                    v_tma_op,
                    kv_smem + UInt32(stage_sz) * write_idx,
                    p_mbar,
                    kv_head_idx=position.kv_head_idx(),
                    elect=e,
                )
                state.step()

            var write_state = PipelineState[pipeline_stages]()

            # Empty partition (kv_start == kv_end): consumers early-exit, so the
            # load thread must issue NO TMA, else it expect_bytes a stage no
            # consumer drains.  Inert for num_partitions==1 (n_tiles >= 1).
            if n_tiles > 0:
                # Preheader: Q0, K0 share mbar[0].
                produced_mbar_kv[0].expect_bytes(Int32(q_bytes + kv_bytes))
                comptime for d_i in range(ceildiv(depth, swizzle_granularity)):
                    comptime d: Int = d_i * swizzle_granularity
                    comptime smem_offset = q_smem_layout_consumer(
                        IntTuple(0, d)
                    )
                    q_tma_op.async_copy_4d(
                        q_producer(UInt32(smem_offset)),
                        produced_mbar_kv[0],
                        (
                            d,
                            0,
                            Int(position.kv_head_idx()),
                            Int(position.q_row),
                        ),
                    )
                # K0 writes into mbar[0] (already expecting q+kv bytes); do_wait
                # False (no prior consumer arrival), no extra expect.
                var base0 = block_base_row(0)
                emit_k(base0, write_state, False)

                var prev_base = base0
                var t: Int = 1
                while t < n_tiles:
                    # Body: K_t (current), V_{t-1} (prev).
                    var cur_base = block_base_row(t)
                    emit_k(cur_base, write_state, True)
                    emit_v(prev_base, write_state)
                    prev_base = cur_base
                    t += 1

                # Exit: V_{n_tiles-1}.
                emit_v(prev_base, write_state)

        elif warp_id() == 0:  # warp id == 0: Q @ K'
            # Iterate this partition's topk slice [kv_start, kv_end) step BN.
            # NullMask + decode => always PARTIAL (no FULL_MASK skips), so every
            # tile is processed; masking uses num_keys_arg == topk tokens.
            var kv_tile_start_row: UInt32 = kv_start
            var end: UInt32 = kv_end

            comptime if PartitionType.do_partition:
                # we exit before allocating so we don't need to deallocate
                if kv_tile_start_row >= end:
                    return

            comptime if use_p_smem:
                # Two-bank: bank 0 = O (depth cols), bank 1 = S (num_s*BN cols)
                comptime assert num_s * MMA_N0 <= max_tmem_cols
                comptime assert MMA_N1 <= max_tmem_cols
            else:
                comptime tmem_cols = num_s * MMA_N0 + (MMA_N0 // 2) + MMA_N1
                comptime assert tmem_cols <= max_tmem_cols
            tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)

            qk_desc = UMMA0Type.mma_descriptors(
                q_smem.as_unsafe_any_origin(), kv_smem.as_unsafe_any_origin()
            )

            named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)]()
            if tid != 0:
                return
            q_desc = qk_desc.get_a()
            k_desc = qk_desc.get_b()
            var tmem_addr: UInt32 = ptr_tmem_addr[0]
            var s_tmem: UInt32
            # var o_tmem: UInt32
            comptime if use_p_smem:
                # o_tmem = tmem_addr  # bank 0
                s_tmem = tmem_addr + UInt32(1 << 20)  # bank 1
            else:
                s_tmem = tmem_addr
                # o_tmem = tmem_addr + UInt32(MMA_N0 * num_s)
            s_accumulator = UMMA0Type.c_t(s_tmem)

            @parameter
            @always_inline
            def q_mul_k(read_idx: UInt32, read_phase: UInt32):
                q = q_desc
                k = k_desc + Int(
                    UInt32(BN * config.padded_depth * size_of[kv_type]())
                    * read_idx
                )
                umma_0.wait_for_tmem()
                produced_mbar_kv[read_idx].wait(read_phase)

                umma_0.mma(
                    rebind[UMMA0Type.a_t](q),
                    rebind[UMMA0Type.b_t](k),
                    s_accumulator,
                    0,
                )

            var mask_status: TileMaskStatus
            while True:
                mask_status = position.mask_status(mask, kv_tile_start_row)
                if mask_status != TileMaskStatus.FULL_MASK:
                    break
                kv_tile_start_row += UInt32(BN)

            kv_pipeline_states = PipelineState[pipeline_stages]()
            # s_pipeline_states = PipelineState[pipeline_stages]()
            q_mul_k(
                kv_pipeline_states.index(),
                kv_pipeline_states.phase(),
            )
            kv_pipeline_states.step()

            # Consumption order:
            # Preheader: Q0, K0
            # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
            # Exit: V{-1}
            while True:
                kv_tile_start_row += UInt32(BN)
                if kv_tile_start_row >= end:
                    break
                mask_status = position.mask_status(mask, kv_tile_start_row)
                if mask_status == TileMaskStatus.FULL_MASK:
                    continue

                # new pipeline states
                # start ummas
                q_mul_k(
                    kv_pipeline_states.index(), kv_pipeline_states.phase()
                )  # can't rw `p_reg_tile`
                kv_pipeline_states.step()
                kv_pipeline_states.step()

        elif warp_id() == 1:  # warp id 1: P @ V
            # Iterate this partition's topk slice [kv_start, kv_end) step BN
            # (see warp 0).
            var kv_tile_start_row: UInt32 = kv_start
            var end: UInt32 = kv_end

            comptime if PartitionType.do_partition:
                if kv_tile_start_row >= end:
                    return

            named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)]()
            var tmem_addr: UInt32 = ptr_tmem_addr[0]
            if tid == 32:
                var s_tmem: UInt32 = 0
                var o_tmem: UInt32 = 0
                var p_tmem: UInt32 = 0

                @parameter
                @always_inline("nodebug")
                def p_mul_v(
                    read_idx: UInt32,
                    read_phase: UInt32,
                    scale_c: UInt32,
                    kv_row: UInt32,
                ):
                    comptime offset_elems_per = BN * config.padded_depth
                    comptime offset_bytes_per = offset_elems_per * size_of[
                        kv_type
                    ]()
                    comptime if use_p_smem:
                        comptime assert UMMA1TypeSS.num_n_mmas == 2
                        o_tmem = tmem_addr  # bank 0
                        output_accumulator = UMMA1Type.c_t(o_tmem)
                        s_tmem = tmem_addr + UInt32(1 << 20)  # bank 1
                        # SS MMA: both P and V from SMEM
                        pv_descs = UMMA1TypeSS.mma_descriptors(
                            p_smem.as_unsafe_any_origin(),
                            kv_smem.as_unsafe_any_origin(),
                        )
                        p_desc_a = pv_descs.get_a()
                        v_desc = pv_descs.get_b()
                        v = v_desc + Int(UInt32(offset_bytes_per) * read_idx)
                        umma_1_ts.wait_for_tmem()
                        produced_mbar_kv[read_idx].wait(read_phase)
                        umma_1_ss.mma(
                            rebind[UMMA1TypeSS.a_t](p_desc_a),
                            rebind[UMMA1TypeSS.b_t](v),
                            rebind[UMMA1TypeSS.c_t](output_accumulator),
                            scale_c,
                        )
                        # Fence ensures SS MMA has finished reading P
                        # from SMEM before we signal P SMEM is free.
                        tcgen05_fence_after()
                    else:
                        s_tmem = tmem_addr
                        o_tmem = tmem_addr + UInt32(MMA_N0 * num_s)
                        output_accumulator = UMMA1Type.c_t(o_tmem)
                        p_tmem = (
                            tmem_addr + UInt32(MMA_N0 * num_s) + UInt32(MMA_N1)
                        )
                        p_desc = UMMA1Type.a_mma_descriptor(p_tmem)
                        v_desc = UMMA1Type.b_mma_descriptor(
                            kv_smem.as_unsafe_any_origin()
                        )
                        v = v_desc + Int(UInt32(offset_bytes_per) * read_idx)
                        umma_1_ts.wait_for_tmem()
                        produced_mbar_kv[read_idx].wait(read_phase)
                        umma_1_ts.mma(
                            rebind[UMMA1Type.a_t](p_desc),
                            rebind[UMMA1Type.b_t](v),
                            output_accumulator,
                            scale_c,
                        )

                var mask_status: TileMaskStatus
                while True:
                    mask_status = position.mask_status(mask, kv_tile_start_row)
                    if mask_status != TileMaskStatus.FULL_MASK:
                        break
                    kv_tile_start_row += UInt32(BN)

                kv_pipeline_states = PipelineState[pipeline_stages]()
                kv_pipeline_states.step()
                comptime assert pipeline_stages >= 2

                var output_scale: UInt32 = 0
                while True:
                    kv_tile_start_row += UInt32(BN)
                    if kv_tile_start_row >= end:
                        break
                    mask_status = position.mask_status(mask, kv_tile_start_row)
                    if mask_status == TileMaskStatus.FULL_MASK:
                        continue

                    # copy new pfrag, used by `p_mul_v` on next iter
                    # start ummas
                    kv_pipeline_states.step()
                    # read_idx_v = 0, phase = 1
                    p_mul_v(
                        kv_pipeline_states.index(),
                        kv_pipeline_states.phase(),
                        output_scale,
                        0,
                    )  # can't rw output or pfrag
                    output_scale = 1
                    kv_pipeline_states.step()

                p_mul_v(
                    kv_pipeline_states.index(),
                    kv_pipeline_states.phase(),
                    output_scale,
                    kv_tile_start_row,
                )
            tcgen05_release_allocation_lock[cta_group]()
            tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)

    else:  # softmax
        warpgroup_reg_alloc[num_softmax_regs]()

        # arrive to unblock the producers
        # TODO: skip this by not waiting on the first set
        comptime for i in range(pipeline_stages):
            _ = producer_mbar_kv[i].arrive()
        umma_0.tmem_arrive_init()

        var warp_id: UInt32 = warp.broadcast((tid - 128) // UInt32(WARP_SIZE))

        # Coordinates of the current warp.
        var elect_one_warp = warp_id == 0

        var lane = UInt32(lane_id())

        var warp_y: UInt32 = warp_id  # // num_warps_n

        comptime if num_softmax_threads > 128:
            warp_y = 2 * (warp_y % 4) + (warp_y // 4)
        comptime warp_x: UInt32 = 0
        comptime assert num_softmax_warps == 4 or num_softmax_warps == 8

        # Mask global memory iterator.

        mask_warp_row = warp_y * UInt32(WM)
        var scale_log2e: Scalar[accum_type] = (
            scale.cast[
                accum_type
            ]() if MaskType.apply_log2e_after_mask else scale.cast[accum_type]()
            * log2e
        )

        # layout is
        # shape  = (2, num_m_blocks_per_warp) x (2, num_n_mmas)
        # stride = (2, 4*num_n_mmas) x (1, 4)

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
        def vectorize_p_reg_tile(
            out result: VecPType,
        ):
            result = {p_reg_tile.ptr}

        @parameter
        @always_inline
        def vectorize_o_reg_tile(
            out result: VecOType,
        ):
            result = {output_reg_tile.ptr}

        @parameter
        @always_inline
        def apply_mask(
            position: PositionType,
            mask_status: TileMaskStatus,
            kv_tile_start_row: UInt32,
        ):
            var max_len: UInt32 = num_keys_arg
            _apply_mask[WM, MMA_N0, num_m_mmas, num_n_mmas](
                mask_warp_row,
                position,
                lane,
                max_len,
                scale_log2e,
                kv_tile_start_row,
                mask,
                mask_status,
                vectorize_p_reg_tile(),
            )

            # Unselected (`-1`) block: this whole tile is one block, redirected
            # to block 0 by `block_base_row`, so its loaded scores are real but
            # must be dropped.  Poison every fragment column to -inf before the
            # rowmax.  Same thread-gate / fragment walk as the causal poison below.
            comptime if mask_unselected:
                if d_idx_base[Int(kv_tile_start_row) // BN] < 0:
                    comptime n_groups = min(2, ceildiv(group, 8))
                    if warp_id <= UInt32((group - 1) // 16) and lane < UInt32(
                        4 * group
                    ):
                        var p_vec = vectorize_p_reg_tile()
                        comptime for i in range(n_groups):
                            comptime for m_mma in range(num_m_mmas):
                                comptime for n_mma in range(num_n_mmas):
                                    comptime for j in range(MMA_N0 // 8):
                                        var v = p_vec[i, m_mma, j, n_mma]
                                        comptime for c in range(frag_simdwidth):
                                            v[c] = min_or_neg_inf[accum_type]()
                                        p_vec[i, m_mma, j, n_mma] = v

            # In-kernel causal masking.  `slot` is the topk TOKEN column; its
            # block is `slot // BN` and `kv_logical_pos[block]` is that block's
            # logical START position (per-block, `[head_kv, total_q,
            # topk_blocks]`).  The slot's logical KV position is `block_start +
            # slot % BN`; poison `> q_pos` to -inf before the rowmax.
            # `causal_q_offset = seqlen_k - seqlen_q` folds host-side into
            # `q_pos` / `kv_logical_pos`.  Decode fragment-col arithmetic (the
            # kernel only runs decode tiles).
            comptime if causal:
                # Skip threads outside the GQA group (they hold no live score).
                comptime num_groups_per_thread = min(2, ceildiv(group, 8))
                if warp_id <= UInt32((group - 1) // 16) and lane < UInt32(
                    4 * group
                ):
                    var p_vec = vectorize_p_reg_tile()
                    var pos_row = kv_logical_pos + kv_logical_base
                    var frag_col: UInt32 = (
                        lane * UInt32(frag_simdwidth) % UInt32(MMA_N0)
                    ) % 8
                    comptime for i in range(num_groups_per_thread):
                        comptime for m_mma in range(num_m_mmas):
                            comptime for n_mma in range(num_n_mmas):
                                comptime for j in range(MMA_N0 // 8):
                                    var col0 = (
                                        kv_tile_start_row
                                        + frag_col
                                        + UInt32(n_mma * MMA_N0)
                                        + UInt32(j * 8)
                                    )
                                    var v = p_vec[i, m_mma, j, n_mma]
                                    comptime for c in range(frag_simdwidth):
                                        var slot = col0 + UInt32(c)
                                        # Out-of-topk columns are already masked
                                        # by `_apply_mask` (num_keys == topk).
                                        if slot < topk:
                                            var blk = slot // UInt32(BN)
                                            var col_in_blk = slot % UInt32(BN)
                                            var blk_start = pos_row[Int(blk)]
                                            # `-1` (unselected block) keeps its
                                            # zero-fill convention; only mask
                                            # valid future positions.
                                            if (
                                                blk_start >= 0
                                                and Int32(
                                                    blk_start
                                                    + Int32(col_in_blk)
                                                )
                                                > q_pos
                                            ):
                                                v[c] = min_or_neg_inf[
                                                    accum_type
                                                ]()
                                    p_vec[i, m_mma, j, n_mma] = v

        @parameter
        @always_inline
        def correct_output(correction: type_of(rowmax), vout: VecOType):
            # Rescale O by the online-softmax correction factor.
            comptime for row in range(num_rows_per_warp):
                c = SIMD[accum_type, element_layout.size()](
                    rebind[Scalar[accum_type]](correction[row])
                )

                comptime for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * c

        @parameter
        @always_inline
        def write_output(
            position: PositionType,
            rowsum_inv: type_of(rowsum),
            vout: VecOType,
        ):
            # Apply softmax denumerator.
            comptime for row in range(num_rows_per_warp):
                rs_inv = vout.element_type(rowsum_inv[row][0])

                comptime for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * rs_inv

            var output_ptr: UnsafePointer[
                Scalar[output_type], MutAnyOrigin
            ] = o_ptr_arg

            comptime if PartitionType.do_partition:
                output_ptr = output_ptr + (
                    UInt32(depth * num_heads)
                    * batch_size
                    * position.prompt_offset
                )
            output_gmem_tile = position.q_out_gmem_tensor(output_ptr)

            # Write to global memory.
            comptime assert (
                output_type.is_half_float()
            ), "we don't support Float32 output"
            comptime assert size_of[kv_type]() <= size_of[output_type]()
            comptime swizzle = make_swizzle[
                num_rows=WM // 2, row_size=BN, access_size=8
            ]()
            # Reuse a_smem for c tile in smem
            comptime q_tile_size: UInt32 = q_smem_size // 2
            accum_smem_tile = LayoutTensor[
                output_type,
                Layout.row_major(BM, config.padded_depth),
                address_space=AddressSpace.SHARED,
            ]((q_smem).bitcast[Scalar[output_type]]())
            accum_smem_warp_tile = accum_smem_tile.tile[WM, BN](
                Int(warp_y), Int(warp_x)
            )

            # ensure all threads have finished reading `q_smem`
            named_barrier[Int32(num_softmax_threads)]()

            copy_local_to_shared[
                thread_layout=mma_thread_layout, swizzle=swizzle
            ](
                accum_smem_warp_tile.vectorize[1, 2](),
                UMMA1Type.c_t.rows_of_frags(output_reg_tile)
                .vectorize[1, 2]()
                .transpose(),
            )
            fence_async_view_proxy()
            # Guard writing to shared memory.
            named_barrier[Int32(num_softmax_threads)]()
            # Vectorized copy from shared to global memory, during which every 2 FP32
            # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
            # vector and stored using 16B store instruction.
            comptime out_simd_size = simd_width_of[output_type]()
            copy_sram_to_dram[
                thread_layout=Layout.row_major(
                    num_softmax_threads * out_simd_size // depth,
                    depth // out_simd_size,
                ),
                swizzle=swizzle,
            ](
                output_gmem_tile.vectorize[1, out_simd_size](),
                accum_smem_tile.vectorize[1, out_simd_size](),
            )

        comptime if PartitionType.do_partition:  # we may have an empty partition
            if kv_tile_start_row >= end:
                if umod(thread_idx.x, 4) == 0 and thread_idx.x < (
                    4 * min(group, 8) + 128
                ):
                    exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                        partition, batch_size
                    )
                    var q_heads = get_q_head_idx(position, lane)

                    comptime for i in range(q_heads.size):
                        var q_head_idx = q_heads[i]
                        exp_sum_ptr[q_head_idx] = Scalar[
                            PartitionType.accum_dtype
                        ](0)
                        qk_max_ptr[q_head_idx] = min_or_neg_inf[
                            PartitionType.accum_dtype
                        ]()

                write_output(position, rowsum, vectorize_o_reg_tile().fill(0))
                return

        named_barrier[Int32(num_softmax_threads + 2 * WARP_SIZE)]()
        var tmem_addr = ptr_tmem_addr[0]

        var s_tmem: UInt32
        var o_tmem: UInt32
        var p_tmem: UInt32 = 0
        comptime if use_p_smem:
            o_tmem = tmem_addr  # bank 0
            s_tmem = tmem_addr + UInt32(1 << 20)  # bank 1
        else:
            comptime if num_softmax_warps > 4:
                if warp_group_idx != 1:  # elect_one_warp will be false
                    tmem_addr += 1 << 20
            s_tmem = tmem_addr
            o_tmem = tmem_addr + UInt32(MMA_N0 * num_s)
            p_tmem = tmem_addr + UInt32(MMA_N0 * num_s) + UInt32(MMA_N1)
        p_accumulator = UMMA0Type.c_t(s_tmem)
        # Use UMMA1Type.c_t for output_accumulator in all branches
        # (both TS and SS have the same c_t shape since MMA_N was aligned)
        output_accumulator = UMMA1Type.c_t(o_tmem)
        p_desc = UMMA1Type.a_mma_descriptor(p_tmem)

        @parameter
        @always_inline
        def wait_for_q_mul_k(read_idx: UInt32):
            p_acc = umma_0.wait_for_mma(p_accumulator)  # P is available
            _ = producer_mbar_kv[read_idx].arrive()
            comptime if use_p_smem:
                tcgen05_fence_after()
            p_acc.copy_to(p_reg_tile)
            umma_0.tmem_arrive()

        @parameter
        @always_inline
        def wait_for_p_mul_v(read_idx: UInt32):
            umma_1_ts.wait_for_mma()  # output is available
            _ = producer_mbar_kv[read_idx].arrive()
            comptime if use_p_smem:
                tcgen05_fence_after()
            output_accumulator.copy_to(output_reg_tile)

        var mask_status: TileMaskStatus
        while True:
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status != TileMaskStatus.FULL_MASK:
                break
            kv_tile_start_row += UInt32(BN)

        kv_pipeline_states = PipelineState[pipeline_stages]()
        # q_mul_k must wait on fetching q and k
        # therefore, we find `kv_tile_start_row` first.
        var read_idx_q: UInt32 = kv_pipeline_states.index()
        # q_mul_k(
        #     read_idx_q,
        #     kv_pipeline_states.phase(),
        # )
        kv_pipeline_states.step()

        wait_for_q_mul_k(read_idx_q)
        apply_mask(position, mask_status, kv_tile_start_row)

        comptime if not SinkType.is_null:
            # Include sink_weights in rowmax computation if present
            var q_head_indices = get_q_head_idx(position, lane)

            comptime for i in range(q_head_indices.size):
                var head_idx = q_head_indices[i]
                var sink_weight = sink_weights_ptr[head_idx] * log2e
                rowmax[i] = sink_weight.cast[accum_type]()

        # Compute initial rowmax
        var attention_rowmax = _rowmax_online_softmax[
            1, mma_thread_layout, use_exp2=True
        ](vectorize_p_reg_tile(), rowmax, init_rowmax=SinkType.is_null)

        rowmax.copy_from(attention_rowmax)

        comptime assert p_vec_output_layout.size() > 0, "layout: " + String(
            p_vec_output_layout
        )

        # Compute rowsum
        var attention_rowsum = _rowsum[mma_thread_layout](
            vectorize_p_reg_tile()
        )

        # Add sink weight contribution to rowsum
        comptime if not SinkType.is_null:
            var q_head_indices = get_q_head_idx(position, lane)

            comptime for i in range(q_head_indices.size):
                var head_idx = q_head_indices[i]
                var sink_weight = (
                    sink_weights_ptr[head_idx].cast[accum_type]() * log2e
                )
                var sink_contribution = exp2(sink_weight - rowmax[i])
                attention_rowsum[i] += sink_contribution[0]

        rowsum.copy_from(attention_rowsum)

        while True:
            kv_tile_start_row += UInt32(BN)
            if kv_tile_start_row >= end:
                break
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status == TileMaskStatus.FULL_MASK:
                continue

            named_barrier[Int32(num_softmax_threads)](5)

            # copy new pfrag, used by `p_mul_v` on next iter
            comptime if use_p_smem:
                # Wait for warp 1's SS MMA to finish reading P from SMEM
                # before we overwrite it.
                comptime p_swizzle = (
                    make_swizzle[
                        kv_type, TensorMapSwizzle.SWIZZLE_64B
                    ]() if kv_type.is_float8() else make_swizzle[
                        num_rows=WM // 2, row_size=MMA_N0, access_size=8
                    ]()
                )
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
                named_barrier[Int32(num_softmax_threads)]()
                umma_1_ts.tmem_arrive()
            else:
                p_desc.copy_from(UMMA0Type.c_t.rows_of_frags(p_reg_tile))
                umma_1_ts.tmem_arrive()

            # new pipeline states
            var read_idx_q: UInt32 = kv_pipeline_states.index()
            # start ummas
            # q_mul_k(
            #     read_idx_q, kv_pipeline_states.phase()
            # )  # can't rw `p_reg_tile`
            kv_pipeline_states.step()
            var read_idx_v: UInt32 = kv_pipeline_states.index()
            # p_mul_v(
            #     read_idx_v, kv_pipeline_states.phase(), output_scale
            # )  # can't rw output or pfrag
            # output_scale = 1
            kv_pipeline_states.step()
            wait_for_q_mul_k(read_idx_q)

            apply_mask(position, mask_status, kv_tile_start_row)
            # Compute rowmax for current scores
            var current_rowmax = _rowmax_online_softmax[
                1, mma_thread_layout, use_exp2=True
            ](vectorize_p_reg_tile(), rowmax, False)

            score_frag_rowmax = current_rowmax
            score_frag_rowsum = rebind[type_of(rowsum)](
                _rowsum[mma_thread_layout](vectorize_p_reg_tile())
            )

            _online_softmax_correction[use_exp2=True](rowmax, score_frag_rowmax)
            # rowmax now holds score_frag_rowmax
            # score_frag_rowmax now holds the correction

            comptime for i in range(num_rows_per_warp):
                rowsum[i] = (
                    rowsum[i] * score_frag_rowmax[i] + score_frag_rowsum[i]
                )

            wait_for_p_mul_v(read_idx_v)  # can rw output and pfrag
            correct_output(score_frag_rowmax, vectorize_o_reg_tile())
            output_accumulator.copy_from(output_reg_tile)

        # Final P write
        comptime if use_p_smem:
            # Wait for warp 1's SS MMA to finish reading P from SMEM
            comptime p_swizzle = (
                make_swizzle[
                    kv_type, TensorMapSwizzle.SWIZZLE_64B
                ]() if kv_type.is_float8() else make_swizzle[
                    num_rows=WM // 2, row_size=MMA_N0, access_size=8
                ]()
            )
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
            named_barrier[Int32(num_softmax_threads)](6)
            umma_1_ts.tmem_arrive()
        else:
            p_desc.copy_from(UMMA0Type.c_t.rows_of_frags(p_reg_tile))
            umma_1_ts.tmem_arrive()

        # p_mul_v(
        #     kv_pipeline_states.index(),
        #     kv_pipeline_states.phase(),
        #     output_scale,
        # )

        comptime if PartitionType.do_partition:
            # Only the first thread of each row
            if umod(thread_idx.x, 4) == 0 and thread_idx.x < (
                4 * min(group, 8) + 128
            ):
                exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                    partition, batch_size
                )
                var q_heads = get_q_head_idx(position, lane)

                comptime for i in range(q_heads.size):
                    var q_head_idx = q_heads[i]
                    exp_sum_ptr[q_head_idx] = rebind[
                        Scalar[PartitionType.accum_dtype]
                    ](rowsum[i])
                    qk_max_ptr[q_head_idx] = rebind[
                        Scalar[PartitionType.accum_dtype]
                    ](rowmax[i])

        comptime for row in range(num_rows_per_warp):
            comptime if mask_unselected:
                # All-`-1` query: every column poisoned -> rowmax==-inf,
                # rowsum==0, so recip would be inf and O would be NaN.  Fold the
                # reciprocal to 0 (O -> 0), matching the prefill / combine
                # numerator-zero convention (C==0 -> O=0, LSE=-inf).  Only the
                # degenerate row is affected; a real row has rowsum > 0.
                var rs = rowsum[row][0]
                rowsum[row] = recip(rs)[0] if rs > Scalar[accum_type](0) else (
                    Scalar[accum_type](0)
                )
            else:
                rowsum[row] = recip(rowsum[row])[0]

        umma_1_ts.wait_for_mma()
        comptime if use_p_smem:
            tcgen05_fence_after()
        output_accumulator.copy_to(output_reg_tile)

        comptime assert type_of(output_reg_tile).layout[1].size() > 1, (
            "output_reg_tile.layout = "
            + String(type_of(output_reg_tile).layout)
            + "\n"
        )
        write_output(position, rowsum, vectorize_o_reg_tile())
        # don't arrive


# ===-----------------------------------------------------------------------===#
# Enqueue + dispatch chain (mirrors mha_1q's 4-stage threading)
# ===-----------------------------------------------------------------------===#


@always_inline
def _msa_sm100_enqueue[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
    per_token_index: Bool = False,
    causal: Bool = False,
    mask_unselected: Bool = False,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM=config.block_m(),
        depth=config.depth,
        group=group,
        decoding=_is_decoding[MaxSeqLenType](),
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
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    d_indices: UnsafePointer[Int32, MutAnyOrigin],
    indices_stride: Int,
    scale: Float32,
    batch_size: UInt32,
    total_q: UInt32,
    max_seq_len: MaxSeqLenType,
    kv_logical_pos: UnsafePointer[Int32, MutAnyOrigin],
    q_positions: UnsafePointer[Int32, MutAnyOrigin],
    valid_length: ValidLengthType,
    kv_input_row_offsets: KVRowOffsetsType,
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    ctx: DeviceContext,
    # Real-token count of the topk band when the last gathered block is partial
    # (cache_length % BN != 0).  `None` => full band (topk*BN).  Host contract:
    # the partial block must be the LAST gathered slot and all earlier slots
    # full, so the valid columns are the contiguous prefix [0, valid_key) and
    # the dense mask's `score_col < num_keys` compare trims the pad with no
    # extra work.  Only narrows masking; gather/loop/split-K keep topk*BN.
    valid_key: OptionalReg[UInt32] = None,
) raises:
    comptime kernel_sm100 = _msa_sm100[
        KVLUTType,
        output_type,
        MaskType,
        SchedulerType,
        config,
        group,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        _is_cache_length_accurate,
        MaxSeqLenType,
        PartitionType,
        swizzle_mode,
        # split_k is a comptime property of PartitionType (False NoPartition,
        # True SplitKPartition), so this stays a compile-time specialization.
        split_k=PartitionType.do_partition,
        sink=False,
        extra_kv=False,
        variable_topk=False,
        fp8=False,
        per_token_index=per_token_index,
        causal=causal,
        mask_unselected=mask_unselected,
    ]
    comptime PackType = Pack[
        MaskType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxSeqLenType,
        PartitionType,
    ]
    var pack: PackType = {
        mask,
        scheduler,
        valid_length,
        sink_weights,
        kv_input_row_offsets,
        max_seq_len,
        partition,
    }

    var block_x: UInt32 = partition.num_partitions()

    comptime max_tmem_cols = 512
    comptime BN = config.block_n()
    comptime BM_enq = config.block_m()

    # `indices_stride` counts BLOCKS; the kernel's key (column) axis is in TOKENS
    # = topk_blocks * BN.  Derive num_keys here (single source of truth) so both
    # dispatch entries stay block-unit; `_apply_mask` / `position.num_keys` then
    # see the token count and OOB-mask columns >= topk tokens.
    var num_keys_tokens: UInt32 = UInt32(indices_stride) * UInt32(BN)
    # The mask bound (`position.num_keys`) gets the shrunken `valid_key` when the
    # last gathered block is partial; the gather/loop/split-K keep
    # `num_keys_tokens` (topk*BN).  Off => full band => bit-identical.
    var num_keys_mask: UInt32 = (
        valid_key.value() if valid_key else num_keys_tokens
    )
    comptime use_p_smem = config.padded_depth + BN // 2 > max_tmem_cols
    comptime num_s = (
        max_tmem_cols
        // BN if use_p_smem else (
            max_tmem_cols - (BN // 2) - config.padded_depth
        )
        // BN
    )
    comptime p_smem_bytes = BM_enq * BN * size_of[
        config.dtype
    ]() if use_p_smem else 0
    comptime assert size_of[output_type]() >= size_of[config.dtype]()
    comptime q_extra_for_output_bytes = BM_enq * config.padded_depth * (
        size_of[output_type]() - size_of[config.dtype]()
    )
    # Dense extra SMEM only (block-sparse load reuses kv_smem + the dense mbars;
    # no separate index SMEM/mbars).
    comptime extra_B200_smem = (
        (2 * num_s + 3) * 8 + p_smem_bytes + q_extra_for_output_bytes
    )
    comptime smem_use = config.shared_mem_bytes[
        True, sm_90=True
    ]() + extra_B200_smem
    comptime num_threads = config.num_threads[True]()
    logger.info("------ Dispatching to SM100 Sparse MHA (MSA)-1Q ------")
    logger.info(
        "QKV Type: ",
        KVLUTType.dtype,
        "Depth:",
        config.depth,
        "Number of Q // KV Heads:",
        config.num_heads,
        "//",
        config.num_heads // group,
        "Batch Size:",
        batch_size,
        "topk:",
        indices_stride,
    )
    # Per-token prefill puts the token axis on grid.x: grid (total_q,
    # num_kv_heads, 1).  This removes the 65535 grid.z cap a single long prompt
    # would hit with the decode layout (token on grid.z).  Decode keeps the
    # scheduler layout (block_x partitions, num_kv_heads, batch on grid.z).
    var grid: Tuple[Int, Int, Int]
    comptime if per_token_index:
        comptime num_kv_heads = config.num_heads // group
        grid = (Int(total_q), num_kv_heads, 1)
    else:
        grid = SchedulerType.grid_dim(batch_size, block_x)
    ctx.enqueue_function[kernel_sm100](
        q_tma_op,
        k_tma_op,
        v_tma_op,
        o_ptr_arg,
        kv_lut,
        d_indices,
        indices_stride,
        scale,
        batch_size,
        total_q,
        num_keys_mask,
        kv_logical_pos,
        q_positions,
        pack,
        grid_dim=grid,
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


@always_inline
def msa_sm100_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
    # True once an indexer can pad `d_indices` with `-1`: skip those blocks
    # (load block 0, poison their columns).  False => bit-identical to HEAD.
    mask_unselected: Bool = False,
](
    output: DeviceBuffer[output_type],
    q_arg: DeviceBuffer[q_type],
    k: KVType,
    v: KVType,
    d_indices: UnsafePointer[Int32, MutAnyOrigin],
    indices_stride: Int,  # topk in BLOCKS
    num_rows_q: Int,
    mask: MaskType,
    valid_length: DeviceBuffer[DType.uint32],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[DType.uint32]],
    batch_size_arg: Int,
    partition: PartitionType,
    ctx: DeviceContext,
    # Causal inputs.  `kv_logical_pos` is `[head_kv, total_q, topk]`
    # Int32 (logical KV position per slot); `q_positions` is `[total_q]` Int32
    # (query logical position per token).  Both `None` => no causal masking
    # (decode is usually past-only, so causal is a no-op unless the indexer
    # over-selects future slots).
    kv_logical_pos: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    q_positions: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    # Partial-last-block pad mask.  Pass `(topk_blocks-1)*BN + valid_cols` when
    # cache_length % BN != 0; the dense mask then trims the partial block's pad.
    # Host contract: the partial block is the LAST gathered slot (earlier slots
    # full), so the valid columns are a contiguous prefix.  `None` => full band.
    valid_key: OptionalReg[UInt32] = None,
) raises:
    """Dispatch entry for the SM100 block-sparse MHA decode kernel.

    The KV bulk-TMA tiles are dense, but each tile's row base is chosen by a
    block id from `d_indices` (`indices_stride` == topk BLOCKS).  `KVType` is
    generic over `MHAOperand`, so the same path serves a flat
    `LayoutTensorMHAOperand` or a whole-block paged `KVCacheMHAOperand`
    (page_size == BN; the page table resolves each block).  Split-K supported
    (pass a `SplitKPartition`; reduce with the shared `mha_splitk_reduce`).
    BF16 only, fixed topk-blocks.
    """
    comptime assert _is_decoding[MaxPromptLenType](), "msa_1q is decode-only"
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match."
    comptime assert config.dtype.is_half_float(), "BF16 only"

    comptime KVPtrT = UnsafePointer[Int32, MutAnyOrigin]

    comptime new_config = MHAConfig[config.dtype](
        config.num_heads,
        config.depth,
        num_queries_per_block=64,
        num_keys_per_block=config.num_keys_per_block,
        BK=config.BK,
        num_pipeline_stages=2 if config.padded_depth >= 512 else 4,
    )
    comptime BM = new_config.block_m()
    comptime BK = new_config.padded_depth
    comptime BN = new_config.block_n()
    comptime assert BM % 64 == 0, "SM90 requires BM%64==0"
    comptime assert BK % 64 == 0, "B200 requires BK%64 (128B swizzle)"
    comptime num_threads = new_config.num_threads[True]()
    comptime assert num_threads % 128 == 0
    comptime assert new_config.algorithm == FlashAttentionAlgorithm(3)

    var q = rebind[UnsafePointer[Scalar[KVType.dtype], MutAnyOrigin]](q_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    _ = max_cache_valid_length_arg  # decode num_keys derived from topk-blocks

    comptime num_scheduler_heads = config.num_heads // group
    comptime swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

    q_tma_op = rebind[
        QTMATile[
            KVType.dtype,
            swizzle_mode,
            BM=new_config.block_m(),
            depth=new_config.depth,
            group=group,
            decoding=_is_decoding[MaxPromptLenType](),
        ]
    ](
        q_tma[
            swizzle_mode,
            BM=BM,
            depth=new_config.depth,
            q_num_heads=new_config.num_heads,
            group=group,
            decoding=_is_decoding[MaxPromptLenType](),
        ](ctx, q, num_rows_q)
    )

    # Block-sparse KV descriptors: contiguous bulk-TMA tiles (BF16 D=128
    # SWIZZLE_128B), same as dense MHA -- each tile loads one 128-token block.
    comptime kv_sub_BN = _kv_sub_tile_rows(
        new_config.block_n(), KVType.page_size
    )
    k_tma_op = k.create_tma_tile[
        swizzle_mode,
        BN=kv_sub_BN,
        depth=new_config.depth,
        BK=new_config.padded_depth,
    ](ctx)
    v_tma_op = v.create_tma_tile[
        swizzle_mode,
        BN=kv_sub_BN,
        depth=new_config.depth,
        BK=new_config.padded_depth,
    ](ctx)

    comptime SchedulerType = TransientScheduler[
        UInt32(1),
        UInt32(num_scheduler_heads),
        flip_prompt_idx=False,
    ]
    var scheduler: SchedulerType = SchedulerType()

    comptime SinkType = NullPointer[KVType.dtype]
    comptime sink_ptr: SinkType = {}

    @parameter
    @always_inline
    def with_kv_offsets[
        KVRowOffsetsType: OptionalPointer
    ](kv_row_offsets: KVRowOffsetsType) raises:
        @parameter
        @always_inline
        def with_valid_length[
            ValidLengthType: OptionalPointer
        ](valid_len: ValidLengthType) raises:
            @parameter
            @always_inline
            def with_causal[causal: Bool](kv_pos: KVPtrT, q_pos: KVPtrT) raises:
                _msa_sm100_enqueue[
                    SchedulerType=SchedulerType,
                    KVLUTType=KVType,
                    output_type=output_type,
                    MaxSeqLenType=MaxPromptLenType,
                    PartitionType=PartitionType,
                    MaskType=MaskType,
                    config=new_config,
                    group=group,
                    SinkType=SinkType,
                    ValidLengthType=ValidLengthType,
                    KVRowOffsetsType=KVRowOffsetsType,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    swizzle_mode=swizzle_mode,
                    causal=causal,
                    mask_unselected=mask_unselected,
                ](
                    scheduler,
                    q_tma_op,
                    k_tma_op,
                    v_tma_op,
                    output,
                    k,
                    d_indices,
                    indices_stride,
                    scale,
                    batch_size,
                    # total_q: decode keys the index on prompt_idx
                    # (per_token_index defaults False), so the token-axis stride
                    # is never read; pass batch_size to satisfy the shared kernel
                    # signature.
                    batch_size,
                    max_prompt_len_arg,
                    kv_pos,
                    q_pos,
                    valid_len,
                    kv_row_offsets,
                    sink_ptr,
                    partition,
                    mask,
                    ctx,
                    valid_key,
                )

            # Causal needs both inputs; either missing => no masking (kernel args
            # go unread, so dangling is safe).
            if kv_logical_pos and q_positions:
                with_causal[True](kv_logical_pos.value(), q_positions.value())
            else:
                with_causal[False](
                    KVPtrT.unsafe_dangling(), KVPtrT.unsafe_dangling()
                )

        comptime if ragged:
            with_valid_length[NonNullPointer[DType.uint32]]({valid_length})
        else:
            with_valid_length[NullPointer[DType.uint32]]({})

    if kv_input_row_offsets:
        with_kv_offsets[NonNullPointer[DType.uint32]](
            {kv_input_row_offsets.value().ptr}
        )
    else:
        with_kv_offsets[NullPointer[DType.uint32]]({})


@always_inline
def msa_sm100_prefill_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    output_type: DType,
    PartitionType: MHAPartitionScheme,
    //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: DeviceBuffer[q_type],
    k: KVType,
    v: KVType,
    d_indices: UnsafePointer[Int32, MutAnyOrigin],
    indices_stride: Int,  # topk in BLOCKS
    total_q: Int,  # B*S query tokens (rows of the [head_kv, total_q, topk] idx)
    mask: MaskType,
    valid_length: DeviceBuffer[DType.uint32],
    scale: Float32,
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[DType.uint32]],
    partition: PartitionType,
    ctx: DeviceContext,
    # Causal inputs.  `kv_logical_pos` is `[head_kv, total_q, topk]`
    # Int32 (logical KV position per slot); `q_positions` is `[total_q]` Int32
    # (query logical position per token, indexed by the global query row).  Both
    # `None` => no causal masking (causality baked into the indices).
    kv_logical_pos: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    q_positions: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    # Partial-last-block pad mask (same contract as the decode entry): pass
    # `(topk_blocks-1)*BN + valid_cols` with the partial block last.  `None` =>
    # full band.  Decode is the usual partial-block path; threaded here for
    # parity (a per-token prefill caller can pass it too).
    valid_key: OptionalReg[UInt32] = None,
) raises:
    """Per-token sparse MHA prefill for SM100.

    One CTA per (query token, kv-head): the `total_q` query tokens are enumerated
    on grid.x through the `_msa_sm100` body (`StaticInt[1]` tiles).  Each CTA
    reads its own index row from `d_indices` (`[head_kv, total_q, topk]`), keyed
    on the global query row.  Causality is either baked into the indices (both
    causal inputs `None`) or applied in-kernel (pass `kv_logical_pos` /
    `q_positions`).

    `ragged` is a host-packing concern only: each query row owns its own index
    row, so the kernel maps `block_idx.x` straight to the global row and never
    consults `cu_seqlens` (total_q rides grid.x, no 65535 grid.z cap).  No
    split-K.  BF16 only, fixed topk-blocks.
    """
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match."
    comptime assert config.dtype.is_half_float(), "BF16 only"
    comptime assert (
        not PartitionType.do_partition
    ), "per-token prefill split-K unsupported"

    comptime KVPtrT = UnsafePointer[Int32, MutAnyOrigin]

    # Reuse the decode tile shape: a query token's `group` heads are one M-tile.
    comptime new_config = MHAConfig[config.dtype](
        config.num_heads,
        config.depth,
        num_queries_per_block=64,
        num_keys_per_block=config.num_keys_per_block,
        BK=config.BK,
        num_pipeline_stages=2 if config.padded_depth >= 512 else 4,
    )
    comptime BM = new_config.block_m()
    comptime BK = new_config.padded_depth
    comptime BN = new_config.block_n()
    comptime assert BM % 64 == 0, "SM90 requires BM%64==0"
    comptime assert BK % 64 == 0, "B200 requires BK%64 (128B swizzle)"
    comptime num_threads = new_config.num_threads[True]()
    comptime assert num_threads % 128 == 0
    comptime assert new_config.algorithm == FlashAttentionAlgorithm(3)

    var q = rebind[UnsafePointer[Scalar[KVType.dtype], MutAnyOrigin]](q_arg)
    # Scheduler enumerates one work-unit per (query token, kv-head): batch_size
    # = total_q, head dim = num_kv_heads, max_seq_len = 1 (decode-shaped tile).
    var total_q_u: UInt32 = UInt32(total_q)

    comptime num_scheduler_heads = config.num_heads // group
    comptime swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    comptime MaxSeqLenType = StaticInt[1]

    # Q descriptor is decode-shaped (`MaxSeqLenType == StaticInt[1]`); with
    # max_seq_len == 1 the per-token global row is `prompt_idx`, indexing the
    # [total_q, num_q_heads, depth] Q buffer exactly as
    # [total_q, num_q_heads//group, group, depth].  Build it with the same
    # `decoding=_is_decoding[MaxSeqLenType]()` expression the enqueue's param
    # uses, so the TMA-tile type folds to a consistent rank.
    q_tma_op = rebind[
        QTMATile[
            KVType.dtype,
            swizzle_mode,
            BM=new_config.block_m(),
            depth=new_config.depth,
            group=group,
            decoding=_is_decoding[MaxSeqLenType](),
        ]
    ](
        q_tma[
            swizzle_mode,
            BM=BM,
            depth=new_config.depth,
            q_num_heads=new_config.num_heads,
            group=group,
            decoding=_is_decoding[MaxSeqLenType](),
        ](ctx, q, total_q)
    )

    # Block-sparse KV descriptors: contiguous bulk-TMA tiles, one 128-token block
    # per tile (same as dense MHA).
    comptime kv_sub_BN = _kv_sub_tile_rows(
        new_config.block_n(), KVType.page_size
    )
    k_tma_op = k.create_tma_tile[
        swizzle_mode,
        BN=kv_sub_BN,
        depth=new_config.depth,
        BK=new_config.padded_depth,
    ](ctx)
    v_tma_op = v.create_tma_tile[
        swizzle_mode,
        BN=kv_sub_BN,
        depth=new_config.depth,
        BK=new_config.padded_depth,
    ](ctx)

    comptime SchedulerType = TransientScheduler[
        UInt32(1),
        UInt32(num_scheduler_heads),
        flip_prompt_idx=False,
    ]
    var scheduler: SchedulerType = SchedulerType()

    comptime SinkType = NullPointer[KVType.dtype]
    comptime sink_ptr: SinkType = {}

    @parameter
    @always_inline
    def with_kv_offsets[
        KVRowOffsetsType: OptionalPointer
    ](kv_row_offsets: KVRowOffsetsType) raises:
        @parameter
        @always_inline
        def with_valid_length[
            ValidLengthType: OptionalPointer
        ](valid_len: ValidLengthType) raises:
            @parameter
            @always_inline
            def with_causal[causal: Bool](kv_pos: KVPtrT, q_pos: KVPtrT) raises:
                _msa_sm100_enqueue[
                    SchedulerType=SchedulerType,
                    KVLUTType=KVType,
                    output_type=output_type,
                    MaxSeqLenType=MaxSeqLenType,
                    PartitionType=PartitionType,
                    MaskType=MaskType,
                    config=new_config,
                    group=group,
                    SinkType=SinkType,
                    ValidLengthType=ValidLengthType,
                    KVRowOffsetsType=KVRowOffsetsType,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    swizzle_mode=swizzle_mode,
                    per_token_index=True,
                    causal=causal,
                ](
                    scheduler,
                    q_tma_op,
                    k_tma_op,
                    v_tma_op,
                    output,
                    k,
                    d_indices,
                    indices_stride,
                    scale,
                    total_q_u,  # batch_size: one work-unit per query token
                    total_q_u,  # total_q: index row stride over the token axis
                    MaxSeqLenType(),
                    kv_pos,
                    q_pos,
                    valid_len,
                    kv_row_offsets,
                    sink_ptr,
                    partition,
                    mask,
                    ctx,
                    valid_key,
                )

            # Causal needs both inputs; either missing => no masking (kernel args
            # go unread, so dangling is safe).
            if kv_logical_pos and q_positions:
                with_causal[True](kv_logical_pos.value(), q_positions.value())
            else:
                with_causal[False](
                    KVPtrT.unsafe_dangling(), KVPtrT.unsafe_dangling()
                )

        comptime if ragged:
            with_valid_length[NonNullPointer[DType.uint32]]({valid_length})
        else:
            with_valid_length[NullPointer[DType.uint32]]({})

    if kv_input_row_offsets:
        with_kv_offsets[NonNullPointer[DType.uint32]](
            {kv_input_row_offsets.value().ptr}
        )
    else:
        with_kv_offsets[NullPointer[DType.uint32]]({})
