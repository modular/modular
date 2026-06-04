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
"""Softmax warp group logic for FA4 (SM100 Flash Attention)."""

from std.math import exp2, recip, align_up
from std.math.constants import log2e
from std.memory import bitcast
from std.sys import size_of, get_defined_int
from std.sys.info import _accelerator_arch
import std.gpu.primitives.warp as warp
from std.gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from std.gpu.memory import AddressSpace, fence_async_view_proxy
from std.gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    umma_arrive_leader_cta,
)
from std.gpu.primitives.cluster import block_rank_in_cluster
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_release_allocation_lock,
    tcgen05_store_wait,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TMEM_LOWER_ROW_OFFSET,
)
from std.gpu.primitives.warp import _vote_nvidia_helper
from layout import row_major, stack_allocation as tt_stack_allocation
from layout.swizzle import make_swizzle
from layout.tma_async import RaggedTMA3DTile
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    EnableForcedOrdering,
    EnableEarlyAdd,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    MBarType,
    TMemTile,
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    STMatrixLayout,
    STMatrixOffsets,
    break_into_powers_of_two,
    elect,
    llvm_opaque_tid,
    add_ftz,
    sub_ftz,
    mul_ftz,
    fma_ftz,
    exp2_emulation,
    maximum,
    apply_mask,
    peel_mask,
)
from nn.attention.gpu.nvidia.sm90.attention import (
    MHAPosition,
    NullPointer,
    OptionalPointer,
    _LocalTT,
    _SharedMemTT,
    output_reg_to_smem_st_matrix,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus, MaskStrategy
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo
from nn.attention.mha_utils import OptionallyStaticInt, _is_decoding
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from .smem import SM100AttentionSMem


@always_inline
def fa4_scale_write_output[
    output_type: DType,
    //,
    config: FA4Config,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    inv_row_sum: Float32,
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    o_tmem_arg: TMemTile[
        DType.float32, config.BM // config.num_qo, config.padded_ov_depth
    ],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        # `config.BM // config.num_qo` is "rows this WG writes": 128 in
        # 2Q (BM=256, two WGs split rows) and 128 in 1Q (BM=128, one WG
        # writes the full set in the T==1 fast path; the multi-tile 1Q
        # path uses fa4_lse_combine_write instead). Same numeric value
        # in both modes.
        BM=config.BM // config.num_qo,
        BN=config.ov_depth,
        group=config.group if config.fuse_gqa else 1,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    comptime accum_dtype = DType.float32

    comptime swizzle_granularity = output_swizzle_mode.bytes() // size_of[
        output_type
    ]()
    comptime iters = config.padded_ov_depth // swizzle_granularity
    # Rows this WG writes. 128 in both 2Q (BM=256, two WGs split rows
    # in half) and 1Q (BM=128, one WG owns all rows in the T==1
    # fast path).
    comptime bm_per_q = config.BM // config.num_qo

    comptime ST = STMatrixLayout[
        bm_per_q,
        swizzle_granularity,
        num_threads=WARPGROUP_SIZE,
        accum_dtype_size=4,
    ]
    comptime num_rows = ST.vec_local_layout[0].size()

    comptime swizzle = make_swizzle[output_type, output_swizzle_mode]()

    comptime swizzle_block_size: UInt32 = UInt32(
        WARP_SIZE * swizzle_granularity
    )

    e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Allocate register tiles for double-buffered pipeline.
    comptime ChunkTMemType = TMemTile[
        accum_dtype, bm_per_q, swizzle_granularity
    ]
    var o_cur = ChunkTMemType.allocate_register_tile[
        num_threads=WARPGROUP_SIZE
    ]()

    # --- Composable pipeline primitives, parameterized by m_half ---

    @always_inline
    @parameter
    def load_chunk[col: Int, m_half: Int](dst: type_of(o_cur)):
        """Async tmem load for one M-half of column `col`."""
        comptime load_dtype = DType.uint32
        chunk_tmem_addr = o_tmem_arg.tmem_addr + UInt32(
            col * swizzle_granularity
        )

        @parameter
        @always_inline
        def load_fn[pow_two: Int, local_offset: Int]():
            comptime assert pow_two + local_offset <= ST.repeat
            comptime if pow_two > 0:
                comptime offsets = STMatrixOffsets[
                    bm_per_q,
                    swizzle_granularity,
                    num_threads=WARPGROUP_SIZE,
                    accum_dtype_size=4,
                    curr_repeat=pow_two,
                    cumulative_repeat=local_offset,
                    m_mma=m_half,
                ]()
                comptime assert (
                    offsets.local_frag_size_b32 % 2 == 0
                ), "local_frag_size_b32 must be even for f32x2 stores"
                tmem = chunk_tmem_addr + UInt32(offsets.tmem_offset)
                frag = tcgen05_ld[
                    datapaths=16,
                    bits=ST.bits,
                    repeat=pow_two,
                    dtype=load_dtype,
                    pack=False,
                    width=offsets.local_frag_size_b32,
                ](tmem)

                # Store as f32x2 pairs so SROA decomposes the alloca
                # into individual f32x2 pieces instead of <8 x i32>.
                comptime for _i in range(offsets.local_frag_size_b32 // 2):
                    var pair = SIMD[DType.float32, 2](
                        bitcast[DType.float32](frag[2 * _i]),
                        bitcast[DType.float32](frag[2 * _i + 1]),
                    )
                    dst.ptr.store(
                        offsets.ptr_offset + 2 * _i,
                        pair,
                    )

        comptime max_value = 64 if ST.bits == 128 else 32
        break_into_powers_of_two[
            func=load_fn, N=ST.repeat, max_value=max_value
        ]()

    load_chunk[0, 0](o_cur)
    inv_row_sums = tt_stack_allocation[
        dtype=accum_dtype, address_space=AddressSpace.LOCAL
    ](row_major[num_rows]())
    lane = local_row % 32
    lane_row = lane // 4

    comptime for i in range(num_rows):
        inv_row_sums[i] = warp.shuffle_idx(
            inv_row_sum, lane_row + UInt32(8 * i)
        )
    o_smem = o_smem_arg + local_warp_idx * swizzle_block_size

    @always_inline
    @parameter
    def scale_half[m_half: Int](o: type_of(o_cur)):
        """Scale one M-half's registers by `inv_row_sum`."""
        comptime rows_per_half = ST.num_row_blocks_per_mma
        comptime start = m_half * rows_per_half
        comptime for i in range(start, start + rows_per_half):
            irs = o.element_type(inv_row_sums[i])
            comptime for k in range(o.layout[1].size()):
                o[i, k] *= irs

    @always_inline
    @parameter
    def write_to_smem[j: Int, m_half: Int](o: type_of(o_cur)):
        """Write one M-half of column `j` to smem."""
        comptime datapath_offset: UInt32 = UInt32(
            16 * m_half * swizzle_granularity
        )
        comptime ofs = m_half * ST.frag_size
        comptime reg_layout = row_major[1, ST.frag_size]()
        var rows_of_o_frags = _LocalTT[accum_dtype, reg_layout](
            o.ptr + ofs, reg_layout
        )

        comptime warp_smem_offset: UInt32 = datapath_offset + UInt32(
            j * bm_per_q * swizzle_granularity
        )
        comptime smem_layout = row_major[16, swizzle_granularity]()
        var accum_smem_warp_tile = _SharedMemTT[output_type, smem_layout](
            o_smem + warp_smem_offset, smem_layout
        )

        output_reg_to_smem_st_matrix[
            BM=16,
            swizzle=swizzle,
            num_consumer=1,
        ](
            lane,
            local_warp_group_idx=0,
            output_reg_tile=rows_of_o_frags,
            accum_smem_tile=accum_smem_warp_tile,
        )

    @always_inline
    @parameter
    def sync_and_tma_store[j: Int]():
        """Barrier sync + TMA store for column `j`."""
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

        if local_warp_idx == 0:
            if e != 0:
                fence_async_view_proxy()
            if e != 0:
                ragged_tma_store.async_copy_from_col[j](
                    o_smem_arg,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                )
            if e != 0:
                cp_async_bulk_commit_group()

    # --- Pipeline loop ---

    # Prologue: load column 0, m_half=1 into o_cur (m_half=0 was already
    # loaded above).
    load_chunk[0, 1](o_cur)

    comptime for iter in range(iters):
        # Each 'iter' processes one column (column 'iter') in two M-halves.
        comptime next_iter = iter + 1
        scale_half[0](o_cur)
        write_to_smem[iter, 0](o_cur)

        comptime if next_iter < iters:
            load_chunk[next_iter, 0](o_cur)

        scale_half[1](o_cur)
        write_to_smem[iter, 1](o_cur)

        comptime if next_iter < iters:
            load_chunk[next_iter, 1](o_cur)

        sync_and_tma_store[iter]()

    # Wait for all TMA stores to complete
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_lse_combine_write[
    output_type: DType,
    //,
    config: FA4Config,
    wg_j_offset: Int,
    iters_per_wg: Int,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    final_scale_local: Float32,
    final_scale_peer: Float32,
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    own_o_tmem: TMemTile[DType.float32, config.BM, config.padded_ov_depth],
    peer_o_tmem: TMemTile[DType.float32, config.BM, config.padded_ov_depth],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        # 1Q only: equals config.BM (= 128); kept as `config.BM //
        # config.num_qo` for typewise consistency with the fa4_softmax
        # signature and kernel.mojo construction (which use the same
        # expression to give 128 in both 2Q and 1Q).
        BM=config.BM // config.num_qo,
        BN=config.ov_depth,
        group=config.group if config.fuse_gqa else 1,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """LSE-combine two TMEM_O fragments and TMA-store a depth-column slice.

    1Q-only sibling of `fa4_scale_write_output`. Each WG handles a disjoint
    range `j in [wg_j_offset, wg_j_offset + iters_per_wg)` of swizzle-block
    columns. For each `j`, the WG loads both its own and the peer's TMEM_O
    fragments, combines them in registers via per-row scales
    (`final_scale_local` for own, `final_scale_peer` for peer), writes the
    combined output to the shared `o_smem_arg` at the `j` slot, then
    TMA-stores that slot to gmem. Both WGs target the same `BM` Q rows but
    disjoint depth columns, so smem and gmem regions never overlap.

    The caller must have already waited on both `pipeline_o0` and
    `pipeline_o1` producer barriers (and issued `tcgen05_fence_after()`)
    before invoking this helper, so the TMEM fragments are visible.
    """
    comptime assert config.num_qo == 1
    comptime accum_dtype = DType.float32

    comptime swizzle_granularity = output_swizzle_mode.bytes() // size_of[
        output_type
    ]()
    comptime iters = config.padded_ov_depth // swizzle_granularity
    # Per-WG j-range is (wg_j_offset, iters_per_wg) and must stay in
    # bounds. Even iters → both WGs get iters/2 (the typical case for
    # depth >= 72 with swizzle 128B bf16, where padded_ov_depth in
    # {128, 256} gives iters in {2, 4}). Odd iters → ceil/floor split:
    # WG0 takes ceil(iters/2) starting at j=0 and WG1 takes floor(iters/2)
    # starting at j=ceil(iters/2). For iters == 1 (depth=64 single
    # swizzle block) WG0 gets the only block (iters_per_wg=1) and WG1
    # is skipped by the caller (iters_per_wg=0 here would underflow the
    # prologue load, so the caller must not invoke this helper for
    # iters_per_wg=0).
    comptime assert iters_per_wg >= 1, (
        "fa4_lse_combine_write requires at least one column block per"
        " call; the caller must skip WG1 when iters_per_wg would be 0"
        " (e.g. iters == 1 / depth=64)."
    )
    comptime assert wg_j_offset + iters_per_wg <= iters

    # Same STMatrixLayout config as fa4_scale_write_output, but with the
    # full `config.BM` (= 128 in 1Q) so one WG addresses all BM rows. The
    # numerical layout matches the 2Q half-BM helper because that one also
    # used BM=128 (config.BM // 2 in 2Q).
    comptime bm = config.BM
    comptime ST = STMatrixLayout[
        bm,
        swizzle_granularity,
        num_threads=WARPGROUP_SIZE,
        accum_dtype_size=4,
    ]
    comptime num_rows = ST.vec_local_layout[0].size()

    comptime swizzle = make_swizzle[output_type, output_swizzle_mode]()

    comptime swizzle_block_size: UInt32 = UInt32(
        WARP_SIZE * swizzle_granularity
    )

    e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Two register tiles, one per source (own + peer). Combined result
    # lives in `o_own` after each combine_half call.
    comptime ChunkTMemType = TMemTile[accum_dtype, bm, swizzle_granularity]
    var o_own = ChunkTMemType.allocate_register_tile[
        num_threads=WARPGROUP_SIZE
    ]()
    var o_peer = ChunkTMemType.allocate_register_tile[
        num_threads=WARPGROUP_SIZE
    ]()

    @always_inline
    @parameter
    def load_chunk[
        col: Int, m_half: Int
    ](tmem_base: UInt32, dst: type_of(o_own)):
        """Async tmem load of one M-half of column `col` from `tmem_base`.

        Same body as fa4_scale_write_output's load_chunk, but
        parameterized on a runtime base so it serves both the own and
        peer fragments.
        """
        comptime load_dtype = DType.uint32
        chunk_tmem_addr = tmem_base + UInt32(col * swizzle_granularity)

        @parameter
        @always_inline
        def load_fn[pow_two: Int, local_offset: Int]():
            comptime assert pow_two + local_offset <= ST.repeat
            comptime if pow_two > 0:
                comptime offsets = STMatrixOffsets[
                    bm,
                    swizzle_granularity,
                    num_threads=WARPGROUP_SIZE,
                    accum_dtype_size=4,
                    curr_repeat=pow_two,
                    cumulative_repeat=local_offset,
                    m_mma=m_half,
                ]()
                comptime assert (
                    offsets.local_frag_size_b32 % 2 == 0
                ), "local_frag_size_b32 must be even for f32x2 stores"
                tmem = chunk_tmem_addr + UInt32(offsets.tmem_offset)
                frag = tcgen05_ld[
                    datapaths=16,
                    bits=ST.bits,
                    repeat=pow_two,
                    dtype=load_dtype,
                    pack=False,
                    width=offsets.local_frag_size_b32,
                ](tmem)

                comptime for _i in range(offsets.local_frag_size_b32 // 2):
                    var pair = SIMD[DType.float32, 2](
                        bitcast[DType.float32](frag[2 * _i]),
                        bitcast[DType.float32](frag[2 * _i + 1]),
                    )
                    dst.ptr.store(
                        offsets.ptr_offset + 2 * _i,
                        pair,
                    )

        comptime max_value = 64 if ST.bits == 128 else 32
        break_into_powers_of_two[
            func=load_fn, N=ST.repeat, max_value=max_value
        ]()

    # Prologue (early): load m_half=0 of column `wg_j_offset` for both
    # own and peer. m_half=1 is loaded after the per-row scale setup,
    # mirroring the latency-hide in fa4_scale_write_output.
    load_chunk[wg_j_offset, 0](own_o_tmem.tmem_addr, o_own)
    load_chunk[wg_j_offset, 0](peer_o_tmem.tmem_addr, o_peer)

    # Broadcast per-row scales (lane row index, 8 row blocks per lane).
    fsl_stack = tt_stack_allocation[
        dtype=accum_dtype, address_space=AddressSpace.LOCAL
    ](row_major[num_rows]())
    fsp_stack = tt_stack_allocation[
        dtype=accum_dtype, address_space=AddressSpace.LOCAL
    ](row_major[num_rows]())
    lane = local_row % 32
    lane_row = lane // 4

    comptime for i in range(num_rows):
        fsl_stack[i] = warp.shuffle_idx(
            final_scale_local, lane_row + UInt32(8 * i)
        )
        fsp_stack[i] = warp.shuffle_idx(
            final_scale_peer, lane_row + UInt32(8 * i)
        )

    # WGs share the same `o_smem_arg` base; each warp offsets into its
    # warp slot. WG0/WG1 collisions are prevented at the `j` level
    # because the iteration ranges are disjoint.
    o_smem = o_smem_arg + local_warp_idx * swizzle_block_size

    @always_inline
    @parameter
    def combine_half[m_half: Int](own: type_of(o_own), peer: type_of(o_peer)):
        """Combine: own[i, k] = own[i, k] * fsl[i] + peer[i, k] * fsp[i].

        Stores the result back into `own`'s storage so the subsequent
        write_to_smem reuses the same code path as the 2Q helper.
        """
        comptime rows_per_half = ST.num_row_blocks_per_mma
        comptime start = m_half * rows_per_half
        comptime for i in range(start, start + rows_per_half):
            fsl_i = own.element_type(fsl_stack[i])
            fsp_i = peer.element_type(fsp_stack[i])
            comptime for k in range(own.layout[1].size()):
                own[i, k] = own[i, k] * fsl_i + peer[i, k] * fsp_i

    @always_inline
    @parameter
    def write_to_smem[j: Int, m_half: Int](o: type_of(o_own)):
        """Write one M-half of column `j` to the shared smem slot."""
        comptime datapath_offset: UInt32 = UInt32(
            16 * m_half * swizzle_granularity
        )
        comptime ofs = m_half * ST.frag_size
        comptime reg_layout = row_major[1, ST.frag_size]()
        var rows_of_o_frags = _LocalTT[accum_dtype, reg_layout](
            o.ptr + ofs, reg_layout
        )

        comptime warp_smem_offset: UInt32 = datapath_offset + UInt32(
            j * bm * swizzle_granularity
        )
        comptime smem_layout = row_major[16, swizzle_granularity]()
        var accum_smem_warp_tile = _SharedMemTT[output_type, smem_layout](
            o_smem + warp_smem_offset, smem_layout
        )

        output_reg_to_smem_st_matrix[
            BM=16,
            swizzle=swizzle,
            num_consumer=1,
        ](
            lane,
            local_warp_group_idx=0,
            output_reg_tile=rows_of_o_frags,
            accum_smem_tile=accum_smem_warp_tile,
        )

    @always_inline
    @parameter
    def sync_and_tma_store[j: Int]():
        """Per-WG named-barrier sync + TMA store for column `j`."""
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

        if local_warp_idx == 0:
            if e != 0:
                fence_async_view_proxy()
            if e != 0:
                ragged_tma_store.async_copy_from_col[j](
                    o_smem_arg,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                )
            if e != 0:
                cp_async_bulk_commit_group()

    # Prologue (late): load m_half=1 of column `wg_j_offset` for both
    # own and peer.
    load_chunk[wg_j_offset, 1](own_o_tmem.tmem_addr, o_own)
    load_chunk[wg_j_offset, 1](peer_o_tmem.tmem_addr, o_peer)

    # Pipeline loop over this WG's depth-column range.
    comptime for iter in range(iters_per_wg):
        comptime next_iter = iter + 1
        comptime j_global = wg_j_offset + iter
        comptime next_j_global = wg_j_offset + next_iter
        combine_half[0](o_own, o_peer)
        write_to_smem[j_global, 0](o_own)

        comptime if next_iter < iters_per_wg:
            load_chunk[next_j_global, 0](own_o_tmem.tmem_addr, o_own)
            load_chunk[next_j_global, 0](peer_o_tmem.tmem_addr, o_peer)

        combine_half[1](o_own, o_peer)
        write_to_smem[j_global, 1](o_own)

        comptime if next_iter < iters_per_wg:
            load_chunk[next_j_global, 1](own_o_tmem.tmem_addr, o_own)
            load_chunk[next_j_global, 1](peer_o_tmem.tmem_addr, o_peer)

        sync_and_tma_store[j_global]()

    # Wait for all TMA stores to complete.
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_softmax[
    QScaleType: OptionalPointer,
    KScaleType: OptionalPointer,
    qkv_dtype: DType,
    rope_dtype: DType,
    scale_dtype: DType,
    output_type: DType,
    MaskType: MHAMask,
    //,
    KVLUTType: MHAOperand,
    config: FA4Config[
        qkv_dtype, rope_dtype=rope_dtype, scale_dtype=scale_dtype
    ],
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
](
    smem: SM100AttentionSMem[config],
    score_row: UInt32,
    seq_info: SeqInfo,
    mask: MaskType,
    num_keys: UInt32,
    scale: Float32,
    max_seq_len: UInt32,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        _,
        # 2Q: BM=128 (one Q-half per WG). 1Q: BM=128 (both WGs cover the
        # full BM=128 and write disjoint depth-column ranges). Use
        # `config.BM // config.num_qo` so the type is consistent across
        # both modes; in 2Q this equals the historical `config.BM // 2`.
        BM=config.BM // config.num_qo,
        BN=config.ov_depth,
        group=config.group if config.fuse_gqa else 1,
    ],
    sink_weights: SinkType,
    q_scale: QScaleType = NullPointer[DType.float32, AddressSpace.SHARED](),
    k_scale: KScaleType = NullPointer[DType.float32, AddressSpace.SHARED](),
):
    # Local aliases matching SM100MHA2Q comptime members
    comptime qkv_type = KVLUTType.dtype
    comptime accum_dtype = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime group = config.group
    comptime fuse_gqa = config.fuse_gqa
    comptime BM_mask: Int = config.PairBM_eff()
    comptime padded_ov_depth = config.padded_ov_depth
    comptime page_size = KVLUTType.page_size
    comptime ragged = not ValidLengthType.is_null
    comptime cta_group = config.cta_group()

    var mbars = smem.misc_mbars()
    comptime MiscMBarsType = type_of(mbars)

    # MMA types for TMEM access
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        qkv_type,
        accum_dtype,
        MMA_M=config.MMA_M,
        MMA_N=BN,
        BK=align_up(config.qk_depth, config.MMA_K),
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=True,
        num_stages=config.num_qk_stages,
        cta_group=cta_group,
    ]
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        qkv_type,
        accum_dtype,
        MMA_M=config.MMA_M,
        MMA_N=padded_ov_depth,
        BK=BN,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        num_stages=config.num_pv_stages,
        cta_group=cta_group,
    ]
    comptime PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.qk_depth,
        config.padded_qk_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    var tmem_addr: UInt32 = smem.tmem_addr_ptr()[]
    var o_smem = smem.o_smem[output_type]()
    var o_prod_mbar: MBarType = (
        mbars.mbar_base + MiscMBarsType.O_producer_offset
    )
    var s_tmem: UInt32 = tmem_addr + UInt32(config.TMEM_S0)

    # var tid = UInt32(thread_idx.x)
    var tid = llvm_opaque_tid()
    var row = tid % 128
    var warp_idx: UInt32 = warp.broadcast(tid // 32)
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)
    # Per-thread BM row within the current Q tile.
    # 2Q (BM = 256): WG0 covers BM rows [0, 128) and WG1 covers
    # [128, 256), so `tid` directly indexes the BM row.
    # 1Q (BM = 128): both WGs share the same BM rows [0, 128); folding
    # WG1's `tid` (128..255) back to [0, 128) via `tid % BM` ensures
    # the per-thread (Q row, head) mapping is identical across WGs.
    # Using bare `tid` in 1Q would shift WG1's score_row by `BM_eff`,
    # which leaks OOB K positions into the softmax for tiles whose
    # `score_row + BM_eff` exceeds `num_keys` (the OOB columns are not
    # masked by SlidingWindow's UPPER|LOWER strategy and TMA-padded
    # K=0 / V=0 then dilutes the output toward 0).
    var thread_tile_row: UInt32 = tid % UInt32(config.BM)

    var cta_q_offset: UInt32 = 0
    comptime if config.pair_cta:
        cta_q_offset = UInt32(
            warp.broadcast(block_rank_in_cluster()) % 2
        ) * UInt32(config.BM_eff())

    # 2-Q path: S1 is at +BN columns
    s_tmem += UInt32(config.BN) * warp_group_idx

    p_tmem = s_tmem
    c_tmem = p_tmem + UInt32(config.BN // 2)
    s_tile = UMMA0Type.CType(s_tmem)
    p_tile = UMMA1Type.AType(p_tmem)

    var pipeline_s = mbars.consumer_s(warp_group_idx)
    pipeline_c = mbars.producer_c(warp_group_idx)
    var order_phase: UInt32 = 1 - warp_group_idx

    var order_s_wait: Optional[MBarType] = None
    var order_s_arrive: Optional[MBarType] = None
    comptime if EnableForcedOrdering:
        order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
        order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)

    # When fuse_gqa, head_idx is a kv_head_idx
    # the output will match, so `head_idx` is what we use for writing
    # sink and mask want q_head_idx
    var head_idx: UInt32 = seq_info.head_idx
    var q_head_idx: UInt32 = head_idx
    comptime if config.fuse_gqa:
        q_head_idx = UInt32(config.group) * head_idx + row % UInt32(
            config.group
        )

    var scale_log2e: Scalar[accum_dtype] = scale
    var correction_smem = smem.correction_smem() + tid

    comptime if not MaskType.apply_log2e_after_mask:
        scale_log2e *= log2e

    # Fuse scale*log2e multiplication and row_max subtraction into a
    # single FMA in store_exp. Only valid on the default scaling path
    # where apply_log2e_after_mask is off.
    # Disabled when sink weights are used because the sink logit lives
    # in a different domain (scaled by log2e only, not scale*log2e).
    # To disable for NaN debugging, set use_fma = False.
    comptime use_fma = (
        not MaskType.apply_log2e_after_mask
    ) and SinkType.is_null and QScaleType.is_null

    @parameter
    @always_inline
    def mask_row[
        BN: Int, //, mask_strategy: MaskStrategy
    ](mut s: InlineArray[Scalar[accum_dtype], BN], kv_row: UInt32):
        apply_mask[
            mask_strategy=mask_strategy,
            skip_scale=use_fma,
        ](
            s,
            mask,
            scale_log2e,
            prompt_idx=seq_info.prompt_idx,
            q_head_idx=q_head_idx,
            kv_tile_start_row=Int32(kv_row),
            max_seq_len=max_seq_len,
            num_keys=Int32(num_keys),
            score_row=Int32(
                score_row
                + cta_q_offset
                + (
                    thread_tile_row
                    // UInt32(group) if fuse_gqa else thread_tile_row
                )
            ),
        )

    # while waiting, offset output
    #
    # Q-tile geometry:
    # - `per_qo_BM` is the row count of one output tile. 2Q emits two
    #   BM/2-row outputs (one per WG); 1Q emits one full-BM output
    #   combined across both WGs. Both modes have per_qo_BM == 128.
    # - `wg_row_offset` is the gap between WG0's and WG1's row ranges
    #   in BM-direct units (used for q_scale indexing). 2Q: BM/2. 1Q: 0
    #   (both WGs share the same Q rows).
    # - `wg_row_offset_seq` is the same gap in seq-space units
    #   (fuse_gqa-aware, used for num_output_rows / gmem_row).
    comptime per_qo_BM = BM // config.num_qo
    comptime per_qo_BM_seq = per_qo_BM // group if fuse_gqa else per_qo_BM
    comptime wg_row_offset: Int = (BM // 2) if config.num_qo == 2 else 0
    comptime wg_row_offset_seq: Int = (
        wg_row_offset // group if fuse_gqa else wg_row_offset
    )
    num_output_rows = min(
        Int32(seq_info.seq_len)
        - Int32(seq_info.prompt_offset)
        - Int32(cta_q_offset)
        - Int32(warp_group_idx) * Int32(wg_row_offset_seq),
        Int32(per_qo_BM_seq),
    )

    gmem_row = PositionType.get_q_gmem_row[ragged=ragged](seq_info, max_seq_len)
    var s = InlineArray[Scalar[accum_dtype], config.BN](uninitialized=True)

    # Per-token k_scale buffer offset. The load warp cycles k_scale through
    # num_k_scale_bufs staged buffers (each BN elements wide). The softmax
    # must advance this offset after each K tile to read the correct buffer.
    var k_scale_off: UInt32 = 0
    comptime k_scale_wrap = config.num_k_scale_bufs() * config.BN
    comptime assert KScaleType.is_null == (k_scale_wrap == 0), String(
        "KScaleType.is_null = ",
        KScaleType.is_null,
        "\nconfig.num_k_scale_bufs() = ",
        config.num_k_scale_bufs(),
        "\nBN = ",
        config.BN,
    )

    comptime max_unroll = 8

    comptime f32x2 = SIMD[DType.float32, 2]

    @parameter
    @always_inline
    def apply_k_scale[
        N: Int, //, offset: Int
    ](mut s0: InlineArray[Float32, N], k_scale_off: UInt32):
        comptime if not QScaleType.is_null:
            comptime for n in range(0, N, 2):
                var k_sc: f32x2 = (
                    k_scale.value()
                    .load[width=2](k_scale_off + UInt32(n + offset))
                    .cast[accum_dtype]()
                )
                sn = mul_ftz(k_sc, f32x2(s0[n], s0[n + 1]))
                s0[n] = sn[0]
                s0[n + 1] = sn[1]

    @parameter
    @always_inline
    def load_mask_max_impl[
        *, mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> StaticTuple[Float32, max_unroll]:
        comptime if EnableForcedOrdering:
            order_s_wait.unsafe_value()[].wait(order_phase)
        # break up into sets of 32
        # minimize wait time by using smallest first
        comptime BM = config.BM // 2
        comptime batch_size = 32
        comptime has_remainder = (config.BN % batch_size) != 0
        comptime first_cols = (
            config.BN % batch_size
        ) if has_remainder else batch_size
        s0 = TMemTile[accum_dtype, BM, first_cols](s_tmem).load_async()
        apply_k_scale[0](s0, k_scale_off)
        s1 = TMemTile[accum_dtype, BM, batch_size](
            s_tmem + UInt32(first_cols)
        ).load_async()
        mask_row[mask_strategy=mask_strategy](s0, kv_row)
        vrow_max = maximum[width=max_unroll](s0)

        comptime for _i in range(first_cols):
            s[_i] = s0[_i]
        comptime cols = config.BN - first_cols + batch_size

        comptime for i in range(cols // (2 * batch_size)):
            comptime offset0 = first_cols + batch_size * (2 * i)
            comptime offset1 = first_cols + batch_size * (2 * i + 1)
            comptime offset2 = first_cols + batch_size * (2 * i + 2)

            comptime if offset1 >= config.BN:
                apply_k_scale[offset0](s1, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s1, kv_row + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]
            else:
                s2 = TMemTile[accum_dtype, BM, batch_size](
                    s_tmem + UInt32(offset1)
                ).load_async()
                apply_k_scale[offset0](s1, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s1, kv_row + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]

                comptime if offset2 < config.BN:
                    s1 = TMemTile[accum_dtype, BM, batch_size](
                        s_tmem + UInt32(offset2)
                    ).load_async()
                apply_k_scale[offset1](s2, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s2, kv_row + UInt32(offset1)
                )
                vrow_max = maximum(s2, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset1 + _i] = s2[_i]

        comptime if not KScaleType.is_null:
            k_scale_off = (k_scale_off + UInt32(config.BN)) if (
                k_scale_off != UInt32(k_scale_wrap - config.BN)
            ) else 0
        return vrow_max

    @parameter
    @always_inline
    def init_load_mask_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> Float32:
        return maximum(load_mask_max_impl[mask_strategy=mask_strategy](kv_row))

    @parameter
    @always_inline
    def load_mask_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32, old_max: Float32) -> Float32:
        pipeline_s.wait()
        tcgen05_fence_after()
        return maximum(
            load_mask_max_impl[mask_strategy=mask_strategy](kv_row), old_max
        )

    @parameter
    @always_inline
    def store_exp(row_max: Float32) -> f32x2:
        comptime exp_simd = 2
        comptime vs_len = config.BN // exp_simd  # 128 // 2 = 64
        comptime assert (vs_len % config.num_pv_stages) == 0
        comptime use_3_then_1_split = UMMA1Type.use_3_then_1_split
        comptime batch_size = 32 if config.num_pv_stages == 1 else vs_len // (
            4 if use_3_then_1_split else config.num_pv_stages
        )
        comptime num_batch_iters, remainder = divmod(vs_len, batch_size)
        comptime assert num_batch_iters > 0
        comptime BatchTileType = TMemTile[
            qkv_type, config.BM // 2, batch_size * exp_simd
        ]
        comptime RemainderTileType = TMemTile[
            qkv_type, config.BM // 2, remainder * exp_simd
        ]
        comptime assert (config.BN % exp_simd) == 0

        @parameter
        @always_inline
        def s_load[i: Int]() -> f32x2:
            return f32x2(s[2 * i], s[2 * i + 1])

        @parameter
        @always_inline
        def s_store[i: Int](v: f32x2):
            s[2 * i] = v[0]
            s[2 * i + 1] = v[1]

        var vrow_max: f32x2
        var vscale: f32x2
        var vneg_max_scaled: f32x2

        comptime if use_fma:
            vscale = f32x2(scale_log2e)
            vneg_max_scaled = f32x2(-row_max * scale_log2e)
            vrow_max = f32x2(0)  # unused
        else:
            vrow_max = f32x2(row_max)
            vscale = f32x2(0)  # unused
            vneg_max_scaled = f32x2(0)  # unused

        @parameter
        @always_inline
        def score_to_logit(score: f32x2) -> f32x2:
            comptime if use_fma:
                return fma_ftz(score, vscale, vneg_max_scaled)
            else:
                return sub_ftz(score, vrow_max)

        # --- Experiment parameters ---
        # Schedule the score-to-logit conversion `ratio` iterations ahead
        # of its corresponding exp2 to hide latency.  1 = strict
        # interleave, 4 = ~4-iteration prefetch (current tuned value).
        comptime score_to_logit_ratio: Int = 4
        # Number of exp2s per pass to route through the polynomial
        # emulation path (`exp2_emulation`) rather than hardware
        # `ex2.approx`.  Default 16 on sm_100; disabled on sm_103 where
        # the emulation does not pay off.
        comptime default_emulate_count: Int = 0 if "sm_103" in _accelerator_arch() else 16
        # `default_emulate_count` is calibrated at vs_len=64; the
        # `// 64` normalizes it back to that reference so non-default
        # vs_len scales the count proportionally.  Override at compile
        # time with `-D EXP2_EMULATE_COUNT=N`.
        comptime num_emulated: Int = (
            get_defined_int["EXP2_EMULATE_COUNT", default_emulate_count]()
            * vs_len
        ) // 64  # target emulated exp2s out of vs_len
        comptime emulation_start: Int = batch_size  # emulation window start
        # comptime emulation_start: Int = vs_len // score_to_logit_ratio  # emulation window start
        comptime emulation_end: Int = 0 if num_emulated == 0 else vs_len  # emulation window end
        comptime order_arrive_offset: Int = batch_size - 1  # within last batch
        # Derived: stride to distribute ~num_emulated across [emul_start, emul_end)
        comptime emulation_window: Int = 0 if num_emulated == 0 else emulation_end - emulation_start
        comptime emulation_stride_freq = 1 if num_emulated == 0 else emulation_window // num_emulated
        # num_emulated = emulation_window_freq / emulation_stride_freq
        # +  (emulation_window - emulation_window_freq) / (emulation_stride_freq + 1)
        #
        # num_emulated * emulation_stride_freq * (emulation_stride_freq + 1)
        #   = (emulation_stride_freq + 1)*emulation_window_freq
        #    + emulation_stride_freq * (emulation_window - emulation_window_freq)
        #   = (emulation_stride_freq + 1)*emulation_window_freq
        #    + emulation_stride_freq * emulation_window
        #    - emulation_stride_freq * emulation_window_freq
        #   = emulation_window_freq + emulation_stride_freq * emulation_window
        #
        # Thus:
        comptime emulation_window_freq = num_emulated * emulation_stride_freq * (
            emulation_stride_freq + 1
        ) - emulation_stride_freq * emulation_window
        comptime emulation_window_unfreq_start = emulation_start + emulation_window_freq
        comptime assert vs_len % score_to_logit_ratio == 0
        comptime assert (
            num_emulated >= 0
            and num_emulated <= emulation_window
            and emulation_window >= 0
        )
        comptime assert (
            num_emulated
            == emulation_window_freq // emulation_stride_freq
            + (emulation_window - emulation_window_freq)
            // (emulation_stride_freq + 1)
        )

        @parameter
        @always_inline
        def exp_iter[idx: Int]():
            comptime if idx < vs_len // score_to_logit_ratio:
                comptime for i in range(score_to_logit_ratio):
                    comptime j = score_to_logit_ratio * idx + i
                    s_store[j](score_to_logit(s_load[j]()))

            var x = s_load[idx]()
            comptime if (
                (
                    idx >= emulation_start
                    and (idx < emulation_window_unfreq_start)
                    and ((idx - emulation_start) % emulation_stride_freq == 0)
                )
                or (
                    idx >= emulation_window_unfreq_start
                    and (idx < emulation_end)
                    and (
                        (idx - emulation_start) % (emulation_stride_freq + 1)
                        == 0
                    )
                )
            ):
                x = exp2_emulation(x)
            else:
                x = exp2(x)
            s_store[idx](x)

        # --- Batch 0 ---
        comptime for idx in range(batch_size):
            exp_iter[idx]()

        var acc = s_load[0]()
        comptime if EnableEarlyAdd:
            comptime for i in range(1, batch_size // 2):
                acc = add_ftz(acc, s_load[i]())

        BatchTileType(p_tmem).store_async(s)

        comptime for b in range(1, num_batch_iters):
            comptime offset = batch_size * b

            comptime if use_3_then_1_split:
                comptime if 4 * b == 3 * num_batch_iters:
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                    comptime if config.pair_cta:
                        umma_arrive_leader_cta(pipeline_s.consumer_mbar[0]())
                    else:
                        pipeline_s.release_no_step[0]()
            elif config.num_pv_stages > 1:
                comptime assert config.num_pv_stages == num_batch_iters
                tcgen05_store_wait()
                tcgen05_fence_before()

                comptime assert config.num_pv_stages == num_batch_iters
                comptime if config.pair_cta:
                    umma_arrive_leader_cta(pipeline_s.consumer_mbar[b - 1]())
                else:
                    pipeline_s.release_no_step[b - 1]()

            comptime for idx in range(offset, offset + batch_size):
                exp_iter[idx]()
                comptime if (
                    EnableForcedOrdering
                    and b == max(1, num_batch_iters - 1)
                    and idx == offset + order_arrive_offset
                ):
                    _ = order_s_arrive.unsafe_value()[].arrive()
                    order_phase ^= 1

            comptime el_offset = offset * exp_simd
            comptime tmem_offset = (el_offset * size_of[qkv_type]()) // size_of[
                accum_dtype
            ]()
            BatchTileType(p_tmem + UInt32(tmem_offset)).store_async[
                src_offset=el_offset
            ](s)

        comptime if remainder > 0:
            comptime offset = batch_size * num_batch_iters

            comptime for idx in range(offset, offset + remainder):
                exp_iter[idx]()

            comptime el_offset = offset * exp_simd
            comptime tmem_offset = (el_offset * size_of[qkv_type]()) // size_of[
                accum_dtype
            ]()
            RemainderTileType(p_tmem + UInt32(tmem_offset)).store_async[
                src_offset=el_offset
            ](s)

        tcgen05_store_wait()
        tcgen05_fence_before()
        comptime if config.pair_cta:
            umma_arrive_leader_cta(
                pipeline_s.consumer_mbar[config.num_pv_stages - 1]()
            )
            pipeline_s.step()
        else:
            pipeline_s.release[config.num_pv_stages - 1]()

        pipeline_c.acquire()
        # now we can sum the remaining elements of `acc`
        comptime add_offset = batch_size // 2 if EnableEarlyAdd else 0
        var acc0: f32x2
        var acc1: f32x2
        var acc2: f32x2
        var acc3: f32x2

        comptime if EnableEarlyAdd:
            acc0 = acc
            acc1 = s_load[batch_size // 2]()
            acc2 = s_load[batch_size // 2 + 1]()
            acc3 = add_ftz(
                s_load[batch_size // 2 + 2](),
                s_load[batch_size // 2 + 3](),
            )
        else:
            acc0 = acc
            acc1 = s_load[1]()
            acc2 = s_load[2]()
            acc3 = s_load[3]()

        comptime for i in range(add_offset + 4, vs_len, 4):
            acc0 = add_ftz(acc0, s_load[i]())
            acc1 = add_ftz(acc1, s_load[i + 1]())
            acc2 = add_ftz(acc2, s_load[i + 2]())
            acc3 = add_ftz(acc3, s_load[i + 3]())
        return add_ftz(add_ftz(acc0, acc1), add_ftz(acc2, acc3))

    var kv_row: UInt32 = mask.start_column[BM_mask, BN, page_size](
        seq_info.prompt_idx, score_row
    )
    # 1Q: WG0 takes even-indexed K/V tiles (start = kv_row); WG1 takes
    # odd-indexed (+BN). Both advance by 2*BN per main-loop iter (set
    # below). 2Q: both WGs share the same kv_row stride of BN.
    comptime if config.num_qo == 1:
        kv_row += warp_group_idx * UInt32(config.BN)
    comptime mask_sets = MaskType.nonfull_sets[BM_mask, BN]()
    comptime mask_strategies = MaskType.mask_strategies[BM_mask, BN]()
    comptime num_sets = len(mask_strategies)
    comptime assert len(mask_sets) == num_sets

    var row_max: Float32
    var mask_iters: StaticTuple[UInt32, num_sets] = {}

    # `total_iters_combined` is the combined K-tile count across both
    # WGs in 1Q (= MMA's `mask.total_iters` view). Needed for the peer
    # `o_prod_mbar` wait phase in the 1Q LSE combine below.
    var total_iters_combined: UInt32 = 0

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        mask_ends = mask.masked_set_ends[
            BM=BM_mask, BN=BN, page_size=page_size
        ](seq_info.prompt_idx, score_row, num_keys)
        mask_iters[0] = mask_ends[0]

        comptime for i in range(1, num_sets):
            mask_iters[i] = mask_ends[i] - mask_ends[i - 1]

        comptime if config.num_qo == 1:
            total_iters_combined = mask_ends[num_sets - 1]
            # Per-WG split with cumulative-parity carry. WG0 owns
            # combined indices with parity 0 (even cumulative position);
            # WG1 owns parity 1. Within set i starting at cumulative
            # combined index `cum`:
            #   parity=0: WG0 takes ceil(iters_combined_i/2), WG1 floor.
            #   parity=1: WG0 takes floor, WG1 ceil.
            var cumulative: UInt32 = 0
            comptime for i in range(num_sets):
                iters_combined_i = mask_iters[i]
                parity = cumulative & UInt32(1)
                if warp_group_idx == UInt32(0):
                    mask_iters[i] = (
                        iters_combined_i + UInt32(1) - parity
                    ) // UInt32(2)
                else:
                    mask_iters[i] = (iters_combined_i + parity) // UInt32(2)
                cumulative += iters_combined_i
    else:
        comptime if config.num_qo == 1:
            # Unmasked-only path has no precomputed mask_ends. Derive
            # the combined K-tile count from the [start_column, num_keys)
            # range, matching MMA's `mask.total_iters` view.
            total_iters_combined = mask.total_iters[BM_mask, BN, page_size](
                seq_info.prompt_idx, score_row, num_keys
            )

    comptime assert num_sets >= 1 and num_sets <= 3
    comptime assert num_sets == 1 or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK

    # 1Q T==1 fast path: WG1 owns the odd-indexed K-tiles but the
    # sequence has only K_e[0], so WG1 has zero work. MMA never
    # commits to s1; pipeline_s.wait() below would hang. peel_mask
    # (num_sets==1 form) would also underflow `mask_iters[0]` from 0.
    # Skip everything WG1 would do (s wait, peel_mask, main loop,
    # LSE-exchange, output write) and drop straight to the final
    # cross-WG sync that gates TMEM dealloc. The dealloc (`warp_idx
    # == 0`) is WG0's responsibility and runs there after the sync.
    comptime if config.num_qo == 1:
        if total_iters_combined == UInt32(1) and warp_group_idx == UInt32(1):
            named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
            return

    pipeline_s.wait()
    tcgen05_fence_after()
    # Apply per-token q_scale
    comptime if not QScaleType.is_null:
        scale_log2e *= q_scale.value()[
            warp_group_idx * UInt32(wg_row_offset) + row
        ].cast[accum_dtype]()

    var row_max: Float32 = peel_mask[
        rebind[StaticTuple[MaskStrategy, num_sets]](mask_strategies),
        init_load_mask_max,
    ](mask_iters, kv_row)
    var sink_weight: Scalar[accum_dtype]

    comptime if not SinkType.is_null:
        var sink_weights_ptr = rebind[
            UnsafePointer[Scalar[qkv_type], ImmutAnyOrigin]
        ](sink_weights.value())

        comptime if use_fma:
            sink_weight = sink_weights_ptr[q_head_idx].cast[accum_dtype]()
        else:
            sink_weight = (
                sink_weights_ptr[q_head_idx].cast[accum_dtype]() * log2e
            )
        row_max = max(row_max, sink_weight)
    else:
        sink_weight = 0.0

    var row_sum: f32x2 = store_exp(row_max)

    var o_phase: UInt32 = 0  # initial wait is phase 0

    comptime if not SinkType.is_null:
        comptime if use_fma:
            row_sum[0] += exp2((sink_weight - row_max) * scale_log2e)
        else:
            row_sum[0] += exp2(sink_weight - row_max)

    # Lazy-rescale gate for online softmax: only re-scale the accumulator
    # (and adopt the new running max) when `new_row_max - old_max > 8` in
    # log2 domain — i.e., `old_max - new_row_max < rescale_threshold`.
    # Below that, we keep the stale max and skip the rescale; the new
    # exp2(score - old_max) terms stay within 2^8 = 256× of the existing
    # scale, which fp32 accumulation can absorb without meaningful loss.
    # For FP8 inputs (`size_of < 2`), set threshold to 0 to force a
    # rescale on every actual max update.
    comptime rescale_threshold: Float32 = Float32(-8) if size_of[
        qkv_type
    ]() >= 2 else Float32(0)

    # 1Q advances kv_row by 2*BN (each WG strides over its half of the
    # K/V stream); 2Q advances by BN (each WG processes every K tile).
    comptime kv_row_stride: Int = (
        2 * config.BN if config.num_qo == 1 else config.BN
    )

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        comptime for i in range(num_sets):
            comptime mask_status = mask_sets[i]
            comptime mask_strategy = mask_strategies[i]
            var iters: UInt32

            iters = warp.broadcast(mask_iters[i])
            while iters != 0:
                iters -= 1
                kv_row += UInt32(kv_row_stride)
                # calculate rowmax
                old_max = row_max
                var new_row_max: Float32 = load_mask_max[mask_strategy](
                    kv_row, old_max
                )

                diff = sub_ftz(old_max, new_row_max)

                comptime if use_fma:
                    diff = mul_ftz(diff, scale_log2e)
                var correction: Float32

                comptime if rescale_threshold < 0:
                    # old_max - new_row_max < -8
                    # 8 < new_row_max - old_max
                    if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                        row_max = new_row_max
                        correction = exp2(diff)
                    else:
                        correction = 1
                else:
                    row_max = new_row_max
                    correction = exp2(diff)
                correction_smem[] = correction
                pipeline_c.commit()
                # update s->p
                local_rowsum = store_exp(row_max)
                row_sum = fma_ftz(row_sum, f32x2(correction), local_rowsum)
                o_phase ^= 1
    else:
        while True:
            kv_row += UInt32(kv_row_stride)
            if kv_row >= num_keys:
                break
            cur_mask_status = mask.status(
                seq_info.prompt_idx,
                Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                Index[dtype=DType.int32](BM_mask, BN),
            )
            if cur_mask_status == TileMaskStatus.FULL_MASK:
                continue
            # calculate rowmax
            old_max = row_max
            var new_row_max: Scalar[accum_dtype]
            if cur_mask_status == TileMaskStatus.PARTIAL_MASK:
                new_row_max = load_mask_max[
                    MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS
                ](kv_row, old_max)
            else:
                new_row_max = load_mask_max[MaskStrategy.OUT_OF_BOUNDS](
                    kv_row, old_max
                )

            diff = sub_ftz(old_max, new_row_max)

            comptime if use_fma:
                diff = mul_ftz(diff, scale_log2e)
            var correction: Float32

            comptime if rescale_threshold < 0:
                # old_max - new_row_max < -8
                # 8 < new_row_max - old_max
                if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                    row_max = new_row_max
                    correction = exp2(diff)
                else:
                    correction = 1
            else:
                row_max = new_row_max
                correction = exp2(diff)
            correction_smem[] = correction
            pipeline_c.commit()
            # update s->p
            local_rowsum = store_exp(row_max)
            row_sum = fma_ftz(row_sum, f32x2(correction), local_rowsum)
            o_phase ^= 1
    # Do the final correction and write.
    comptime assert size_of[output_type]() >= size_of[qkv_type]()

    comptime if config.num_qo == 2:
        # 2Q: each WG writes its row half independently.
        inv_row_sum = recip(row_sum.reduce_add())
        # `BM // config.num_qo` matches the helper's signature
        # (`config.BM // config.num_qo`) at the comptime-expression level;
        # numerically identical to HalfBM = BM // 2 inside this 2Q branch.
        o_tile = TMemTile[accum_dtype, BM // config.num_qo, padded_ov_depth](
            tmem_addr
            + UInt32(config.TMEM_O0)
            + warp_group_idx * UInt32(padded_ov_depth)
        )
        # wait on the o_pipeline producer
        if num_output_rows > 0:
            o_prod_mbar[warp_group_idx].wait(o_phase)  # consumer wait
            tcgen05_fence_after()  # example 1
            # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?

            fa4_scale_write_output[config](
                row,
                warp_idx & 3,
                warp_group_idx,
                inv_row_sum,
                o_smem + warp_group_idx * UInt32(HalfBM * padded_ov_depth),
                o_tile,
                ragged_tma_store,
                num_output_rows,
                head_idx,
                gmem_row
                + cta_q_offset
                + warp_group_idx
                * UInt32(HalfBM // group if fuse_gqa else HalfBM),
            )
    else:
        # 1Q output. T==1 takes a fast path (WG0 has the full output
        # in TMEM_O0 and WG1 has no work / already returned); T>=2
        # combines per-WG partials via LSE exchange.
        if total_iters_combined == UInt32(1):
            # T==1 fast path: skip LSE-exchange entirely and reuse the
            # 2Q row-scale + stmatrix + TMA helper directly. No peer
            # partial to combine; no per-WG smem/gmem-row offsets.
            # `BM // config.num_qo` is the helper's expected row count
            # and numerically equals config.BM (= 128) in 1Q.
            inv_row_sum = recip(row_sum.reduce_add())
            o_tile = TMemTile[
                accum_dtype, BM // config.num_qo, padded_ov_depth
            ](tmem_addr + UInt32(config.TMEM_O0))
            if num_output_rows > 0:
                # Only o0 is produced (MMA skipped the o1 commit at T==1).
                o_prod_mbar[0].wait(o_phase)
                tcgen05_fence_after()
                fa4_scale_write_output[config](
                    row,
                    warp_idx & 3,
                    UInt32(0),
                    inv_row_sum,
                    o_smem,
                    o_tile,
                    ragged_tma_store,
                    num_output_rows,
                    head_idx,
                    gmem_row + cta_q_offset,
                )
            # WG1 already participated in `named_barrier[2*WG](2)` and
            # returned; WG0 must hit it here so the pair-WG sync resolves
            # before TMEM dealloc. Mirrors the unconditional sync below.
            named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
            comptime if not config.pair_cta:
                if warp_idx == 0:
                    tcgen05_release_allocation_lock[Int32(cta_group)]()
                    tcgen05_dealloc[Int32(cta_group)](
                        tmem_addr, UInt32(config.sm100_tmem_cols)
                    )
            return

        # 1Q: LSE-combine both WGs' TMEM_O fragments into the shared
        # o_smem in depth-column slices, then both WGs TMA-store
        # disjoint column ranges to gmem. Both WGs cover the same Q
        # rows; no per-WG row offset on the write side.

        # 1. WG-local LSE reduce.
        row_sum_total = row_sum.reduce_add()

        # 2. Wait on OWN pipeline_o producer. After this, MMA1 has finished
        # its last V·P, so the last P in own's s_tmem has been consumed and
        # the slot is safe to repurpose for the cross-WG LSE exchange below.
        # The wait is unconditional (independent of num_output_rows) because
        # MMA1 always runs and the TMEM reuse requires it; the peer-side
        # wait stays inside the num_output_rows guard since it only gates
        # the TMEM_O read in fa4_lse_combine_write.
        o_prod_mbar[warp_group_idx].wait(o_phase)
        tcgen05_fence_after()

        # 3. LSE exchange through the (now-dead) s_tmem slot. Each WG writes
        # (row_max, row_sum_total) into the first two TMEM columns of its
        # s_tmem slot; the peer reads those two columns from the other WG's
        # slot. Replaces an earlier smem-aliased exchange buffer that had
        # to overlay the K region and could collide with the load warp's K
        # TMA writes.
        # TMEM layout (per Q row r):
        #     col TMEM_S0+0     = WG0 row_max
        #     col TMEM_S0+1     = WG0 row_sum_total
        #     col TMEM_S0+BN+0  = WG1 row_max
        #     col TMEM_S0+BN+1  = WG1 row_sum_total
        var own_lse: InlineArray[Scalar[accum_dtype], 2] = [
            row_max,
            row_sum_total,
        ]
        TMemTile[accum_dtype, BM, 2](s_tmem).store_async(own_lse)
        tcgen05_store_wait()
        tcgen05_fence_before()
        named_barrier[Int32(2 * WARPGROUP_SIZE)](5)
        tcgen05_fence_after()

        # 4. Read peer's slice from the peer WG's s_tmem.
        peer_wg = UInt32(1) - warp_group_idx
        var peer_s_tmem: UInt32 = (tmem_addr + UInt32(config.TMEM_S0)) + UInt32(
            config.BN
        ) * peer_wg
        var peer_lse = TMemTile[accum_dtype, BM, 2](peer_s_tmem).load_async()
        peer_max = peer_lse[0]
        peer_sum = peer_lse[1]

        global_max = max(row_max, peer_max)
        # Match the per-WG online softmax convention: when `use_fma`,
        # `row_max` is tracked in raw (unscaled) score units and the
        # inner-loop diff is multiplied by `scale_log2e` before `exp2`
        # (see `diff = mul_ftz(diff, scale_log2e)` above). The LSE
        # combine must apply the same conversion, otherwise the
        # cross-WG weights are `exp2(raw_diff)` instead of
        # `exp2(raw_diff * scale_log2e)` and the 1Q output drifts ~1
        # ULP whenever the two WGs' raw maxes differ. Without this
        # scaling the bug is masked when K is constant (raw maxes
        # equal across WGs) or V is constant (per-WG O ∝ row_sum so
        # the wrong weights cancel through global_sum normalization).
        var diff_local: Float32 = row_max - global_max
        var diff_peer: Float32 = peer_max - global_max
        comptime if use_fma:
            diff_local *= scale_log2e
            diff_peer *= scale_log2e
        scale_local = exp2(diff_local)
        scale_peer = exp2(diff_peer)
        global_sum = row_sum_total * scale_local + peer_sum * scale_peer
        inv_global_sum = recip(global_sum)
        final_scale_local = scale_local * inv_global_sum
        final_scale_peer = scale_peer * inv_global_sum

        # 5. Wait on PEER pipeline_o producer so peer's TMEM_O is safe to
        # read. Per-pipeline iter counts differ by
        # `total_iters_combined & 1` for odd combined-T, so peer's phase
        # XORs in that bit. (Own's producer was already waited on above
        # before the LSE exchange.)
        if num_output_rows > 0:
            peer_phase = o_phase ^ (total_iters_combined & UInt32(1))
            o_prod_mbar[peer_wg].wait(peer_phase)
            tcgen05_fence_after()

            # 5. Build own + peer TMEM tiles at full-BM extent.
            own_o_tile = TMemTile[accum_dtype, BM, padded_ov_depth](
                tmem_addr
                + UInt32(config.TMEM_O0)
                + warp_group_idx * UInt32(padded_ov_depth)
            )
            peer_o_tile = TMemTile[accum_dtype, BM, padded_ov_depth](
                tmem_addr
                + UInt32(config.TMEM_O0)
                + peer_wg * UInt32(padded_ov_depth)
            )

            # 6. Per-WG comptime j-range specialization for the helper.
            # Ceil/floor split: WG0 takes ceil(iters/2) blocks starting
            # at j=0, WG1 takes floor(iters/2) starting at j=ceil(iters/2).
            # Even iters → both WGs get iters/2. Odd iters (depth=64 with
            # iters=1) → WG0 takes the only block; WG1 skips the helper
            # entirely (its iters_per_wg would be 0, which the helper
            # rejects via comptime assert).
            comptime swizzle_granularity = (
                config.swizzle_mode.bytes() // size_of[output_type]()
            )
            comptime iters_total = padded_ov_depth // swizzle_granularity
            comptime iters_per_wg0 = (iters_total + 1) // 2
            comptime iters_per_wg1 = iters_total // 2
            # In 1Q both WGs write the same Q rows; no per-WG gmem-row
            # offset (the depth column j drives the gmem position).
            out_row_idx = gmem_row + cta_q_offset
            if warp_group_idx == UInt32(0):
                fa4_lse_combine_write[
                    config,
                    wg_j_offset=0,
                    iters_per_wg=iters_per_wg0,
                ](
                    row,
                    warp_idx & 3,
                    warp_group_idx,
                    final_scale_local,
                    final_scale_peer,
                    o_smem,
                    own_o_tile,
                    peer_o_tile,
                    ragged_tma_store,
                    num_output_rows,
                    head_idx,
                    out_row_idx,
                )
            else:
                comptime if iters_per_wg1 > 0:
                    fa4_lse_combine_write[
                        config,
                        wg_j_offset=iters_per_wg0,
                        iters_per_wg=iters_per_wg1,
                    ](
                        row,
                        warp_idx & 3,
                        warp_group_idx,
                        final_scale_local,
                        final_scale_peer,
                        o_smem,
                        own_o_tile,
                        peer_o_tile,
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        out_row_idx,
                    )
    named_barrier[Int32(2 * WARPGROUP_SIZE)](4)
    # Pair-CTA: dealloc is deferred to the kernel after cluster_sync so that
    # the peer CTA cannot exit while cluster-scoped stmatrix is in flight.
    comptime if not config.pair_cta:
        if warp_idx == 0:
            tcgen05_release_allocation_lock[Int32(cta_group)]()
            tcgen05_dealloc[Int32(cta_group)](
                tmem_addr, UInt32(config.sm100_tmem_cols)
            )
