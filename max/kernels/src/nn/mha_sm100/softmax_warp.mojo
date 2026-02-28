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

from math import exp2, recip, align_up
from math.constants import log2e
from sys import size_of
import gpu.primitives.warp as warp
from gpu import thread_idx
from gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from gpu.memory import AddressSpace, CacheEviction, fence_async_view_proxy
from gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from gpu.compute.arch.tcgen05 import (
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_store_wait,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.barriers import (
    WarpGroupBarrier,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TMEM_LOWER_ROW_OFFSET,
    TmemAllocation,
)
from gpu.primitives.warp import _vote_nvidia_helper
from layout._layout import row_major
from layout import stack_allocation as tt_stack_allocation
from layout.swizzle import make_swizzle
from layout.tma_async import RaggedTMA3DTile, SharedMemBarrier
from nn.fa4_config import FA4Config, EnableForcedOrdering, EnableEarlyAdd
from nn.sm100_attention_utils import (
    LocalTensor,
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
    FA4MiscMBars,
)
from nn.mha_fa3_utils import (
    MHAPosition,
    OptionalPointer,
    _LocalTT,
    _SharedMemTT,
    output_reg_to_smem_st_matrix,
)
from nn.mha_mask import MHAMask, TileMaskStatus, MaskStrategy
from nn.mha_operand import MHAOperand
from nn.mha_tile_scheduler import SeqInfo
from nn.mha_utils import OptionallyStaticInt, _is_decoding
from utils.index import Index
from utils.static_tuple import StaticTuple


@always_inline
fn fa4_scale_write_output[
    qkv_type: DType,
    output_type: DType,
    config: FA4Config,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    inv_row_sum: Float32,
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    o_tmem_arg: TMemTile[DType.float32, config.BM // 2, config.padded_depth],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        config.swizzle_mode,
        BM = config.BM // 2,
        BN = config.depth,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    comptime accum_type = DType.float32
    comptime BM = config.BM
    comptime padded_depth = config.padded_depth

    comptime swizzle_granularity = config.swizzle_mode.bytes() // size_of[
        output_type
    ]()
    comptime iters = padded_depth // swizzle_granularity

    comptime ST = STMatrixLayout[
        BM // 2,
        swizzle_granularity,
        num_threads=WARPGROUP_SIZE,
        accum_type_size=4,
    ]
    comptime num_rows = ST.vec_local_layout[0].size()

    comptime swizzle = make_swizzle[output_type, config.swizzle_mode]()

    comptime swizzle_block_size: UInt32 = UInt32(
        WARP_SIZE * swizzle_granularity
    )

    e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Allocate register tiles for double-buffered pipeline.
    comptime ChunkTMemType = TMemTile[accum_type, BM // 2, swizzle_granularity]
    var o_cur = ChunkTMemType.allocate_register_tile[
        num_threads=WARPGROUP_SIZE
    ]()

    # --- Composable pipeline primitives, parameterized by m_half ---

    @always_inline
    @parameter
    fn load_chunk[col: Int, m_half: Int](dst: type_of(o_cur)):
        """Async tmem load for one M-half of column `col`."""
        comptime load_dtype = DType.uint32
        var ptr = rebind[
            UnsafePointer[
                Scalar[load_dtype],
                MutAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ]
        ](dst.ptr)
        chunk_tmem_addr = o_tmem_arg.tmem_addr + UInt32(
            col * swizzle_granularity
        )

        @parameter
        @always_inline
        fn load_fn[pow_two: Int, local_offset: Int]():
            comptime assert pow_two + local_offset <= ST.repeat
            comptime if pow_two > 0:
                comptime offsets = STMatrixOffsets[
                    BM // 2,
                    swizzle_granularity,
                    num_threads=WARPGROUP_SIZE,
                    accum_type_size=4,
                    curr_repeat=pow_two,
                    cumulative_repeat=local_offset,
                    m_mma=m_half,
                ]()
                tmem = chunk_tmem_addr + UInt32(offsets.tmem_offset)
                frag = tcgen05_ld[
                    datapaths=16,
                    bits = ST.bits,
                    repeat=pow_two,
                    dtype=load_dtype,
                    pack=False,
                    width = offsets.local_frag_size_b32,
                ](tmem)
                ptr.store(offsets.ptr_offset, frag)

        comptime max_value = 64 if ST.bits == 128 else 32
        break_into_powers_of_two[
            func=load_fn, N = ST.repeat, max_value=max_value
        ]()

    load_chunk[0, 0](o_cur)
    inv_row_sums = tt_stack_allocation[
        dtype=accum_type, address_space = AddressSpace.LOCAL
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
    fn scale_half[m_half: Int](o: type_of(o_cur)):
        """Scale one M-half's registers by `inv_row_sum`."""
        comptime rows_per_half = ST.num_row_blocks_per_mma
        comptime start = m_half * rows_per_half
        comptime for i in range(start, start + rows_per_half):
            irs = o.element_type(rebind[Scalar[accum_type]](inv_row_sums[i]))
            comptime for k in range(o.layout[1].size()):
                o[i, k] *= irs

    @always_inline
    @parameter
    fn write_to_smem[j: Int, m_half: Int](o: type_of(o_cur)):
        """Write one M-half of column `j` to smem."""
        comptime datapath_offset: UInt32 = UInt32(
            16 * m_half * swizzle_granularity
        )
        comptime ofs = m_half * ST.frag_size
        comptime reg_layout = row_major[1, ST.frag_size]()
        var rows_of_o_frags = _LocalTT[accum_type, reg_layout](
            o.ptr + ofs, reg_layout
        )

        comptime warp_smem_offset: UInt32 = datapath_offset + UInt32(
            j * (BM // 2) * swizzle_granularity
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
    fn sync_and_tma_store[j: Int]():
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
fn fa4_softmax[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    config: FA4Config,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
](
    mbars: FA4MiscMBars[
        num_qk_stages = config.num_qk_stages,
        num_pv_stages = config.num_pv_stages,
        num_kv_stages = config.num_kv_stages,
        separate_kv=True,
        use_order_barriers=EnableForcedOrdering,
    ],
    score_row: UInt32,
    seq_info: SeqInfo,
    mask: MaskType,
    num_keys: UInt32,
    scale: Float32,
    max_seq_len: UInt32,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        config.swizzle_mode,
        BM = config.BM // 2,
        BN = config.depth,
    ],
    sink_weights: SinkType,
):
    # Local aliases matching SM100MHA2Q comptime members
    comptime qkv_type = KVLUTType.dtype
    comptime accum_type = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime padded_depth = config.padded_depth
    comptime page_size = KVLUTType.page_size
    comptime ragged = not ValidLengthType.is_null
    comptime cta_group = 1

    # Offset calculations
    comptime q_offset: Int32 = 0
    comptime kv_offset: Int32 = q_offset + Int32(
        config.BM * config.padded_depth
    )
    comptime correction_offset: Int32 = (
        kv_offset
        + Int32(2 * config.num_kv_stages * config.padded_depth * config.BN)
    ) * Int32(size_of[qkv_type]()) // Int32(size_of[DType.float32]())
    comptime mbar_offset = (correction_offset + Int32(config.BM)) * Int32(
        size_of[DType.float32]()
    ) // Int32(size_of[SharedMemBarrier]())

    comptime MiscMBarsType = type_of(mbars)

    # MMA types for TMEM access
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        qkv_type,
        accum_type,
        MMA_M=HalfBM,
        MMA_N=BN,
        BK = align_up(config.depth, config.MMA_K),
        swizzle_a = config.swizzle_mode,
        swizzle_b = config.swizzle_mode,
        transpose_b=True,
        num_stages = config.num_qk_stages,
    ]
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        qkv_type,
        accum_type,
        MMA_M=HalfBM,
        MMA_N=padded_depth,
        BK=BN,
        swizzle_b = config.swizzle_mode,
        transpose_b=False,
        num_stages = config.num_pv_stages,
    ]
    comptime PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.depth,
        config.padded_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    var tmem_addr: UInt32 = (
        mbars.mbar_base + MiscMBarsType.num_mbars()
    ).bitcast[UInt32]()[]
    var o_smem: SharedMemPointer[Scalar[output_type]] = (
        (mbars.mbar_base - mbar_offset)
        .bitcast[Scalar[qkv_type]]()
        .bitcast[Scalar[output_type]]()
    )
    var o_prod_mbar: MBarType = (
        mbars.mbar_base + MiscMBarsType.O_producer_offset
    )
    var s_tmem: UInt32 = tmem_addr + UInt32(config.TMEM_S0)

    # var tid = UInt32(thread_idx.x)
    var tid = llvm_opaque_tid()
    var row = tid % 128
    var warp_idx: UInt32 = warp.broadcast(tid // 32)
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)

    comptime if config.split_m:
        # split-M: second S is (+16 rows) in st-matrix space
        s_tmem += TMEM_LOWER_ROW_OFFSET * warp_group_idx
    else:
        # 2-Q path: S1 is at +BN columns
        s_tmem += UInt32(config.BN) * warp_group_idx

    p_tmem = s_tmem
    c_tmem = p_tmem + UInt32(config.BN // 2)
    s_tile = UMMA0Type.CType(s_tmem)
    p_tile = UMMA1Type.AType(p_tmem)

    var pipeline_s = mbars.consumer_s(warp_group_idx)
    pipeline_c = mbars.producer_c(warp_group_idx)
    var order_phase: UInt32 = 1 - warp_group_idx

    comptime if EnableForcedOrdering:
        order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
        order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)
    else:
        order_s_wait = MBarType()
        order_s_arrive = MBarType()

    var q_head_idx: UInt32 = seq_info.head_idx
    var scale_log2e: Scalar[accum_type] = scale
    var correction_smem = (
        (mbars.mbar_base - mbar_offset).bitcast[Float32]() + correction_offset
    ) + tid

    comptime if not MaskType.apply_log2e_after_mask:
        scale_log2e *= log2e

    # Fuse scale*log2e multiplication and row_max subtraction into a
    # single FMA in store_exp. Only valid on the default scaling path
    # where apply_log2e_after_mask is off.
    # Disabled when sink weights are used because the sink logit lives
    # in a different domain (scaled by log2e only, not scale*log2e).
    # To disable for NaN debugging, set use_fma = False.
    comptime use_fma = not (
        MaskType.apply_log2e_after_mask or not SinkType.is_null
    )

    @parameter
    @always_inline
    fn mask_row[
        BN: Int, //, mask_strategy: MaskStrategy
    ](s: LocalTensor[accum_type, row_major[BN]()], kv_row: UInt32,):
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
            score_row=Int32(score_row + tid),
        )

    # while waiting, offset output
    comptime splitBM = BM // 2
    var num_output_rows = min(
        Int32(seq_info.seq_len)
        - Int32(seq_info.prompt_offset)
        - Int32(warp_group_idx) * Int32(splitBM),
        Int32(splitBM),
    )

    gmem_row = PositionType.get_q_gmem_row[ragged=ragged](seq_info, max_seq_len)
    s = tt_stack_allocation[
        dtype=accum_type, address_space = AddressSpace.LOCAL
    ](row_major[config.BN]())

    comptime max_unroll = 8

    @parameter
    @always_inline
    fn load_mask_max_impl[
        *, mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> StaticTuple[Float32, max_unroll]:
        comptime if EnableForcedOrdering:
            order_s_wait[].wait(order_phase)
        pipeline_s.wait()
        tcgen05_fence_after()
        # break up into sets of 32
        # minimize wait time by using smallest first
        comptime BM = config.BM // 2
        comptime batch_size = 32
        comptime has_remainder = (config.BN % batch_size) != 0
        comptime first_cols = (
            config.BN % batch_size
        ) if has_remainder else batch_size
        s0 = TMemTile[accum_type, BM, first_cols](s_tmem).load_async()
        s1 = TMemTile[accum_type, BM, batch_size](
            s_tmem + UInt32(first_cols)
        ).load_async()
        mask_row[mask_strategy=mask_strategy](s0, kv_row)
        vrow_max = maximum[width=max_unroll](s0)

        s.ptr.store(s0.ptr.load[width=first_cols]())
        comptime cols = config.BN - first_cols + batch_size

        comptime for i in range(cols // (2 * batch_size)):
            comptime offset0 = first_cols + batch_size * (2 * i)
            comptime offset1 = first_cols + batch_size * (2 * i + 1)
            comptime offset2 = first_cols + batch_size * (2 * i + 2)

            comptime if offset1 >= config.BN:
                mask_row[mask_strategy=mask_strategy](
                    s1, kv_row + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)
                s.ptr.store(offset0, s1.ptr.load[width=batch_size]())
            else:
                s2 = TMemTile[accum_type, BM, batch_size](
                    s_tmem + UInt32(offset1)
                ).load_async()
                mask_row[mask_strategy=mask_strategy](
                    s1, kv_row + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)
                s.ptr.store(offset0, s1.ptr.load[width=batch_size]())

                comptime if offset2 < config.BN:
                    s1 = TMemTile[accum_type, BM, batch_size](
                        s_tmem + UInt32(offset2)
                    ).load_async()
                mask_row[mask_strategy=mask_strategy](
                    s2, kv_row + UInt32(offset1)
                )
                vrow_max = maximum(s2, vrow_max)
                s.ptr.store(offset1, s2.ptr.load[width=batch_size]())

        return vrow_max

    @parameter
    @always_inline
    fn load_mask_max[*, mask_strategy: MaskStrategy](kv_row: UInt32) -> Float32:
        return maximum(load_mask_max_impl[mask_strategy=mask_strategy](kv_row))

    @parameter
    @always_inline
    fn load_mask_max[
        *, mask_strategy: MaskStrategy
    ](kv_row: UInt32, old_max: Float32) -> Float32:
        return maximum(
            load_mask_max_impl[mask_strategy=mask_strategy](kv_row), old_max
        )

    comptime f32x2 = SIMD[DType.float32, 2]

    @parameter
    @always_inline
    fn store_exp(row_max: Float32) -> f32x2:
        comptime exp_simd = 2
        comptime vs_len = config.BN // exp_simd  # 128 // 2 = 64
        comptime assert (vs_len % config.num_pv_stages) == 0
        comptime use_3_then_1_split = UMMA1Type.use_3_then_1_split
        comptime batch_size = 32 if config.num_pv_stages == 1 else vs_len // (
            4 if use_3_then_1_split else config.num_pv_stages
        )
        comptime num_batch_iters = vs_len // batch_size
        comptime remainder = vs_len % batch_size
        comptime assert num_batch_iters > 0
        comptime BatchTileType = TMemTile[
            qkv_type, config.BM // 2, batch_size * exp_simd
        ]
        comptime RemainderTileType = TMemTile[
            qkv_type, config.BM // 2, remainder * exp_simd
        ]
        comptime assert (config.BN % exp_simd) == 0

        vs = s.vectorize[exp_simd]()
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
        fn score_to_logit(score: f32x2) -> f32x2:
            comptime if use_fma:
                return fma_ftz(score, vscale, vneg_max_scaled)
            else:
                return sub_ftz(score, vrow_max)

        var acc: f32x2 = exp2(score_to_logit(rebind[f32x2](vs[0])))
        vs[0] = rebind[vs.ElementType](acc)
        vsi = exp2(score_to_logit(rebind[f32x2](vs[1])))
        vs[1] = rebind[vs.ElementType](vsi)

        comptime if EnableEarlyAdd:
            acc = add_ftz(acc, vsi)
        comptime exp2_emulation_freq = 3

        comptime for i in range(2, 8):
            vs[i] = rebind[vs.ElementType](score_to_logit(rebind[f32x2](vs[i])))

        comptime for i in range(2, 8):
            vsi = exp2(rebind[f32x2](vs[i]))
            vs[i] = rebind[vs.ElementType](vsi)

            comptime if EnableEarlyAdd:
                acc = add_ftz(acc, vsi)

        comptime for i in range(8, batch_size // 2):
            diff = score_to_logit(rebind[f32x2](vs[i]))
            vsi = exp2(diff)
            vs[i] = rebind[vs.ElementType](vsi)

            comptime if EnableEarlyAdd:
                acc = add_ftz(acc, vsi)

        # at this point, we need 32 fewer fp32 registers but 16 more u32
        comptime for i in range(batch_size // 2, batch_size):
            diff = score_to_logit(rebind[f32x2](vs[i]))
            vs[i] = rebind[vs.ElementType](exp2(diff))

        BatchTileType(p_tmem).store_async(
            LocalTensor[accum_type, row_major[batch_size * exp_simd]()](
                s.ptr, row_major[batch_size * exp_simd]()
            )
        )

        comptime for b in range(1, num_batch_iters):
            comptime offset = batch_size * b

            comptime if use_3_then_1_split:
                comptime if 4 * b == 3 * num_batch_iters:
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                    pipeline_s.release_no_step[0]()
            elif config.num_pv_stages > 1:
                comptime assert config.num_pv_stages == num_batch_iters
                tcgen05_store_wait()
                tcgen05_fence_before()

                comptime assert config.num_pv_stages == num_batch_iters
                pipeline_s.release_no_step[b - 1]()

            comptime for i in range(offset, offset + batch_size):
                diff = score_to_logit(rebind[f32x2](vs[i]))

                comptime if i % exp2_emulation_freq == 0:
                    vs[i] = rebind[vs.ElementType](exp2_emulation(diff))
                else:
                    vs[i] = rebind[vs.ElementType](exp2(diff))

            comptime el_offset = offset * exp_simd
            comptime tmem_offset = (el_offset * size_of[qkv_type]()) // size_of[
                accum_type
            ]()
            BatchTileType(p_tmem + UInt32(tmem_offset)).store_async(
                LocalTensor[accum_type, row_major[batch_size * exp_simd]()](
                    s.ptr + el_offset, row_major[batch_size * exp_simd]()
                )
            )

        comptime if remainder > 0:
            comptime offset = batch_size * num_batch_iters

            comptime for i in range(offset, offset + remainder):
                diff = score_to_logit(rebind[f32x2](vs[i]))

                comptime if i % exp2_emulation_freq == 0:
                    vs[i] = rebind[vs.ElementType](exp2_emulation(diff))
                else:
                    vs[i] = rebind[vs.ElementType](exp2(diff))

            comptime el_offset = offset * exp_simd
            comptime tmem_offset = (el_offset * size_of[qkv_type]()) // size_of[
                accum_type
            ]()
            RemainderTileType(p_tmem + UInt32(tmem_offset)).store_async(
                LocalTensor[accum_type, row_major[remainder * exp_simd]()](
                    s.ptr + el_offset, row_major[remainder * exp_simd]()
                )
            )

        tcgen05_store_wait()
        tcgen05_fence_before()
        pipeline_s.release[config.num_pv_stages - 1]()

        comptime if EnableForcedOrdering:
            _ = order_s_arrive[].arrive()
            order_phase ^= 1
        pipeline_c.acquire()
        # now we can sum the remaining elements of `acc`
        comptime add_offset = batch_size // 2 if EnableEarlyAdd else 0
        var acc0: f32x2
        var acc1: f32x2
        var acc2: f32x2
        var acc3: f32x2

        comptime if EnableEarlyAdd:
            acc0 = acc
            acc1 = rebind[f32x2](vs[batch_size // 2])
            acc2 = rebind[f32x2](vs[batch_size // 2 + 1])
            acc3 = add_ftz(
                rebind[f32x2](vs[batch_size // 2 + 2]),
                rebind[f32x2](vs[batch_size // 2 + 3]),
            )
        else:
            acc0 = acc
            acc1 = rebind[f32x2](vs[1])
            acc2 = rebind[f32x2](vs[2])
            acc3 = rebind[f32x2](vs[3])

        comptime for i in range(add_offset + 4, vs_len, 4):
            acc0 = add_ftz(acc0, rebind[f32x2](vs[i]))
            acc1 = add_ftz(acc1, rebind[f32x2](vs[i + 1]))
            acc2 = add_ftz(acc2, rebind[f32x2](vs[i + 2]))
            acc3 = add_ftz(acc3, rebind[f32x2](vs[i + 3]))
        return add_ftz(add_ftz(acc0, acc1), add_ftz(acc2, acc3))

    var kv_row: UInt32 = mask.start_column[BM, BN, page_size](score_row)
    comptime mask_sets = MaskType.nonfull_sets[BM, BN]()
    comptime mask_strategies = MaskType.mask_strategies[BM, BN]()
    comptime num_sets = len(mask_sets)

    var row_max: Float32
    var mask_iters: StaticTuple[UInt32, num_sets] = {}

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        mask_ends = mask.masked_set_ends[BM=BM, BN=BN, page_size=page_size](
            score_row, num_keys
        )
        mask_iters[0] = mask_ends[0]

        comptime for i in range(1, num_sets):
            mask_iters[i] = mask_ends[i] - mask_ends[i - 1]

    comptime assert num_sets >= 1 and num_sets <= 3
    comptime assert num_sets == 1 or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK

    comptime if num_sets == 1:
        row_max = load_mask_max[mask_strategy = mask_strategies[0]](kv_row)
        mask_iters[0] -= 1
    else:
        # find out which strategy to apply
        if mask_iters[0] > 0:
            row_max = load_mask_max[mask_strategy = mask_strategies[0]](kv_row)
            mask_iters[0] -= 1
        else:
            comptime if num_sets == 2:
                row_max = load_mask_max[mask_strategy = mask_strategies[1]](
                    kv_row
                )
                mask_iters[1] -= 1
            else:
                if mask_iters[1] > 1:
                    row_max = load_mask_max[mask_strategy = mask_strategies[1]](
                        kv_row
                    )
                    mask_iters[1] -= 1
                else:
                    row_max = load_mask_max[mask_strategy = mask_strategies[2]](
                        kv_row
                    )
                    mask_iters[2] -= 1
    var sink_weights_ptr = UnsafePointer[Scalar[qkv_type], ImmutAnyOrigin]()
    var sink_weight: Scalar[accum_type]

    comptime if not SinkType.is_null:
        sink_weights_ptr = rebind[
            UnsafePointer[Scalar[qkv_type], ImmutAnyOrigin]
        ](sink_weights.value())
        var head_idx: UInt32 = seq_info.head_idx

        comptime if use_fma:
            sink_weight = sink_weights_ptr[head_idx].cast[accum_type]()
        else:
            sink_weight = sink_weights_ptr[head_idx].cast[accum_type]() * log2e
        row_max = max(row_max, sink_weight)
    else:
        sink_weights_ptr = {}
        sink_weight = 0.0

    var row_sum: f32x2 = store_exp(row_max)

    var o_phase: UInt32 = 0  # initial wait is phase 0

    comptime if not SinkType.is_null:
        comptime if use_fma:
            row_sum[0] += exp2((sink_weight - row_max) * scale_log2e)
        else:
            row_sum[0] += exp2(sink_weight - row_max)

    comptime rescale_threshold: Float32 = Float32(-8) if size_of[
        qkv_type
    ]() >= 2 else Float32(0)

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        comptime for i in range(num_sets):
            comptime mask_status = mask_sets[i]
            comptime mask_strategy = mask_strategies[i]
            var iters: UInt32

            iters = warp.broadcast(mask_iters[i])
            while iters != 0:
                iters -= 1
                kv_row += UInt32(config.BN)
                # calculate rowmax
                old_max = row_max
                var new_row_max: Float32 = load_mask_max[
                    mask_strategy=mask_strategy
                ](kv_row, old_max)

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
            kv_row += UInt32(config.BN)
            if kv_row >= num_keys:
                break
            cur_mask_status = mask.status(
                Index[dtype = DType.int32](Int(score_row), Int(kv_row)),
                Index[dtype = DType.int32](BM, BN),
            )
            if cur_mask_status == TileMaskStatus.FULL_MASK:
                continue
            # calculate rowmax
            old_max = row_max
            var new_row_max: Scalar[accum_type]
            if cur_mask_status == TileMaskStatus.PARTIAL_MASK:
                new_row_max = load_mask_max[
                    mask_strategy = MaskStrategy.COMPUTED
                    | MaskStrategy.OUT_OF_BOUNDS
                ](kv_row, old_max)
            else:
                new_row_max = load_mask_max[
                    mask_strategy = MaskStrategy.OUT_OF_BOUNDS
                ](kv_row, old_max)

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
    # Do the final correction and write
    inv_row_sum = recip(row_sum.reduce_add())
    o_tile = UMMA1Type.CType(
        tmem_addr
        + UInt32(config.TMEM_O0)
        + warp_group_idx * UInt32(padded_depth)
    )
    # wait on the o_pipeline producer
    comptime assert size_of[output_type]() == size_of[qkv_type]()
    if num_output_rows > 0:
        o_prod_mbar[warp_group_idx].wait(o_phase)  # consumer wait
        tcgen05_fence_after()  # example 1
        # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?

        fa4_scale_write_output[qkv_type, output_type, config](
            row,
            warp_idx & 3,
            warp_group_idx,
            inv_row_sum,
            o_smem + warp_group_idx * UInt32(HalfBM * padded_depth),
            o_tile,
            ragged_tma_store,
            num_output_rows,
            q_head_idx,
            gmem_row + warp_group_idx * UInt32(HalfBM),
        )
    WarpGroupBarrier[2 * WARPGROUP_SIZE, 2].sync()
    if warp_idx == 0:
        var tmem = TmemAllocation[cta_group](tmem_addr)
        tmem.release_lock()
        tmem.deallocate()
