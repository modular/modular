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
from structured_kernels.barriers import (
    WarpGroupBarrier,
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
    o_tmem_arg: TMemTile[DType.float32, config.BM // 2, config.padded_ov_depth],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        BM=config.BM // 2,
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
    comptime half_bm = config.BM // 2

    comptime ST = STMatrixLayout[
        half_bm,
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
    comptime ChunkTMemType = TMemTile[accum_dtype, half_bm, swizzle_granularity]
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
                    half_bm,
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
            irs = o.element_type(rebind[Scalar[accum_dtype]](inv_row_sums[i]))
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
            j * half_bm * swizzle_granularity
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
        BM=config.BM // 2,
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
                + (tid // UInt32(group) if fuse_gqa else tid)
            ),
        )

    # while waiting, offset output
    comptime splitBM = BM // 2
    comptime splitBM_seq = splitBM // group if fuse_gqa else splitBM
    num_output_rows = min(
        Int32(seq_info.seq_len)
        - Int32(seq_info.prompt_offset)
        - Int32(cta_q_offset)
        - Int32(warp_group_idx) * Int32(splitBM_seq),
        Int32(splitBM_seq),
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
        comptime score_to_logit_ratio: Int = 4  # 1=interleaved, 4=4x ahead
        comptime default_emulate_count: Int = 0 if "sm_103" in _accelerator_arch() else 16
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

    var kv_row: UInt32 = mask.start_column[BM_mask, BN, page_size](score_row)
    comptime mask_sets = MaskType.nonfull_sets[BM_mask, BN]()
    comptime mask_strategies = MaskType.mask_strategies[BM_mask, BN]()
    comptime num_sets = len(mask_strategies)
    comptime assert len(mask_sets) == num_sets

    var row_max: Float32
    var mask_iters: StaticTuple[UInt32, num_sets] = {}

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        mask_ends = mask.masked_set_ends[
            BM=BM_mask, BN=BN, page_size=page_size
        ](score_row, num_keys)
        mask_iters[0] = mask_ends[0]

        comptime for i in range(1, num_sets):
            mask_iters[i] = mask_ends[i] - mask_ends[i - 1]

    comptime assert num_sets >= 1 and num_sets <= 3
    comptime assert num_sets == 1 or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK

    pipeline_s.wait()
    tcgen05_fence_after()
    # Apply per-token q_scale
    comptime if not QScaleType.is_null:
        scale_log2e *= q_scale.value()[
            warp_group_idx * UInt32(splitBM) + row
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
            kv_row += UInt32(config.BN)
            if kv_row >= num_keys:
                break
            cur_mask_status = mask.status(
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
    # Do the final correction and write
    inv_row_sum = recip(row_sum.reduce_add())
    o_tile = TMemTile[accum_dtype, HalfBM, padded_ov_depth](
        tmem_addr
        + UInt32(config.TMEM_O0)
        + warp_group_idx * UInt32(padded_ov_depth)
    )
    # wait on the o_pipeline producer
    comptime assert size_of[output_type]() >= size_of[qkv_type]()
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
            + warp_group_idx * UInt32(HalfBM // group if fuse_gqa else HalfBM),
        )
    WarpGroupBarrier[2 * WARPGROUP_SIZE, 2].sync()
    # Pair-CTA: dealloc is deferred to the kernel after cluster_sync so that
    # the peer CTA cannot exit while cluster-scoped stmatrix is in flight.
    comptime if not config.pair_cta:
        if warp_idx == 0:
            tcgen05_release_allocation_lock[Int32(cta_group)]()
            tcgen05_dealloc[Int32(cta_group)](tmem_addr, UInt32(512))
