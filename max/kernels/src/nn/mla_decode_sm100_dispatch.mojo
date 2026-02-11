# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from math import ceildiv, clamp
from sys import size_of
from gpu.host import DeviceContext, FuncAttribute
from gpu.memory import AddressSpace
from gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout.layout import (
    Layout,
)
from logger import Logger

from layout.layout_tensor import (
    LayoutTensor,
)
from nn.mha_fa3_utils import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
)
from nn.mha_mask import MHAMask
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_utils import (
    MHAConfig,
)
from nn.mha_fa3_utils import KVTMATile
from layout.runtime_layout import RuntimeLayout
from utils.numerics import get_accum_type
from utils.index import Index

comptime logger = Logger()
from nn.mla_decode_sm100_utils import (
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    tma_tile_qo,
    MLA_Decode_Pack,
    num_matrix_view_rows_decode,
    OffsetPosition,
    SharedMemPointer,
    MBarType,
    SharedMemTensor,
    KVPipelineGeneric,
    KVLoad2CvtProducer,
    KVLoad2CvtConsumer,
    KVCvt2MmaProducer,
    KVCvt2MmaConsumer,
    DecodeSM100MiscMBars,
    DecodeSProducer,
    DecodeSConsumer,
    DecodePConsumer,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeSM100QKTSS,
    DecodeSM100PVSS,
    ld_shared_v4_u32,
    cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4,
    st_shared_v4_b32_at_bf16_elem_off,
)
from nn.mla_decode_sm100_kv_bf16 import MLA_SM100_Decode_KV_BF16
from nn.mla_decode_sm100_kv_fp8 import MLA_SM100_Decode_KV_FP8
from nn.mla_decode_sm100_combine import mla_decode_combine_partial_outputs


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------
fn mla_decode_sm100_dispatch[
    q_type: DType,
    q_layout: Layout,
    k_t: MHAOperand,
    output_type: DType,
    output_layout: Layout,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    valid_layout: Layout,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    *,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    q: LayoutTensor[
        q_type, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: k_t,
    output: LayoutTensor[
        output_type, output_layout, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    q_max_seq_len: Int,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    ctx: DeviceContext,
) raises:
    comptime hw_info = ctx.default_device_info
    comptime sm_count = hw_info.sm_count
    var available_SMs = ceildiv(
        sm_count,
        (q_max_seq_len * ceildiv(num_heads, 64) * batch_size),
    )
    var page_size = 128

    # CRITICAL: The kernel's OffsetPosition adds q_max_seq_len to num_keys when
    # _is_cache_length_accurate=False. We must use the same effective num_keys
    # here to compute num_partitions correctly, otherwise there's a mismatch
    # between how the dispatcher divides work and how the kernel sees it.
    var effective_max_cache_len = max_cache_valid_length

    @parameter
    if not _is_cache_length_accurate:
        effective_max_cache_len += q_max_seq_len

    # This can get threshold like min 8 pages etc.
    # require heuristinc to test
    var num_kv_cache_pages = ceildiv(effective_max_cache_len, page_size)
    # Clamp num_partitions to:
    # 1. MAX_SPLITS (96) - combine kernel supports up to 96 splits via multi-LSE-per-thread
    # 2. num_kv_cache_pages - ensure at least 1 page per split to avoid empty splits
    #    (empty splits cause hangs due to barrier deadlocks or infinite loops)
    # 3. At least 1 partition (even if num_kv_cache_pages is 0, we need 1 partition)
    var num_partitions = clamp(available_SMs, 1, min(96, num_kv_cache_pages))
    # Eliminate empty splits caused by ceil division mismatch.
    # The main kernel uses pages_per_split = ceildiv(total_pages, num_partitions),
    # which means only ceildiv(total_pages, pages_per_split) splits actually have
    # work. Splits beyond that have start_page >= total_pages and return early
    # with uninitialized LSE, causing combine kernel corruption.
    # Recompute to ensure every split has at least 1 page of work.
    if num_partitions > 1 and num_kv_cache_pages > 0:
        var pages_per_split = ceildiv(num_kv_cache_pages, num_partitions)
        num_partitions = ceildiv(num_kv_cache_pages, pages_per_split)
    var block_z = batch_size * num_partitions

    comptime AccumType = get_accum_type[output.dtype]()
    comptime v_depth = depth - 64

    if num_partitions > 1:
        comptime SplitAccumType = NonNullPointer[AccumType]
        # Create partial output buffer (same type as output - bfloat16)
        # Each split writes its partial attention result here
        # Note: Output dimension is v_depth (512), not depth (576)
        o_accum_split_data = ctx.enqueue_create_buffer[output_type](
            Int(
                num_partitions
                * batch_size
                * q_max_seq_len
                * num_heads
                * v_depth
            )
        )
        var o_accum_split = LayoutTensor[output_type, Layout.row_major[5]()](
            o_accum_split_data.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[5]()].row_major(
                Index(
                    num_partitions,
                    batch_size,
                    q_max_seq_len,
                    Int(num_heads),
                    Int(v_depth),
                )
            ),
        )
        # Create LSE accumulator buffer (AccumType = float32 for numerical stability)
        var lse_accum_data = ctx.enqueue_create_buffer[AccumType](
            Int(num_partitions * batch_size * q_max_seq_len * num_heads)
        )
        var lse_accum_split = LayoutTensor[AccumType, Layout.row_major[4]()](
            lse_accum_data.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[4]()].row_major(
                Index(
                    num_partitions,
                    batch_size,
                    q_max_seq_len,
                    Int(num_heads),
                )
            ),
        )
        var lse_accum_split_ptr: SplitAccumType = {
            lse_accum_split.to_device_buffer(ctx).unsafe_ptr()
        }

        # Launch main MLA decode kernel (writes partial results to accumulators)
        mla_decode_sm100_sink_split_k[
            q_type=q_type,
            q_layout=q_layout,
            k_t=k_t,
            output_type=output_type,
            mask_t=mask_t,
            score_mod_t=score_mod_t,
            valid_layout=valid_layout,
            config=config,
            depth=depth,
            num_heads=num_heads,
            SplitAccumType=SplitAccumType,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=True,
        ](
            q,
            k,
            o_accum_split,
            lse_accum_split_ptr,
            scale,
            batch_size,
            block_z,
            num_partitions,
            max_cache_valid_length,
            q_max_seq_len,
            valid_length,
            mask,
            score_mod,
            ctx,
        )

        # Get input_row_offsets pointer for combine kernel's ragged output writes.
        var input_row_offsets_ptr = rebind[
            UnsafePointer[Scalar[DType.uint32], origin=MutAnyOrigin]
        ](valid_length.to_device_buffer(ctx).unsafe_ptr())

        # Dispatch to specialized kernel based on num_partitions for compile-time unrolling.
        # Supports up to 96 splits (max_splits) to allow higher SM utilization on B200.
        # For batch_size=1 on B200 (148 SMs),
        @parameter
        fn launch_combine[n_splits: Int]() raises:
            mla_decode_combine_partial_outputs[
                output_type=output_type,
                accum_type=AccumType,
                head_dim=v_depth,
                num_splits=n_splits,
                ragged=ragged,
            ](
                o_accum_split,
                lse_accum_split,
                output,
                input_row_offsets_ptr,
                batch_size,
                q_max_seq_len,
                Int(num_heads),
                ctx,
            )

        # Dispatch to the appropriate compile-time specialization.
        # we go from 2 to 96 inclusively
        @parameter
        for i in range(2, 97):
            if num_partitions == i:
                launch_combine[i]()
    else:
        comptime SplitAccumType = NullPointer[AccumType]
        var lse_accum_split_ptr: SplitAccumType = {}

        mla_decode_sm100_sink_split_k[
            q_type=q_type,
            q_layout=q_layout,
            k_t=k_t,
            output_type=output_type,
            mask_t=mask_t,
            score_mod_t=score_mod_t,
            valid_layout=valid_layout,
            config=config,
            depth=depth,
            num_heads=num_heads,
            SplitAccumType=SplitAccumType,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=False,
        ](
            q,
            k,
            output,
            lse_accum_split_ptr,
            scale,
            batch_size,
            block_z,
            num_partitions,
            max_cache_valid_length,
            q_max_seq_len,
            valid_length,
            mask,
            score_mod,
            ctx,
        )


fn mla_decode_sm100_sink_split_k[
    q_type: DType,
    q_layout: Layout,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    score_mod_t: ScoreModTrait,
    valid_layout: Layout,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    SplitAccumType: OptionalPointer,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
    decoding_warp_split_k: Bool,
](
    q: LayoutTensor[
        q_type, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: k_t,
    output: LayoutTensor[address_space = AddressSpace.GENERIC, ...],
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    q_max_seq_len: Int,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    ctx: DeviceContext,
) raises:
    comptime mla_config = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,  # num_q_heads/h_k(1)
        depth=(depth - 64),  # 512
        q_depth=depth,  # 576
        dtype_size=size_of[q_type](),
        kv_type_size=size_of[k_t.dtype](),
        swizzle_mode=config.swizzle_mode,
        kv_mma_swizzle_mode=config.swizzle_mode,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
    )
    var num_rows_q = num_matrix_view_rows_decode(q)
    q_ptr = rebind[UnsafePointer[Scalar[q_type], origin=MutAnyOrigin]](
        q.to_device_buffer(ctx).unsafe_ptr()
    )
    q_tma_op = tma_tile_qo[
        swizzle_mode = mla_config.swizzle_mode,
        BM = mla_config.BM,  # tile_m =64
        BK = mla_config.BK0,  # tile_n =576
        depth = mla_config.q_depth,
    ](ctx, q_ptr, num_rows_q)

    k_tma_op = k.create_tma_tile[
        BN = mla_config.BK1,  # tile_m =64
        depth = mla_config.q_depth,
        BK = mla_config.BK0,  # tile_n =576
        swizzle_mode = mla_config.kv_tma_swizzle_mode,
    ](ctx)
    comptime output_tile_width = (mla_config.BN // 2) * (
        4 // size_of[output_type]()
    )
    o_ptr = rebind[UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]](
        output.to_device_buffer(ctx).unsafe_ptr()
    )
    var num_rows_o = num_matrix_view_rows_decode(output)
    o_tma_op = tma_tile_qo[
        swizzle_mode = mla_config.swizzle_mode,
        BM = mla_config.out_rows,
        BK = mla_config.BN,
        depth = mla_config.depth,
    ](ctx, o_ptr, num_rows_o)

    if ragged:
        comptime ValidLengthType = NonNullPointer[DType.uint32]
        var valid_len: ValidLengthType = {
            valid_length.to_device_buffer(ctx).unsafe_ptr()
        }
        launch_mla_sm100_decode_enqueue_kernel[
            q_type=q_type,
            KVLUTType=k_t,
            output_type=output_type,
            SplitAccumType=SplitAccumType,
            MaskType=mask_t,
            ScoreModType=score_mod_t,
            config=mla_config,
            use_score_mod=use_score_mod,
            ValidLengthType=ValidLengthType,
            ragged=True,
            _is_cache_length_accurate=_is_cache_length_accurate,
        ](
            q_tma_op,
            k_tma_op,
            o_tma_op,
            k,
            lse_accum_split_ptr,
            scale,
            batch_size,
            block_z,
            num_partitions,
            max_cache_valid_length,
            q_max_seq_len,
            valid_len,
            mask,
            score_mod,
            ctx,
        )
    else:
        comptime ValidLengthType = NullPointer[DType.uint32]
        var valid_len: ValidLengthType = {}
        launch_mla_sm100_decode_enqueue_kernel[
            q_type=q_type,
            KVLUTType=k_t,
            output_type=output_type,
            SplitAccumType=SplitAccumType,
            MaskType=mask_t,
            ScoreModType=score_mod_t,
            config=mla_config,
            use_score_mod=use_score_mod,
            ValidLengthType=ValidLengthType,
            ragged=False,
            _is_cache_length_accurate=_is_cache_length_accurate,
        ](
            q_tma_op,
            k_tma_op,
            o_tma_op,
            k,
            lse_accum_split_ptr,
            scale,
            batch_size,
            block_z,
            num_partitions,
            max_cache_valid_length,
            q_max_seq_len,
            valid_len,
            mask,
            score_mod,
            ctx,
        )


@always_inline
fn launch_mla_sm100_decode_enqueue_kernel[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    config: MLA_SM100_Decode_Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM = config.BM,  # tile_m =64
        BK = config.BK0,  # tile_n =576
        swizzle_mode = config.swizzle_mode,
    ],
    k_tma: KVTMATile[
        dtype = KVLUTType.dtype,
        swizzle_mode = config.kv_tma_swizzle_mode,
        BN = config.BK1,  # tile_m =64
        BK = config.BK0,  # tile_n =576
    ],
    o_tma: QOTMATile[
        dtype=output_type,
        BM = config.out_rows,
        BK = config.BN,
        swizzle_mode = config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
) raises:
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        ScoreModType=ScoreModType,
        SplitAccumType=SplitAccumType,
    ](mask, score_mod, valid_len, lse_accum_split_ptr)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    # we have 3 warp groups:
    # - one for load/store/2xMMA
    # - one for compute softmax
    # - one for compute correction
    var block_dim = (config.num_threads, 1, 1)
    logger.info(
        "block_dim:",
        block_dim[0],
        block_dim[1],
        block_dim[2],
        "grid_dim:",
        grid_dim[0],
        grid_dim[1],
        grid_dim[2],
        "config.smem_used:",
        config.smem_used,
        "config.num_q_heads:",
        config.num_q_heads,
        "config.num_kv_heads:",
        config.num_kv_heads,
        "config.num_threads:",
        config.num_threads,
        "config.num_kv_stages:",
        config.num_kv_stages,
        "config.BM:",
        config.BM,
        "config.BN:",
        config.BN,
        "config.BK0:",
        config.BK0,
        "config.BK1:",
        config.BK1,
        "config.q_depth:",
        config.q_depth,
        "config.depth:",
        config.depth,
        "config.padded_depth:",
        config.padded_depth,
        "config.padded_q_depth:",
        config.padded_q_depth,
        "config.rope_depth:",
        config.rope_depth,
        "config.swizzle_mode:",
        config.swizzle_mode,
        "max_cache_valid_length:",
        max_cache_valid_length,
        "output_tile_width:",
        (config.BN // 2) * (4 // size_of[output_type]()),
    )

    logger.info("------ Dispatching to SM100 MLA-SM100-DECODE ------")
    logger.info(
        "QK Type:",
        KVLUTType.dtype,
        "Q Depth:",
        config.q_depth,
        "Number of Q // KV Heads:",
        config.num_q_heads,
        "//",
        config.num_kv_heads,
        "Batch Size:",
        block_z,
        "Num Partitions:",
        num_partitions,
        "Max Cache Valid Length:",
        max_cache_valid_length,
    )

    comptime kernel = MLA_SM100_Decode_KV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        ScoreModType=ScoreModType,
        config=config,
        use_score_mod=use_score_mod,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
    ].kernel if KVLUTType.dtype == DType.float8_e4m3fn else MLA_SM100_Decode_KV_BF16[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        ScoreModType=ScoreModType,
        config=config,
        use_score_mod=use_score_mod,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
    ].kernel
    # Enable PDL (Programmatic Dependent Launch) for split-K mode to chain
    # the MLA decode kernel with the combine kernel, reducing host synchronization.
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    ctx.enqueue_function[kernel, kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        batch_size,
        q_max_seq_len,
        num_partitions,
        max_cache_valid_length,
        mla_decode_pack,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )
