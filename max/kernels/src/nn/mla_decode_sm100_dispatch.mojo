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

from math import ceildiv
from sys import size_of
from gpu.host import DeviceContext, FuncAttribute
from gpu.memory import AddressSpace
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


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------
fn mla_decode_sm100_dispatch[
    q_type: DType,
    q_layout: Layout,
    k_t: MHAOperand,
    output_type: DType,
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
    output: LayoutTensor[address_space = AddressSpace.GENERIC, ...],
    scale: Float32,
    batch_size: Int,
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
    var num_rows_qo = num_matrix_view_rows_decode(q)
    q_ptr = rebind[UnsafePointer[Scalar[q_type], origin=MutAnyOrigin]](
        q.to_device_buffer(ctx).unsafe_ptr()
    )
    q_tma_op = tma_tile_qo[
        swizzle_mode = mla_config.swizzle_mode,
        BM = mla_config.BM,  # tile_m =64
        BK = mla_config.BK0,  # tile_n =576
        depth = mla_config.q_depth,
    ](ctx, q_ptr, num_rows_qo)

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
    o_tma_op = tma_tile_qo[
        swizzle_mode = mla_config.swizzle_mode,
        BM = mla_config.out_rows,
        BK = mla_config.BN,
        depth = mla_config.depth,
    ](ctx, o_ptr, num_rows_qo)

    if ragged:
        comptime ValidLengthType = NonNullPointer[DType.uint32]
        var valid_len: ValidLengthType = {
            valid_length.to_device_buffer(ctx).unsafe_ptr()
        }
        launch_mla_sm100_decode_enqueue_kernel[
            q_type=q_type,
            KVLUTType=k_t,
            output_type=output_type,
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
            scale,
            batch_size,
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
            scale,
            batch_size,
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
    scale: Float32,
    batch_size: Int,
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
    ](mask, score_mod, valid_len)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    # TODO: this should be seq_len and batch to be distributed across the grid
    var block_y = q_max_seq_len
    # TODO: # currently this is fixed for batch size, Next step is to modify it
    # to support the split k when the KVcachesize is varrable length per batch so
    # the num_partitions would create more load balanced work per block based on
    # the KV cache size per batch.
    var block_z = batch_size
    var grid_dim = (block_x, block_y, block_z)
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
        batch_size,
        "Num Partitions:",
        num_partitions,
        "Max Cache Valid Length:",
        max_cache_valid_length,
    )

    comptime kernel = MLA_SM100_Decode_KV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
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
        MaskType=MaskType,
        ScoreModType=ScoreModType,
        config=config,
        use_score_mod=use_score_mod,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
    ].kernel
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
            config.smem_used
        ),
    )
