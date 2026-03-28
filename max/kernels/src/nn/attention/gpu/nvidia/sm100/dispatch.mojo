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

from std.collections import OptionalReg
from std.math import ceildiv
from std.gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from nn.attention.gpu.nvidia.sm90.attention import ImmutTileTensor1D
from layout.tma_async import RaggedTMA3DTile
from std.logger import Logger
from nn.attention.gpu.nvidia.sm100.attention import FA4Config
from nn.attention.gpu.nvidia.sm90.attention import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    Pack,
    q_tma,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import TransientScheduler
from nn.attention.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
)
from .kernel import SM100MHA2Q


comptime logger = Logger()


@always_inline
def mha_sm100_dispatch[
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
    sink: Bool,
    _is_cache_length_accurate: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: UnsafePointer[Scalar[q_type], _],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask: MaskType,
    valid_length: UnsafePointer[UInt32, _],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[DType.uint32]],
    batch_size_arg: Int,
    partition: PartitionType,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
) raises:
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match for FA3."
    comptime decoding: Bool = _is_decoding[MaxPromptLenType]()
    comptime assert (
        not decoding
    ), "this implementation does not support decoding"
    comptime fa4_config = FA4Config[KVType.dtype](
        num_q_heads=Int(config.num_heads),
        group=group,
        qk_depth=Int(config.depth),
        ov_depth=Int(config.depth),
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
        is_mla=False,
    )
    comptime swizzle_mode = fa4_config.swizzle_mode
    comptime BM = fa4_config.BM
    comptime num_threads = fa4_config.num_threads
    var q = rebind[UnsafePointer[Scalar[KVType.dtype], q_arg.origin]](q_arg)

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)

    comptime RaggedStoreType = RaggedTMA3DTile[
        output_type,
        swizzle_mode,
        BM=BM // 2,
        BN=fa4_config.ov_depth,
    ]

    var ragged_tma_store = RaggedStoreType.create(
        ctx,
        output.unsafe_ptr(),
        rows=num_rows_q,
        middle_dim=fa4_config.num_q_heads,
    )

    q_tma_op = q_tma[
        swizzle_mode,
        BM=BM // 2,
        depth=fa4_config.qk_depth,
        q_num_heads=fa4_config.num_q_heads,
        group=fa4_config.group,
        decoding=False,
        num_qk_stages=fa4_config.num_qk_stages,
    ](ctx, q, num_rows_q)
    k_tma_op = k.create_tma_tile[
        fa4_config.swizzle_mode,
        BN=fa4_config.BN,
        depth=fa4_config.qk_depth,
        BK=fa4_config.BK0,
    ](ctx)
    v_tma_op = v.create_tma_tile[
        fa4_config.swizzle_mode,
        BN=fa4_config.BN,
        depth=fa4_config.ov_depth,
        BK=fa4_config.padded_ov_depth,
    ](ctx)
    comptime assert BM == 256
    comptime SchedulerType = TransientScheduler[
        UInt32(BM),
        UInt32(fa4_config.num_q_heads),
        flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
    ]
    var scheduler: SchedulerType = SchedulerType()

    @parameter
    @always_inline
    def with_sink[SinkType: OptionalPointer](sink_ptr: SinkType) raises:
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
                # the pack contains all possibly 0-sized objects
                comptime PackType = Pack[
                    MaskType,
                    SchedulerType,
                    ValidLengthType,
                    SinkType,
                    KVRowOffsetsType,
                    MaxPromptLenType,
                    PartitionType,
                ]
                var pack: PackType = {
                    mask,
                    scheduler,
                    valid_len,
                    sink_ptr,
                    kv_row_offsets,
                    max_prompt_len_arg,
                    partition,
                }

                var max_num_prompt_tiles: UInt32 = ceildiv(
                    max_prompt_len_arg.as_uint32(), UInt32(BM)
                )
                var block_x: UInt32 = (
                    max_num_prompt_tiles * partition.num_partitions()
                )
                logger.info("------ Dispatching to SM100 FMHA-2Q ------")
                logger.info(
                    "QKV Type:",
                    KVType.dtype,
                    "Depth:",
                    fa4_config.qk_depth,
                    "Number of Q // KV Heads:",
                    fa4_config.num_q_heads,
                    "//",
                    fa4_config.num_kv_heads,
                    "Batch Size:",
                    batch_size,
                    "Max Num Prompt Tiles:",
                    max_num_prompt_tiles,
                )

                comptime smem_use = fa4_config.smem_used

                comptime kernel = SM100MHA2Q[
                    KVType,
                    output_type,
                    MaskType,
                    SchedulerType,
                    fa4_config,
                    ValidLengthType,
                    SinkType,
                    KVRowOffsetsType,
                    _is_cache_length_accurate,
                    MaxPromptLenType,
                    PartitionType,
                ].kernel

                ctx.enqueue_function[kernel, kernel](
                    q_tma_op,
                    k_tma_op,
                    v_tma_op,
                    ragged_tma_store,
                    k,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    pack,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(num_threads, 1, 1),
                    shared_mem_bytes=smem_use,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(smem_use)
                    ),
                )

            # --- ragged dispatch ---
            comptime if ragged:
                with_valid_length[NonNullPointer[DType.uint32]]({valid_length})
            else:
                with_valid_length[NullPointer[DType.uint32]]({})

        # --- kv_input_row_offsets dispatch ---
        if kv_input_row_offsets:
            with_kv_offsets[NonNullPointer[DType.uint32]](
                {kv_input_row_offsets.value().ptr}
            )
        else:
            with_kv_offsets[NullPointer[DType.uint32]]({})

    # --- sink dispatch ---
    comptime if sink:
        with_sink[NonNullPointer[KVType.dtype]](
            {
                rebind[UnsafePointer[Scalar[KVType.dtype], ImmutAnyOrigin]](
                    sink_weights.value().ptr
                )
            }
        )
    else:
        with_sink[NullPointer[KVType.dtype]]({})
