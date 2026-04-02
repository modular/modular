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
"""Dispatch for depth=512 pair-CTA SM100 (Blackwell) MHA prefill.

Creates the Depth512SM100Config, TMA tile descriptors, and launches the
pair-CTA kernel with cluster_dim=(2,1,1). The TransientScheduler uses
pair_cta=True so that both CTAs in a cluster derive the same tile index
from block_idx.x >> 1.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from layout.tma_async import RaggedTMA3DTile
from std.logger import Logger
from nn.attention.gpu.nvidia.sm90.attention import (
    ImmutTileTensor1D,
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
from .config import Depth512SM100Config
from .kernel import SM100MHADepth512


comptime logger = Logger()


@always_inline
def mha_sm100_depth512_dispatch[
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
) raises:
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match."
    comptime decoding: Bool = _is_decoding[MaxPromptLenType]()
    comptime assert not decoding, "depth512 pair-CTA does not support decoding"

    comptime d512_config = Depth512SM100Config[KVType.dtype](
        num_q_heads=Int(config.num_heads),
        group=group,
        qk_depth=Int(config.depth),
        ov_depth=Int(config.depth),
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
    )
    comptime assert d512_config.supported(), d512_config.description()
    comptime swizzle_mode = d512_config.swizzle_mode
    comptime PairBM = d512_config.BM * 2  # 128
    comptime num_threads = d512_config.num_threads  # 384

    var q = rebind[UnsafePointer[Scalar[KVType.dtype], q_arg.origin]](q_arg)
    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)

    # ---- TMA tile descriptors ------------------------------------------------

    # Output store: BM=64 per CTA, full ov_depth=512.
    comptime RaggedStoreType = RaggedTMA3DTile[
        output_type,
        swizzle_mode,
        BM=d512_config.BM,
        BN=d512_config.ov_depth,
    ]
    var ragged_tma_store = RaggedStoreType.create(
        ctx,
        output.unsafe_ptr(),
        rows=num_rows_q,
        middle_dim=d512_config.num_q_heads,
    )

    # Q: BM=64 per CTA (not halved like 2Q).
    q_tma_op = q_tma[
        swizzle_mode,
        BM=d512_config.BM,
        depth=d512_config.qk_depth,
        q_num_heads=d512_config.num_q_heads,
        group=d512_config.group,
        decoding=False,
        num_qk_stages=d512_config.num_qk_stages,
    ](ctx, q, num_rows_q)

    # K: each CTA loads BN//2 rows, BK0 depth per stage.
    k_tma_op = k.create_tma_tile[
        d512_config.swizzle_mode,
        BN=d512_config.BN // 2,
        depth=d512_config.qk_depth,
        BK=d512_config.BK0,
    ](ctx)

    # V: BK1 rows x ov_depth//4 columns (heavily sub-tiled for SMEM).
    v_tma_op = v.create_tma_tile[
        d512_config.swizzle_mode,
        BN=d512_config.BK1,
        depth=d512_config.ov_depth,
        BK=d512_config.ov_depth // 4,
    ](ctx)

    # ---- Scheduler -----------------------------------------------------------

    comptime SchedulerType = TransientScheduler[
        UInt32(PairBM),
        UInt32(d512_config.num_q_heads),
        flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
        pair_cta=True,
    ]
    var scheduler: SchedulerType = SchedulerType()

    # ---- Nested closure dispatch (no sink) -----------------------------------

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
            comptime PackType = Pack[
                MaskType,
                SchedulerType,
                ValidLengthType,
                NullPointer[DType.float32],  # no sink
                KVRowOffsetsType,
                MaxPromptLenType,
                PartitionType,
            ]
            var pack: PackType = {
                mask,
                scheduler,
                valid_len,
                NullPointer[DType.float32](),
                kv_row_offsets,
                max_prompt_len_arg,
                partition,
            }

            var max_num_prompt_tiles: UInt32 = ceildiv(
                max_prompt_len_arg.as_uint32(), UInt32(PairBM)
            )
            var block_x: UInt32 = max_num_prompt_tiles
            # SchedulerType.grid_dim doubles block_x (pair_cta=True).

            logger.info("------ Dispatching to SM100 Depth512 Pair-CTA ------")
            logger.info(
                "QKV Type:",
                KVType.dtype,
                "Depth:",
                d512_config.qk_depth,
                "Number of Q // KV Heads:",
                d512_config.num_q_heads,
                "//",
                d512_config.num_kv_heads,
                "Batch Size:",
                batch_size,
                "Max Num Prompt Tiles:",
                max_num_prompt_tiles,
            )

            comptime smem_use = d512_config.smem_used

            comptime kernel = SM100MHADepth512[
                KVType,
                output_type,
                MaskType,
                SchedulerType,
                d512_config,
                ValidLengthType,
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
