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
from std.math import align_up, ceildiv, recip
from std.math.uutils import umod, ufloordiv, udivmod
from nn.attention.mha_utils import DynamicInt
from std.math.constants import log2e
from std.sys import (
    align_of,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    simd_width_of,
    size_of,
    is_nvidia_gpu,
    is_amd_gpu,
    CompilationTarget,
)

from nn.attention.gpu.mha import mha_splitk_reduce, q_num_matrix_view_rows
from nn.attention.gpu.mha_decode_partition_heuristic import (
    mha_decoding_num_partitions,
)
import std.gpu.primitives.warp as warp
from std.algorithm.functional import (
    _elementwise_impl_gpu,
    tile_and_unswitch,
    unswitch,
)
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    thread_idx,
    block_idx,
    global_idx,
    lane_id,
    warp_id,
)
from std.gpu.host import (
    DeviceContext,
    FuncAttribute,
    get_gpu_target,
    DeviceBuffer,
    Dim as LaunchDim,
)
from std.gpu.host.info import A100, H100, B200, _is_sm10x_gpu
from std.gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from kv_cache.types import KVCacheT
from layout import (
    Coord,
    Idx,
    IntTuple,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size, get_mma_shape
from layout.tile_tensor import NullableTileTensor
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from std.memory import stack_allocation
from nn._ragged_utils import get_batch_from_row_offsets
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    MHAOperand,
    LayoutTensorMHAOperand,
    RaggedMHAOperand,
)
from nn.attention.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    _copy_frag_to_smem,
    _kernel_mask,
    DynamicInt,
)
from std.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.mha_utils import get_start_and_end_for_partitions
from nn.softmax import _online_softmax_iter_for_mma_output
from .amd_structured.mla_decode import Attention
from .amd_structured.mla_prefill import Attention
from .nvidia.sm100.mla_prefill import mla_sm100_prefill
from std.gpu.host.info import B200, _is_sm10x_gpu
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    mla_decode_sm100_dispatch,
)
from .nvidia.sm100.mla_prefill_per_token_scale import (
    mla_sm100_prefill_per_token_scale,
)


# ===-----------------------------------------------------------------------===#
# GPU Multi-head Latent Attention (MLA) decoding implementations
# ===-----------------------------------------------------------------------===#


# entrypoint for MLA decoding kernels
@always_inline
def flare_mla_decoding[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    //,
    config: MHAConfig[dtype],
    ragged: Bool = False,
    decoding_warp_split_k: Bool = False,
    per_token_scale_rope_aware: Bool = False,
    sparse: Bool = False,
    # Sparse-only routing flag: True selects the BF16-rope sparse kernel
    # (split FP8 nope + BF16 rope). False (default) selects the all-FP8
    # sparse kernel. This is the production default; True is only used by
    # internal BF16-rope-kernel tests.
    rope_aware_kv_sparse: Bool = False,
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    k: cache_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    scalar_args_buf: NullableTileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    q_max_seq_len: OptionalReg[Int] = None,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
    num_partitions: Optional[Int] = None,
    # Per-token Q scale pointer: float32 array with one scale per Q token.
    # sigma_Q[q_token_idx] is folded into scale_log2e inside the Softmax function.
    # Default is null (sigma_Q = 1.0, no effect).
    q_scale_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    # Sparse indices: when non-null, the kernel uses gather4 TMA with
    # pre-computed physical row indices instead of page-table lookups.
    # d_indices[batch * indices_stride + token] = physical KV row index.
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    # Per-batch topk lengths: when non-null, topk_lengths[batch_idx] gives
    # the actual number of valid sparse indices for that batch. indices_stride
    # is the allocation stride (max topk across all batches).
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    # Extra KV: separate always-attend cache. Tokens from extra_k are
    # appended after the topk tokens in a unified attention loop.
    extra_k: OptionalReg[cache_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
) raises:
    """MLA decoding kernel that would only be called in the optimized compute
    graph.

    The Q input has a shape of [seq_len, num_heads, depth].
    The K input has a shape of [seq_len, 1, depth].
    The V tensor is derived by reusing K, where V = K[:, :, :depth_v].

    Specifically, for DeepSeek V2/3, depth = 576 and depth_v = 512.

    When per_token_scale_rope_aware is True, Q and KV cache have an interleaved
    FP8+BF16 layout: FP8 content (512 bytes) + BF16 rope (128 bytes) = 640
    bytes/row. Q's last dimension is 640 (FP8 elements) but represents 576
    logical dimensions (512 nope + 64 rope).

    This kernel computes attention without needing to load V twice. This kernel
    only handles decoding requests. In this case q_max_seq_len = 1.

    This kernel handles batches with different valid lengths (i.e., before the
    padding). Such lengths are passed in valid_length argument.
    """
    comptime assert (
        ragged or rank == 4
    ), "only support rank 4 inputs for non-ragged inputs."
    comptime assert (
        not ragged or rank == 3
    ), "only support rank 3 inputs for ragged inputs."
    # Q and output may differ for native FP8 path: Q is float8_e4m3fn,
    # output is bfloat16. Both half-float Q (bfloat16) and FP8 Q are valid.
    comptime assert q.dtype == output.dtype or (
        q.dtype == DType.float8_e4m3fn and output.dtype == DType.bfloat16
    ), (
        "Q and output must have same type, or Q=float8_e4m3fn with"
        " output=bfloat16."
    )

    @always_inline
    @parameter
    def description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_decoding",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        comptime kv_num_heads = cache_t.kv_params.num_heads

        var max_prompt_len: Int
        var num_keys = Int(k.max_context_length())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k.max_prompt_length())

        var k_operand = KVCacheMHAOperand(k)

        # For per_token_scale_rope_aware: Q's last dim is 640 (interleaved FP8+BF16)
        # but the logical depth is 576. Override config to use 576.
        comptime if per_token_scale_rope_aware:
            comptime rope_aware_config = MHAConfig[dtype](config.num_heads, 576)
            # Build extra_k_operand when extra_k is provided.
            var extra_k_operand: OptionalReg[type_of(k_operand)] = None
            if extra_k is not None:
                extra_k_operand = KVCacheMHAOperand(extra_k.value())

            flare_mla_decoding_dispatch[
                kv_num_heads=kv_num_heads,
                config=rope_aware_config,
                ragged=ragged,
                decoding_warp_split_k=decoding_warp_split_k,
                per_token_scale_rope_aware=True,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
            ](
                output,
                q,
                k_operand,
                mask_functor,
                valid_length,
                max_prompt_len,
                num_keys,
                scale,
                ctx,
                scalar_args_buf,
                kv_input_row_offsets,
                num_partitions,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k_operand,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
            )
        else:
            # Build extra_k_operand when extra_k is provided.
            var extra_k_operand: OptionalReg[type_of(k_operand)] = None
            if extra_k is not None:
                extra_k_operand = KVCacheMHAOperand(extra_k.value())

            flare_mla_decoding_dispatch[
                kv_num_heads=kv_num_heads,
                config=config,
                ragged=ragged,
                decoding_warp_split_k=decoding_warp_split_k,
                per_token_scale_rope_aware=False,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
            ](
                output,
                q,
                k_operand,
                mask_functor,
                valid_length,
                max_prompt_len,
                num_keys,
                scale,
                ctx,
                scalar_args_buf,
                kv_input_row_offsets,
                num_partitions,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k_operand,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
            )


# entrypoint for TileTensor as K input, used by tests and benchmarks.
def flare_mla_decoding[
    mask_t: MHAMask,
    dtype: DType,
    //,
    config: MHAConfig[dtype],
    decoding_warp_split_k: Bool = False,
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    k: TileTensor[address_space=AddressSpace.GENERIC, ...],
    mask_functor: mask_t,
    scale: Float32,
    ctx: DeviceContext,
    scalar_args_buf: NullableTileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    # if not set, we select num_partitions based on heuristics
    num_partitions: Optional[Int] = None,
) raises:
    comptime assert q.rank == 4, "only support rank 4 inputs."

    comptime kv_num_heads = type_of(k).static_shape[2]

    # Runtime dimensions.
    var num_keys = Int(k.dim[1]())

    var k_lt = k.to_layout_tensor()
    var k_operand = LayoutTensorMHAOperand(
        LayoutTensor[k_lt.dtype, k_lt.layout, k_lt.origin](
            k_lt.ptr,
            RuntimeLayout[k_lt.layout].row_major(
                k_lt.runtime_layout.shape.value.canonicalize()
            ),
        )
    )

    var valid_length = TileTensor(
        UnsafePointer[
            Scalar[DType.uint32], MutExternalOrigin
        ].unsafe_dangling(),
        row_major(Coord(Idx(0))),
    )

    flare_mla_decoding_dispatch[
        kv_num_heads=kv_num_heads,
        config=config,
        ragged=False,
        _is_cache_length_accurate=True,
        _use_valid_length=False,
        decoding_warp_split_k=decoding_warp_split_k,
    ](
        output,
        q,
        k_operand,
        mask_functor,
        valid_length,
        Int(q.dim[1]()),
        num_keys,
        scale,
        ctx,
        scalar_args_buf,
        kv_input_row_offsets=None,
        num_partitions=num_partitions,
    )


@always_inline
def flare_mla_decoding_dispatch[
    k_t: MHAOperand,
    mask_t: MHAMask,
    dtype: DType,
    //,
    kv_num_heads: Int,
    config: MHAConfig[dtype],
    ragged: Bool = False,
    # Work arounds to unify KVCache and TileTensor inputs:
    # Differentiate two cases, KV cache's length is before adding the latest
    # tokens e.g. zero for CE, and KV TileTensor's length is the latest length
    # e.g. prompt length for CE.
    _is_cache_length_accurate: Bool = False,
    # valid_length is needed for KV cache inputs and is empty for TileTensor
    # inputs to avoid overhead in benchmark.
    _use_valid_length: Bool = True,
    decoding_warp_split_k: Bool = False,
    per_token_scale_rope_aware: Bool = False,
    sparse: Bool = False,
    # Sparse-only routing flag: True selects the BF16-rope sparse kernel,
    # False (default) selects the all-FP8 sparse kernel.
    rope_aware_kv_sparse: Bool = False,
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    k: k_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: Int,
    max_cache_valid_length: Int,
    scale: Float32,
    ctx: DeviceContext,
    scalar_args_buf: NullableTileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
    num_partitions: Optional[Int] = None,
    q_scale_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    # Extra KV: separate always-attend cache operand.
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
) raises:
    comptime num_heads = config.num_heads
    comptime depth = config.depth
    comptime group = config.num_heads // kv_num_heads
    comptime assert num_heads == type_of(q).static_shape[q.rank - 2]

    # only A100 or H100 have the enough smem to store the full BM * head_dim Q tensor.
    comptime has_enough_smem = ctx.default_device_info == A100 or ctx.default_device_info == H100

    # For per_token_scale_rope_aware: Q's physical last dim is 640 (interleaved
    # FP8+BF16) but the logical depth (from config) is 576. Only validate
    # the config depth; the Q physical dim is checked separately.
    comptime if per_token_scale_rope_aware:
        comptime assert (
            depth == 576
        ), "per_token_scale_rope_aware requires logical depth == 576."
        comptime assert (
            type_of(q).static_shape[q.rank - 1] == 640
        ), "per_token_scale_rope_aware requires Q physical dim == 640."
    else:
        comptime assert (
            depth == type_of(q).static_shape[q.rank - 1] == 576
        ), "flareMLA_decoding only supports head_dim == 576."
    comptime assert (
        kv_num_heads == 1
    ), "flareMLA_decoding only supports kv_num_heads == 1."
    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "flareMLA_decoding currently only supports Nvidia and AMD GPUs."

    comptime assert (
        q.dtype.is_half_float() or q.dtype == DType.float8_e4m3fn
    ), "Only support half precision or float8_e4m3fn Q."

    # TileTensor always has static shapes for the last two dims.

    comptime if _is_sm10x_gpu(ctx.default_device_info):
        if scalar_args_buf.ptr:
            # Capturable path: GPU buffer is pre-computed, compute host-side
            # dispatch args from inputs.
            var batch_size: Int
            comptime if ragged:
                batch_size = Int(valid_length.dim[0]()) - 1
            else:
                batch_size = Int(q.dim(0))
            if batch_size == 0:
                return
            mla_decode_sm100_dispatch[
                q.dtype,
                k_t,
                output.dtype,
                mask_t,
                config=config,
                depth=depth,
                num_heads=num_heads,
                group=group,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=decoding_warp_split_k,
                per_token_scale_rope_aware=per_token_scale_rope_aware,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
            ](
                q,
                k,
                output,
                scale,
                valid_length,
                mask_functor,
                scalar_args_buf.value(),
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                ctx,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
            )
        else:
            # Legacy path: compute dispatch params and GPU buffer from inputs.
            var batch_size: Int
            comptime if ragged:
                batch_size = Int(valid_length.dim[0]()) - 1
            else:
                batch_size = Int(q.dim(0))
            if batch_size == 0:
                return

            comptime num_heads_val = type_of(q).static_shape[q.rank - 2]
            comptime _is_fp8_kv = (k_t.dtype == DType.float8_e4m3fn)
            var local_args = MLADispatchScalarArgs[
                num_heads=num_heads_val,
                _is_cache_length_accurate=_is_cache_length_accurate,
                is_fp8_kv=_is_fp8_kv,
            ](batch_size, max_cache_valid_length, max_prompt_len, ctx)
            mla_decode_sm100_dispatch[
                q.dtype,
                k_t,
                output.dtype,
                mask_t,
                config=config,
                depth=depth,
                num_heads=num_heads,
                group=group,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=decoding_warp_split_k,
                per_token_scale_rope_aware=per_token_scale_rope_aware,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
            ](
                q,
                k,
                output,
                scale,
                valid_length,
                mask_functor,
                TileTensor(
                    rebind[
                        UnsafePointer[Scalar[DType.int64], origin=MutAnyOrigin]
                    ](local_args.gpu_buf.unsafe_ptr()),
                    row_major[3](),
                ),
                local_args.batch_size,
                local_args.q_max_seq_len,
                max_cache_valid_length,
                ctx,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
            )
            _ = local_args^

    else:
        var batch_size: Int
        comptime if ragged:
            batch_size = Int(valid_length.dim[0]()) - 1
        else:
            batch_size = Int(q.dim(0))

        if batch_size == 0:
            return

        # only A100 or H100 have the enough smem to store the full BM * head_dim Q tensor.
        comptime has_enough_smem = ctx.default_device_info == A100 or ctx.default_device_info == H100

        # AMD FP8 MLA decode uses 32x32x64 MMA (depth=576 is not a multiple
        # of 128, so the 16x16x128 path is unusable). That requires BM>=32
        # and BK>=64. BF16 AMD MLA keeps BM=16, BK=32 (16x16x32 MMA).
        # For amd_fp8 we always use BM=32 even when num_heads<32; the top
        # num_heads rows of the tile carry real work and the bottom rows
        # do wasted MMA work (OOB Q reads return 0 via AMD buffer_load).
        comptime amd_fp8 = has_amd_gpu_accelerator() and q.dtype.is_float8()
        comptime preferred_BM = 32 if amd_fp8 else (
            16 if (not has_enough_smem or has_amd_gpu_accelerator()) else 32
        )
        comptime BM = preferred_BM if (
            amd_fp8 or preferred_BM <= num_heads
        ) else num_heads
        comptime BN = 64 if has_nvidia_gpu_accelerator() else 128
        comptime BK = 64 if (
            has_nvidia_gpu_accelerator() or amd_fp8
        ) else 32  # need 8 mma_tile per row to resolve the bank conflict on nvidia
        comptime WM = BM
        comptime WN = 16 if has_nvidia_gpu_accelerator() else 32
        # num warps in M and N, multiplied by warp size.
        comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

        comptime accum_type = get_accum_type[q.dtype]()
        comptime num_pipeline_stages = 6
        # smem for q
        var shared_mem_bytes = BM * depth * size_of[q.dtype]()

        shared_mem_bytes += BN * depth * size_of[k_t.dtype]()

        comptime num_warps = ceildiv(num_threads, WARP_SIZE)

        # smem for p and warp_scratch
        shared_mem_bytes += (
            BM * BN * size_of[k_t.dtype]()
            + 2 * num_warps * BM * size_of[accum_type]()
        )

        shared_mem_bytes = (
            shared_mem_bytes if has_nvidia_gpu_accelerator() else 0
        )

        comptime num_blocks_y = ceildiv(num_heads, BM)

        comptime depth_v = type_of(output).static_shape[output.rank - 1]

        comptime kernel = mla_decoding[
            q.dtype,
            k_t,
            output.dtype,
            mask_t,
            type_of(valid_length).LayoutType,
            BM=BM,
            BN=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            depth=depth,
            depth_v=depth_v,
            num_heads=num_heads,
            num_threads=num_threads,
            num_pipeline_stages=num_pipeline_stages,
            group=group,
            ragged=ragged,
            _use_valid_length=_use_valid_length,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=decoding_warp_split_k,
        ]

        # Pick num_partitions for split-K. Only AMD has a tuned heuristic
        # today; non-SM10x NVIDIA stays at 1 to preserve existing behavior.
        var num_partitions_value: Int
        if num_partitions:
            num_partitions_value = num_partitions.value()
        else:
            comptime if has_amd_gpu_accelerator():
                # MLA: kv_num_heads == 1, so heads_per_group == num_heads.
                num_partitions_value = mha_decoding_num_partitions(
                    batch_size, max_cache_valid_length, num_heads, ctx
                )
            else:
                num_partitions_value = 1

        var q_device = DeviceBuffer[q.dtype](
            ctx, q.ptr, q.num_elements(), owning=False
        )
        var output_device = DeviceBuffer[output.dtype](
            ctx, output.ptr, output.num_elements(), owning=False
        )

        if num_partitions_value == 1:
            # Single-partition fast path — no intermediate buffers, no reduce.
            var nullptr_device = DeviceBuffer[accum_type].empty(ctx)
            ctx.enqueue_function[kernel](
                q_device,
                k,
                output_device,
                nullptr_device,
                nullptr_device,
                scale,
                batch_size,
                num_partitions_value,
                max_cache_valid_length,
                valid_length,
                mask_functor,
                grid_dim=(1, num_blocks_y, batch_size),
                block_dim=(num_threads, 1, 1),
                shared_mem_bytes=shared_mem_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(
                        ctx.default_device_info.shared_memory_per_multiprocessor
                        - 4096
                    )
                ),
            )
        else:
            # Split-K: per-partition output (depth_v) and softmax stats.
            comptime intermediate_dtype = output.dtype

            var output_intermediate_data = ctx.enqueue_create_buffer[
                intermediate_dtype
            ](num_heads * depth_v * batch_size * num_partitions_value)

            var data_len = num_heads * batch_size * num_partitions_value
            var exp_sum_qk_max_data = ctx.enqueue_create_buffer[accum_type](
                2 * data_len
            )
            var exp_sum_device = DeviceBuffer[accum_type](
                ctx,
                exp_sum_qk_max_data.unsafe_ptr(),
                data_len,
                owning=False,
            )
            var qk_max_device = DeviceBuffer[accum_type](
                ctx,
                exp_sum_qk_max_data.unsafe_ptr() + data_len,
                data_len,
                owning=False,
            )

            ctx.enqueue_function[kernel](
                q_device,
                k,
                output_intermediate_data,
                exp_sum_device,
                qk_max_device,
                scale,
                batch_size,
                num_partitions_value,
                max_cache_valid_length,
                valid_length,
                mask_functor,
                grid_dim=(
                    num_partitions_value,
                    num_blocks_y,
                    batch_size,
                ),
                block_dim=(num_threads, 1, 1),
                shared_mem_bytes=shared_mem_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(
                        ctx.default_device_info.shared_memory_per_multiprocessor
                        - 4096
                    )
                ),
            )

            # AMD softmax always uses exp2; CUDA non-FA3 path uses exp.
            comptime reduce_use_exp2 = has_amd_gpu_accelerator()
            comptime kernel_reduce = mha_splitk_reduce[
                intermediate_dtype,
                output.dtype,
                depth=depth_v,
                num_heads=num_heads,
                num_threads=WARP_SIZE,
                use_exp2=reduce_use_exp2,
            ]
            ctx.enqueue_function[kernel_reduce](
                output_intermediate_data,
                output_device,
                exp_sum_device,
                qk_max_device,
                batch_size,
                num_partitions_value,
                grid_dim=(1, num_heads, batch_size),
                block_dim=(WARP_SIZE, 1, 1),
            )
            _ = exp_sum_qk_max_data^
            _ = output_intermediate_data^


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"mla_decoding_{q_type}_{output_type}_{BM}x{BN}x{BK}_{ragged}_nqh{num_heads}_nkvh{num_heads // group}",
    mangle=True,
)
def mla_decoding[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    ValidLT: TensorLayout,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    depth_v: Int,
    num_heads: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    group: Int = 1,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    batch_size: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    valid_length_tt: TileTensor[
        DType.uint32,
        ValidLT,
        MutAnyOrigin,
    ],  # valid length per batch
    mask: mask_t,
):
    var valid_length = valid_length_tt.to_layout_tensor()
    var batch_idx = block_idx.z

    # split-k offsets
    var partition_idx = block_idx.x
    var output_batch_offset: Int = (
        depth_v * num_heads * batch_idx
        + depth_v * num_heads * batch_size * partition_idx
    )
    var qk_max_offset = (
        num_heads * batch_idx + num_heads * batch_size * partition_idx
    )
    var exp_sum_offset = qk_max_offset

    # split-k intermediate buffers — only used when num_partitions > 1
    var qk_max_batch_ptr = qk_max_ptr
    if num_partitions > 1:
        qk_max_batch_ptr = qk_max_ptr + qk_max_offset

    var exp_sum_batch_ptr = exp_sum_ptr
    if num_partitions > 1:
        exp_sum_batch_ptr = exp_sum_ptr + exp_sum_offset

    var seq_len: Int
    var q_batch_offset: Int

    comptime if ragged:
        # treat valid_lengths as a input_row_offsets
        start_of_seq = Int(valid_length[batch_idx])
        end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq
        q_batch_offset = start_of_seq * depth * num_heads
    elif _use_valid_length:
        # treat valid_lengths as valid lengths
        q_batch_offset = depth * num_heads * batch_idx
        seq_len = Int(valid_length[batch_idx])
    else:
        seq_len = 1
        q_batch_offset = depth * num_heads * batch_idx

    var num_keys = k.cache_length(batch_idx)

    comptime if not _is_cache_length_accurate:
        num_keys += seq_len

    comptime if is_nvidia_gpu():
        mla_decoding_single_batch[
            BM=BM,
            BN=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            depth=depth,
            depth_v=depth_v,
            num_threads=num_threads,
            num_pipeline_stages=num_pipeline_stages,
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            q_ptr + q_batch_offset,
            k,
            output_ptr + output_batch_offset,
            exp_sum_batch_ptr,
            qk_max_batch_ptr,
            scale,
            num_keys,
            num_partitions,
            mask,
            batch_idx,
        )
    elif is_amd_gpu():
        comptime config = MHAConfig[q_type](
            num_heads,
            depth,
            num_queries_per_block=BM,
            num_keys_per_block=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            num_pipeline_stages=num_pipeline_stages,
            k_group_size=group,
        )

        var attention = Attention[
            config,
            group,
            False,  # sink
            token_gen=True,
            q_depth=depth,
            output_depth=depth_v,
            mla_mode=True,
            # K==V in MLA — load once, let PV reuse K's SMEM.
            mla_kv_alias=True,
        ](
            output_ptr + output_batch_offset,
            q_ptr + q_batch_offset,
            k,
            k,
            mask,
            None,
            batch_idx,
            scale,
            seq_len,
            num_keys,
            0,
        )
        attention.mla_decode(
            exp_sum_batch_ptr,
            qk_max_batch_ptr,
            num_partitions,
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@always_inline
def mla_decoding_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    depth_v: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    decoding_warp_split_k: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    num_keys: Int,
    num_partitions: Int,
    mask: mask_t,
    batch_idx: Int,
):
    """Flash attention v2 algorithm."""
    comptime k_type = k_t.dtype
    comptime assert q_type == k_type

    comptime simd_size = simd_width_of[q_type]()

    comptime WN_O = 128
    comptime nope_dim = depth_v
    comptime rope_dim = depth - depth_v

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN

    comptime assert num_warps_m * num_warps_n == (
        num_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    comptime assert (
        not decoding_warp_split_k
    ), "mla_decoding doesn't support warp split-k."

    var tid = thread_idx.x
    var warp_id = warp.broadcast(ufloordiv(tid, WARP_SIZE))
    var lane = lane_id()

    # Coordinates of the current warp.
    var warp_y, warp_x = udivmod(warp_id, num_warps_n)

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )

    comptime kv_smem_size = BN * depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()

    # For MLA, We define V = K[:, :nope_dim], thus we split the K tensor
    # in two parts when storing it in the smem: K[:, :nope_dim] and
    # K[:, nope_dim:(nope_dim+rope_dim)].
    # Instead of initializing the tiled iterator with a row-major layout
    # (BN, BK) like standard mha kernels, we manually set the following
    # layout. This ensures that once Q @ K calculation is complete, the
    # K[:, :nope_dim] tensor stored continuously in the smem.
    comptime IteratorTypeKV = LayoutTensorIter[
        k_type,
        Layout(IntTuple(BN, BK), IntTuple(nope_dim, 1)),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var kv_nope_smem_iter = IteratorTypeKV(
        k_smem,
        IteratorTypeKV.layout_uint_type(nope_dim),
        stride=IteratorTypeKV.layout_uint_type(BK),
    )

    # view the K[:, :nope_dim] as V tensor.
    comptime IteratorTypeV = LayoutTensorIter[
        k_type,
        Layout.row_major(BK, nope_dim),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        k_smem, IteratorTypeV.layout_uint_type(BN * nope_dim)
    )

    # smem for the last rope_dim of each head, will only be used during
    # Q @ K calculation.
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var k_rope_smem_iter = IteratorTypeK(
        k_smem + BN * nope_dim, IteratorTypeK.layout_uint_type(BN * rope_dim)
    )

    comptime mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime accum_type = get_accum_type[q_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    comptime num_output_rows = num_m_mmas * (WN_O // MMA_N)  # num_n_mmas
    comptime num_output_rows_full = num_output_rows
    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_output_rows_full, p_frag_size),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    # Rowwise max and sum for online softmax
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

    comptime for i in range(WM):
        rowmax[i] = min_or_neg_inf[accum_type]()
        rowsum[i] = 0.0

    # Shared memory for P = Q * K^t
    var p_smem = (k_smem + kv_smem_size).bitcast[Scalar[k_type]]()
    comptime p_smem_size = BM * BN
    comptime IteratorTypeP = LayoutTensorIter[
        k_type,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space=AddressSpace.SHARED,
    ]((p_smem + BM * BN).bitcast[Scalar[accum_type]]())

    comptime kv_num_heads = 1
    comptime kv_head_idx = 0
    var q_head_group = block_idx.y

    var q_offset = depth * BM * q_head_group

    comptime q_gmem_layout = Layout.row_major(BM, depth)
    var q_gmem_block = LayoutTensor[q_type, q_gmem_layout](q_ptr + q_offset)
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    start, end = get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
    )

    # Mask global memory iterator, seq_len = 1
    comptime seq_len = 1
    var mask_warp_col = warp_x * WN + start

    comptime q_num_vecs = BM * BK // simd_size

    comptime async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK,
        BK // simd_size,
    )

    comptime for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.layout_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        async_copy_commit_group()

        q_gmem_iter._incr()

    @always_inline
    @parameter
    def loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        var k_ptr = k.block_paged_ptr[BN](
            UInt32(batch_idx), UInt32(kv_tile_start_row), kv_head_idx, 0
        )

        comptime kv_gmem_layout = Layout(
            IntTuple(BN, depth),
            IntTuple(kv_num_heads * depth, 1),
        )
        var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[
            element_type=DType.int32, linear_idx_type=DType.int32
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                kv_num_heads * depth, 1
            ),
        )

        _ = p_reg_tile.fill(0)

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=not not_last_iter,
        ](
            k_ptr,
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        # load K[:, nope_dim:(nope_dim+rope_dim)], this would be used later
        comptime k_rope_num_ves = BN * rope_dim // simd_size
        comptime async_copy_k_rope_layout = Layout.row_major(
            ufloordiv(
                min(num_threads, k_rope_num_ves) * simd_size,
                k_rope_smem_iter.layout.shape[1].value(),
            ),
            k_rope_smem_iter.layout.shape[1].value() // simd_size,
        )

        comptime for k_id in range(ufloordiv(rope_dim, BK)):
            var k_rope_smem_tile = k_rope_smem_iter.next_unsafe(
                k_rope_smem_iter.layout_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_rope_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_rope_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter.next(ufloordiv(nope_dim, BK) + k_id)[].vectorize[
                    1, simd_size
                ](),
            )

        # Calculate Q[:, :nope_dim] @ K[:, :nope_dim] (K transposed)
        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=True,
            static_num_iters=ufloordiv(nope_dim, BK),
        ](
            p_reg_tile,
            q_smem_iter,
            k_gmem_iter,
            q_smem_iter,
            kv_nope_smem_iter,
            ufloordiv(nope_dim, BK),
        )

        # Calculate the last `rope_dim` part of Q @ K
        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            1,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=ufloordiv(rope_dim, BK),
        ](
            p_reg_tile,
            q_smem_iter.next_unsafe(
                q_smem_iter.linear_uint_type(ufloordiv(nope_dim, BK))
            ),
            k_rope_smem_iter,
            q_smem_iter.next_unsafe(
                q_smem_iter.linear_uint_type(ufloordiv(nope_dim, BK))
            ),
            k_rope_smem_iter,
            ufloordiv(nope_dim, BK),
        )

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        @parameter
        def _apply_mask[masked: Bool]():
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[
                    accum_type
                ]() if mask_t.apply_log2e_after_mask else scale.cast[
                    accum_type
                ]()
                * log2e
            )

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var q_head_idx = q_head_group * (BM + m_mma) * MMA_M
                    var mask_frag_col = mask_warp_col + n_mma * MMA_N

                    # Offset to current thread's fragment
                    mask_frag_col += umod(lane * p_frag_simdwidth, MMA_N)

                    # Offset to current thread's head idx
                    q_head_idx += ufloordiv(
                        lane, ufloordiv(MMA_N, p_frag_simdwidth)
                    )

                    comptime for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_col = mask_frag_col

                        var score_head_idx = q_head_idx + (i * MMA_M // 2)

                        var score_row_with_start_pos = num_keys - 1
                        var score_row = (
                            0  # this is a decoding kernel with seq_len = 1
                        )

                        comptime if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type=DType.uint32](
                                    block_idx.z,
                                    score_head_idx,
                                    score_row_with_start_pos,
                                    score_col,
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )
                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        comptime if mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[2, element_type=DType.uint32](
                                    score_row, score_col
                                ),
                                IndexList[2, element_type=DType.uint32](
                                    seq_len,
                                    num_keys,
                                ),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch[_apply_mask](
            mask.status(
                Index[dtype=DType.uint32](
                    num_keys,
                    kv_tile_start_row,
                ),
                Index[dtype=DType.uint32](1, BN),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )

        comptime output_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * (WN_O // MMA_N), 2
        )
        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[output_layout_by_mma_unit]().vectorize[
                1, 2
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[num_warps_n, WM](0, warp_y),
            rowmax,
            rowsum,
        )

        # Copy score fragments to shared memory with swizzling to resolve bank
        # conflicts for ldmatrix in the 2nd matmul.
        # warp_split_k does not need the copy as warps don't perform reduction
        # iterating across tiles, but use extra registers to perform MMAs
        # with warp-local data.
        _copy_frag_to_smem[BM, BN, BK, WM, WN, MMA_M, MMA_N, p_frag_simdwidth](
            p_smem_iter, p_reg_tile, UInt32(warp_x), UInt32(warp_y)
        )

        async_copy_wait_all()
        barrier()

        # S[m, :] @ V[:, (0:WN) + n*WN]
        multistage_mma[
            BM,
            nope_dim,
            BK,
            WM,
            WN_O,
            num_threads,
            num_pipeline_stages,
            False,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=ufloordiv(BN, BK),
        ](
            output_reg_tile,
            p_smem_iter,
            v_smem_iter,
            p_smem_iter,
            v_smem_iter,
            ufloordiv(BN, BK),
        )

        barrier()

    tile_and_unswitch[loop_over_kvcache, [BN]](start, end)

    # Apply softmax denumerator.
    comptime for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        comptime for n_mma in range(WN_O // 8):
            comptime for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    var o_offset = nope_dim * BM * q_head_group

    comptime output_gmem_layout = Layout(
        IntTuple(BM, nope_dim), IntTuple(nope_dim, 1)
    )
    var output_gmem_tile = LayoutTensor[output_type, output_gmem_layout](
        output_ptr + o_offset,
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN_O](warp_y, warp_x)

    # Write to global memory.
    comptime if output_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=nope_dim, access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, nope_dim),
            address_space=AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN_O](
            warp_y, warp_x
        )

        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout=Layout.row_major(
                WARP_SIZE * simd_size // WN_O, WN_O // simd_size
            ),
            swizzle=swizzle,
        ](
            output_gmem_warp_tile.vectorize[1, simd_size](),
            accum_smem_warp_tile.vectorize[1, simd_size](),
        )

    else:
        copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


# ===-----------------------------------------------------------------------===#
# GPU Multi-head Latent Attention (MLA) prefill implementations
# ===-----------------------------------------------------------------------===#


# entrypoint for MLA prefill kernels
@always_inline
def flare_mla_prefill[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    output_type: DType,
    //,
](
    output: TileTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    q: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    k: LayoutTensor[mut=False, _, address_space=AddressSpace.GENERIC, ...],
    v: LayoutTensor[mut=False, _, address_space=AddressSpace.GENERIC, ...],
    k_rope: cache_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ] = None,
) raises:
    """MLA prefill kernel that would only be called in the optimized compute
    graph. Only supports ragged Q/K/V inputs.

    The Q input has a shape of [seq_len, num_heads, q_depth].
    The K and V input has a shape of [cache_len, num_heads, depth].
    The K_rope input is retrieved from the KV cache, with a shape of
    [cache_len, 1, q_depth - depth].

    Specifically, for DeepSeek V2/3, depth = 128 and q_depth = 192.

    When computing attention scores (Q @ K), each head of K is smaller than Q
    head. The missing 64 elements of K are retrieved from the K cache, and
    broadcasted to all the heads. This kernel also handles that output has
    reduced dimension compared to input Q.

    This kernel handles batches with different valid lengths (i.e., before the
    padding). Such lengths are passed in valid_length argument.
    """
    comptime assert rank == 3, "only support ragged inputs"

    comptime if q.dtype == DType.bfloat16 and cache_t.dtype == DType.bfloat16:
        comptime assert (
            q.dtype == k.dtype == v.dtype == cache_t.dtype == output.dtype
        ), "Q, K, V, output should have same type if q.dtype is bfloat16"
    elif q.dtype == DType.bfloat16 and cache_t.dtype == DType.float8_e4m3fn:
        comptime assert q.dtype == k.dtype == v.dtype == output.dtype, (
            "Q, K, V, output should have same type if q.dtype is bfloat16 and"
            " k_rope.dtype is float8_e4m3fn"
        )
    elif q.dtype == DType.float8_e4m3fn and cache_t.dtype == DType.float8_e4m3fn:
        comptime assert (
            q.dtype == k.dtype == v.dtype == cache_t.dtype
            and output.dtype == DType.bfloat16
        ), (
            "Q, K, V, output should have same type if q.dtype is float8_e4m3fn"
            " and k_rope.dtype is float8_e4m3fn and output.dtype is bfloat16"
        )
    else:
        comptime assert False, "Q, K, V, output dtype combination not supported"

    @always_inline
    @parameter
    def description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg("k", k.runtime_layout.shape.value),
                    trace_arg("v", v.runtime_layout.shape.value),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k_rope.max_prompt_length())

        # Build row-major LayoutTensor from TileTensor for RaggedMHAOperand.
        comptime cache_row_offsets_layout = Layout.row_major(UNKNOWN_VALUE)
        var cache_row_offsets_lt = LayoutTensor[
            DType.uint32,
            cache_row_offsets_layout,
            cache_row_offsets.origin,
        ](
            cache_row_offsets.ptr,
            RuntimeLayout[cache_row_offsets_layout].row_major(
                coord_to_index_list(
                    cache_row_offsets.layout.shape_coord()
                ).canonicalize()
            ),
        )
        var k_operand = RaggedMHAOperand(
            LayoutTensor[k.dtype, k.layout, k.origin](
                k.ptr,
                RuntimeLayout[k.layout].row_major(
                    k.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var v_operand = RaggedMHAOperand(
            LayoutTensor[v.dtype, v.layout, v.origin](
                v.ptr,
                RuntimeLayout[v.layout].row_major(
                    v.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var k_rope_operand = KVCacheMHAOperand(k_rope)

        comptime kv_num_heads = cache_t.kv_params.num_heads
        comptime cache_depth = cache_t.kv_params.head_size
        comptime q_depth = type_of(q).static_shape[rank - 1]

        comptime num_keys_per_block = 64

        comptime mha_config = MHAConfig[dtype](
            type_of(q).static_shape[rank - 2],  # num_heads
            Int(k.layout.shape[rank - 1]),  # depth
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )

        flare_mla_prefill_dispatch[
            kv_num_heads=kv_num_heads,
            q_depth=q_depth,
            cache_depth=cache_depth,
            config=mha_config,
        ](
            output,
            q,
            k_operand,
            v_operand,
            k_rope_operand,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
            cache_offsets,
        )


# entrypoint for TileTensor as K_rope input, used by tests.
@always_inline
def flare_mla_prefill[
    rank: Int,
    mask_t: MHAMask,
    dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    k: TileTensor[address_space=AddressSpace.GENERIC, ...],
    v: TileTensor[address_space=AddressSpace.GENERIC, ...],
    k_rope: TileTensor[address_space=AddressSpace.GENERIC, ...],
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: TileTensor[
        mut=True, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ] = None,
) raises:
    comptime assert rank == 3, "only support ragged inputs"

    comptime if q.dtype == DType.bfloat16 and k_rope.dtype == DType.bfloat16:
        comptime assert (
            q.dtype == k.dtype == v.dtype == k_rope.dtype == output.dtype
        ), "Q, K, V, output should have same type if q.dtype is bfloat16"
    elif q.dtype == DType.bfloat16 and k_rope.dtype == DType.float8_e4m3fn:
        comptime assert q.dtype == k.dtype == v.dtype == output.dtype, (
            "Q, K, V, output should have same type if q.dtype is bfloat16 and"
            " k_rope.dtype is float8_e4m3fn"
        )
    elif q.dtype == DType.float8_e4m3fn and k_rope.dtype == DType.float8_e4m3fn:
        comptime assert (
            q.dtype == k.dtype == v.dtype == k_rope.dtype
            and output.dtype == DType.bfloat16
        ), (
            "Q, K, V, output should have same type if q.dtype is float8_e4m3fn"
            " and k_rope.dtype is float8_e4m3fn and output.dtype is bfloat16"
        )
    else:
        comptime assert False, "Q, K, V, output dtype combination not supported"

    @always_inline
    @parameter
    def description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int = Int(q.dim[0]())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        var cache_row_offsets_lt = cache_row_offsets.to_layout_tensor()
        var k_lt = k.to_layout_tensor()
        var v_lt = v.to_layout_tensor()
        var k_rope_lt = k_rope.to_layout_tensor()
        var k_operand = RaggedMHAOperand(
            LayoutTensor[k_lt.dtype, k_lt.layout, k_lt.origin](
                k_lt.ptr,
                RuntimeLayout[k_lt.layout].row_major(
                    k_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var v_operand = RaggedMHAOperand(
            LayoutTensor[v_lt.dtype, v_lt.layout, v_lt.origin](
                v_lt.ptr,
                RuntimeLayout[v_lt.layout].row_major(
                    v_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var k_rope_operand = LayoutTensorMHAOperand(
            LayoutTensor[k_rope_lt.dtype, k_rope_lt.layout, k_rope_lt.origin](
                k_rope_lt.ptr,
                RuntimeLayout[k_rope_lt.layout].row_major(
                    k_rope_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )

        comptime output_type = output.dtype
        comptime kv_num_heads = type_of(k_rope).static_shape[2]
        comptime cache_depth = type_of(k_rope).static_shape[3]
        comptime q_depth = type_of(q).static_shape[q.rank - 1]
        comptime num_keys_per_block = 64
        comptime mha_config = MHAConfig[dtype](
            type_of(q).static_shape[rank - 2],
            type_of(k).static_shape[rank - 1],
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )
        flare_mla_prefill_dispatch[
            kv_num_heads=kv_num_heads,
            config=mha_config,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=True,
        ](
            output,
            q,
            k_operand,
            v_operand,
            k_rope_operand,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
            cache_offsets,
        )


# entrypoint for TileTensor as K_rope input with scales, used by tests.
@always_inline
def flare_mla_prefill[
    rank: Int,
    mask_t: MHAMask,
    dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    k: TileTensor[address_space=AddressSpace.GENERIC, ...],
    v: TileTensor[address_space=AddressSpace.GENERIC, ...],
    k_rope: TileTensor[address_space=AddressSpace.GENERIC, ...],
    k_rope_scales: TileTensor[address_space=AddressSpace.GENERIC, ...],
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: TileTensor[
        mut=True, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ] = None,
) raises:
    comptime assert rank == 3, "only support ragged inputs"
    comptime assert (
        q.dtype == k.dtype == v.dtype == k_rope.dtype == output.dtype
    ) if k_rope.dtype == DType.bfloat16 else (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), (
        "Q, K, V, output should have same type if k_rope.dtype is bfloat16,"
        " otherwise only Q, K, V should have same type."
    )
    comptime assert (
        q.dtype == DType.float32 or q.dtype.is_half_float()
    ), "Only support single and half precision."

    @always_inline
    @parameter
    def description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int = Int(q.dim[0]())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        var cache_row_offsets_lt = cache_row_offsets.to_layout_tensor()
        var k_lt = k.to_layout_tensor()
        var v_lt = v.to_layout_tensor()
        var k_rope_lt = k_rope.to_layout_tensor()
        var k_rope_scales_lt = k_rope_scales.to_layout_tensor()
        var k_operand = RaggedMHAOperand(
            LayoutTensor[k_lt.dtype, k_lt.layout, k_lt.origin](
                k_lt.ptr,
                RuntimeLayout[k_lt.layout].row_major(
                    k_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var v_operand = RaggedMHAOperand(
            LayoutTensor[v_lt.dtype, v_lt.layout, v_lt.origin](
                v_lt.ptr,
                RuntimeLayout[v_lt.layout].row_major(
                    v_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var k_rope_operand = LayoutTensorMHAOperand(
            LayoutTensor[k_rope_lt.dtype, k_rope_lt.layout, k_rope_lt.origin](
                k_rope_lt.ptr,
                RuntimeLayout[k_rope_lt.layout].row_major(
                    k_rope_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[
                k_rope_scales_lt.dtype,
                k_rope_scales_lt.layout,
                k_rope_scales_lt.origin,
            ](
                k_rope_scales_lt.ptr,
                RuntimeLayout[k_rope_scales_lt.layout].row_major(
                    k_rope_scales_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )

        comptime output_type = output.dtype
        comptime kv_num_heads = type_of(k_rope).static_shape[2]
        comptime cache_depth = type_of(k_rope).static_shape[3]
        comptime q_depth = type_of(q).static_shape[q.rank - 1]
        comptime num_keys_per_block = 64
        comptime mha_config = MHAConfig[dtype](
            type_of(q).static_shape[rank - 2],
            type_of(k).static_shape[rank - 1],
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )
        flare_mla_prefill_dispatch[
            kv_num_heads=kv_num_heads,
            q_depth=q_depth,
            cache_depth=cache_depth,
            config=mha_config,
            _ndbuffer_mha_operand=True,
        ](
            output,
            q,
            k_operand,
            v_operand,
            k_rope_operand,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
            cache_offsets,
        )


@always_inline
def flare_mla_prefill[
    rank: Int,
    mask_t: MHAMask,
    dtype: DType,
    scale_dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    q_nope: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    q_rope: TileTensor[address_space=AddressSpace.GENERIC, ...],
    q_scale: TileTensor[scale_dtype, address_space=AddressSpace.GENERIC, ...],
    k: TileTensor[address_space=AddressSpace.GENERIC, ...],
    k_scales: TileTensor[scale_dtype, address_space=AddressSpace.GENERIC, ...],
    v: TileTensor[address_space=AddressSpace.GENERIC, ...],
    k_rope: TileTensor[address_space=AddressSpace.GENERIC, ...],
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: TileTensor[
        mut=True, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ] = None,
) raises:
    @always_inline
    @parameter
    def description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q_nope",
                        coord_to_index_list(q_nope.layout.shape_coord()),
                    ),
                    trace_arg(
                        "q_rope",
                        coord_to_index_list(q_rope.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int = Int(q_nope.dim[0]())

        comptime assert q_scale.rank == 2, (
            "q_scale must be a per token scale 2D tensor of [batch_size *"
            " seq_len, 1]"
        )
        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        var cache_row_offsets_lt = cache_row_offsets.to_layout_tensor()

        comptime assert k_scales.rank == 2, (
            "k_scales must be a per token scale 2D tensor of [batch_size *"
            " num_keys, 1]"
        )

        var k_lt = k.to_layout_tensor()
        var k_scales_lt = k_scales.to_layout_tensor()
        var v_lt = v.to_layout_tensor()
        var k_rope_lt = k_rope.to_layout_tensor()
        var q_rope_lt = q_rope.to_layout_tensor()
        var q_scale_lt = q_scale.to_layout_tensor()
        var k_operand = RaggedMHAOperand(
            LayoutTensor[k_lt.dtype, k_lt.layout, k_lt.origin](
                k_lt.ptr,
                RuntimeLayout[k_lt.layout].row_major(
                    k_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[k_scales_lt.dtype, k_scales_lt.layout, ImmutAnyOrigin](
                k_scales_lt.ptr,
                RuntimeLayout[k_scales_lt.layout].row_major(
                    k_scales_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )

        var v_operand = RaggedMHAOperand(
            LayoutTensor[v_lt.dtype, v_lt.layout, v_lt.origin](
                v_lt.ptr,
                RuntimeLayout[v_lt.layout].row_major(
                    v_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var k_rope_operand = LayoutTensorMHAOperand(
            LayoutTensor[k_rope_lt.dtype, k_rope_lt.layout, k_rope_lt.origin](
                k_rope_lt.ptr,
                RuntimeLayout[k_rope_lt.layout].row_major(
                    k_rope_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )

        var batch_size: Int = Int(valid_length.dim[0]()) - 1

        if batch_size == 0 or max_prompt_len == 0:
            return

        comptime output_type = output.dtype
        comptime kv_num_heads = type_of(k_rope).static_shape[2]
        comptime cache_depth = type_of(k_rope).static_shape[3]
        comptime q_depth = type_of(q_nope).static_shape[
            q_nope.rank - 1
        ] + type_of(q_rope).static_shape[q_rope.rank - 1]
        comptime num_keys_per_block = 64
        comptime mha_config = MHAConfig[dtype](
            type_of(q_nope).static_shape[rank - 2],
            type_of(k).static_shape[rank - 1],
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )
        mla_sm100_prefill_per_token_scale[
            config=mha_config,
            group=mha_config.num_heads // kv_num_heads,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=True,
        ](
            output,
            q_nope,
            q_rope_lt,
            q_scale_lt,
            k_operand,
            k_rope_operand,
            v_operand,
            mask_functor,
            valid_length,
            DynamicInt(max_prompt_len),
            scale,
            batch_size,
            ctx,
        )


# entrypoint for paged K_rope (KVCacheT) with per-token Q/K scales,
# used by tests. Mirrors the contiguous per-token-scale entrypoint
# above but swaps K_rope from a TileTensor to a paged KVCacheT
# operand (matching the generic paged entrypoint at ~:1517).
@always_inline
def flare_mla_prefill[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    scale_dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    q_nope: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    q_rope: TileTensor[address_space=AddressSpace.GENERIC, ...],
    q_scale: TileTensor[scale_dtype, address_space=AddressSpace.GENERIC, ...],
    k: TileTensor[address_space=AddressSpace.GENERIC, ...],
    k_scales: TileTensor[scale_dtype, address_space=AddressSpace.GENERIC, ...],
    v: TileTensor[address_space=AddressSpace.GENERIC, ...],
    k_rope: cache_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: TileTensor[
        mut=True, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ] = None,
) raises:
    @always_inline
    @parameter
    def description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q_nope",
                        coord_to_index_list(q_nope.layout.shape_coord()),
                    ),
                    trace_arg(
                        "q_rope",
                        coord_to_index_list(q_rope.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k_rope.max_prompt_length())

        comptime assert q_scale.rank == 2, (
            "q_scale must be a per token scale 2D tensor of [batch_size *"
            " seq_len, 1]"
        )
        comptime assert k_scales.rank == 2, (
            "k_scales must be a per token scale 2D tensor of [batch_size *"
            " num_keys, 1]"
        )

        var cache_row_offsets_lt = cache_row_offsets.to_layout_tensor()
        var k_lt = k.to_layout_tensor()
        var k_scales_lt = k_scales.to_layout_tensor()
        var v_lt = v.to_layout_tensor()
        var q_rope_lt = q_rope.to_layout_tensor()
        var q_scale_lt = q_scale.to_layout_tensor()
        var k_operand = RaggedMHAOperand(
            LayoutTensor[k_lt.dtype, k_lt.layout, k_lt.origin](
                k_lt.ptr,
                RuntimeLayout[k_lt.layout].row_major(
                    k_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[k_scales_lt.dtype, k_scales_lt.layout, ImmutAnyOrigin](
                k_scales_lt.ptr,
                RuntimeLayout[k_scales_lt.layout].row_major(
                    k_scales_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )

        var v_operand = RaggedMHAOperand(
            LayoutTensor[v_lt.dtype, v_lt.layout, v_lt.origin](
                v_lt.ptr,
                RuntimeLayout[v_lt.layout].row_major(
                    v_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var k_rope_operand = KVCacheMHAOperand(k_rope)

        var batch_size: Int = Int(valid_length.dim[0]()) - 1

        if batch_size == 0 or max_prompt_len == 0:
            return

        comptime kv_num_heads = cache_t.kv_params.num_heads
        comptime cache_depth = cache_t.kv_params.head_size
        comptime q_depth = type_of(q_nope).static_shape[
            q_nope.rank - 1
        ] + type_of(q_rope).static_shape[q_rope.rank - 1]
        comptime num_keys_per_block = 64
        comptime mha_config = MHAConfig[dtype](
            type_of(q_nope).static_shape[rank - 2],
            type_of(k).static_shape[rank - 1],
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )
        mla_sm100_prefill_per_token_scale[
            config=mha_config,
            group=mha_config.num_heads // kv_num_heads,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=False,
        ](
            output,
            q_nope,
            q_rope_lt,
            q_scale_lt,
            k_operand,
            k_rope_operand,
            v_operand,
            mask_functor,
            valid_length,
            DynamicInt(max_prompt_len),
            scale,
            batch_size,
            ctx,
        )


@always_inline
def flare_mla_prefill_dispatch[
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    mask_t: MHAMask,
    dtype: DType,
    output_type: DType,
    //,
    kv_num_heads: Int,
    config: MHAConfig[dtype],
    q_depth: Int = 192,
    cache_depth: Int = 576,
    _ndbuffer_mha_operand: Bool = False,
](
    output: TileTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    q: TileTensor[dtype, address_space=AddressSpace.GENERIC, ...],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: Int,
    scale: Float32,
    ctx: DeviceContext,
    cache_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ] = None,
) raises:
    comptime num_heads = config.num_heads
    comptime depth = config.depth
    comptime group = config.num_heads // kv_num_heads
    comptime rank = output.rank

    comptime assert q_depth == type_of(q).static_shape[rank - 1]
    comptime assert num_heads == type_of(q).static_shape[rank - 2]
    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "flareMLA_prefill currently only supports Nvidia and AMD GPUs."

    var batch_size: Int = Int(valid_length.dim[0]()) - 1

    if batch_size == 0 or max_prompt_len == 0:
        return

    comptime q_half_float = dtype in (DType.float16, DType.bfloat16)

    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()

    comptime q_smem = BM * q_depth
    comptime k_smem = BN * q_depth
    comptime v_smem = BN * depth

    comptime smem_use = (q_smem + k_smem + v_smem) * size_of[
        config.dtype
    ]() if has_nvidia_gpu_accelerator() else 0

    comptime if _is_sm10x_gpu(ctx.default_device_info):
        comptime assert (
            k_rope_t.dtype == DType.bfloat16
            or k_rope_t.dtype == DType.float8_e4m3fn
        ), "Only support bfloat16 or float8_e4m3fn for SM100"

        mla_sm100_prefill[
            config=config,
            group=group,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](
            output,
            q,
            k,
            rebind[type_of(k)](v),
            k_rope,
            mask_functor,
            valid_length,
            DynamicInt(max_prompt_len),
            scale,
            batch_size,
            ctx,
        )

    else:
        comptime assert (
            k_rope_t.dtype == DType.bfloat16 or has_amd_gpu_accelerator()
        ), (
            "Only support bfloat16 for non-SM100 Nvidia GPUs; AMD supports"
            " bfloat16 and float8_e4m3fn"
        )

        var q_device = DeviceBuffer[q.dtype](
            ctx, q.ptr, q.num_elements(), owning=False
        )
        var output_device = DeviceBuffer[output.dtype](
            ctx, output.ptr, output.num_elements(), owning=False
        )

        comptime kernel = mla_prefill[
            config.dtype,
            k_t,
            v_t,
            k_rope_t,
            output.dtype,
            mask_t,
            type_of(valid_length).LayoutType,
            config,
            group=group,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ]
        var grid_dim = LaunchDim(
            ceildiv(max_prompt_len, BM),
            config.num_heads,
            batch_size,
        ) if has_nvidia_gpu_accelerator() else LaunchDim(
            config.num_heads,
            ceildiv(max_prompt_len, BM),
            batch_size,
        )
        ctx.enqueue_function[kernel](
            q_device,
            k,
            v,
            k_rope,
            output_device,
            scale,
            batch_size,
            max_prompt_len,
            valid_length,
            cache_offsets,
            mask_functor,
            grid_dim=grid_dim,
            block_dim=(config.num_threads(), 1, 1),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )


@__llvm_metadata(`rocdl.waves_per_eu`=Int(2))
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
@__name(
    t"mla_prefill_{q_type}_{output_type}_{q_depth}_{cache_depth}_nqh{config.num_heads}_nkvh{config.num_heads // group}",
    mangle=True,
)
def mla_prefill[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    valid_layout: TensorLayout,
    config: MHAConfig,
    group: Int = 128,
    q_depth: Int = 192,
    cache_depth: Int = 576,
    _ndbuffer_mha_operand: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    batch_size: Int,
    seq_len_arg: Int,
    valid_length_tt: TileTensor[
        DType.uint32,
        valid_layout,
        MutAnyOrigin,
    ],
    cache_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ],
    mask: mask_t,
):
    var valid_length = valid_length_tt.to_layout_tensor()
    comptime depth = config.depth
    var batch_idx = block_idx.z

    # mha inputs
    var seq_len: Int
    var max_seq_len = seq_len_arg
    var num_keys: Int
    var start_pos: UInt32 = 0
    var cache_start_pos: UInt32 = 0

    # treat valid_lengths as a input_row_offsets
    start_of_seq = Int(valid_length[batch_idx])
    end_of_seq = Int(valid_length[batch_idx + 1])
    seq_len = end_of_seq - start_of_seq

    @always_inline
    def q_block_idx() -> Int:
        return block_idx.x if is_nvidia_gpu() else block_idx.y

    # @always_inline
    # def head_idx() -> Int:
    #     return block_idx.y if is_nvidia_gpu() else block_idx.x

    if seq_len < q_block_idx() * config.block_m():
        return

    comptime if _ndbuffer_mha_operand:
        num_keys = k_rope.cache_length(batch_idx)
        start_pos = UInt32(num_keys - seq_len)

    else:
        start_pos = UInt32(k_rope.cache_length(batch_idx))
        num_keys = Int(start_pos) + seq_len

    if cache_offsets:
        var cache_offsets_nd = cache_offsets.value()
        cache_start_pos = cache_offsets_nd[batch_idx][0]

    q_batch_offset = start_of_seq * q_depth * config.num_heads
    o_batch_offset = start_of_seq * depth * config.num_heads

    comptime if is_nvidia_gpu():
        mla_prefill_single_batch[
            config=config,
            group=group,
            q_depth=q_depth,
            cache_depth=cache_depth,
        ](
            q_ptr + q_batch_offset,
            k,
            v,
            k_rope,
            output_ptr + o_batch_offset,
            scale,
            seq_len,
            max_seq_len,
            start_pos,
            cache_start_pos,
            num_keys,
            mask,
            batch_idx,
        )
    elif is_amd_gpu():
        var attention = Attention[config, 1, False, q_depth=q_depth](
            output_ptr + o_batch_offset,
            q_ptr + q_batch_offset,
            k,
            v,
            mask,
            None,
            batch_idx,
            scale,
            seq_len,
            num_keys,
            Int(start_pos),
            Int(cache_start_pos),
        )
        attention.mla_prefill(
            k_rope,
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@always_inline
def mla_prefill_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    config: MHAConfig,
    group: Int = 1,
    q_depth: Int = 192,
    cache_depth: Int = 576,
](
    q_ptr: UnsafePointer[mut=True, Scalar[q_type], _],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    output_ptr: UnsafePointer[mut=True, Scalar[output_type], _],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    cache_start_pos: UInt32,
    num_keys: Int,
    mask: mask_t,
    batch_idx: Int,
):
    """MLA for encoding where seqlen > 1."""
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime k_rope_type = k_rope_t.dtype
    comptime assert (
        q_type == k_type and k_type == v_type and k_type == k_rope_type
    )

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()
    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()
    comptime num_heads = config.num_heads
    comptime depth = config.depth

    comptime rope_depth = q_depth - depth

    comptime cache_num_heads = num_heads // group

    comptime assert (
        num_warps_m * num_warps_n == num_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    var tid: Int = thread_idx.x
    var warp_id = UInt32(warp_id[broadcast=True]())
    var lane = UInt32(lane_id())

    # Coordinates of the current warp.
    var warp_y, warp_x = divmod(warp_id, UInt32(num_warps_n))

    # The entire query block (BM x q_depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * q_depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    comptime k_smem_size = BN * q_depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    comptime v_smem_size = BN * depth
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, depth),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    var head_idx = UInt32(block_idx.y)
    var q_tile_idx = UInt32(block_idx.x)

    # Query global memory iterator
    comptime q_gmem_layout = Layout(
        IntTuple(BM, q_depth),
        IntTuple(num_heads * q_depth, 1),
    )
    var q_tile_num_rows = min(
        UInt32(BM), UInt32(seq_len) - q_tile_idx * UInt32(BM)
    )
    var q_offset = UInt32(q_depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )

    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[element_type=DType.int32, linear_idx_type=DType.int32](
            RuntimeTuple[q_gmem_layout.shape, element_type=DType.int32](
                Int(q_tile_num_rows), q_depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type=DType.int32](
                num_heads * q_depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x q_depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    comptime mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime WM = config.WM
    comptime WN = config.WN
    comptime WN_O = depth
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime num_n_mmas_output = WN_O // MMA_N

    comptime accum_type = get_accum_type[q_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas_output, p_frag_size),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    # Rowwise max and sum for online softmax
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

    comptime for i in range(0, WM, 2):
        rowmax.store(i, SIMD[accum_type, 2](min_or_neg_inf[accum_type]()))
        rowsum.store(i, SIMD[accum_type, 2](0))

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space=AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # Mask global memory iterator.
    var mask_block_row = q_tile_idx * UInt32(BM)
    var mask_warp_row = warp_y * UInt32(WM)
    var mask_warp_col = warp_x * UInt32(WN)

    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime q_num_vecs = BM * BK // simd_size

    comptime async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK,
        BK // simd_size,
    )

    comptime for q_id in range(q_depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.linear_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        async_copy_commit_group()

        q_gmem_iter._incr()

    # Iterate over KV, equivalent to the following with if hoisted out.
    #   ```
    #   for i in range(kv_tile_start_row, seq_len, tile_size):
    #     if i + tile_size >= seq_len:
    #       loop_over_kvcache[tile_size, False]
    #     else:
    #       loop_over_kvcache[tile_size, True]
    #   ```
    # Only the last iteration is doing boundary check.
    @__copy_capture(seq_len, max_seq_len, num_keys, start_pos)
    @always_inline
    @parameter
    def loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        if (
            mask.status(
                Index[dtype=DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    Int(UInt32(kv_tile_start_row) + cache_start_pos),
                ),
                Index[dtype=DType.uint32](BM, BN),
            )
            == TileMaskStatus.FULL_MASK
        ):
            return

        comptime kv_gmem_layout = Layout(
            IntTuple(BN, depth),
            IntTuple(num_heads * depth, 1),
        )

        var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[
            element_type=DType.int32, linear_idx_type=DType.int32
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                num_heads * depth, 1
            ),
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx)),
                0,
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx)),
                0,
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, depth, axis=0](0, 0)

        # here we set up variables for k_rope tensor
        comptime k_rope_gmem_layout = Layout(
            IntTuple(BN, cache_depth),
            IntTuple(cache_num_heads * cache_depth, 1),
        )

        var k_rope_runtime_layout = RuntimeLayout[
            element_type=DType.int32, linear_idx_type=DType.int32
        ](
            RuntimeTuple[k_rope_gmem_layout.shape, element_type=DType.int32](
                kv_tile_num_rows, cache_depth
            ),
            RuntimeTuple[k_rope_gmem_layout.stride, element_type=DType.int32](
                cache_num_heads * cache_depth, 1
            ),
        )

        var k_rope_gmem_block = LayoutTensor[
            k_rope_type,
            k_rope_gmem_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=not not_last_iter,
        ](
            k_rope.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row) + cache_start_pos,
                UInt32(Int(head_idx // UInt32(group))),
                UInt32(cache_depth - rope_depth),
            ),
            k_rope_runtime_layout,
        )
        var k_rope_gmem_iter = k_rope_gmem_block.tiled_iterator[BN, BK, axis=1](
            0, 0
        )

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        @always_inline
        @parameter
        def _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: type_of(tensor)
        ):
            return {
                tensor.ptr,
                {{num_rows, tensor.dim[1]()}, tensor.runtime_layout.stride},
            }

        comptime kv_num_vecs = BN * BK // simd_size
        comptime async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // k_smem_iter.layout.stride[0].value(),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        comptime for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.layout_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter[].vectorize[1, simd_size](),
            )

            async_copy_commit_group()

            k_gmem_iter._incr()

        comptime for k_id in range(depth // BK, q_depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.linear_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_rope_gmem_iter[].vectorize[1, simd_size](),
            )

            async_copy_commit_group()

            k_rope_gmem_iter._incr()

        # synchronize here since we can overlap q tile and first k tile copy
        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=q_depth // BK,
            k_group_size=config.k_group_size,
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            q_depth // BK,
        )

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        @parameter
        def _apply_mask[masked: Bool]():
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[
                    accum_type
                ]() if mask_t.apply_log2e_after_mask else scale.cast[
                    accum_type
                ]()
                * log2e
            )

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + UInt32(m_mma * MMA_M)
                    var mask_frag_col = mask_warp_col + UInt32(n_mma * MMA_N)

                    # Offset to current thread's fragment
                    mask_frag_row += lane // UInt32(MMA_N // p_frag_simdwidth)
                    mask_frag_col += (
                        lane * UInt32(p_frag_simdwidth) % UInt32(MMA_N)
                    )

                    comptime for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row
                            + mask_frag_row
                            + UInt32(i * MMA_M // 2)
                        )
                        var score_col = mask_frag_col

                        score_row_with_start_pos = score_row + start_pos
                        score_col_with_cache_start_pos = (
                            score_col + cache_start_pos
                        )

                        comptime if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type=DType.uint32](
                                    block_idx.z,
                                    block_idx.y,
                                    Int(score_row_with_start_pos),
                                    Int(score_col_with_cache_start_pos),
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )
                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        comptime if mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[2, element_type=DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                IndexList[2, element_type=DType.uint32](
                                    seq_len,
                                    num_keys,
                                ),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch[_apply_mask](
            mask.status(
                Index[dtype=DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    UInt32(kv_tile_start_row) + cache_start_pos,
                ),
                Index[dtype=DType.uint32](BM, BN),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += UInt32(BN)

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )
        comptime reg_output_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas_output, 2
        )

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[reg_output_layout_by_mma_unit]().vectorize[
                1, 2
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        comptime async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // v_smem_iter.layout.stride[0].value(),
            v_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load V tile into smem
        comptime for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(
                v_smem_iter.layout_uint_type(v_id)
            )[]

            comptime if not not_last_iter:
                var num_rows_bound = min(
                    BK, end - (kv_tile_start_row + v_id * BK)
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle=v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_tensor.vectorize[1, simd_size](),
            )

            async_copy_commit_group()

            v_gmem_iter._incr()

        comptime if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                p_frag_simdwidth,
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                BM,
                depth,
                BK,
                WM,
                depth,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters=ufloordiv(BN, BK),
                k_group_size=config.k_group_size,
            ](
                output_reg_tile,
                p_smem_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
            )

        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                BM,
                depth,
                BK,
                WM,
                depth,
                num_threads,
                2,
                False,  # transpose_b
                swizzle_a=False,
                prefetch_init=False,
                static_num_iters=ufloordiv(BN, BK),
                k_group_size=config.k_group_size,
            ](
                output_reg_tile,
                p_reg_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
            )

    tile_and_unswitch[loop_over_kvcache, [BN]](0, num_keys)

    comptime output_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(num_heads * depth, 1)
    )

    var output_offset = UInt32(depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )
    var output_gemm_runtime_layout = RuntimeLayout[
        element_type=DType.int32, linear_idx_type=DType.int32
    ](
        RuntimeTuple[output_gmem_layout.shape, element_type=DType.int32](
            Int(q_tile_num_rows), depth
        ),
        RuntimeTuple[output_gmem_layout.stride, element_type=DType.int32](
            num_heads * depth, 1
        ),
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        output_ptr + Int(output_offset),
        output_gemm_runtime_layout,
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN_O](
        Int(warp_y), Int(warp_x)
    )

    # Apply softmax denumerator.
    comptime for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        comptime for n_mma in range(num_n_mmas_output):
            comptime for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    # Write to global memory.
    comptime if output_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=depth, access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space=AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, depth](
            Int(warp_y), Int(warp_x)
        )
        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout=Layout.row_major(
                num_threads * simd_size // depth,
                depth // simd_size,
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )
    else:
        copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


# ===-----------------------------------------------------------------------===#
# Helper function that creates cache_row_offsets for the MLA prefill kernel
# ===-----------------------------------------------------------------------===#


def set_buffer_lengths_to_zero[
    BufferLengthsLayoutType: TensorLayout,
](
    buffer_lengths: TileTensor[
        mut=True, DType.int32, BufferLengthsLayoutType, MutExternalOrigin
    ],
):
    comptime assert buffer_lengths.flat_rank == 1
    comptime MAX_CHUNKS = buffer_lengths.static_shape[0]

    comptime for chunk_idx in range(MAX_CHUNKS):
        buffer_lengths[chunk_idx] = 0


@always_inline
def mla_prefill_plan[
    cache_t: KVCacheT,
](
    buffer_row_offsets: TileTensor[mut=True, DType.uint32, ...],
    cache_offsets: TileTensor[mut=True, DType.uint32, ...],
    buffer_lengths: TileTensor[mut=True, DType.int32, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    k_cache: cache_t,
    buffer_token_size: UInt32,
    ctx: DeviceContext,
) raises:
    """
    This calls a GPU kernel that plans how to process a batch of sequences with
    varying lengths using a fixed-size buffer.

    Each sequence in the batch has some existing cached tokens and new input
    tokens. The kernel divides the total tokens into chunks of buffer_token_size.

    For each chunk (iteration), it calculates:
        1. Buffer offsets for each sequence in each chunk
        2. Cache offsets for each sequence in each chunk
        3. Total buffer lengths for each processing iteration
    """
    var batch_size: Int = Int(input_row_offsets.dim[0]()) - 1

    if batch_size == 0:
        # Fill buffer lengths with 0
        comptime kernel = set_buffer_lengths_to_zero[buffer_lengths.LayoutType,]
        ctx.enqueue_function[kernel](buffer_lengths, grid_dim=1, block_dim=1)
    else:
        comptime kernel = mla_prefill_plan_kernel[
            buffer_row_offsets.LayoutType,
            cache_offsets.LayoutType,
            buffer_lengths.LayoutType,
            input_row_offsets.LayoutType,
            cache_t,
        ]

        ctx.enqueue_function[kernel](
            buffer_row_offsets,
            cache_offsets,
            buffer_lengths,
            input_row_offsets.as_immut(),
            k_cache,
            buffer_token_size,
            grid_dim=(ceildiv(batch_size, 128), 1, 1),
            block_dim=(128, 1, 1),
        )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
@__name(t"mla_prefill_plan", mangle=True)
def mla_prefill_plan_kernel[
    BufferRowOffsetsLayoutType: TensorLayout,
    CacheOffsetsLayoutType: TensorLayout,
    BufferLengthsLayoutType: TensorLayout,
    InputRowOffsetsLayoutType: TensorLayout,
    cache_t: KVCacheT,
](
    buffer_row_offsets: TileTensor[
        mut=True,
        DType.uint32,
        BufferRowOffsetsLayoutType,
        MutExternalOrigin,
    ],
    cache_offsets: TileTensor[
        mut=True,
        DType.uint32,
        CacheOffsetsLayoutType,
        MutExternalOrigin,
    ],
    buffer_lengths: TileTensor[
        mut=True,
        DType.int32,
        BufferLengthsLayoutType,
        MutExternalOrigin,
    ],
    input_row_offsets: TileTensor[
        DType.uint32,
        InputRowOffsetsLayoutType,
        ImmutExternalOrigin,
    ],
    k_cache: cache_t,
    buffer_token_size: UInt32,
):
    comptime assert buffer_row_offsets.flat_rank == 2
    comptime assert cache_offsets.flat_rank == 2
    comptime assert buffer_lengths.flat_rank == 1
    comptime assert input_row_offsets.flat_rank == 1

    var seq_idx = global_idx.x
    var seq_start_pos = 0
    var batch_size: Int = Int(input_row_offsets.dim[0]()) - 1
    var buffer_size: Int = Int(buffer_token_size)

    comptime MAX_CHUNKS = buffer_lengths.static_shape[0]
    comptime page_size = cache_t.page_size_
    comptime assert page_size != 0, "Only PagedKVCache is supported."

    if seq_idx >= batch_size:
        return

    # Calculate starting position for this sequence.
    # Note: the total cache length of each sequence is aligned to the page size.
    var prev_row_offset = Int(input_row_offsets[0])
    for i in range(seq_idx):
        # The cache length that has been prefilled in previous forward passes.
        var cache_length = k_cache.cache_length(i)

        # account for the new input tokens.
        var row_offset_i = Int(input_row_offsets[i + 1])
        cache_length += row_offset_i - prev_row_offset
        prev_row_offset = row_offset_i

        seq_start_pos += align_up(cache_length, page_size)

    var curr_seq_len = align_up(
        k_cache.cache_length(seq_idx)
        + Int(input_row_offsets[seq_idx + 1])
        - prev_row_offset,
        page_size,
    )

    # which chunk this sequence starts in
    var start_chunk = seq_start_pos // buffer_size
    var processed_seq_len = UInt32(0)
    var seq_len_left = curr_seq_len

    # Fill buffer offsets for this sequence
    comptime for chunk_idx in range(MAX_CHUNKS):
        if chunk_idx < start_chunk:
            buffer_row_offsets[chunk_idx, seq_idx] = UInt32(buffer_size)
        elif chunk_idx == start_chunk:
            buffer_row_offsets[chunk_idx, seq_idx] = UInt32(
                seq_start_pos % buffer_size
            )
        else:
            buffer_row_offsets[chunk_idx, seq_idx] = 0

        cache_offsets[chunk_idx, seq_idx] = processed_seq_len

        var chunk_len = min(
            seq_len_left,
            buffer_size - Int(buffer_row_offsets[chunk_idx, seq_idx]),
        )
        processed_seq_len += UInt32(chunk_len)
        seq_len_left -= chunk_len

    # If this is the last sequence in the batch
    if seq_idx == batch_size - 1:
        var seq_end_pos = seq_start_pos + curr_seq_len
        var end_chunk = (seq_end_pos + buffer_size - 1) // buffer_size - 1

        # Set buffer lengths for all chunks
        comptime for chunk_idx in range(MAX_CHUNKS):
            if chunk_idx < end_chunk:
                buffer_row_offsets[chunk_idx, seq_idx + 1] = UInt32(buffer_size)
                buffer_lengths[chunk_idx] = Int32(buffer_size)
            elif chunk_idx == end_chunk:
                var last_chunk_len = seq_end_pos - end_chunk * buffer_size
                buffer_row_offsets[chunk_idx, seq_idx + 1] = UInt32(
                    last_chunk_len
                )
                buffer_lengths[chunk_idx] = Int32(last_chunk_len)
            else:
                buffer_row_offsets[chunk_idx, seq_idx + 1] = 0
                buffer_lengths[chunk_idx] = -1


# ===-----------------------------------------------------------------------===#
# Helper function that copies K cache to a contiguous buffer
# ===-----------------------------------------------------------------------===#


@always_inline
def _k_cache_to_buffer[
    dtype: DType,
    cache_t: KVCacheT,
    BufferRowOffsetsLayoutType: TensorLayout,
    CacheOffsetsLayoutType: TensorLayout,
](
    buffer_row_offsets: TileTensor[
        DType.uint32, BufferRowOffsetsLayoutType, ...
    ],
    cache_offsets: TileTensor[DType.uint32, CacheOffsetsLayoutType, ...],
    k_cache: cache_t,
    length: Int32,
    buffer: TileTensor[mut=True, dtype=dtype, ...],
    context: DeviceContext,
) raises:
    comptime num_heads = cache_t.kv_params.num_heads
    comptime assert num_heads == 1, "num_heads should be equal to 1"
    comptime assert buffer.rank == 2, "buffer should be rank 2"
    comptime assert buffer_row_offsets.flat_rank == 1
    comptime assert cache_offsets.flat_rank == 1

    @always_inline
    @parameter
    @__copy_capture(k_cache, buffer_row_offsets, cache_offsets)
    def copy_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        comptime assert rank == 2, "rank should be equal to 2"

        var global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            buffer_row_offsets, global_token_idx
        )

        var token_idx = (
            global_token_idx
            - Int(buffer_row_offsets[batch_idx])
            + Int(cache_offsets[batch_idx])
        )

        var head_dim_idx = idx[1]

        var cache_val = rebind[SIMD[dtype, width]](
            k_cache.load[width=width](
                batch_idx, 0, token_idx, head_dim_idx
            ).cast[dtype]()
        )

        buffer.store_linear(idx, cache_val)

    var launch_shape = IndexList[2](
        Int(length),
        Int(buffer.dim[1]()),
    )
    comptime target_simd_width = simd_width_of[dtype, target=get_gpu_target()]()

    def copy_fn_unified[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) register_passable:
        copy_fn[width, rank, alignment](idx)

    _elementwise_impl_gpu[simd_width=target_simd_width](
        copy_fn_unified, shape=launch_shape, ctx=context
    )
