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

from math import ceildiv, recip
from math.constants import log2e
from collections import OptionalReg
from sys import (
    CompilationTarget,
    align_of,
    env_get_bool,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
)
from sys.info import _cdna_4_or_newer
import gpu.primitives.warp as warp
from algorithm import elementwise
from algorithm.functional import tile_and_unswitch, unswitch, vectorize
from bit import next_power_of_two
from buffer import DimList, NDBuffer
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host import Dim as LaunchDim
from gpu.host import FuncAttribute
from gpu.host.info import A100, B200, H100, GPUInfo
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from kv_cache.types import KVCacheT
from layout import Layout
from layout.int_tuple import IntTuple, UNKNOWN_VALUE
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.bmm import batched_matmul
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from linalg.transpose import transpose
from memory import stack_allocation

from .attention.gpu.amd.mha_gfx942 import MHAAttentionConfig
from .attention.gpu.amd.mha_gfx950 import Attention
from nn.mha_mask import MaterializedMask, MHAMask, TileMaskStatus
from nn.mha_operand import (
    KVCacheMHAOperand,
    MHAOperand,
    LayoutTensorMHAOperand,
    RaggedMHAOperand,
)
from nn.mha_score_mod import IdentityScoreMod, ScoreModTrait
from nn.mha_sm90 import mha_sm90_dispatch
from nn.mha_sm100_1q import mha_sm100_dispatch as mha_sm100_1q_dispatch
from nn.mha_sm100_2q import mha_sm100_dispatch as mha_sm100_2q_dispatch
from nn.mha_utils import (
    DynamicInt,
    FlashAttentionAlgorithm,
    MHAConfig,
    NoPartition,
    SplitKPartition,
    StaticInt,
    _copy_frag_to_smem,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple

from .softmax import (
    _exp2_concrete,
    _exp_concrete,
    _online_softmax_iter_for_mma_output,
    _online_softmax_iter_for_mma_output_split_warp_reduce,
    _softmax_gpu,
    softmax,
)

# ===-----------------------------------------------------------------------===#
# Flash attention
# ===-----------------------------------------------------------------------===#


fn flash_attention[
    dtype: DType,
    q_layout: Layout,
    //,
    use_score_mod: Bool = False,
    config: MHAConfig[dtype] = {
        UInt(Int(q_layout.shape[2])),
        UInt(Int(q_layout.shape[3])),
    },
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
    sink: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, ...],
    q: LayoutTensor[
        mut=False, dtype, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[mut=False, address_space = AddressSpace.GENERIC, ...],
    v: LayoutTensor[mut=False, address_space = AddressSpace.GENERIC, ...],
    mask: LayoutTensor[mut=False, address_space = AddressSpace.GENERIC, ...],
    scale: Float32,
    context: DeviceContextPtr = DeviceContextPtr(),
    num_partitions: Optional[Int] = None,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    # TODO docstring
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg("k", k.runtime_layout.shape.value),
                    trace_arg("v", v.runtime_layout.shape.value),
                    trace_arg("output", output.runtime_layout.shape.value),
                ]
            )
        )

    var ctx = context.get_device_context()

    with Trace[TraceLevel.OP, target = ctx.default_device_info.api](
        "flash_attention",
        Trace[
            TraceLevel.OP, target = ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        return flash_attention[
            use_score_mod=use_score_mod,
            config=config,
            decoding_warp_split_k=decoding_warp_split_k,
            naive_kernel=naive_kernel,
            sink=sink,
        ](
            output,
            q,
            k,
            v,
            MaterializedMask(
                LayoutTensor[
                    mask.dtype,
                    Layout.row_major(mask.layout.shape),
                    MutAnyOrigin,
                ](
                    mask.ptr,
                    RuntimeLayout[
                        Layout.row_major(mask.layout.shape)
                    ].row_major(mask.runtime_layout.shape.value.canonicalize()),
                )
            ),
            IdentityScoreMod(),
            scale,
            context.get_device_context(),
            num_partitions,
            sink_weights=sink_weights,
        )


fn get_mha_decoding_num_partitions[
    num_heads: Int, group: Int
](batch_size: Int, num_keys: Int, ctx: DeviceContext) -> Int:
    comptime sm_count = ctx.default_device_info.sm_count

    @parameter
    if has_amd_gpu_accelerator():
        # AMD split-k strategy: scale partitioning based on occupancy
        # 256: min context length where split-k overhead is worthwhile
        if num_keys <= 256:
            return 1

        # Compute total work items (occupancy)
        work_items = batch_size * (num_heads // group)

        # High occupancy when work_items >= sm_count (≥1 work item per CU)
        if work_items >= sm_count:
            # High occupancy: scale partition size to avoid over-partitioning
            # 128: base partition size matching kernel block (BN=128)
            # 64: scaling factor - reduces partitions as occupancy increases
            occupancy_scale = work_items // 64
            return min(ceildiv(num_keys, 256 * occupancy_scale), WARP_SIZE)
        else:
            # Low occupancy: aggressive partitioning for more parallelism
            # 128: keys per partition (matches kernel BN=128)
            # WARP_SIZE (64): max partitions (AMD wavefront size, reduction limit)
            return min(ceildiv(num_keys, 256), WARP_SIZE)
    else:
        if num_keys > 512:
            return min(
                next_power_of_two(
                    min(
                        sm_count // (batch_size * (num_heads // group)),
                        num_keys // 512,
                    )
                ),
                32,
            )
    return 1


fn flash_attention_hw_supported[qkv_type: DType]() -> Bool:
    return has_nvidia_gpu_accelerator() or (
        has_amd_gpu_accelerator() and qkv_type == DType.bfloat16
    )


fn depth_supported_by_gpu[
    depth: Int,
    mask_t: MHAMask,
    config: MHAConfig,
    info: GPUInfo,
]() -> Bool:
    comptime is_sm90or100 = (info == H100) or (info == B200)
    comptime head_depth_supported = depth == 128 or (
        depth == 64
        and (is_sm90or100 or info == A100 or has_amd_gpu_accelerator())
    ) or (
        depth == 256
        and (
            has_amd_gpu_accelerator()
            or (is_sm90or100 and mask_t.mask_safe_out_of_bounds)
        )
    ) or (
        depth in (72, 80, 96)
        and is_sm90or100
        and config.algorithm == FlashAttentionAlgorithm(3)
    )
    return head_depth_supported


# Entry point for flash_attention with batch_size > 1.
@always_inline
fn flash_attention[
    cache_t: KVCacheT,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    q_layout: Layout,
    //,
    use_score_mod: Bool = False,
    config: MHAConfig[dtype] = {
        UInt(Int(q_layout.shape[q_layout.rank() - 2])),
        UInt(Int(q_layout.shape[q_layout.rank() - 1])),
    },
    ragged: Bool = False,
    sink: Bool = False,
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, ...],
    q: LayoutTensor[
        mut=False, dtype, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: Optional[Int] = None,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
    num_partitions: Optional[Int] = None,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    """Flash attention 2 algorithm.
    Compute:
        (1) Transpose (Q) BSHD -> BHSD;
        (2) Transpose (K) BSHD -> BHSD;
        (3) Transpose (V) BSHD -> BHSD;
        (4) P = Bmm(Q, K), P is also called "score";
        (5) P = P * scale + mask;
        (6) P = softmax(P);
        (7) O = Bmm(P, V)
        (8) Output = Transpose(O).

    B, S, H, D denote batch size, sequence length, head count and depth, respectively.
    (1), (2), (3) happens while loading the data into shared memory.
    (8) happens when writing output to global memory.

    All inputs (query, key, and value) must have BSHD layout. The mask can be
    BSS or BHSS.

    This kernel also handles grouped attention optimization. In this case the shape of
    K and V are BShD where h = H / num_groups.

    This kernels handles batches with different valid lengths (i.e., before the
    padding). Such lengths are passed in valid_length argument.
    """
    comptime assert (
        ragged or q.rank == 4
    ), "only support rank 4 inputs for non-ragged inputs."
    comptime assert (
        not ragged or q.rank == 3
    ), "only support rank 3 inputs for ragged inputs."
    comptime assert (
        q.dtype == cache_t.dtype == output.dtype
    ), "Q, K, V, output should have same type."
    comptime assert (
        q.dtype == DType.float32 or q.dtype.is_half_float()
    ), "Only support single and half precision."

    # TODO docstring
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg("output", output.runtime_layout.shape.value),
                ]
            )
        )

    with Trace[TraceLevel.OP, target = ctx.default_device_info.api](
        "flash_attention",
        Trace[
            TraceLevel.OP, target = ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        # TODO: This helps differentiate between CE/TG. Not batch-specific.
        #       We'll just implement a flag on the cache object which is true
        #       when the batch contains all cache_lens == 0. Remove this when
        #       such flag (part of ContiguousKVCache) is implemented.
        var is_token_generation = (
            k.max_prompt_length() == 1 and not k.empty_cache()
        )

        var max_prompt_len: Int
        var num_keys = Int(k.max_context_length())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k.max_prompt_length())

        # Whether head and depth are static. With BSHD, B and S are dynamic.
        # H and D are always known for opaque KVCache types, we only check Q.
        # fmt: off
        comptime head_depth_known = q.layout.shape.all_known[q.rank-2, q.rank]()
        comptime depth = Int(q.layout.shape[q.rank-1])
        comptime gpu_info = ctx.default_device_info
        comptime head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
        comptime flash_attention_applicable = flash_attention_hw_supported[dtype]() and head_depth_known and head_depth_supported and not naive_kernel
        # fmt: on
        comptime kv_num_heads = cache_t.kv_params.num_heads

        var k_operand = KVCacheMHAOperand(k)
        var v_operand = KVCacheMHAOperand(v)

        flash_attention_dispatch[
            kv_num_heads = Int(kv_num_heads),
            use_score_mod=use_score_mod,
            config=config,
            ragged=ragged,
            sink=sink,
            _is_flash_attention_applicable=flash_attention_applicable,
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            output,
            q,
            k_operand,
            v_operand,
            mask_functor,
            score_mod_functor,
            max_prompt_len,
            num_keys,
            scale,
            is_token_generation,
            ctx,
            rebind[
                LayoutTensor[
                    DType.uint32,
                    Layout.row_major(UNKNOWN_VALUE),
                    ImmutAnyOrigin,
                ]
            ](valid_length),
            kv_input_row_offsets,
            num_partitions,
            sink_weights,
        )


@always_inline
fn q_num_matrix_view_rows[
    dtype: DType, //
](q: LayoutTensor[mut=False, dtype, ...]) -> Int:
    # for tma if decoding, we view q as a rows x depth matrix
    # otherwise, we view q as a rows x (depth*num_heads) matrix
    var num_rows: Int = q.dim[0]()

    @parameter
    for i in range(1, q.rank - 2):
        num_rows *= q.dim[i]()
    return num_rows


@always_inline
fn flash_attention_dispatch[
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    q_layout: Layout,
    //,
    kv_num_heads: Int,
    use_score_mod: Bool = False,
    config: MHAConfig[dtype] = {
        UInt(Int(q_layout.shape[q_layout.rank() - 2])),
        UInt(Int(q_layout.shape[q_layout.rank() - 1])),
    },
    ragged: Bool = False,
    sink: Bool = False,
    _is_flash_attention_applicable: Bool = True,
    # Work arounds to unify KVCache and NDBuffer inputs:
    # Differentiate two cases, KV cache's length is before adding the latest
    # tokens e.g. zero for CE, and KV NBuffer's length is the latest length
    # e.g. prompt length for CE.
    _is_cache_length_accurate: Bool = False,
    # valid_length is needed for KV cache inputs and is empty for homogeneous
    # NDBuffer inputs to avoid overhead in benchmark.
    _use_valid_length: Bool = True,
    # we might also want to use valid length for padded NDBuffer inputs
    _padded_ndbuffer: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, ...],
    q: LayoutTensor[
        mut=False, dtype, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    max_prompt_len: Int,
    max_cache_valid_length: Int,
    scale: Float32,
    is_token_generation: Bool,
    ctx: DeviceContext,
    valid_length: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
    num_partitions: Optional[Int] = None,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    comptime num_heads = config.num_heads
    comptime depth = config.depth
    comptime group = config.num_heads // UInt(kv_num_heads)

    # K V smem is only separate for GPUs with shared memory greater or equal to A100's.
    comptime is_shared_kv = ctx.default_device_info.shared_memory_per_multiprocessor < A100.shared_memory_per_multiprocessor

    comptime assert depth == UInt(Int(q.layout.shape[q.rank - 1]))
    comptime assert num_heads == UInt(Int(q.layout.shape[q.rank - 2]))
    var batch_size: Int

    @parameter
    if ragged:
        batch_size = valid_length.value().dim[0]() - 1
    # This branch holds for both KVCache and NDBuffer inputs.
    # Q is BSHD, S is either homogeneous or padded to same length.
    else:
        batch_size = q.dim[0]()

    comptime q_half_float = dtype in (DType.float16, DType.bfloat16)
    comptime q_half_float_or_fp32 = dtype == DType.float32 or q_half_float

    var q_device = DeviceBuffer[q.dtype](ctx, q.ptr, q.size(), owning=False)
    var output_device = DeviceBuffer[output.dtype](
        ctx, output.ptr, output.size(), owning=False
    )

    @parameter
    if _is_flash_attention_applicable:
        comptime is_sm90 = ctx.default_device_info == H100
        comptime is_sm100 = ctx.default_device_info == B200
        if not is_token_generation:
            # TODO note that we have to handle mask tensor alignment here.
            # Choose matmul parameters based on dtype.
            @parameter
            if (
                (is_sm90 or is_sm100)
                and q_half_float
                and (ragged or not _use_valid_length)
                and config.algorithm == FlashAttentionAlgorithm(3)
            ):
                num_rows_q = q_num_matrix_view_rows(q)

                @parameter
                if is_sm90:
                    mha_sm90_dispatch[
                        config=config,
                        group = Int(group),
                        use_score_mod=use_score_mod,
                        ragged=ragged,
                        sink=sink,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                    ](
                        output.to_device_buffer(ctx),
                        q.to_device_buffer(ctx),
                        k,
                        rebind[k_t](v),
                        num_rows_q,
                        mask_functor,
                        score_mod_functor,
                        valid_length.value().to_device_buffer(ctx),
                        DynamicInt(max_prompt_len),
                        max_cache_valid_length,
                        scale,
                        kv_input_row_offsets,
                        batch_size,
                        NoPartition[get_accum_type[q.dtype]()](),
                        ctx,
                        sink_weights,
                    )
                else:
                    comptime assert is_sm100

                    @parameter
                    if depth == 256 or not env_get_bool["ENABLE_FA4", True]():
                        mha_sm100_1q_dispatch[
                            config=config,
                            group = Int(group),
                            use_score_mod=use_score_mod,
                            ragged=ragged,
                            sink=sink,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                        ](
                            output.to_device_buffer(ctx),
                            q.to_device_buffer(ctx),
                            k,
                            rebind[k_t](v),
                            num_rows_q,
                            mask_functor,
                            score_mod_functor,
                            valid_length.value().to_device_buffer(ctx),
                            DynamicInt(max_prompt_len),
                            max_cache_valid_length,
                            scale,
                            kv_input_row_offsets,
                            batch_size,
                            NoPartition[get_accum_type[q.dtype]()](),
                            ctx,
                            sink_weights,
                        )
                    else:
                        mha_sm100_2q_dispatch[
                            config=config,
                            group = Int(group),
                            use_score_mod=use_score_mod,
                            ragged=ragged,
                            sink=sink,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                        ](
                            output.to_device_buffer(ctx),
                            q.to_device_buffer(ctx).unsafe_ptr(),
                            k,
                            rebind[k_t](v),
                            num_rows_q,
                            mask_functor,
                            score_mod_functor,
                            valid_length.value()
                            .to_device_buffer(ctx)
                            .unsafe_ptr(),
                            DynamicInt(max_prompt_len),
                            max_cache_valid_length,
                            scale,
                            kv_input_row_offsets,
                            batch_size,
                            NoPartition[get_accum_type[q.dtype]()](),
                            ctx,
                            sink_weights,
                        )

            else:
                comptime BM = config.block_m()
                comptime smem_use = config.shared_mem_bytes[is_shared_kv]()
                comptime kernel = mha[
                    config.dtype,
                    k_t,
                    v_t,
                    output.dtype,
                    mask_t,
                    score_mod_t,
                    type_of(valid_length.value()).layout,
                    config,
                    group = Int(group),
                    use_score_mod=use_score_mod,
                    ragged=ragged,
                    is_shared_kv=is_shared_kv,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    _padded_ndbuffer=_padded_ndbuffer,
                ]

                var grid_dim = LaunchDim(
                    ceildiv(max_prompt_len, Int(BM)),
                    Int(config.num_heads),
                    batch_size,
                ) if has_nvidia_gpu_accelerator() else LaunchDim(
                    Int(config.num_heads),
                    ceildiv(max_prompt_len, Int(BM)),
                    batch_size,
                )

                ctx.enqueue_function[kernel, kernel](
                    q_device,
                    k,
                    v,
                    output_device,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length.value(),
                    kv_input_row_offsets,
                    sink_weights,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=grid_dim,
                    block_dim=(Int(config.num_threads()), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(smem_use)
                    ),
                )
        # FA3 decoding impl only support half precision, while fp32 is supported
        # for fp32 as well.
        elif q_half_float_or_fp32 and is_token_generation:
            comptime BM = 16
            comptime BN = depth
            comptime BK = 32 if has_amd_gpu_accelerator() else (
                16 if q.dtype == DType.float32 else 32
            )
            comptime WM = BM
            comptime WN = 32
            # num warps in M and N, multiplied by warp size.
            comptime num_threads = (BM // WM) * Int(BN // WN) * WARP_SIZE

            comptime accum_type = get_accum_type[q.dtype]()
            comptime num_pipeline_stages = 4
            # smem for q
            var shared_mem_bytes = BM * Int(depth) * size_of[q.dtype]()

            # separate KV smem if we have enough smem
            @parameter
            if not is_shared_kv:
                shared_mem_bytes += (
                    2 * Int(BN) * Int(depth) * size_of[k_t.dtype]()
                )
            else:
                shared_mem_bytes += (
                    num_pipeline_stages * Int(BN) * BK * size_of[k_t.dtype]()
                )

            comptime num_warps = ceildiv(num_threads, WARP_SIZE)

            # smem for p and warp_scratch
            shared_mem_bytes += (
                BM * Int(BN) * size_of[k_t.dtype]()
                + 2 * num_warps * BM * size_of[accum_type]()
            )
            comptime num_blocks_y = num_heads // group

            var num_partitions_value = num_partitions.value() if num_partitions else get_mha_decoding_num_partitions[
                Int(num_heads), Int(group)
            ](
                batch_size, max_cache_valid_length, ctx
            )

            comptime use_fa3_kernel = (
                (is_sm90 or is_sm100)
                and q_half_float
                and (ragged or not _use_valid_length)
                and mask_t.mask_safe_out_of_bounds
                and config.algorithm == FlashAttentionAlgorithm(3)
            )

            @parameter
            if (not use_fa3_kernel) and (depth % 64) != 0:
                # FA2 kernel only supports depth % 64 == 0
                # Assumes BSHD.
                mha_gpu_naive[
                    ragged=ragged,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                ](
                    q,
                    k,
                    v,
                    mask_functor,
                    output,
                    valid_length.value(),
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    Int(num_heads),
                    Int(depth),
                    Int(group),
                    ctx,
                    sink_weights,
                )
            else:
                comptime kernel = mha_decoding[
                    q.dtype,
                    k_t,
                    v_t,
                    output.dtype,
                    mask_t,
                    score_mod_t,
                    type_of(valid_length.value()).layout,
                    BM=BM,
                    BN=BN,
                    BK = UInt(BK),
                    WM=WM,
                    WN=WN,
                    depth=depth,
                    num_heads=num_heads,
                    num_threads = UInt(num_threads),
                    num_pipeline_stages = UInt(num_pipeline_stages),
                    group=group,
                    use_score_mod=use_score_mod,
                    ragged=ragged,
                    is_shared_kv=is_shared_kv,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    decoding_warp_split_k=decoding_warp_split_k,
                ]

                if num_partitions_value == 1:

                    @parameter
                    if use_fa3_kernel:
                        num_rows_q = q_num_matrix_view_rows(q)

                        @parameter
                        if is_sm90:
                            mha_sm90_dispatch[
                                config=config,
                                group = Int(group),
                                use_score_mod=use_score_mod,
                                ragged=ragged,
                                sink=sink,
                                _is_cache_length_accurate=_is_cache_length_accurate,
                            ](
                                output.to_device_buffer(ctx),
                                q.to_device_buffer(ctx),
                                k,
                                rebind[k_t](v),
                                num_rows_q,
                                mask_functor,
                                score_mod_functor,
                                valid_length.value().to_device_buffer(ctx),
                                StaticInt[1](),
                                max_cache_valid_length,
                                scale,
                                kv_input_row_offsets,
                                batch_size,
                                NoPartition[accum_type](),
                                ctx,
                                sink_weights,
                            )
                        else:
                            mha_sm100_1q_dispatch[
                                config=config,
                                group = Int(group),
                                use_score_mod=use_score_mod,
                                ragged=ragged,
                                sink=sink,
                                _is_cache_length_accurate=_is_cache_length_accurate,
                            ](
                                output.to_device_buffer(ctx),
                                q.to_device_buffer(ctx),
                                k,
                                rebind[k_t](v),
                                num_rows_q,
                                mask_functor,
                                score_mod_functor,
                                valid_length.value().to_device_buffer(ctx),
                                StaticInt[1](),
                                max_cache_valid_length,
                                scale,
                                kv_input_row_offsets,
                                batch_size,
                                NoPartition[accum_type](),
                                ctx,
                                sink_weights,
                            )
                    else:
                        comptime nullptr = UnsafePointer[
                            Scalar[accum_type], MutAnyOrigin
                        ]()

                        var nullptr_device = DeviceBuffer[accum_type](
                            ctx, nullptr, 0, owning=False
                        )
                        ctx.enqueue_function[kernel, kernel](
                            q_device,
                            k,
                            v,
                            output_device,
                            nullptr_device,
                            nullptr_device,
                            scale,
                            batch_size,
                            num_partitions_value,
                            max_cache_valid_length,
                            valid_length.value(),
                            sink_weights,
                            mask_functor,
                            score_mod_functor,
                            grid_dim=(
                                1,
                                Int(num_blocks_y),
                                batch_size,
                            ),
                            block_dim=(num_threads, 1, 1),
                            shared_mem_bytes=shared_mem_bytes if has_nvidia_gpu_accelerator() else 0,
                            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                                UInt32(
                                    (
                                        ctx.default_device_info.shared_memory_per_multiprocessor
                                        - 4096
                                    ) if has_nvidia_gpu_accelerator() else 0
                                )
                            ),
                        )
                    return

                else:
                    # We split partitions and then reduce
                    # allocate memory for intermediate results
                    # q # [B, S, H, D]

                    # Determine intermediate buffer type based on platform
                    # AMD uses float32 for higher precision with aggressive split-k
                    comptime intermediate_dtype = output.dtype

                    var output_intermediate_data = ctx.enqueue_create_buffer[
                        intermediate_dtype
                    ](
                        Int(
                            num_heads
                            * depth
                            * UInt(batch_size)
                            * UInt(num_partitions_value)
                        )
                    )

                    var output_intermediate = LayoutTensor[
                        intermediate_dtype, Layout.row_major[4]()
                    ](
                        output_intermediate_data.unsafe_ptr(),
                        RuntimeLayout[Layout.row_major[4]()].row_major(
                            Index(
                                num_partitions_value,
                                batch_size,
                                Int(num_heads),
                                Int(depth),
                            )
                        ),
                    )

                    var data_len = (
                        num_heads
                        * UInt(batch_size)
                        * UInt(num_partitions_value)
                    )
                    var data_dim = Index(
                        num_partitions_value,
                        batch_size,
                        Int(num_heads),
                    )
                    var exp_sum_qk_max_data = ctx.enqueue_create_buffer[
                        accum_type
                    ](2 * Int(data_len))

                    var exp_sum = LayoutTensor[
                        accum_type, Layout.row_major[3]()
                    ](
                        exp_sum_qk_max_data.unsafe_ptr(),
                        RuntimeLayout[Layout.row_major[3]()].row_major(
                            data_dim
                        ),
                    )

                    var qk_max = LayoutTensor[
                        accum_type, Layout.row_major[3]()
                    ](
                        exp_sum_qk_max_data.unsafe_ptr() + data_len,
                        RuntimeLayout[Layout.row_major[3]()].row_major(
                            data_dim
                        ),
                    )

                    var exp_sum_device = DeviceBuffer[accum_type](
                        ctx, exp_sum.ptr, exp_sum.size(), owning=False
                    )
                    var qk_max_device = DeviceBuffer[accum_type](
                        ctx, qk_max.ptr, qk_max.size(), owning=False
                    )

                    @parameter
                    if use_fa3_kernel:
                        num_rows_q = q_num_matrix_view_rows(q)

                        @parameter
                        if is_sm90:
                            mha_sm90_dispatch[
                                config=config,
                                group = Int(group),
                                use_score_mod=use_score_mod,
                                ragged=ragged,
                                sink=sink,
                                _is_cache_length_accurate=_is_cache_length_accurate,
                            ](
                                output_intermediate.to_device_buffer(ctx),
                                q.to_device_buffer(ctx),
                                k,
                                rebind[k_t](v),
                                num_rows_q,
                                mask_functor,
                                score_mod_functor,
                                valid_length.value().to_device_buffer(ctx),
                                StaticInt[1](),
                                max_cache_valid_length,
                                scale,
                                kv_input_row_offsets,
                                batch_size,
                                SplitKPartition(
                                    exp_sum_qk_max_data.unsafe_ptr(),
                                    UInt32(num_partitions_value),
                                ),
                                ctx,
                                sink_weights,
                            )
                        else:
                            mha_sm100_1q_dispatch[
                                config=config,
                                group = Int(group),
                                use_score_mod=use_score_mod,
                                ragged=ragged,
                                sink=sink,
                                _is_cache_length_accurate=_is_cache_length_accurate,
                            ](
                                output_intermediate.to_device_buffer(ctx),
                                q.to_device_buffer(ctx),
                                k,
                                rebind[k_t](v),
                                num_rows_q,
                                mask_functor,
                                score_mod_functor,
                                valid_length.value().to_device_buffer(ctx),
                                StaticInt[1](),
                                max_cache_valid_length,
                                scale,
                                kv_input_row_offsets,
                                batch_size,
                                SplitKPartition(
                                    exp_sum_qk_max_data.unsafe_ptr(),
                                    UInt32(num_partitions_value),
                                ),
                                ctx,
                                sink_weights,
                            )
                    else:
                        # For split-k, instantiate kernel with intermediate dtype
                        comptime kernel_splitk = mha_decoding[
                            q.dtype,
                            k_t,
                            v_t,
                            intermediate_dtype,
                            mask_t,
                            score_mod_t,
                            type_of(valid_length.value()).layout,
                            BM=BM,
                            BN=BN,
                            BK = UInt(BK),
                            WM=WM,
                            WN=WN,
                            depth=depth,
                            num_heads=num_heads,
                            num_threads = UInt(num_threads),
                            num_pipeline_stages = UInt(num_pipeline_stages),
                            group=group,
                            use_score_mod=use_score_mod,
                            ragged=ragged,
                            is_shared_kv=is_shared_kv,
                            sink=sink,
                            _use_valid_length=_use_valid_length,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            decoding_warp_split_k=decoding_warp_split_k,
                        ]

                        ctx.enqueue_function[kernel_splitk, kernel_splitk](
                            q_device,
                            k,
                            v,
                            output_intermediate_data,
                            exp_sum_device,
                            qk_max_device,
                            scale,
                            batch_size,
                            num_partitions_value,
                            max_cache_valid_length,
                            valid_length.value(),
                            sink_weights,
                            mask_functor,
                            score_mod_functor,
                            grid_dim=(
                                num_partitions_value,
                                Int(num_blocks_y),
                                batch_size,
                            ),
                            block_dim=(num_threads, 1, 1),
                            shared_mem_bytes=shared_mem_bytes if has_nvidia_gpu_accelerator() else 0,
                            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                                UInt32(
                                    ctx.default_device_info.shared_memory_per_multiprocessor
                                    - 4096 if has_nvidia_gpu_accelerator() else 0
                                )
                            ),
                        )

                    comptime kernel_reduce = mha_splitk_reduce[
                        intermediate_dtype,
                        output.dtype,
                        depth=depth,
                        num_heads=num_heads,
                        num_threads = UInt(WARP_SIZE),
                        group=group,
                        use_exp2=use_fa3_kernel,
                    ]

                    ctx.enqueue_function[kernel_reduce, kernel_reduce](
                        output_intermediate_data,
                        output_device,
                        exp_sum_device,
                        qk_max_device,
                        batch_size,
                        num_partitions_value,
                        grid_dim=(
                            1,
                            Int(num_heads),
                            batch_size,
                        ),
                        block_dim=(WARP_SIZE, 1, 1),
                    )
                    _ = exp_sum_qk_max_data^
                    _ = output_intermediate_data^
        # Not supported by contexting and decoding, e.g cross-attention or depth != 128
        else:
            # Assumes BSHD.
            mha_gpu_naive[
                ragged=ragged,
                sink=sink,
                _use_valid_length=_use_valid_length,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q,
                k,
                v,
                mask_functor,
                output,
                valid_length.value(),
                scale,
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                Int(num_heads),
                Int(depth),
                Int(group),
                ctx,
                sink_weights,
            )

    # Not supported by fast flash attention kernel.
    else:
        # Assumes BSHD.
        mha_gpu_naive[
            ragged=ragged,
            _use_valid_length=_use_valid_length,
            _is_cache_length_accurate=_is_cache_length_accurate,
            sink=sink,
        ](
            q,
            k,
            v,
            mask_functor,
            output,
            valid_length.value(),
            scale,
            batch_size,
            max_prompt_len,
            max_cache_valid_length,
            Int(num_heads),
            Int(depth),
            Int(group),
            ctx,
            sink_weights,
        )


fn flash_attention[
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    q_layout: Layout,
    //,
    use_score_mod: Bool = False,
    config: MHAConfig[dtype] = {
        UInt(Int(q_layout.shape[2])),
        UInt(Int(q_layout.shape[3])),
    },
    decoding_warp_split_k: Bool = False,
    _use_valid_length: Bool = False,
    _padded_ndbuffer: Bool = False,
    naive_kernel: Bool = False,
    sink: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, ...],
    q: LayoutTensor[
        mut=False, dtype, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[mut=False, address_space = AddressSpace.GENERIC, ...],
    v: LayoutTensor[mut=False, address_space = AddressSpace.GENERIC, ...],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: Optional[Int] = None,
    valid_length: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    # See the kV cache overloads for comments.

    comptime assert q.rank == 4, "only support rank 4 inputs."

    # Runtime dimensions.
    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()

    # Whether head and depth are static. With BSHD, B and S are dynamic.
    # H and D are always known.
    # fmt: off
    comptime head_depth_known = q.layout.shape.all_known[2, 4]() and k.layout.shape[2] != UNKNOWN_VALUE
    comptime depth = Int(q.layout.shape[q.rank-1])
    comptime gpu_info = ctx.default_device_info
    comptime head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
    comptime flash_attention_applicable = flash_attention_hw_supported[dtype]() and head_depth_known and head_depth_supported and not naive_kernel

    comptime q_half_float = q.dtype in (DType.float16, DType.bfloat16)
    comptime kv_num_heads = Int(k.layout.shape[2])
    # fmt: on

    var is_token_generation = seq_len == 1 and num_keys > seq_len

    var k_operand = LayoutTensorMHAOperand(
        LayoutTensor[k.dtype, Layout.row_major(k.layout.shape), MutAnyOrigin](
            k.ptr,
            RuntimeLayout[Layout.row_major(k.layout.shape)].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        )
    )
    var v_operand = LayoutTensorMHAOperand(
        LayoutTensor[v.dtype, Layout.row_major(v.layout.shape), MutAnyOrigin](
            v.ptr,
            RuntimeLayout[Layout.row_major(v.layout.shape)].row_major(
                v.runtime_layout.shape.value.canonicalize()
            ),
        )
    )

    flash_attention_dispatch[
        kv_num_heads=kv_num_heads,
        use_score_mod=use_score_mod,
        config=config,
        ragged=False,
        sink=sink,
        _is_flash_attention_applicable=flash_attention_applicable,
        _is_cache_length_accurate=True,
        _use_valid_length=_use_valid_length,
        _padded_ndbuffer=_padded_ndbuffer,
        decoding_warp_split_k=decoding_warp_split_k,
    ](
        output,
        q,
        k_operand,
        v_operand,
        mask_functor,
        score_mod_functor,
        q.dim[1](),
        num_keys,
        scale,
        is_token_generation,
        ctx,
        valid_length,
        None,
        num_partitions,
        sink_weights,
    )


fn flash_attention_ragged[
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_layout: Layout,
    //,
    use_score_mod: Bool = False,
    config: MHAConfig[type] = {
        UInt(Int(q_layout.shape[q_layout.rank() - 2])),  # num_heads
        UInt(Int(q_layout.shape[q_layout.rank() - 1])),  # head_dim
    },
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, ...],
    q: LayoutTensor[
        mut=False, type, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[mut=False, address_space = AddressSpace.GENERIC, ...],
    v: LayoutTensor[mut=False, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    max_prompt_len: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: Optional[Int] = None,
) raises:
    # See the kV cache overloads for comments.

    comptime assert q.rank == 3, "only support rank 3 inputs for ragged inputs."
    comptime assert (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), "Q, K, V, output should have same type."

    comptime assert (
        q.dtype == DType.float32 or q.dtype.is_half_float()
    ), "Only support single and half precision."

    # Runtime dimensions.
    # For ragged inputs: [total_seq_len, num_heads, head_dim]
    # fmt: off
    comptime head_depth_known = q.layout.shape.all_known[1, 3]() and k.layout.shape[1] != UNKNOWN_VALUE
    comptime depth = Int(q.layout.shape[q.rank - 1])
    comptime gpu_info = ctx.default_device_info
    comptime head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
    comptime flash_attention_applicable = flash_attention_hw_supported[type]() and head_depth_known and head_depth_supported and not naive_kernel
    comptime kv_num_heads = Int(k.layout.shape[1])
    # fmt: on

    var is_token_generation = False

    var cache_row_offsets = input_row_offsets.as_any_origin()

    var k_operand = RaggedMHAOperand(
        LayoutTensor[k.dtype, Layout.row_major(k.layout.shape), MutAnyOrigin](
            k.ptr,
            RuntimeLayout[Layout.row_major(k.layout.shape)].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        ),
        cache_row_offsets,
    )
    var v_operand = RaggedMHAOperand(
        LayoutTensor[v.dtype, Layout.row_major(v.layout.shape), MutAnyOrigin](
            v.ptr,
            RuntimeLayout[Layout.row_major(v.layout.shape)].row_major(
                v.runtime_layout.shape.value.canonicalize()
            ),
        ),
        cache_row_offsets,
    )
    flash_attention_dispatch[
        kv_num_heads=kv_num_heads,
        use_score_mod=use_score_mod,
        config=config,
        ragged=True,
        _is_flash_attention_applicable=flash_attention_applicable,
        _is_cache_length_accurate=True,
        decoding_warp_split_k=decoding_warp_split_k,
    ](
        output,
        q,
        k_operand,
        v_operand,
        mask_functor,
        score_mod_functor,
        Int(max_prompt_len[0]),
        Int(max_prompt_len[0]),
        scale,
        is_token_generation,
        ctx,
        OptionalReg[
            LayoutTensor[
                DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ]
        ](input_row_offsets),
        None,
        num_partitions,
    )


# ===-----------------------------------------------------------------------===#
# Flash attention for context encoding
# ===-----------------------------------------------------------------------===#


# for depth = 128 we want waves_per_eu = 2 and for depth = 256 we want waves_per_eu = 1
# for depth = 64 we want waves_per_eu = 2
# this heuristic may not be valid for other depths
@__llvm_metadata(`rocdl.waves_per_eu`=min(256 // Int(config.depth), 2))
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
fn mha[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    valid_length_layout: Layout,
    config: MHAConfig,
    group: Int = 1,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    is_shared_kv: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    _padded_ndbuffer: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    batch_size: Int,
    seq_len_arg: Int,
    num_keys_arg: Int,
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ],
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
):
    comptime depth = config.depth
    comptime num_heads = config.num_heads
    var batch_idx = block_idx.z

    # mha inputs
    var seq_len: Int
    var max_seq_len = seq_len_arg
    var num_keys: Int
    var mask_tensor_col = num_keys_arg
    var start_pos: UInt32 = 0

    @always_inline
    fn q_block_idx() -> UInt:
        return block_idx.x if is_nvidia_gpu() else block_idx.y

    @parameter
    if ragged:
        # treat valid_lengths as a input_row_offsets
        start_of_seq = Int(valid_length[batch_idx])
        end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq

        if seq_len < Int(q_block_idx() * config.block_m()):
            return

        @parameter
        if not _is_cache_length_accurate:
            start_pos = UInt32(k.cache_length(Int(batch_idx)))

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        if kv_input_row_offsets:
            var kv_row_offsets = kv_input_row_offsets.value()
            kv_seq_start = Int(kv_row_offsets[batch_idx])
            kv_seq_end = Int(kv_row_offsets[batch_idx + 1])
            cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = cur_kv_len + Int(start_pos)
        else:
            num_keys = seq_len + Int(start_pos)

        q_batch_offset = start_of_seq * Int(config.depth * config.num_heads)

    # KVCache inputs, prompt lengths are all padded to the max in batch.
    elif _use_valid_length and not _padded_ndbuffer:
        # treat valid_lengths as valid lengths
        seq_len = Int(valid_length[batch_idx])

        if seq_len < Int(q_block_idx() * config.block_m()):
            return

        @parameter
        if not _is_cache_length_accurate:
            var cache_length = k.cache_length(Int(batch_idx))
            start_pos = UInt32(cache_length)

        num_keys = seq_len + k.cache_length(Int(batch_idx))
        q_batch_offset = Int(
            config.depth * config.num_heads * UInt(max_seq_len) * batch_idx
        )
    # NDBuffer inputs, homogeneous and padded batching.
    else:

        @parameter
        if _padded_ndbuffer:
            seq_len = Int(valid_length[batch_idx])
            num_keys = seq_len
        else:
            seq_len = seq_len_arg
            num_keys = num_keys_arg

        if seq_len < Int(q_block_idx() * config.block_m()):
            return
        q_batch_offset = Int(
            config.depth * config.num_heads * UInt(max_seq_len) * batch_idx
        )

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = UInt32(num_keys - seq_len)

    @parameter
    if is_nvidia_gpu():

        @parameter
        if is_shared_kv:
            mha_single_batch_pipelined[
                config=config,
                group=group,
                use_score_mod=use_score_mod,
                sink=sink,
            ](
                q_ptr + q_batch_offset,
                k,
                v,
                output_ptr + q_batch_offset,
                scale,
                seq_len,
                max_seq_len,
                start_pos,
                num_keys,
                mask_tensor_col,
                mask,
                score_mod,
                Int(batch_idx),
                sink_weights,
            )
        else:
            mha_single_batch[
                config=config,
                group=group,
                use_score_mod=use_score_mod,
                sink=sink,
            ](
                q_ptr + q_batch_offset,
                k,
                v,
                output_ptr + q_batch_offset,
                scale,
                seq_len,
                max_seq_len,
                start_pos,
                num_keys,
                mask_tensor_col,
                mask,
                score_mod,
                Int(batch_idx),
                sink_weights,
            )
    elif is_amd_gpu():
        comptime assert (
            use_score_mod == False
        ), "use_score_mod must be False for AMD flash attention"

        comptime attention_config = MHAAttentionConfig[False, config, group]()
        var attention = Attention[config, group, False, sink](
            attention_config,
            output_ptr + q_batch_offset,
            q_ptr + q_batch_offset,
            k,
            v,
            mask,
            sink_weights,
            Int(batch_idx),
            scale,
            seq_len,
            num_keys,
            Int(start_pos),
        )

        @parameter
        if attention_config.USE_EXPERIMENTAL_CDNA4_MHA_KERNEL:
            attention.mha_prefill_experimental()
        else:
            attention.mha_prefill()
    else:
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
fn mha_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    *,
    config: MHAConfig,
    group: Int = 1,
    use_score_mod: Bool = False,
    sink: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    num_keys: Int,
    mask_tensor_col: Int,  # second dimension of mask tensor
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    comptime accum_type = get_accum_type[q_type]()
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime assert q_type == k_type and k_type == v_type

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()
    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()
    comptime num_heads = config.num_heads
    comptime depth = config.depth

    comptime assert num_warps_m * num_warps_n == (
        num_threads // UInt(WARP_SIZE)
    ), "Number of warps doesn't match warp tile sizes."

    var tid = UInt32(thread_idx.x)
    var warp_id: UInt32 = warp.broadcast(tid // UInt32(WARP_SIZE))
    var lane = UInt32(lane_id())

    # Coordinates of the current warp.
    var warp_y = warp_id // UInt32(num_warps_n)
    var warp_x = warp_id % UInt32(num_warps_n)

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(Int(BM), Int(BK)),
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(Int(BM), Int(BK)),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    comptime k_smem_size = config.k_smem_size()
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(Int(BN), Int(BK)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    comptime v_smem_size = config.v_smem_size()
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(Int(BK), Int(BN)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    var head_idx = UInt32(block_idx.y)
    var q_tile_idx = UInt32(block_idx.x)

    # Query global memory iterator
    comptime q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var q_tile_num_rows = min(
        UInt32(BM), UInt32(seq_len) - q_tile_idx * UInt32(BM)
    )
    var q_offset = UInt32(depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[q_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), Int(depth)
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                Int(num_heads * depth), 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[Int(BM), Int(BK), axis=1](
        0, 0
    )
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    comptime mma_shape = get_mma_shape[q_type, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime WM = config.WM
    comptime WN = config.WN
    comptime num_m_mmas = WM // UInt(MMA_M)
    comptime num_n_mmas = WN // UInt(MMA_N)

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0)
    )

    # Rowwise max and sum for online softmax
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[
        Int(WM), accum_type, alignment=row_alignment
    ]()
    var rowsum = stack_allocation[
        Int(WM), accum_type, alignment=row_alignment
    ]()

    @parameter
    for i in range(0, Int(WM), 2):

        @parameter
        if sink:
            debug_assert(
                Bool(sink_weights),
                "expect sink_weights to be non-null when sink=true",
            )
            var sink_logit_log2 = (
                sink_weights.value()[Int(head_idx)][0].cast[accum_type]()
                * log2e
            )
            rowmax.store(
                i,
                SIMD[accum_type, 2](sink_logit_log2),
            )
            # exp(sink_val-sink_val) = exp(0) = 1
            rowsum.store(i, SIMD[accum_type, 2](1))
        else:
            rowmax.store(i, SIMD[accum_type, 2](min_or_neg_inf[accum_type]()))
            rowsum.store(i, SIMD[accum_type, 2](0))

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(Int(BM), Int(BK)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * Int(num_warps_n), Int(BM)),
        address_space = AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # Mask global memory iterator.
    var mask_block_row = q_tile_idx * UInt32(BM)
    var mask_warp_row = warp_y * UInt32(WM)
    var mask_warp_col = warp_x * UInt32(WN)

    # Account for group query.
    comptime kv_num_heads = num_heads // UInt(group)

    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime q_num_vecs = BM * BK // UInt(simd_size)

    comptime async_copy_q_layout = Layout.row_major(
        Int(min(num_threads, q_num_vecs) * UInt(simd_size) // BK),
        Int(BK // UInt(simd_size)),
    )

    @parameter
    for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.layout_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads = Int(num_threads),
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        # we `async_copy_commit_group()` and after we finish copying `k`.

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
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        if (
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.FULL_MASK
        ):
            mask_warp_col += UInt32(BN)
            return

        comptime kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )
        var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[kv_gmem_layout](
            {kv_tile_num_rows, Int(depth)},
            {Int(kv_num_heads * depth), 1},
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k.block_paged_ptr[Int(BN)](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx // UInt32(group))),
                0,
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[Int(BN), Int(BK), axis=1](
            0, 0
        )

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            v.block_paged_ptr[Int(BN)](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx // UInt32(group))),
                0,
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[Int(BK), Int(BN), axis=0](
            0, 0
        )

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        @always_inline
        @parameter
        fn _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: type_of(tensor)
        ):
            return {
                tensor.ptr,
                type_of(tensor.runtime_layout)(
                    type_of(tensor.runtime_layout.shape)(
                        num_rows, tensor.dim[1]()
                    ),
                    tensor.runtime_layout.stride,
                ),
            }

        comptime kv_num_vecs = BN * BK // UInt(simd_size)
        comptime async_copy_k_layout = Layout.row_major(
            Int(
                min(num_threads, kv_num_vecs)
                * UInt(simd_size)
                // UInt(k_smem_iter.layout.stride[0].value())
            ),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        @parameter
        for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.layout_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads = Int(num_threads),
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter[].vectorize[1, simd_size](),
            )

            k_gmem_iter._incr()

        async_copy_commit_group()
        # synchronize here since we can overlap q tile and first k tile copy
        async_copy_wait_all()
        barrier()

        multistage_mma[
            Int(BM),
            Int(BN),
            Int(BK),
            Int(WM),
            Int(WN),
            Int(num_threads),
            Int(num_pipeline_stages),
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters = Int(depth // BK),
            k_group_size = config.k_group_size,
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            Int(depth // BK),
        )

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        @parameter
        fn _apply_mask[masked: Bool]():
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[accum_type]() if use_score_mod
                or mask_t.apply_log2e_after_mask else scale.cast[accum_type]()
                * log2e
            )

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + UInt32(
                        m_mma * UInt(MMA_M)
                    )
                    var mask_frag_col = mask_warp_col + UInt32(
                        n_mma * UInt(MMA_N)
                    )

                    # Offset to current thread's fragment
                    mask_frag_row += lane // UInt32(MMA_N // p_frag_simdwidth)
                    mask_frag_col += (
                        lane * UInt32(p_frag_simdwidth) % UInt32(MMA_N)
                    )

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row
                            + mask_frag_row
                            + UInt32(i * MMA_M // 2)
                        )
                        var score_col = mask_frag_col

                        score_row_with_start_pos = score_row + start_pos

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32](
                                    Int(block_idx.z),
                                    Int(block_idx.y),
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )
                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        @parameter
                        if use_score_mod:
                            p_reg_vec2[mma_id, i] = (
                                score_mod.score_mod(
                                    IndexList[4, element_type = DType.uint32](
                                        Int(block_idx.z),
                                        Int(block_idx.y),
                                        Int(score_row_with_start_pos),
                                        Int(score_col),
                                    ),
                                    p_reg_vec2[mma_id, i],
                                    max_seq_len,
                                )
                                * log2e
                            )
                        elif mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[2, element_type = DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                IndexList[2, element_type = DType.uint32](
                                    seq_len,
                                    num_keys,
                                ),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch[_apply_mask](
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += UInt32(BN)

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * Int(num_m_mmas * num_n_mmas), 2
        )
        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * Int(num_m_mmas), Int(num_n_mmas)),
            # threads layout by warp
            Layout.row_major(Int(num_warps_m), Int(num_warps_n)),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[2 * Int(num_warps_n), Int(WM)](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        comptime async_copy_v_layout = Layout.row_major(
            Int(
                min(num_threads, kv_num_vecs)
                * UInt(simd_size)
                // UInt(v_smem_iter.layout.stride[0].value())
            ),
            v_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load V tile into smem
        @parameter
        for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(
                v_smem_iter.layout_uint_type(v_id)
            )[]

            @parameter
            if not not_last_iter:
                var num_rows_bound = min(
                    Int(BK), end - (kv_tile_start_row + Int(v_id * BK))
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle = v_smem_tile.dtype.is_half_float(),
                num_threads = Int(num_threads),
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_tensor.vectorize[1, simd_size](),
            )

            v_gmem_iter._incr()

        async_copy_commit_group()

        @parameter
        if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM,
                BN,
                BK,
                WM,
                WN,
                UInt(MMA_M),
                UInt(MMA_N),
                UInt(p_frag_simdwidth),
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters = Int(BN // BK),
                k_group_size = config.k_group_size,
            ](
                output_reg_tile,
                p_smem_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                Int(BN // BK),
            )

        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * Int(num_m_mmas), p_frag_size
            ](0, 0)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                False,  # transpose_b
                swizzle_a=False,
                prefetch_init=False,
                static_num_iters = Int(BN // BK),
                k_group_size = config.k_group_size,
            ](
                output_reg_tile,
                p_reg_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                Int(BN // BK),
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList(Int(BN))](0, num_keys)

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * Int(m_mma)])
        var rowsum_inv1 = recip(rowsum[2 * Int(m_mma) + 1])

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    comptime output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        output_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[output_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), Int(depth)
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
                Int(num_heads * depth), 1
            ),
        ),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[Int(WM), Int(WN)](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size = Int(WN), access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(Int(BM), Int(depth)),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[Int(WM), Int(WN)](
            Int(warp_y), Int(warp_x)
        )
        copy_local_to_shared[
            thread_layout = Layout.row_major(8, 4), swizzle=swizzle
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
            thread_layout = Layout.row_major(
                Int(num_threads * UInt(simd_size) // depth),
                Int(depth // UInt(simd_size)),
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )
    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
fn mha_single_batch_pipelined[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    *,
    config: MHAConfig,
    group: Int = 1,
    use_score_mod: Bool = False,
    sink: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    num_keys: Int,
    mask_tensor_col: Int,  # second dimension of mask tensor
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    comptime accum_type = get_accum_type[q_type]()
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime assert q_type == k_type and k_type == v_type

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()
    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()
    comptime num_heads = config.num_heads
    comptime depth = config.depth

    comptime assert num_warps_m * num_warps_n == (
        num_threads // UInt(WARP_SIZE)
    ), "Number of warps doesn't match warp tile sizes."

    var tid = UInt32(thread_idx.x)
    var warp_id: UInt32 = warp.broadcast(tid // UInt32(WARP_SIZE))
    var lane = UInt32(lane_id())

    # Coordinates of the current warp.
    var warp_y = warp_id // UInt32(num_warps_n)
    var warp_x = warp_id % UInt32(num_warps_n)

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(Int(BM), Int(BK)),
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(Int(BM), Int(BK)),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    comptime k_smem_size = config.kv_smem_size()
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(Int(BN), Int(BK)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    var head_idx = UInt32(block_idx.y)
    var q_tile_idx = UInt32(block_idx.x)

    # Query global memory iterator
    comptime q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var q_tile_num_rows = min(
        UInt32(BM), UInt32(seq_len) - q_tile_idx * UInt32(BM)
    )
    var q_offset = UInt32(depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[q_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), Int(depth)
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                Int(num_heads * depth), 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[Int(BM), Int(BK), axis=1](
        0, 0
    )
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    comptime mma_shape = get_mma_shape[q_type, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime WM = config.WM
    comptime WN = config.WN
    comptime num_m_mmas = WM // UInt(MMA_M)
    comptime num_n_mmas = WN // UInt(MMA_N)

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0)
    )

    # Rowwise max and sum for online softmax
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[
        Int(WM), accum_type, alignment=row_alignment
    ]()
    var rowsum = stack_allocation[
        Int(WM), accum_type, alignment=row_alignment
    ]()

    @parameter
    for i in range(0, Int(WM), p_frag_simdwidth):

        @parameter
        if sink:
            debug_assert(
                Bool(sink_weights),
                "expect sink_weights to be non-null when sink=true",
            )
            var sink_logit_log2 = (
                sink_weights.value()[Int(head_idx)][0].cast[accum_type]()
                * log2e
            )
            rowmax.store(
                i,
                SIMD[accum_type, p_frag_simdwidth](sink_logit_log2),
            )
            # exp(sink_val-sink_val) = exp(0) = 1
            rowsum.store(i, SIMD[accum_type, p_frag_simdwidth](1))
        else:
            rowmax.store(
                i,
                SIMD[accum_type, p_frag_simdwidth](
                    min_or_neg_inf[accum_type]()
                ),
            )
            rowsum.store(i, SIMD[accum_type, p_frag_simdwidth](0))

    # Shared memory for P = Q * K^t
    # Only use BN/BK tiles. Setting circular so that the prefetch in matmul
    # doesn't go OOB at the last tile.
    var p_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(Int(BM), Int(BK)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(p_frag_simdwidth * Int(num_warps_n), Int(BM)),
        address_space = AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # Mask global memory iterator.
    var mask_block_row = q_tile_idx * UInt32(BM)
    var mask_warp_row = warp_y * UInt32(WM)
    var mask_warp_col = warp_x * UInt32(WN)

    # Account for group query.
    comptime kv_num_heads = num_heads // UInt(group)

    comptime num_pipeline_stages = config.num_pipeline_stages
    var is_first_iter = True

    # Iterate over KV, equivalent to the following with if hoisted out.
    #   ```
    #   for i in range(start, end, tile_size):
    #     if i + tile_size >= end:
    #       loop_over_kvcache[tile_size, False]
    #     else:
    #       loop_over_kvcache[tile_size, True]
    #   ```
    # Only the last iteration is doing boundary check.
    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        if (
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.FULL_MASK
        ):
            mask_warp_col += UInt32(BN)
            return

        comptime kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )
        var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type = DType.int32](
                kv_tile_num_rows, Int(depth)
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type = DType.int32](
                Int(kv_num_heads * depth), 1
            ),
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k.block_paged_ptr[Int(BN)](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx // UInt32(group))),
                0,
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[Int(BN), Int(BK), axis=1](
            0, 0
        )

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            v.block_paged_ptr[Int(BN)](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx // UInt32(group))),
                0,
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[Int(BK), Int(BN), axis=0](
            0, 0
        )

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        var num_b_rows = Optional[Int]() if not_last_iter else Optional[Int](
            kv_tile_num_rows
        )

        # First iteration load q from global memory to shared memory.
        if is_first_iter:
            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                True,  # transpose_b
                swizzle_a=True,
                continue_prefetch_b=True,
                b_next_smem_layout = Layout.row_major(Int(BK), Int(BN)),
                next_op_b_iter_masked = type_of(v_gmem_iter).masked,
                next_op_b_layout_int_type = type_of(
                    v_gmem_iter
                ).layout_int_type,
                next_op_b_linear_idx_type = type_of(
                    v_gmem_iter
                ).linear_idx_type,
                k_group_size = config.k_group_size,
            ](
                p_reg_tile,
                q_gmem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                Int(depth // BK),
                next_op_b_iter=v_gmem_iter.bitcast[k_type](),
                num_b_rows=num_b_rows,
            )
            is_first_iter = False
        # Subsequent iterations just use q in share memory.
        # TODO: Figure out a better function interface instead of passing in
        # shared memory iterator twice.
        else:
            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                True,  # transpose_b
                swizzle_a=True,
                continue_prefetch_b=True,
                b_next_smem_layout = Layout.row_major(Int(BK), Int(BN)),
                next_op_b_iter_masked = type_of(v_gmem_iter).masked,
                next_op_b_layout_int_type = type_of(
                    v_gmem_iter
                ).layout_int_type,
                next_op_b_linear_idx_type = type_of(
                    v_gmem_iter
                ).linear_idx_type,
                k_group_size = config.k_group_size,
            ](
                p_reg_tile,
                # Pass shared memory iterator to hint not loading from global memory.
                q_smem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                Int(depth // BK),
                next_op_b_iter=v_gmem_iter.bitcast[k_type](),
                num_b_rows=num_b_rows,
            )

        # Increment V iterator since it's prefetched inside 1st matmul.
        v_gmem_iter += num_pipeline_stages - 1

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        @parameter
        fn _apply_mask[masked: Bool]():
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[accum_type]() if use_score_mod
                or mask_t.apply_log2e_after_mask else scale.cast[accum_type]()
                * log2e
            )

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + UInt32(m_mma) * UInt32(
                        MMA_M
                    )

                    var mask_frag_col = mask_warp_col + UInt32(n_mma) * UInt32(
                        MMA_N
                    )

                    mask_frag_row += lane // UInt32(MMA_N // p_frag_simdwidth)
                    mask_frag_col += (
                        lane * UInt32(p_frag_simdwidth) % UInt32(MMA_N)
                    )

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row
                            + mask_frag_row
                            + UInt32(i * MMA_M // 2)
                        )
                        var score_col = mask_frag_col

                        var score_row_with_start_pos = score_row + start_pos

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32](
                                    Int(block_idx.z),
                                    Int(block_idx.y),
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )

                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        @parameter
                        if use_score_mod:
                            p_reg_vec2[mma_id, i] = (
                                score_mod.score_mod(
                                    IndexList[4, element_type = DType.uint32](
                                        Int(block_idx.z),
                                        Int(block_idx.y),
                                        Int(score_row_with_start_pos),
                                        Int(score_col),
                                    ),
                                    p_reg_vec2[mma_id, i],
                                    max_seq_len,
                                )
                                * log2e
                            )
                        elif mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[
                                    2,
                                    element_type = DType.uint32,
                                ](Int(score_row), Int(score_col)),
                                IndexList[
                                    2,
                                    element_type = DType.uint32,
                                ](seq_len, num_keys),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch[_apply_mask](
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += UInt32(BN)

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * Int(num_m_mmas * num_n_mmas), 2
        )

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * Int(num_m_mmas), Int(num_n_mmas)),
            # threads layout by warp
            Layout.row_major(Int(num_warps_m), Int(num_warps_n)),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, p_frag_simdwidth
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, p_frag_simdwidth
            ](),
            warp_scratch.tile[2 * Int(num_warps_n), Int(WM)](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        # V reuse K's smem iterator. They has same smem footage expect for different layouts.
        var v_smem_iter = k_smem_iter.reshape[
            Layout.row_major(Int(BK), Int(BN))
        ]().bitcast[v_type]()

        @parameter
        if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM,
                BN,
                BK,
                WM,
                WN,
                UInt(MMA_M),
                UInt(MMA_N),
                UInt(p_frag_simdwidth),
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)
            barrier()

            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                k_group_size = config.k_group_size,
            ](
                output_reg_tile,
                p_smem_iter,
                v_gmem_iter,
                p_smem_iter,
                v_smem_iter,
                Int(BN // BK),
                num_b_rows=num_b_rows,
            )
        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * Int(num_m_mmas), p_frag_size
            ](0, 0)

            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                False,  # transpose_b
                swizzle_a=True,
                static_num_iters = Int(BN // BK),
                prefetch_init=False,
                k_group_size = config.k_group_size,
            ](
                output_reg_tile,
                p_reg_iter,
                v_gmem_iter,
                p_smem_iter,
                v_smem_iter,
                Int(BN // BK),
                num_b_rows=num_b_rows,
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList(Int(BN))](0, num_keys)

    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * Int(m_mma)])
        var rowsum_inv1 = recip(rowsum[2 * Int(m_mma) + 1])

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    comptime output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        output_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[output_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), Int(depth)
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
                Int(num_heads * depth), 1
            ),
        ),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[Int(WM), Int(WN)](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(Int(BM), Int(depth)),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[Int(WM), Int(WN)](
            Int(warp_y), Int(warp_x)
        )

        comptime swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size = Int(WN), access_size=MMA_N
        ]()
        copy_local_to_shared[
            thread_layout = Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )
        barrier()
        copy_sram_to_dram[
            thread_layout = Layout.row_major(
                Int(num_threads * UInt(simd_size) // depth),
                Int(depth // UInt(simd_size)),
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )

        # Guard writing to shared memory.

        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.

    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


# ===-----------------------------------------------------------------------===#
# Flash decoding for token generation
# ===-----------------------------------------------------------------------===#


# Entry point for mha_decoding with batch_size > 1.
@__llvm_metadata(`rocdl.waves_per_eu`=Int(4))
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn mha_decoding[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    valid_length_layout: Layout,
    BM: UInt,  # number of queries per block
    BN: UInt,  # number of keys per block
    BK: UInt,  # tile size in depth dimension
    WM: UInt,
    WN: UInt,
    depth: UInt,
    num_heads: UInt,
    num_threads: UInt,
    num_pipeline_stages: UInt,
    group: UInt = 1,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    is_shared_kv: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    batch_size: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],  # valid length per batch
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
):
    comptime accum_type = get_accum_type[q_type]()
    var batch_idx = block_idx.z

    # split-k offsets
    var partition_idx = block_idx.x
    var output_batch_offset = (
        depth * num_heads * batch_idx
        + depth * num_heads * UInt(batch_size) * partition_idx
    )
    var qk_max_offset = (
        num_heads * batch_idx + num_heads * UInt(batch_size) * partition_idx
    )
    var exp_sum_offset = qk_max_offset

    # split-k intermediate buffers
    var qk_max_batch_ptr = type_of(qk_max_ptr)()
    if qk_max_ptr:
        qk_max_batch_ptr = qk_max_ptr + qk_max_offset

    var exp_sum_batch_ptr = type_of(exp_sum_ptr)()
    if exp_sum_ptr:
        exp_sum_batch_ptr = exp_sum_ptr + exp_sum_offset

    var seq_len: Int
    var q_batch_offset: Int
    var start_pos: UInt32 = 0

    @parameter
    if ragged:
        # treat valid_lengths as a input_row_offsets
        start_of_seq = Int(valid_length[batch_idx])
        end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq
        q_batch_offset = start_of_seq * Int(depth * num_heads)
    elif _use_valid_length:
        # treat valid_lengths as valid lengths
        q_batch_offset = Int(depth * num_heads * batch_idx)
        seq_len = Int(valid_length[batch_idx])
    else:
        seq_len = 1
        q_batch_offset = Int(depth * num_heads * batch_idx)

    var num_keys = k.cache_length(Int(batch_idx))

    @parameter
    if not _is_cache_length_accurate:
        num_keys += seq_len

    @parameter
    if is_nvidia_gpu():

        @parameter
        if is_shared_kv:
            mha_decoding_single_batch_pipelined[
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                depth=depth,
                num_heads=num_heads,
                num_threads=num_threads,
                num_pipeline_stages=num_pipeline_stages,
                group=group,
                use_score_mod=use_score_mod,
                decoding_warp_split_k=decoding_warp_split_k,
                sink=sink,
            ](
                q_ptr + q_batch_offset,
                k,
                v,
                output_ptr + output_batch_offset,
                exp_sum_batch_ptr,
                qk_max_batch_ptr,
                scale,
                UInt(num_keys),
                UInt(num_partitions),
                UInt(max_cache_valid_length),
                sink_weights,
                mask,
                score_mod,
                Int(batch_idx),
            )
        else:
            mha_decoding_single_batch[
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                depth=depth,
                num_heads=num_heads,
                num_threads=num_threads,
                num_pipeline_stages=num_pipeline_stages,
                group=group,
                use_score_mod=use_score_mod,
                decoding_warp_split_k=decoding_warp_split_k,
                sink=sink,
            ](
                q_ptr + q_batch_offset,
                k,
                v,
                output_ptr + output_batch_offset,
                exp_sum_batch_ptr,
                qk_max_batch_ptr,
                scale,
                UInt(num_keys),
                UInt(num_partitions),
                UInt(max_cache_valid_length),
                mask,
                score_mod,
                Int(batch_idx),
                sink_weights,
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
        comptime assert (
            use_score_mod == False
        ), "use_score_mod must be False for AMD flash attention"
        var sink_weights_lt: OptionalReg[
            LayoutTensor[
                q_ptr.type.dtype,
                Layout.row_major(UNKNOWN_VALUE),
                ImmutAnyOrigin,
            ]
        ] = None
        if sink_weights:
            sink_weights_lt = LayoutTensor[
                q_ptr.type.dtype,
                Layout.row_major(UNKNOWN_VALUE),
                ImmutAnyOrigin,
            ](
                sink_weights.value().ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    IndexList[1](sink_weights.value().size())
                ),
            )

        comptime attention_config = MHAAttentionConfig[
            True, config, Int(group)
        ]()
        var attention = Attention[config, Int(group), True, sink](
            attention_config,
            output_ptr + output_batch_offset,
            q_ptr + q_batch_offset,
            k,
            v,
            mask,
            sink_weights_lt,
            Int(batch_idx),
            scale,
            1,
            num_keys,
            0,
        )
        attention.mha_decoding(
            exp_sum_batch_ptr,
            qk_max_batch_ptr,
            num_partitions,
        )
    else:
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@always_inline
fn scale_and_mask_helper[
    p_type: DType,
    p_layout: Layout,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    group: Int,
    num_n_mmas: Int,
    WN: Int,
    MMA_N: Int,
    simd_width: Int,
    use_score_mod: Bool = False,
](
    p_reg_tile: LayoutTensor[
        mut=True, p_type, p_layout, address_space = AddressSpace.LOCAL
    ],
    scale_log2e: Float32,
    num_keys: UInt,
    bound: UInt,
    lane: UInt,
    warp: UInt,
    mask: mask_t,
    score_mod: score_mod_t,
    kv_tile_start_row: Int,
    mask_stride: UInt,
    max_seq_len: Int,  # max_prompt_len + max_cache_len
):
    # Apply mask and scale to mma result. Only the first row (lane 0-3) has
    # meaningful data, other fragments are zero. The mask is an 1D vector.
    # The dimension of mask are assumed dynamic here so still using index calculation.
    # TODO: check if the explicit index calculation can be avoided.

    # For mma output, thread 0-3 are on the first row, 4-7 second row, etc.
    if lane >= UInt(4 * group):
        return
    var batch_cache_valid_length = num_keys - 1
    var warp_offset = warp * UInt(WN)

    # Number of groups updated by each thread. E.g. for group=16 and 16x8x16 mma,
    # Each thread updates 2 rows in mma output, mapped to 2 groups.
    # When group % 8 != 0, some work are OOB, e.g. updating 15-th row when there are
    # only 12 groups. Such results are ignored when output to global memory.
    comptime num_groups_per_thread = ceildiv(group, 8)

    @parameter
    for n_mma in range(num_n_mmas):
        # offset in fragment
        var frag_offset = n_mma * MMA_N
        # Current thread's offset mapped in num_keys dim
        var key_offset = Int(warp_offset) + frag_offset
        # Current thread's index in current mma tile, e.g. T1 and T5 are 1 in 16x8 mma output.
        var frag_lane_col = Int((lane % 4) * UInt(simd_width))

        @parameter
        for i_group in range(num_groups_per_thread):
            var group_idx = UInt(i_group) * 8 + lane // 4
            var q_head_idx = block_idx.y * UInt(group) + group_idx

            @parameter
            for i in range(simd_width):
                var score_row = batch_cache_valid_length
                var score_col = (
                    kv_tile_start_row + key_offset + frag_lane_col + i
                )

                p_reg_tile[n_mma, i + i_group * simd_width] = mask.mask(
                    Index(
                        Int(block_idx.z),
                        Int(q_head_idx),
                        Int(score_row),
                        score_col,
                    ),
                    p_reg_tile[n_mma, i + i_group * simd_width]
                    * scale_log2e.cast[p_type](),
                )

                @parameter
                if use_score_mod:
                    p_reg_tile[n_mma, i + i_group * simd_width] = (
                        score_mod.score_mod(
                            Index(
                                Int(block_idx.z),
                                Int(q_head_idx),
                                Int(score_row),
                                score_col,
                            ),
                            p_reg_tile[n_mma, i + i_group * simd_width],
                            max_seq_len,
                        )
                        * log2e
                    )
                elif mask_t.apply_log2e_after_mask:
                    p_reg_tile[n_mma, i + i_group * simd_width] = (
                        p_reg_tile[n_mma, i + i_group * simd_width] * log2e
                    )

                p_reg_tile[n_mma, i + i_group * simd_width] = _kernel_mask(
                    Index(score_row, score_col),
                    Index(
                        batch_cache_valid_length + 1,
                        # The following setting ensures that out of bound check happens at
                        # every function call, also it corrects the bounds to be exact.
                        # Previous version was using batch_cache_valid_length + 1 which was fine
                        # with the non-split-k based mha as the ooo would have been triggered only
                        # for the last iteration of the outer loop. So while the bound was not exact, it
                        # led to correct output.
                        kv_tile_start_row + Int(bound),
                    ),
                    p_reg_tile[n_mma, i + i_group * simd_width],
                )


fn mha_decoding_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    *,
    BM: UInt,  # number of queries per block
    BN: UInt,  # number of keys per block
    BK: UInt,  # tile size in depth dimension
    WM: UInt,
    WN: UInt,
    depth: UInt,
    num_heads: UInt,
    num_threads: UInt,
    num_pipeline_stages: UInt,
    group: UInt = 1,
    use_score_mod: Bool = False,
    decoding_warp_split_k: Bool = False,
    sink: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    num_keys: UInt,
    num_partitions: UInt,
    max_cache_valid_length: UInt,  # longest KV cache entry
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
):
    """Flash attention v2 algorithm."""
    comptime accum_type = get_accum_type[q_type]()
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime assert q_type == k_type and k_type == v_type

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN

    comptime assert num_warps_m * num_warps_n == (
        num_threads // UInt(WARP_SIZE)
    ), "Number of warps doesn't match warp tile sizes."

    # It's because in online-softmax we only use the top 8x4 sub-matrix
    # in the 16x8 mma output for Nvidia GPU. It shouldn't matter for AMD
    comptime assert group <= 16, String(
        "Only support GQA with group <= 16 for Nvidia, but got a group = '",
        group,
        "'.",
    )

    var tid = thread_idx.x
    var warp_id = warp.broadcast(tid // UInt(WARP_SIZE))
    var lane = lane_id()

    # Coordinates of the current warp.
    var warp_y, warp_x = divmod(warp_id, num_warps_n)

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(Int(BM), Int(BK)),
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(Int(BM), Int(BK)),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )

    comptime k_smem_size = BN * depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(Int(BN), Int(BK)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    comptime v_smem_size = BN * BN
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(Int(BK), Int(BN)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    var kv_head_idx = block_idx.y
    var q_head_idx = kv_head_idx * group + thread_idx.x // 4
    var partition_idx = block_idx.x

    comptime mma_shape = get_mma_shape[q_type, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = WM // UInt(MMA_M)
    comptime num_n_mmas = WN // UInt(MMA_N)

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    # Note that
    # num_warps_n * num_n_mmas == BN // WN * num_n_mmas
    # so we can use multistage_mma
    comptime num_output_rows = num_m_mmas * num_n_mmas
    comptime num_output_rows_full = num_warps_n * num_output_rows if decoding_warp_split_k else num_output_rows
    # alias num_output_rows = num_warps_n * num_m_mmas * num_n_mmas if decoding_warp_split_k else num_m_mmas * num_n_mmas
    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(Int(num_output_rows_full), p_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0.0)
    )

    # Rowwise max and sum for online softmax
    comptime row_align = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[Int(WM), accum_type, alignment=row_align]()
    var rowsum = stack_allocation[Int(WM), accum_type, alignment=row_align]()

    @parameter
    for i in range(WM):

        @parameter
        if sink:
            debug_assert(
                Bool(sink_weights),
                "expect sink_weights to be non-null when sink=true",
            )
            if thread_idx.x < UInt(4) * group:
                var sink_logit_log2 = (
                    sink_weights.value()[Int(q_head_idx)][0].cast[accum_type]()
                    * log2e
                )
                rowmax[i] = sink_logit_log2
                if partition_idx == 0 and thread_idx.x % 4 == 0:
                    rowsum[i] = 1.0
                else:
                    rowsum[i] = 0.0
            else:
                rowmax[i] = min_or_neg_inf[accum_type]()
                rowsum[i] = 0.0
        else:
            rowmax[i] = min_or_neg_inf[accum_type]()
            rowsum[i] = 0.0

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    comptime p_smem_size = BM * BN
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(Int(BM), Int(BK)),
        address_space = AddressSpace.SHARED,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * Int(num_warps_n), Int(BM)),
        address_space = AddressSpace.SHARED,
    ]((p_smem + BM * BN).bitcast[Scalar[accum_type]]())

    # Mask global memory iterator
    var stride = max_cache_valid_length

    # Account for group query.
    comptime kv_num_heads = num_heads // group

    var q_offset = depth * kv_head_idx * group

    comptime q_gmem_layout = Layout.row_major(Int(BM), Int(depth))
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[q_gmem_layout.shape, element_type = DType.int32](
                Int(group), Int(depth)
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                Int(depth), 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[Int(BM), Int(BK), axis=1](
        0, 0
    )

    start, end = get_start_and_end_for_partitions[Int(BN)](
        Int(num_keys), Int(num_partitions), Int(block_idx.x)
    )

    comptime q_num_vecs = BM * BK // UInt(simd_size)

    comptime async_copy_q_layout = Layout.row_major(
        Int(min(num_threads, q_num_vecs) * UInt(simd_size) // BK),
        Int(BK // UInt(simd_size)),
    )

    @always_inline
    @parameter
    fn _mask_tensor_row(tensor: LayoutTensor, num_rows: Int) -> type_of(tensor):
        return {
            tensor.ptr,
            {{num_rows, tensor.dim[1]()}, tensor.runtime_layout.stride},
        }

    @parameter
    for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.layout_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads = Int(num_threads),
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        # we `async_copy_commit_group()` and after we finish copying `k`.

        q_gmem_iter._incr()

    var scale_log2e: Float32 = (
        scale.cast[DType.float32]() if use_score_mod
        or mask_t.apply_log2e_after_mask else scale.cast[DType.float32]()
        * log2e
    )

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        var k_ptr = k.block_paged_ptr[Int(BN)](
            UInt32(batch_idx), UInt32(kv_tile_start_row), UInt32(kv_head_idx), 0
        )
        var k_gmem_block = LayoutTensor[
            k_type,
            Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            ),
            masked = not not_last_iter,
        ](k_ptr)
        var k_gmem_iter = k_gmem_block.tiled_iterator[Int(BN), Int(BK), axis=1](
            0, 0
        )

        var kv_tile_num_rows = min(Int(BN), end - kv_tile_start_row)

        _ = p_reg_tile.fill(0)

        comptime kv_num_vecs = BN * BK // UInt(simd_size)
        comptime async_copy_k_layout = Layout.row_major(
            Int(
                min(num_threads, kv_num_vecs)
                * UInt(simd_size)
                // UInt(k_smem_iter.layout.stride[0].value())
            ),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        @parameter
        for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.layout_uint_type(k_id)
            )[]

            @parameter
            if not not_last_iter:
                k_tensor = _mask_tensor_row(k_gmem_iter[], kv_tile_num_rows)
            else:
                k_tensor = k_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads = Int(num_threads),
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_tensor.vectorize[1, simd_size](),
            )

            k_gmem_iter._incr()

        async_copy_commit_group()

        async_copy_wait_all()
        barrier()

        multistage_mma[
            Int(BM),
            Int(BN),
            Int(BK),
            Int(WM),
            Int(WN),
            Int(num_threads),
            Int(num_pipeline_stages),
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters = Int(depth // BK),
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            Int(depth // BK),
        )

        scale_and_mask_helper[
            num_n_mmas = Int(num_n_mmas),
            WN = Int(WN),
            MMA_N=MMA_N,
            simd_width=p_frag_simdwidth,
            use_score_mod=use_score_mod,
            group = Int(group),
        ](
            p_reg_tile,
            scale_log2e,
            num_keys,
            UInt(kv_tile_num_rows),
            lane,
            warp_id,
            mask,
            score_mod,
            kv_tile_start_row,
            stride,
            Int(max_cache_valid_length),
        )

        # For 16x8 mma output, group <= 8 only uses the first 8x8 matrix
        # each thread only has one fragment vector of size 2.
        @parameter
        if group <= 8:
            var output_reg_vecs = output_reg_tile.tile[
                Int(num_output_rows_full), p_frag_size // 2
            ](0, 0).vectorize[1, p_frag_simdwidth]()
            var p_reg_vecs = p_reg_tile.tile[
                Int(num_m_mmas * num_n_mmas), p_frag_size // 2
            ](0, 0).vectorize[1, p_frag_simdwidth]()

            _online_softmax_iter_for_mma_output[
                accum_type,
                Layout.row_major(Int(num_m_mmas), Int(num_n_mmas)),
                Layout.row_major(Int(num_warps_m), Int(num_warps_n)),
                Layout.row_major(8, 4),
                warp_split_k=decoding_warp_split_k,
                use_exp2=True,
            ](
                output_reg_vecs,
                p_reg_vecs,
                warp_scratch.tile[2 * Int(num_warps_n), Int(WM)](
                    0, Int(warp_y)
                ),
                rowmax,
                rowsum,
            )
        else:
            var output_reg_vecs = output_reg_tile.reshape[
                Layout.row_major(
                    2 * Int(num_output_rows_full), p_frag_simdwidth
                )
            ]().vectorize[1, p_frag_simdwidth]()
            var p_reg_vecs = p_reg_tile.reshape[
                Layout.row_major(
                    2 * Int(num_m_mmas * num_n_mmas), p_frag_simdwidth
                )
            ]().vectorize[1, p_frag_simdwidth]()

            _online_softmax_iter_for_mma_output[
                accum_type,
                Layout.row_major(2 * Int(num_m_mmas), Int(num_n_mmas)),
                Layout.row_major(Int(num_warps_m), Int(num_warps_n)),
                Layout.row_major(8, 4),
                warp_split_k=decoding_warp_split_k,
                use_exp2=True,
            ](
                output_reg_vecs,
                p_reg_vecs,
                warp_scratch.tile[2 * Int(num_warps_n), Int(WM)](
                    0, Int(warp_y)
                ),
                rowmax,
                rowsum,
            )

        var v_ptr = v.block_paged_ptr[Int(BN)](
            UInt32(batch_idx), UInt32(kv_tile_start_row), UInt32(kv_head_idx), 0
        )
        var v_gmem_block = LayoutTensor[
            v_type,
            Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            ),
            masked = not not_last_iter,
        ](v_ptr)
        var v_gmem_iter = v_gmem_block.tiled_iterator[Int(BK), Int(BN), axis=0](
            0, 0
        )

        comptime async_copy_v_layout = Layout.row_major(
            Int(min(num_threads, kv_num_vecs) * UInt(simd_size) // BN),
            Int(BN // UInt(simd_size)),
        )

        # load V tile into smem
        @parameter
        for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(
                v_smem_iter.layout_uint_type(v_id)
            )[]

            @parameter
            if not not_last_iter:
                var num_rows_bound = max(
                    0, end - (kv_tile_start_row + Int(v_id * BK))
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle = v_smem_tile.dtype.is_half_float(),
                num_threads = Int(num_threads),
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_tensor.vectorize[1, simd_size](),
            )

            v_gmem_iter._incr()

        async_copy_commit_group()

        @parameter
        if not decoding_warp_split_k:
            # Copy score fragments to shared memory with swizzling to resolve bank
            # conflicts for ldmatrix in the 2nd matmul.
            # warp_split_k does not need the copy as warps don't perform reduction
            # iterating across tiles, but use extra registers to perform MMAs
            # with warp-local data.
            _copy_frag_to_smem[
                BM,
                BN,
                BK,
                WM,
                WN,
                UInt(MMA_M),
                UInt(MMA_N),
                UInt(p_frag_simdwidth),
            ](p_smem_iter, p_reg_tile, UInt32(warp_x), UInt32(warp_y))

        async_copy_wait_all()
        barrier()

        # if decoding_warp_split_k:
        #   S[m, (0:WN) + n*WN] @ V[(0:WN) + n*WN, :]
        # else:
        #   S[m, :] @ V[:, (0:WN) + n*WN]
        @parameter
        if decoding_warp_split_k:
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * Int(num_m_mmas), p_frag_size
            ](0, 0)
            comptime IteratorTypeVSub = LayoutTensorIter[
                v_type,
                Layout.row_major(Int(WN), Int(BN)),
                address_space = AddressSpace.SHARED,
                circular=True,
            ]
            var v_smem_sub = IteratorTypeVSub(
                v_smem + BN * WN * warp_x,
                IteratorTypeVSub.layout_uint_type(v_smem_size),
            )
            multistage_mma[
                Int(BM),
                Int(BN),
                Int(WN),  # BK
                Int(WM),
                Int(BN),  # WN
                Int(num_threads),
                Int(num_pipeline_stages),
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters=1,
            ](
                output_reg_tile,
                p_reg_iter,
                v_smem_sub,
                p_smem_iter,
                v_smem_sub,
                1,
            )
        else:
            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters = Int(BN // BK),
            ](
                output_reg_tile,
                p_smem_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                Int(BN // BK),
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList(Int(BN))](start, end)

    @parameter
    if decoding_warp_split_k:
        var output_reg_vecs = output_reg_tile.tile[
            Int(num_warps_n * num_m_mmas * num_n_mmas), p_frag_size // 2
        ](0, 0).vectorize[1, p_frag_size // 2]()
        # offset on the pointer is to avoid possible races
        # with `accum_smem_warp_tile`.
        var o_smem_ptr = q_smem.bitcast[Scalar[accum_type]]()
        var scratch = LayoutTensor[
            accum_type,
            Layout.row_major(2 * Int(num_warps_n), Int(BM)),
            address_space = AddressSpace.SHARED,
        ](o_smem_ptr + num_warps_n * (num_warps_n - 1) * WM * WN)

        # Note: Sink handling is done after warp reduction in partition-specific logic below.
        # The warp reduction just combines warps; sink contribution is added to rowsum later.
        _online_softmax_iter_for_mma_output_split_warp_reduce[
            accum_type,
            Layout.row_major(Int(num_m_mmas), Int(num_n_mmas)),
            Layout.row_major(Int(num_warps_m), Int(num_warps_n)),
            Layout.row_major(8, 4),
            WM,
            WN,
            use_exp2=True,
        ](
            output_reg_vecs,
            scratch.tile[2 * Int(num_warps_n), Int(WM)](0, Int(warp_y)),
            o_smem_ptr,
            rowmax,
            rowsum,
        )

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        if m_mma * UInt(MMA_M) < group:
            var rowsum_inv = recip(rowsum[2 * Int(m_mma)])

            @parameter
            for n_mma in range(num_n_mmas):
                output_reg_tile[n_mma * num_m_mmas + m_mma, 0] *= rowsum_inv
                output_reg_tile[n_mma * num_m_mmas + m_mma, 1] *= rowsum_inv

        @parameter
        if m_mma * UInt(MMA_M) + UInt(MMA_M // 2) < group:
            var rowsum_inv = recip(rowsum[2 * Int(m_mma) + 1])

            @parameter
            for n_mma in range(num_n_mmas):
                output_reg_tile[n_mma * num_m_mmas + m_mma, 2] *= rowsum_inv
                output_reg_tile[n_mma * num_m_mmas + m_mma, 3] *= rowsum_inv

    if num_partitions > 1:
        if thread_idx.x % 4 == 0 and thread_idx.x < UInt(4) * group:
            var row_sum = rowsum[0]
            var row_max = rowmax[0]
            exp_sum_ptr[q_head_idx] = row_sum
            qk_max_ptr[q_head_idx] = row_max

    # Pack results in shared memory for wider simd width.
    var accum_smem_warp_ptr = (
        q_smem.bitcast[Scalar[output_type]]() + warp_id * WM * WN
    )

    @parameter
    if decoding_warp_split_k:
        accum_smem_warp_ptr += (
            (num_warps_n * (num_warps_n - 1))
            * WM
            * WN
            * UInt(size_of[accum_type]())
        ) // UInt(size_of[output_type]())
    var accum_smem_warp_tile = LayoutTensor[
        output_type,
        Layout.row_major(Int(WM), Int(WN)),
        address_space = AddressSpace.SHARED,
    ](accum_smem_warp_ptr)

    comptime swizzle = make_swizzle[
        num_rows = MMA_M // 2, row_size = Int(WN), access_size=MMA_N
    ]()

    @parameter
    if decoding_warp_split_k:
        copy_local_to_shared[
            thread_layout = Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.tile[Int(num_output_rows), p_frag_size](0, 0)
            .vectorize[1, 2]()
            .transpose(),
        )
    else:
        copy_local_to_shared[
            thread_layout = Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

    # Guard writing to shared memory.
    barrier()

    # FIXME: Using RuntimeLayout to override the layout of the output tensor.
    comptime output_gmem_layout = Layout.row_major(Int(BM), Int(depth))
    var output_gmem_runtime_layout = RuntimeLayout[
        output_gmem_layout
    ].row_major(Index(group, depth))
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](output_ptr + q_offset, output_gmem_runtime_layout)
    var output_gmem_warp_tile = output_gmem_tile.tile[Int(WM), Int(WN)](
        Int(warp_y), Int(warp_x)
    )

    copy_sram_to_dram[
        thread_layout = Layout.row_major(
            WARP_SIZE * simd_size // Int(WN), Int(WN // UInt(simd_size))
        ),
        swizzle=swizzle,
    ](
        output_gmem_warp_tile.vectorize[1, simd_size](),
        accum_smem_warp_tile.vectorize[1, simd_size](),
    )


fn mha_decoding_single_batch_pipelined[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    *,
    BM: UInt,  # number of queries per block
    BN: UInt,  # number of keys per block
    BK: UInt,  # tile size in depth dimension
    WM: UInt,
    WN: UInt,
    depth: UInt,
    num_heads: UInt,
    num_threads: UInt,
    num_pipeline_stages: UInt,
    group: UInt = 1,
    use_score_mod: Bool = False,
    decoding_warp_split_k: Bool = False,
    sink: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    num_keys: UInt,
    num_partitions: UInt,
    max_cache_valid_length: UInt,  # longest KV cache entry
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
):
    """Flash attention v2 algorithm."""
    comptime accum_type = get_accum_type[q_type]()
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime assert q_type == k_type and k_type == v_type

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN

    comptime assert num_warps_m * num_warps_n == (
        num_threads // UInt(WARP_SIZE)
    ), "Number of warps doesn't match warp tile sizes."

    comptime assert group <= 8, String(
        "Only support GQA with group <= 8 for Nvidia, but got a group = '",
        group,
        "'.",
    )

    var tid = thread_idx.x
    var warp_id = warp.broadcast(tid // UInt(WARP_SIZE))
    var lane = lane_id()

    # Coordinates of the current warp.
    warp_y, warp_x = divmod(warp_id, num_warps_n)

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(Int(BM), Int(BK)),
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(Int(BM), Int(BK)),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )

    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    comptime k_smem_size = num_pipeline_stages * BN * BK
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(Int(BN), Int(BK)),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    var kv_head_idx = block_idx.y

    comptime mma_shape = get_mma_shape[q_type, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = WM // UInt(MMA_M)
    comptime num_n_mmas = WN // UInt(MMA_N)

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0.0)
    )

    # Account for group query.
    comptime kv_num_heads = num_heads // group
    var q_head_idx = kv_head_idx * group + thread_idx.x // 4

    # Rowwise max and sum for online softmax
    comptime row_align = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[Int(WM), accum_type, alignment=row_align]()
    var rowsum = stack_allocation[Int(WM), accum_type, alignment=row_align]()

    var partition_idx = block_idx.x

    @parameter
    for i in range(WM):

        @parameter
        if sink:
            debug_assert(
                Bool(sink_weights),
                "expect sink_weights to be non-null when sink=true",
            )
            if thread_idx.x < UInt(4) * group:
                var sink_logit_log2 = (
                    sink_weights.value()[Int(q_head_idx)][0].cast[accum_type]()
                    * log2e
                )
                rowmax[i] = sink_logit_log2
                if partition_idx == 0 and thread_idx.x % 4 == 0:
                    rowsum[i] = 1.0
                else:
                    rowsum[i] = 0.0
            else:
                rowmax[i] = min_or_neg_inf[accum_type]()
                rowsum[i] = 0.0
        else:
            rowmax[i] = min_or_neg_inf[accum_type]()
            rowsum[i] = 0.0

    # Share memory tile for Value, reuse K's shared memory tile.
    comptime v_smem_size = num_pipeline_stages * BN * BK
    var v_smem = k_smem.bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(Int(BK), Int(BN)),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    comptime p_smem_size = BM * BN
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(Int(BM), Int(BK)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(p_smem_size)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(p_frag_simdwidth * Int(num_warps_n), Int(BM)),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]((p_smem + BM * BN).bitcast[Scalar[accum_type]]())

    # Mask global memory iterator, seq_len = 1
    var stride = max_cache_valid_length

    var q_offset = depth * kv_head_idx * group

    comptime q_gmem_layout = Layout.row_major(Int(BM), Int(depth))
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[q_gmem_layout.shape, element_type = DType.int32](
                Int(group), Int(depth)
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                Int(depth), 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[Int(BM), Int(BK), axis=1](
        0, 0
    )

    # Loop over Key and Value tiles
    start, end = get_start_and_end_for_partitions[Int(BN)](
        Int(num_keys), Int(num_partitions), Int(block_idx.x)
    )

    var scale_log2e: Float32 = (
        scale.cast[DType.float32]() if use_score_mod
        or mask_t.apply_log2e_after_mask else scale.cast[DType.float32]()
        * log2e
    )

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, seq_len: Int):
        var k_ptr = k.block_paged_ptr[Int(BN)](
            UInt32(batch_idx), UInt32(kv_tile_start_row), UInt32(kv_head_idx), 0
        )
        var k_gmem_block = LayoutTensor[
            k_type,
            Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            ),
            masked = not not_last_iter,
        ](k_ptr)
        var k_gmem_iter = k_gmem_block.tiled_iterator[Int(BN), Int(BK), axis=1](
            0, 0
        )

        var kv_tile_num_rows = min(Int(BN), end - kv_tile_start_row)

        _ = p_reg_tile.fill(0)

        if kv_tile_start_row == start:
            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                True,  # transpose_b
                swizzle_a=True,
            ](
                p_reg_tile,
                q_gmem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                Int(depth // BK),
                num_b_rows=kv_tile_num_rows,
            )
        else:
            multistage_mma[
                Int(BM),
                Int(BN),
                Int(BK),
                Int(WM),
                Int(WN),
                Int(num_threads),
                Int(num_pipeline_stages),
                True,  # transpose_b
                swizzle_a=True,
            ](
                p_reg_tile,
                q_smem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                Int(depth // BK),
                num_b_rows=kv_tile_num_rows,
            )

        scale_and_mask_helper[
            num_n_mmas = Int(num_n_mmas),
            WN = Int(WN),
            MMA_N=MMA_N,
            simd_width=p_frag_simdwidth,
            use_score_mod=use_score_mod,
            group = Int(group),
        ](
            p_reg_tile,
            scale_log2e,
            num_keys,
            UInt(kv_tile_num_rows),
            lane,
            warp_id,
            mask,
            score_mod,
            kv_tile_start_row,
            stride,
            Int(max_cache_valid_length),
        )

        # For 16x8 mma output, only the top 8x4 matrix matters for GQA since
        # G <= 8 typically holds
        var output_reg_vecs = output_reg_tile.tile[
            Int(num_m_mmas * num_n_mmas), p_frag_size // 2
        ](0, 0).vectorize[1, p_frag_size // 2]()
        var p_reg_vecs = p_reg_tile.tile[
            Int(num_m_mmas * num_n_mmas), p_frag_size // 2
        ](0, 0).vectorize[1, p_frag_size // 2]()

        _online_softmax_iter_for_mma_output[
            accum_type,
            Layout.row_major(Int(num_m_mmas), Int(num_n_mmas)),
            Layout.row_major(Int(num_warps_m), Int(num_warps_n)),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_vecs,
            p_reg_vecs,
            warp_scratch.tile[2 * Int(num_warps_n), Int(WM)](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        var v_ptr = v.block_paged_ptr[Int(BN)](
            UInt32(batch_idx), UInt32(kv_tile_start_row), UInt32(kv_head_idx), 0
        )
        var v_gmem_block = LayoutTensor[
            v_type,
            Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            ),
            masked = not not_last_iter,
        ](v_ptr)
        var v_gmem_iter = v_gmem_block.tiled_iterator[Int(BK), Int(BN), axis=0](
            0, 0
        )

        # Copy score fragments to shared memory with swizzling to resolve bank
        # conflicts for ldmatrix in the 2nd matmul.
        _copy_frag_to_smem[
            BM, BN, BK, WM, WN, UInt(MMA_M), UInt(MMA_N), UInt(p_frag_simdwidth)
        ](p_smem_iter, p_reg_tile, UInt32(warp_x), UInt32(warp_y))
        barrier()

        multistage_mma[
            Int(BM),
            Int(BN),
            Int(BK),
            Int(WM),
            Int(WN),
            Int(num_threads),
            Int(num_pipeline_stages),
            False,  # transpose_b
            swizzle_a=True,
        ](
            output_reg_tile,
            p_smem_iter,
            v_gmem_iter,
            p_smem_iter,
            v_smem_iter,
            Int(BN // BK),
            num_b_rows=kv_tile_num_rows,
        )

    tile_and_unswitch[loop_over_kvcache, VariadicList(Int(BN))](start, end)

    # Apply softmax denumerator.

    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = 1.0 / rowsum[2 * Int(m_mma)]

        @parameter
        for n_mma in range(num_n_mmas):
            output_reg_tile[n_mma, 0] *= rowsum_inv0
            output_reg_tile[n_mma, 1] *= rowsum_inv0

    if num_partitions > 1:
        if thread_idx.x % 4 == 0 and thread_idx.x < UInt(4) * group:
            var row_sum = rowsum[0]
            var row_max = rowmax[0]
            var q_head_idx = kv_head_idx * group + thread_idx.x // 4
            exp_sum_ptr[q_head_idx] = row_sum
            qk_max_ptr[q_head_idx] = row_max

    # Pack results in shared memory for wider simd width.
    var accum_smem_warp_tile = LayoutTensor[
        output_type,
        Layout.row_major(Int(WM), Int(WN)),
        address_space = AddressSpace.SHARED,
    ](q_smem.bitcast[Scalar[output_type]]() + warp_id * WM * WN)

    comptime swizzle = make_swizzle[
        num_rows = MMA_M // 2, row_size = Int(WN), access_size=MMA_N
    ]()
    copy_local_to_shared[
        thread_layout = Layout.row_major(8, 4), swizzle=swizzle
    ](
        accum_smem_warp_tile.vectorize[1, 2](),
        output_reg_tile.vectorize[1, 2]().transpose(),
    )
    # Guard writing to shared memory.
    barrier()
    comptime output_gmem_layout = Layout.row_major(Int(BM), Int(depth))
    var output_gmem_runtime_layout = RuntimeLayout[
        element_type = DType.int32, linear_idx_type = DType.int32
    ](
        RuntimeTuple[output_gmem_layout.shape, element_type = DType.int32](
            Int(group), Int(depth)
        ),
        RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
            Int(depth), 1
        ),
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        Layout.row_major(Int(BM), Int(depth)),
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](output_ptr + q_offset, output_gmem_runtime_layout)
    var output_gmem_warp_tile = output_gmem_tile.tile[Int(WM), Int(WN)](
        Int(warp_y), Int(warp_x)
    )
    copy_sram_to_dram[
        thread_layout = Layout.row_major(
            WARP_SIZE * simd_size // Int(WN), Int(WN // UInt(simd_size))
        ),
        swizzle=swizzle,
    ](
        output_gmem_warp_tile.vectorize[1, simd_size](),
        accum_smem_warp_tile.vectorize[1, simd_size](),
    )


fn mha_splitk_reduce[
    intermediate_type: DType,
    output_type: DType,
    depth: UInt,
    num_heads: UInt,
    num_threads: UInt,
    group: UInt = 1,
    use_exp2: Bool = False,
](
    intermediate_ptr: UnsafePointer[Scalar[intermediate_type], ImmutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[
        Scalar[get_accum_type[output_type]()], MutAnyOrigin
    ],
    qk_max_ptr: UnsafePointer[
        Scalar[get_accum_type[output_type]()], MutAnyOrigin
    ],
    batch_size: Int,
    num_partitions: Int,
):
    # we only reduce over a warp so limit number of warps to 1
    comptime assert num_threads == UInt(WARP_SIZE), (
        "num_threads: "
        + String(num_threads)
        + " should be equal to the warp_size:"
        + String(WARP_SIZE)
    )
    debug_assert(
        block_dim.x == UInt(WARP_SIZE),
        "block_dim.x should be equal to the warp_size",
    )

    comptime accum_type = get_accum_type[output_type]()
    var batch_idx = block_idx.z
    var q_head_idx = block_idx.y

    debug_assert(
        num_partitions <= WARP_SIZE,
        "number of partitions should be less than or equal to the warp_size",
    )
    var partition_idx = thread_idx.x

    var qk_max_offset = (
        num_heads * batch_idx
        + num_heads * UInt(batch_size) * partition_idx
        + q_head_idx
    )
    var l = min_or_neg_inf[accum_type]()
    if partition_idx < UInt(num_partitions):
        l = qk_max_ptr[qk_max_offset]

    var qk_max = warp.lane_group_max_and_broadcast[WARP_SIZE](l)

    # since num_partitions <= WARP_SIZE, allocate buffer using WARP_SIZE
    var exp_sums = LayoutTensor[
        accum_type,
        Layout(WARP_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    comptime intermediate_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, Int(num_heads), Int(depth)
    )
    var intermediate_output = LayoutTensor[
        intermediate_type, intermediate_layout
    ](
        intermediate_ptr,
        RuntimeLayout[intermediate_layout].row_major(
            Index(num_partitions, batch_size, num_heads, depth)
        ),
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, Int(num_heads), Int(depth)
    )
    var output = LayoutTensor[output_type, output_layout](
        output_ptr,
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, num_heads, depth)
        ),
    )

    var rescaled_exp_sum: Scalar[accum_type] = 0
    comptime exp_fn = _exp2_concrete if use_exp2 else _exp_concrete
    if partition_idx < UInt(num_partitions):
        rescaled_exp_sum = exp_sum_ptr[qk_max_offset] * exp_fn(l - qk_max)
        exp_sums[partition_idx] = rescaled_exp_sum

    # ensure exp_sums is written to before reading
    barrier()

    var exp_sum = warp.shuffle_idx(warp.sum(rescaled_exp_sum), 0)

    var inv_global_exp_sum = 1.0 / exp_sum

    comptime width = next_power_of_two(ceildiv(depth, num_threads))
    constrained[depth % width == 0, "depth must be divisible by width"]()
    constrained[
        width * num_threads >= depth,
        "width * num_threads must be greater than or equal to depth",
    ]()

    var acc = SIMD[accum_type, Int(width)](0)
    # Kahan summation compensation for improved precision with many partitions
    var compensation = SIMD[accum_type, Int(width)](0)
    var depth_idx = thread_idx.x * width

    # Precompute base pointer and partition stride to avoid ptr_at_offset in inner loop
    # Layout is [num_partitions, batch_size, num_heads, depth] in row-major
    var partition_stride = batch_size * Int(num_heads) * Int(depth)
    var base_offset = (
        Int(batch_idx) * Int(num_heads) * Int(depth)
        + Int(q_head_idx) * Int(depth)
        + Int(depth_idx)
    )
    var base_ptr = intermediate_output.ptr + base_offset

    @parameter
    fn accum_fn[simd_width: Int](partition_idx: Int) unified {mut}:
        var partition_exp_sum = exp_sums.vectorize[simd_width]()[
            partition_idx // simd_width
        ]

        @parameter
        for i in range(simd_width):
            var ptr = base_ptr + (partition_idx + i) * partition_stride
            var x_load = ptr.load[
                width = Int(width),
                alignment = Int(width) * size_of[intermediate_type](),
            ]().cast[accum_type]()
            var scale = partition_exp_sum[i]
            var mask = SIMD[DType.bool, Int(width)](fill=scale > 0)
            var safe_load = mask.select(x_load, type_of(x_load)(0))
            var term = safe_load * type_of(safe_load)(scale)

            # Kahan summation: compensate for lost low-order bits
            var y = term - compensation
            var t = acc + y
            compensation = (t - acc) - y
            acc = t

    if depth_idx < depth:
        # simd_width=8 is based on experimentation
        # we may want to use a lower value if number of partitions are lower
        vectorize[8](num_partitions, accum_fn)

        acc *= inv_global_exp_sum

        var ptr = output.ptr_at_offset(
            IndexList[3](Int(batch_idx), Int(q_head_idx), Int(depth_idx))
        )
        ptr.store[alignment = Int(width) * size_of[output_type](),](
            acc.cast[output_type]()
        )


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead attention supporting flexible dimensions and
# batch_size > 1.
# ===-----------------------------------------------------------------------===#

comptime _NAIVE_BMM_BLOCK_DIM = LaunchDim(32, 16, 1)
comptime _NAIVE_BMM_BLOCK_TUPLE = StaticTuple[Int32, 1](
    Int32(
        _NAIVE_BMM_BLOCK_DIM.x()
        * _NAIVE_BMM_BLOCK_DIM.y()
        * _NAIVE_BMM_BLOCK_DIM.z()
    )
)


fn mha_gpu_naive[
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    q: LayoutTensor[mut=False, address_space = AddressSpace.GENERIC, ...],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    output: LayoutTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, ...
    ],
    valid_length: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[
            mut=False, q.dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
) raises:
    comptime q_type = q.dtype
    comptime k_type = k_t.dtype
    comptime v_type = k_type

    var num_keys = max_cache_size

    if batch_size == 0 or num_keys == 0 or max_prompt_len == 0:
        return

    comptime p_type = get_accum_type[q_type]()
    var p_device = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * max_prompt_len * num_keys
    )
    # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
    var p_buffer = LayoutTensor[p_type, Layout.row_major[3]()](
        p_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size * num_heads, max_prompt_len, num_keys)
        ),
    )
    var q_device = DeviceBuffer[q.dtype](ctx, q.ptr, q.size(), owning=False)
    var output_device = DeviceBuffer[output.dtype](
        ctx, output.ptr, output.size(), owning=False
    )
    comptime kernel = _bmm0_bs[
        q_type,
        k_t,
        mask_t,
        p_type,
        type_of(valid_length).layout,
        ragged=ragged,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
    ]

    ctx.enqueue_function[kernel, kernel](
        p_device,
        q_device,
        k,
        valid_length,
        scale,
        batch_size,
        max_prompt_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        mask_functor,
        grid_dim=(
            ceildiv(num_keys, 32),
            ceildiv(max_prompt_len, 16),
            num_heads * batch_size,
        ),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
    )

    @parameter
    @__copy_capture(p_buffer)
    fn input_fn_device[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[p_type, _simd_width]:
        return p_buffer.load[width=_simd_width](coords)

    _softmax_gpu[p_type, 1, 3, input_fn_device, sink=sink](
        Index(batch_size * num_heads, max_prompt_len, num_keys),
        p_buffer,
        2,
        ctx,
        sink_weights=sink_weights,
    )
    comptime kernel_1 = _bmm1_bs[
        output_type,
        p_type,
        v_t,
        type_of(valid_length).layout,
        ragged=ragged,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
    ]
    ctx.enqueue_function[kernel_1, kernel_1](
        output_device,
        p_device,
        v,
        valid_length,
        max_prompt_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        grid_dim=(
            ceildiv(depth, 32),
            ceildiv(max_prompt_len, 16),
            num_heads * batch_size,
        ),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
    )

    _ = p_device^


@always_inline
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=_NAIVE_BMM_BLOCK_TUPLE)
fn _bmm0_bs[
    q_type: DType,
    k_t: MHAOperand,
    mask_t: MHAMask,
    p_type: DType,
    valid_length_layout: Layout,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    p_ptr: UnsafePointer[Scalar[p_type], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    mask_functor: mask_t,
):
    # In the num_keys dim.
    var x = global_idx.x
    # In the prompt length dim.
    var y = global_idx.y

    comptime k_type = k_t.dtype

    var batch_head = block_idx.z
    var batch, head = divmod(batch_head, UInt(num_heads))

    var cur_query_len: Int
    var q_offset: Int
    var cur_cache_len: Int
    var padded_num_keys = max_cache_size
    var p_offset = batch_head * UInt(max_prompt_len) * UInt(padded_num_keys)
    var start_pos: UInt32 = 0

    @parameter
    if ragged:

        @parameter
        if not _is_cache_length_accurate:
            start_pos = UInt32(k.cache_length(Int(batch)))

        seq_start = Int(valid_length[batch])
        seq_end = Int(valid_length[batch + 1])
        cur_query_len = seq_end - seq_start
        q_offset = depth * (seq_start * num_heads + Int(head))
        cur_cache_len = Int(start_pos) + cur_query_len
    elif _use_valid_length:
        cur_query_len = Int(valid_length[batch])
        q_offset = depth * (Int(head) + num_heads * max_prompt_len * Int(batch))
        cur_cache_len = k.cache_length(Int(batch)) + cur_query_len
    # When inputs are all NDBuffers i.e. all sequences in batch have the same
    # length and same cache length
    else:
        cur_query_len = max_prompt_len
        q_offset = depth * (Int(head) + num_heads * max_prompt_len * Int(batch))
        cur_cache_len = max_cache_size
        p_offset = batch_head * UInt(max_prompt_len) * UInt(max_cache_size)

    debug_assert(cur_query_len <= max_prompt_len, "Invalid cur_query_len")
    debug_assert(
        cur_cache_len <= padded_num_keys,
        "Invalid cur_cache_len",
    )

    if x >= UInt(padded_num_keys) or y >= UInt(max_prompt_len):
        return

    var q = q_ptr + q_offset

    var kv_head = Int(head // UInt(group))

    var p = p_ptr + Int(p_offset)

    var accum = Scalar[p_type](0.0)

    if x < UInt(cur_cache_len) and y < UInt(cur_query_len):
        var k_ptr = k.block_paged_ptr[1](
            UInt32(batch), UInt32(x), UInt32(kv_head), 0
        )

        # TODO: The AMD-specific path is to handle Llama shapes, similar
        #       to how things were before #53433. Once flash attention is
        #       supported on AMD, this stopgap AMD path should be eliminated to
        #       function as a generic fall-back (i.e., without vectorization).
        #       REL: KERN-1343.
        @parameter
        if is_amd_gpu():
            var accum_vec = SIMD[p_type, simd_width_of[p_type]()](0)

            fn accum_fn[width: Int](offset: Int) unified {mut}:
                comptime alignment = align_of[SIMD[p_type, width]]()
                var q_val = q.load[width=width, alignment=alignment](
                    y * UInt(num_heads) * UInt(depth) + UInt(offset)
                ).cast[k_type]()
                var k_val = k_ptr.load[width=width, alignment=alignment](offset)
                var qk_val = (q_val * k_val).cast[p_type]()

                @parameter
                if width == 1:
                    accum += rebind[type_of(accum)](qk_val)
                else:
                    accum_vec += rebind[type_of(accum_vec)](qk_val)

            vectorize[simd_width_of[p_type]()](depth, accum_fn)
            accum += accum_vec.reduce_add()
        else:
            for d in range(depth):
                var q_val = q[y * UInt(num_heads) * UInt(depth) + UInt(d)]
                var k_val = k_ptr[d]
                accum += q_val.cast[p_type]() * k_val.cast[p_type]()

    var score_row = y + UInt(cur_cache_len) - UInt(cur_query_len)
    var score_col = x
    p[y * UInt(padded_num_keys) + x] = mask_functor.mask(
        Index(
            Int(batch),
            Int(head),
            Int(score_row),
            Int(score_col),
        ),
        accum * scale.cast[p_type](),
    )

    if x >= UInt(cur_cache_len) or y >= UInt(cur_query_len):
        p[y * UInt(padded_num_keys) + x] = min_or_neg_inf[p_type]()


@always_inline
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=_NAIVE_BMM_BLOCK_TUPLE)
fn _bmm1_bs[
    output_type: DType,
    p_type: DType,
    v_t: MHAOperand,
    valid_length_layout: Layout,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    p_ptr: UnsafePointer[Scalar[p_type], MutAnyOrigin],
    v: v_t,
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
):
    comptime v_type = v_t.dtype

    # In the depth dim.
    var x = global_idx.x
    # IN the sequence length dim.
    var y = global_idx.y

    var batch_head = block_idx.z
    var batch, head = divmod(batch_head, UInt(num_heads))

    var cur_query_len: Int
    var output_offset: Int
    var cur_cache_len: Int
    var padded_num_keys = max_cache_size
    var p_offset = batch_head * UInt(max_prompt_len) * UInt(padded_num_keys)
    var start_pos: UInt32 = 0

    @parameter
    if ragged:

        @parameter
        if not _is_cache_length_accurate:
            start_pos = UInt32(v.cache_length(Int(batch)))

        seq_start = Int(valid_length[batch])
        seq_end = Int(valid_length[batch + 1])
        cur_query_len = seq_end - seq_start
        output_offset = (seq_start * num_heads + Int(head)) * depth
        cur_cache_len = cur_query_len + Int(start_pos)
    elif _use_valid_length:
        cur_query_len = Int(valid_length[batch])
        output_offset = depth * (
            Int(head) + num_heads * max_prompt_len * Int(batch)
        )
        cur_cache_len = cur_query_len + v.cache_length(Int(batch))
    # When inputs are all NDBuffers i.e. all sequences in batch have the same
    # length and same cache length
    else:
        cur_query_len = max_prompt_len
        output_offset = depth * (
            Int(head) + num_heads * max_prompt_len * Int(batch)
        )
        cur_cache_len = max_cache_size
        p_offset = batch_head * UInt(max_prompt_len) * UInt(max_cache_size)

    debug_assert(cur_query_len <= max_prompt_len, "Invalid cur_query_len")

    if x >= UInt(depth) or y >= UInt(cur_query_len):
        return

    var p = p_ptr + p_offset

    var kv_head = Int(head // UInt(group))
    var output = output_ptr + output_offset

    var accum = Float32(0.0)

    for i in range(cur_cache_len):
        var v_ptr = v.block_paged_ptr[1](
            UInt32(batch), UInt32(i), UInt32(kv_head), UInt32(x)
        )
        accum += (
            p[y * UInt(padded_num_keys) + UInt(i)].cast[DType.float32]()
            * v_ptr[0].cast[DType.float32]()
        )

    output[y * UInt(num_heads) * UInt(depth) + x] = accum.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead attention supporting flexible dimensions.
# ===-----------------------------------------------------------------------===#


fn mha_gpu_naive[
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    mask_type: DType,
    //,
    sink: Bool = False,
](
    q: LayoutTensor[
        mut=False, q_type, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[
        mut=False, k_type, address_space = AddressSpace.GENERIC, ...
    ],
    v: LayoutTensor[
        mut=False, v_type, address_space = AddressSpace.GENERIC, ...
    ],
    mask: LayoutTensor[
        mut=False, mask_type, address_space = AddressSpace.GENERIC, ...
    ],
    output: LayoutTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    mha_gpu_naive[sink=sink](
        q,
        k,
        v,
        MaterializedMask(
            LayoutTensor[
                mask_type,
                Layout.row_major(mask.layout.shape),
                MutAnyOrigin,
            ](
                mask.ptr,
                RuntimeLayout[Layout.row_major(mask.layout.shape)].row_major(
                    mask.runtime_layout.shape.value.canonicalize()
                ),
            )
        ),
        output,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
        sink_weights,
    )


fn mha_gpu_naive[
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    MaskType: MHAMask,
    //,
    sink: Bool = False,
](
    q: LayoutTensor[
        mut=False, q_type, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[
        mut=False, k_type, address_space = AddressSpace.GENERIC, ...
    ],
    v: LayoutTensor[
        mut=False, v_type, address_space = AddressSpace.GENERIC, ...
    ],
    mask: MaskType,
    output: LayoutTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    var k_operand = LayoutTensorMHAOperand(
        LayoutTensor[k.dtype, Layout.row_major(k.layout.shape), MutAnyOrigin](
            k.ptr,
            RuntimeLayout[Layout.row_major(k.layout.shape)].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        )
    )
    var v_operand = LayoutTensorMHAOperand(
        LayoutTensor[v.dtype, Layout.row_major(v.layout.shape), MutAnyOrigin](
            v.ptr,
            RuntimeLayout[Layout.row_major(v.layout.shape)].row_major(
                v.runtime_layout.shape.value.canonicalize()
            ),
        )
    )
    var null_valid_length = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        UnsafePointer[UInt32, MutAnyOrigin](),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    mha_gpu_naive[_is_cache_length_accurate=True, sink=sink](
        q,
        k_operand,
        v_operand,
        mask,
        output,
        null_valid_length,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
        sink_weights,
    )


fn mha_gpu_naive[
    q_type: DType,
    output_type: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    //,
    ragged: Bool = False,
    sink: Bool = False,
](
    q: LayoutTensor[
        mut=False, q_type, address_space = AddressSpace.GENERIC, ...
    ],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    output: LayoutTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, ...
    ],
    valid_length: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    var k_operand = KVCacheMHAOperand(k)
    var v_operand = KVCacheMHAOperand(v)

    mha_gpu_naive[
        _use_valid_length=True, _is_cache_length_accurate=False, sink=sink
    ](
        q,
        k_operand,
        v_operand,
        mask_functor,
        output,
        valid_length,
        scale,
        batch_size,
        max_prompt_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        ctx,
        sink_weights,
    )


# ===-----------------------------------------------------------------------===#
# Naive CPU MHA as reference
# ===-----------------------------------------------------------------------===#


fn _naive_attention_with_transpose[
    dtype: DType,
    transpose_k: Bool = False,
](
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    v: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    mask: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
) raises:
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    Layouts:
        q: BSHD
        k, v: BKHD
        output: BSHD
        mask: SK
    B, S, K, H, D stand for batch size, sequence length, number of keys,
    number of heads, and depth per head, respectively.
    """
    comptime simd_size = simd_width_of[dtype]()

    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()
    var num_heads = q.dim[2]()
    var depth = q.dim[3]()

    # Q, K, V transposed
    var qt_ptr = alloc[Scalar[dtype]](q.size())
    var kt_ptr = alloc[Scalar[dtype]](k.size())
    var vt_ptr = alloc[Scalar[dtype]](v.size())
    # Score = softmax(Q * K)
    var score_size = batch_size * num_heads * seq_len * num_keys
    var score_ptr = alloc[Scalar[dtype]](score_size)
    # O = Score * V. It's transposed and will be transposed back to output.
    var ot_ptr = alloc[Scalar[dtype]](output.size())

    var qt = NDBuffer[dtype, 4](
        qt_ptr, Index(batch_size, num_heads, seq_len, depth)
    )
    var kt = NDBuffer[dtype, 4](
        kt_ptr, Index(batch_size, num_heads, depth, num_keys)
    )
    var vt = NDBuffer[dtype, 4](
        vt_ptr, Index(batch_size, num_heads, num_keys, depth)
    )
    var ot = NDBuffer[dtype, 4](
        ot_ptr, Index(batch_size, num_heads, seq_len, depth)
    )

    comptime layout_4d = Layout.row_major[4]()
    var qt_lt = LayoutTensor[dtype, layout_4d](
        qt_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, seq_len, depth)
        ),
    )
    var kt_lt = LayoutTensor[dtype, layout_4d](
        kt_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, depth, num_keys)
        ),
    )
    var vt_lt = LayoutTensor[dtype, layout_4d](
        vt_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, num_keys, depth)
        ),
    )
    var ot_lt = LayoutTensor[dtype, layout_4d](
        ot_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, seq_len, depth)
        ),
    )

    # BSHD -> BHSD
    var q_perm_stack = InlineArray[Scalar[DType.int], 4](uninitialized=True)
    var q_perm = LayoutTensor[DType.int, Layout(4)](q_perm_stack)
    q_perm[0] = 0
    q_perm[1] = 2
    q_perm[2] = 1
    q_perm[3] = 3

    # BSHD -> BHDS
    var k_perm_stack = InlineArray[Scalar[DType.int], 4](uninitialized=True)
    var k_perm = LayoutTensor[DType.int, Layout(4)](k_perm_stack)
    k_perm[0] = 0
    k_perm[1] = 2
    k_perm[2] = 3
    k_perm[3] = 1

    # BHSD -> BSHD
    var o_perm_stack = InlineArray[Scalar[DType.int], 4](uninitialized=True)
    var o_perm = LayoutTensor[DType.int, Layout(4)](o_perm_stack)
    o_perm[0] = 0
    o_perm[1] = 2
    o_perm[2] = 1
    o_perm[3] = 3

    transpose(
        qt,
        NDBuffer[q.dtype, 4, q.origin](
            q.ptr,
            rebind[IndexList[4]](q.runtime_layout.shape.value.canonicalize()),
        ),
        q_perm.ptr,
    )
    transpose(
        kt,
        NDBuffer[k.dtype, 4, k.origin](
            k.ptr,
            rebind[IndexList[4]](k.runtime_layout.shape.value.canonicalize()),
        ),
        k_perm.ptr,
    )
    transpose(
        vt,
        NDBuffer[v.dtype, 4, v.origin](
            v.ptr,
            rebind[IndexList[4]](v.runtime_layout.shape.value.canonicalize()),
        ),
        q_perm.ptr,
    )

    _naive_attention[dtype, transpose_k](
        ot_lt, qt_lt, kt_lt, vt_lt, mask, scale
    )

    transpose(
        NDBuffer[output.dtype, 4, output.origin](
            output.ptr,
            rebind[IndexList[4]](
                output.runtime_layout.shape.value.canonicalize()
            ),
        ),
        ot,
        o_perm.ptr,
    )

    qt_ptr.free()
    kt_ptr.free()
    vt_ptr.free()
    score_ptr.free()
    ot_ptr.free()


fn _naive_attention[
    dtype: DType,
    transpose_k: Bool = False,
](
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    v: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    mask: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
) raises:
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    """
    comptime simd_size = simd_width_of[dtype]()

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var seq_len = q.dim[2]()
    var num_keys = v.dim[2]()

    # Allocate intermediate memory buffer.
    var score_size = batch_size * num_heads * seq_len * num_keys
    var score_ptr = alloc[Scalar[dtype]](score_size)
    var score = NDBuffer[dtype, 4](
        score_ptr, Index(batch_size, num_heads, seq_len, num_keys)
    )
    comptime layout_4d = Layout.row_major[4]()
    var score_lt = LayoutTensor[dtype, layout_4d](
        score_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
    )

    batched_matmul[transpose_b=transpose_k](
        score,
        NDBuffer[q.dtype, 4, q.origin](
            q.ptr,
            rebind[IndexList[4]](q.runtime_layout.shape.value.canonicalize()),
        ),
        NDBuffer[k.dtype, 4, k.origin](
            k.ptr,
            rebind[IndexList[4]](k.runtime_layout.shape.value.canonicalize()),
        ),
    )

    @__copy_capture(score)
    @parameter
    @always_inline
    fn scale_and_mask[
        width: Int, _rank: Int, alignment: Int = 1
    ](coords: IndexList[_rank]):
        var vec = score.load[width=width](rebind[IndexList[4]](coords))
        vec = vec * scale.cast[dtype]()
        vec = vec + mask.load[width=width](
            Index(coords[_rank - 2], coords[_rank - 1])
        )
        score.store[width=width](rebind[IndexList[4]](coords), vec)

    elementwise[scale_and_mask, simd_size](
        score_lt.runtime_layout.shape.value.canonicalize()
    )

    softmax[dtype, simd_size, 4](
        score_lt,
        score_lt,
        axis=3,
    )

    batched_matmul[transpose_b=False](
        NDBuffer[output.dtype, 4, output.origin](
            output.ptr,
            rebind[IndexList[4]](
                output.runtime_layout.shape.value.canonicalize()
            ),
        ),
        score,
        NDBuffer[v.dtype, 4, v.origin](
            v.ptr,
            rebind[IndexList[4]](v.runtime_layout.shape.value.canonicalize()),
        ),
    )

    score_ptr.free()
