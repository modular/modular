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

from collections import OptionalReg
from math import ceildiv, recip
from math.constants import log2e
from sys import (
    CompilationTarget,
    align_of,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
    env_get_bool,
)
from sys.info import _cdna_4_or_newer
import gpu.warp as warp
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
from layout.tensor_builder import static
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.bmm import batched_matmul
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from linalg.transpose import transpose
from memory import stack_allocation
from nn.mha_gfx942 import (
    mha_decoding_single_batch_gfx942,
    mha_single_batch_gfx942,
)
from nn.mha_gfx950 import mha_single_batch_gfx950
from nn.mha_mask import MaterializedMask, MHAMask, TileMaskStatus
from nn.mha_operand import (
    KVCacheMHAOperand,
    MHAOperand,
    LayoutTensorMHAOperand,
    RaggedMHAOperand,
)
from nn.mha_score_mod import IdentityScoreMod, ScoreModTrait
from nn.mha_sm90 import mha_sm90_dispatch
from nn.mha_sm100 import mha_sm100_dispatch
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
from tensor_internal import IOUnknown, ManagedTensorSlice
from tensor_internal.managed_tensor_slice import StaticTensorSpec

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
    q_layout: Layout, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        dtype, UInt(Int(q_layout.shape[2])), UInt(Int(q_layout.shape[3]))
    ),
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
    sink: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, **_],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, **_],
    k: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    v: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    mask: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    scale: Float32,
    context: DeviceContextPtr = DeviceContextPtr(),
    num_partitions: OptionalReg[Int] = None,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ] = None,
) raises:
    # TODO docstring
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q.runtime_layout.shape.value),
            trace_arg("k", k.runtime_layout.shape.value),
            trace_arg("v", v.runtime_layout.shape.value),
            trace_arg("output", output.runtime_layout.shape.value),
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
                    MutableAnyOrigin,
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
    alias sm_count = ctx.default_device_info.sm_count
    # TODO: This is dumb, make it more granular as a follow up
    if num_keys > 512 and group <= 8:
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
        has_amd_gpu_accelerator() and qkv_type is DType.bfloat16
    )


fn depth_supported_by_gpu[
    depth: Int,
    mask_t: MHAMask,
    config: MHAConfig,
    info: GPUInfo,
]() -> Bool:
    alias is_sm90or100 = (info is H100) or (info is B200)
    alias head_depth_supported = depth == 128 or (
        depth == 64
        and (is_sm90or100 or info is A100 or has_amd_gpu_accelerator())
    ) or (
        depth == 256
        and (
            has_amd_gpu_accelerator()
            or (is_sm90or100 and mask_t.mask_safe_out_of_bounds)
        )
    ) or (
        depth == 80
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
    q_layout: Layout, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        dtype,
        UInt(Int(q_layout.shape[q_layout.rank() - 2])),
        UInt(Int(q_layout.shape[q_layout.rank() - 1])),
    ),
    ragged: Bool = False,
    sink: Bool = False,
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, **_],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, **_],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin
        ]
    ] = None,
    num_partitions: OptionalReg[Int] = None,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
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
    constrained[
        ragged or q.rank == 4,
        "only support rank 4 inputs for non-ragged inputs.",
    ]()
    constrained[
        not ragged or q.rank == 3,
        "only support rank 3 inputs for ragged inputs.",
    ]()
    constrained[
        q.dtype == cache_t.dtype == output.dtype,
        "Q, K, V, output should have same type.",
    ]()
    constrained[
        q.dtype is DType.float32 or q.dtype.is_half_float(),
        "Only support single and half precision.",
    ]()

    # TODO docstring
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q.runtime_layout.shape.value),
            trace_arg("output", output.runtime_layout.shape.value),
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
        alias head_depth_known = q.layout.shape.all_known[q.rank-2, q.rank]()
        alias depth = Int(q.layout.shape[q.rank-1])
        alias gpu_info = ctx.default_device_info
        alias head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
        alias flash_attention_applicable = flash_attention_hw_supported[dtype]() and head_depth_known and head_depth_supported and not naive_kernel
        # fmt: on
        alias kv_num_heads = cache_t.kv_params.num_heads

        var k_operand = KVCacheMHAOperand(k)
        var v_operand = KVCacheMHAOperand(v)

        flash_attention_dispatch[
            kv_num_heads=kv_num_heads,
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
            valid_length,
            max_prompt_len,
            num_keys,
            scale,
            is_token_generation,
            ctx,
            kv_input_row_offsets,
            num_partitions,
            sink_weights,
        )


@always_inline
fn q_num_matrix_view_rows[
    dtype: DType, //, *, decoding: Bool, depth: Int
](q: LayoutTensor[dtype, **_]) -> Int:
    # for tma if decoding, we view q as a rows x depth matrix
    # otherwise, we view q as a rows x (depth*num_heads) matrix
    var num_rows: Int = q.dim[0]()

    @parameter
    for i in range(1, q.rank - 1 if decoding else q.rank - 2):
        num_rows *= q.dim[i]()
    return num_rows * (depth // 64) if decoding else num_rows


@always_inline
fn flash_attention_dispatch[
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    q_layout: Layout, //,
    kv_num_heads: Int,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        dtype,
        UInt(Int(q_layout.shape[q_layout.rank() - 2])),
        UInt(Int(q_layout.shape[q_layout.rank() - 1])),
    ),
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
    output: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, **_],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    max_prompt_len: Int,
    max_cache_valid_length: Int,
    scale: Float32,
    is_token_generation: Bool,
    ctx: DeviceContext,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin
        ]
    ] = None,
    num_partitions: OptionalReg[Int] = None,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ] = None,
) raises:
    alias num_heads = config.num_heads
    alias depth = config.depth
    alias group = config.num_heads // UInt(kv_num_heads)

    # K V smem is only separate for GPUs with shared memory greater or equal to A100's.
    alias is_shared_kv = ctx.default_device_info.shared_memory_per_multiprocessor < A100.shared_memory_per_multiprocessor

    constrained[depth == UInt(Int(q.layout.shape[q.rank - 1]))]()
    constrained[num_heads == UInt(Int(q.layout.shape[q.rank - 2]))]()
    var batch_size: Int

    @parameter
    if ragged:
        batch_size = valid_length.shape()[0] - 1
    # This branch holds for both KVCache and NDBuffer inputs.
    # Q is BSHD, S is either homogeneous or padded to same length.
    else:
        batch_size = q.dim[0]()

    alias q_half_float = dtype in (DType.float16, DType.bfloat16)
    alias q_half_float_or_fp32 = dtype is DType.float32 or q_half_float

    var q_device = DeviceBuffer[q.dtype](ctx, q.ptr, q.size(), owning=False)
    var output_device = DeviceBuffer[output.dtype](
        ctx, output.ptr, output.size(), owning=False
    )

    @parameter
    if _is_flash_attention_applicable:
        alias is_sm90 = ctx.default_device_info is H100
        alias is_sm100 = ctx.default_device_info is B200
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
                num_rows_q = q_num_matrix_view_rows[
                    decoding=False, depth=depth
                ](q)

                @parameter
                if is_sm90:
                    mha_sm90_dispatch[
                        config=config,
                        group=group,
                        use_score_mod=use_score_mod,
                        ragged=ragged,
                        sink=sink,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                    ](
                        output.ptr,
                        q.ptr,
                        k,
                        rebind[k_t](v),
                        num_rows_q,
                        mask_functor,
                        score_mod_functor,
                        valid_length._ptr.address_space_cast[
                            AddressSpace.GENERIC
                        ](),
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
                    constrained[is_sm100]()
                    mha_sm100_dispatch[
                        config=config,
                        group=group,
                        use_score_mod=use_score_mod,
                        ragged=ragged,
                        sink=sink,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                    ](
                        output.ptr,
                        q.ptr,
                        k,
                        rebind[k_t](v),
                        num_rows_q,
                        mask_functor,
                        score_mod_functor,
                        valid_length._ptr.address_space_cast[
                            AddressSpace.GENERIC
                        ](),
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
                alias BM = config.block_m()
                alias smem_use = config.shared_mem_bytes[is_shared_kv]()
                alias kernel = mha[
                    config.dtype,
                    k_t,
                    v_t,
                    output.dtype,
                    mask_t,
                    score_mod_t,
                    __type_of(valid_length.to_layout_tensor()).layout,
                    config,
                    group=group,
                    use_score_mod=use_score_mod,
                    ragged=ragged,
                    is_shared_kv=is_shared_kv,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    _padded_ndbuffer=_padded_ndbuffer,
                ]

                ctx.enqueue_function_checked[kernel, kernel](
                    q_device,
                    k,
                    v,
                    output_device,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length.to_layout_tensor(),
                    kv_input_row_offsets,
                    sink_weights,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=(
                        Int(ceildiv(max_prompt_len, BM)),
                        Int(config.num_heads),
                        Int(batch_size),
                    ),
                    block_dim=(Int(config.num_threads()), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
        # FA3 decoding impl only support half precision, while fp32 is supported
        # for fp32 as well.
        elif q_half_float_or_fp32 and is_token_generation:
            alias BM = 16
            alias BN = depth
            alias BK = 32 if has_amd_gpu_accelerator() else (
                16 if q.dtype is DType.float32 else 32
            )
            alias WM = BM
            alias WN = 32
            # num warps in M and N, multiplied by warp size.
            alias num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

            alias accum_type = get_accum_type[q.dtype]()
            alias num_pipeline_stages = 4
            # smem for q
            var shared_mem_bytes = BM * depth * size_of[q.dtype]()

            # separate KV smem if we have enough smem
            @parameter
            if not is_shared_kv:
                shared_mem_bytes += 2 * BN * depth * size_of[k_t.dtype]()
            else:
                shared_mem_bytes += (
                    num_pipeline_stages * BN * BK * size_of[k_t.dtype]()
                )

            alias num_warps = ceildiv(num_threads, WARP_SIZE)

            # smem for p and warp_scratch
            shared_mem_bytes += (
                BM * BN * size_of[k_t.dtype]()
                + 2 * num_warps * BM * size_of[accum_type]()
            )
            alias num_blocks_y = num_heads // group

            var num_partitions_value = num_partitions.value() if num_partitions else get_mha_decoding_num_partitions[
                num_heads, group
            ](
                batch_size, max_cache_valid_length, ctx
            )

            alias use_fa3_kernel = (
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
                    valid_length,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    num_heads,
                    depth,
                    group,
                    ctx,
                    sink_weights,
                )
            else:
                alias kernel = mha_decoding[
                    q.dtype,
                    k_t,
                    v_t,
                    output.dtype,
                    mask_t,
                    score_mod_t,
                    __type_of(valid_length.to_layout_tensor()).layout,
                    BM=BM,
                    BN=BN,
                    BK = UInt(BK),
                    WM=WM,
                    WN=WN,
                    depth=depth,
                    num_heads=num_heads,
                    num_threads = UInt(num_threads),
                    num_pipeline_stages = UInt(num_pipeline_stages),
                    group = UInt(group),
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
                        num_rows_q = q_num_matrix_view_rows[
                            decoding=True, depth=depth
                        ](q)

                        @parameter
                        if is_sm90:
                            mha_sm90_dispatch[
                                config=config,
                                group=group,
                                use_score_mod=use_score_mod,
                                ragged=ragged,
                                sink=sink,
                                _is_cache_length_accurate=_is_cache_length_accurate,
                            ](
                                output.ptr,
                                q.ptr,
                                k,
                                rebind[k_t](v),
                                num_rows_q,
                                mask_functor,
                                score_mod_functor,
                                valid_length._ptr.address_space_cast[
                                    AddressSpace.GENERIC
                                ](),
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
                            mha_sm100_dispatch[
                                config=config,
                                group=group,
                                use_score_mod=use_score_mod,
                                ragged=ragged,
                                sink=sink,
                                _is_cache_length_accurate=_is_cache_length_accurate,
                            ](
                                output.ptr,
                                q.ptr,
                                k,
                                rebind[k_t](v),
                                num_rows_q,
                                mask_functor,
                                score_mod_functor,
                                valid_length._ptr.address_space_cast[
                                    AddressSpace.GENERIC
                                ](),
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
                        alias nullptr = UnsafePointer[Scalar[accum_type]]()

                        var nullptr_device = DeviceBuffer[accum_type](
                            ctx, nullptr, 0, owning=False
                        )
                        ctx.enqueue_function_checked[kernel, kernel](
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
                            valid_length.to_layout_tensor(),
                            sink_weights,
                            mask_functor,
                            score_mod_functor,
                            grid_dim=(
                                1,
                                Int(num_blocks_y),
                                Int(batch_size),
                            ),
                            block_dim=(num_threads, 1, 1),
                            shared_mem_bytes=shared_mem_bytes if has_nvidia_gpu_accelerator() else 0,
                            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                                (
                                    ctx.default_device_info.shared_memory_per_multiprocessor
                                    - 4096
                                ) if has_nvidia_gpu_accelerator() else 0
                            ),
                        )
                    return

                else:
                    # We split partitions and then reduce
                    # allocate memory for intermediate results
                    # q # [B, S, H, D]
                    var output_intermediate_data = ctx.enqueue_create_buffer[
                        output.dtype
                    ](
                        num_heads
                        * depth
                        * UInt(batch_size)
                        * UInt(num_partitions_value)
                    )

                    var output_intermediate = LayoutTensor[
                        output.dtype, Layout.row_major[4]()
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
                    ](2 * data_len)

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
                        exp_sum_qk_max_data.unsafe_ptr().offset(data_len),
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
                        num_rows_q = q_num_matrix_view_rows[
                            decoding=True, depth=depth
                        ](q)

                        @parameter
                        if is_sm90:
                            mha_sm90_dispatch[
                                config=config,
                                group=group,
                                use_score_mod=use_score_mod,
                                ragged=ragged,
                                sink=sink,
                                _is_cache_length_accurate=_is_cache_length_accurate,
                            ](
                                output_intermediate.ptr,
                                q.ptr,
                                k,
                                rebind[k_t](v),
                                num_rows_q,
                                mask_functor,
                                score_mod_functor,
                                valid_length._ptr.address_space_cast[
                                    AddressSpace.GENERIC
                                ](),
                                StaticInt[1](),
                                max_cache_valid_length,
                                scale,
                                kv_input_row_offsets,
                                batch_size,
                                SplitKPartition(
                                    exp_sum_qk_max_data.unsafe_ptr(),
                                    num_partitions_value,
                                ),
                                ctx,
                                sink_weights,
                            )
                        else:
                            mha_sm100_dispatch[
                                config=config,
                                group=group,
                                use_score_mod=use_score_mod,
                                ragged=ragged,
                                sink=sink,
                                _is_cache_length_accurate=_is_cache_length_accurate,
                            ](
                                output_intermediate.ptr,
                                q.ptr,
                                k,
                                rebind[k_t](v),
                                num_rows_q,
                                mask_functor,
                                score_mod_functor,
                                valid_length._ptr.address_space_cast[
                                    AddressSpace.GENERIC
                                ](),
                                StaticInt[1](),
                                max_cache_valid_length,
                                scale,
                                kv_input_row_offsets,
                                batch_size,
                                SplitKPartition(
                                    exp_sum_qk_max_data.unsafe_ptr(),
                                    num_partitions_value,
                                ),
                                ctx,
                                sink_weights,
                            )
                    else:
                        ctx.enqueue_function_checked[kernel, kernel](
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
                            valid_length.to_layout_tensor(),
                            sink_weights,
                            mask_functor,
                            score_mod_functor,
                            grid_dim=(
                                num_partitions_value,
                                Int(num_blocks_y),
                                Int(batch_size),
                            ),
                            block_dim=(num_threads, 1, 1),
                            shared_mem_bytes=shared_mem_bytes if has_nvidia_gpu_accelerator() else 0,
                            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                                ctx.default_device_info.shared_memory_per_multiprocessor
                                - 4096 if has_nvidia_gpu_accelerator() else 0
                            ),
                        )

                    alias kernel_reduce = mha_splitk_reduce[
                        output.dtype,
                        depth=depth,
                        num_heads=num_heads,
                        num_threads = UInt(WARP_SIZE),
                        group = UInt(group),
                        use_exp2=use_fa3_kernel,
                    ]

                    ctx.enqueue_function_checked[kernel_reduce, kernel_reduce](
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
                valid_length,
                scale,
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                num_heads,
                depth,
                group,
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
            valid_length,
            scale,
            batch_size,
            max_prompt_len,
            max_cache_valid_length,
            num_heads,
            depth,
            group,
            ctx,
            sink_weights,
        )


fn flash_attention[
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    q_layout: Layout, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        dtype, UInt(Int(q_layout.shape[2])), UInt(Int(q_layout.shape[3]))
    ),
    decoding_warp_split_k: Bool = False,
    _use_valid_length: Bool = False,
    _padded_ndbuffer: Bool = False,
    naive_kernel: Bool = False,
    sink: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, **_],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, **_],
    k: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    v: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: OptionalReg[Int] = None,
    valid_length: OptionalReg[
        ManagedTensorSlice[
            IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ]
    ] = None,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ] = None,
) raises:
    # See the kV cache overloads for comments.

    constrained[q.rank == 4, "only support rank 4 inputs."]()

    # Runtime dimensions.
    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()

    # Whether head and depth are static. With BSHD, B and S are dynamic.
    # H and D are always known.
    # fmt: off
    alias head_depth_known = q.layout.shape.all_known[2, 4]() and k.layout.shape[2] != UNKNOWN_VALUE
    alias depth = Int(q.layout.shape[q.rank-1])
    alias gpu_info = ctx.default_device_info
    alias head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
    alias flash_attention_applicable = flash_attention_hw_supported[dtype]() and head_depth_known and head_depth_supported and not naive_kernel

    alias q_half_float = q.dtype in (DType.float16, DType.bfloat16)
    alias kv_num_heads = Int(k.layout.shape[2])
    # fmt: on

    var is_token_generation = seq_len == 1 and num_keys > seq_len

    var k_operand = LayoutTensorMHAOperand(
        LayoutTensor[
            k.dtype, Layout.row_major(k.layout.shape), MutableAnyOrigin
        ](
            k.ptr,
            RuntimeLayout[Layout.row_major(k.layout.shape)].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        )
    )
    var v_operand = LayoutTensorMHAOperand(
        LayoutTensor[
            v.dtype, Layout.row_major(v.layout.shape), MutableAnyOrigin
        ](
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
        valid_length.or_else(valid_length.T(UnsafePointer[UInt32](), [1], [1])),
        q.dim[1](),
        num_keys,
        scale,
        is_token_generation,
        ctx,
        None,
        num_partitions,
        sink_weights,
    )


fn flash_attention_ragged[
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_layout: Layout, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        type,
        UInt(Int(q_layout.shape[q_layout.rank() - 2])),  # num_heads
        UInt(Int(q_layout.shape[q_layout.rank() - 1])),  # head_dim
    ),
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, **_],
    q: LayoutTensor[type, q_layout, address_space = AddressSpace.GENERIC, **_],
    k: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    v: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    input_row_offsets: ManagedTensorSlice[
        IOUnknown,
        static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
    ],
    max_prompt_len: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, **_
    ],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: OptionalReg[Int] = None,
) raises:
    # See the kV cache overloads for comments.

    constrained[q.rank == 3, "only support rank 3 inputs for ragged inputs."]()
    constrained[
        q.dtype == k.dtype == v.dtype == output.dtype,
        "Q, K, V, output should have same type.",
    ]()

    constrained[
        q.dtype is DType.float32 or q.dtype.is_half_float(),
        "Only support single and half precision.",
    ]()

    # Runtime dimensions.
    # For ragged inputs: [total_seq_len, num_heads, head_dim]
    # fmt: off
    alias head_depth_known = q.layout.shape.all_known[1, 3]() and k.layout.shape[1] != UNKNOWN_VALUE
    alias depth = Int(q.layout.shape[q.rank - 1])
    alias gpu_info = ctx.default_device_info
    alias head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
    alias flash_attention_applicable = flash_attention_hw_supported[type]() and head_depth_known and head_depth_supported and not naive_kernel
    alias kv_num_heads = Int(k.layout.shape[1])
    # fmt: on

    var is_token_generation = False

    var cache_row_offsets = input_row_offsets.to_layout_tensor().as_any_origin()

    var k_operand = RaggedMHAOperand(
        LayoutTensor[
            k.dtype, Layout.row_major(k.layout.shape), MutableAnyOrigin
        ](
            k.ptr,
            RuntimeLayout[Layout.row_major(k.layout.shape)].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        ),
        cache_row_offsets,
    )
    var v_operand = RaggedMHAOperand(
        LayoutTensor[
            v.dtype, Layout.row_major(v.layout.shape), MutableAnyOrigin
        ](
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
        input_row_offsets,
        Int(max_prompt_len[0]),
        Int(max_prompt_len[0]),
        scale,
        is_token_generation,
        ctx,
        None,
        num_partitions,
    )


# ===-----------------------------------------------------------------------===#
# Flash attention for context encoding
# ===-----------------------------------------------------------------------===#


# for depth = 128 we want waves_per_eu = 2 and for depth = 256 we want waves_per_eu = 1
# for depth = 64 we want waves_per_eu = 2
# this heuristic may not be valid for other depths
@__llvm_metadata(`rocdl.waves_per_eu`=min(Int(256 // config.depth), 2))
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    batch_size: Int,
    seq_len_arg: Int,
    num_keys_arg: Int,
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        MutableAnyOrigin,
    ],
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin
        ]
    ],
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
):
    alias depth = config.depth
    alias num_heads = config.num_heads
    var batch_idx = block_idx.z

    # mha inputs
    var seq_len: Int
    var max_seq_len = seq_len_arg
    var num_keys: Int
    var mask_tensor_col = num_keys_arg
    var start_pos: UInt32 = 0

    @parameter
    if ragged:
        # treat valid_lengths as a input_row_offsets
        start_of_seq = Int(valid_length[batch_idx])
        end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq

        if seq_len < block_idx.x * config.block_m():
            return

        @parameter
        if not _is_cache_length_accurate:
            start_pos = k.cache_length(batch_idx)

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

        q_batch_offset = start_of_seq * config.depth * config.num_heads

    # KVCache inputs, prompt lengths are all padded to the max in batch.
    elif _use_valid_length and not _padded_ndbuffer:
        # treat valid_lengths as valid lengths
        seq_len = Int(valid_length[batch_idx])

        if seq_len < block_idx.x * config.block_m():
            return

        @parameter
        if not _is_cache_length_accurate:
            var cache_length = k.cache_length(batch_idx)
            start_pos = cache_length

        num_keys = seq_len + k.cache_length(batch_idx)
        q_batch_offset = (
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

        if seq_len < block_idx.x * config.block_m():
            return
        q_batch_offset = (
            config.depth * config.num_heads * UInt(max_seq_len) * batch_idx
        )

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len

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
                q_ptr.offset(q_batch_offset),
                k,
                v,
                output_ptr.offset(q_batch_offset),
                scale,
                seq_len,
                max_seq_len,
                start_pos,
                num_keys,
                mask_tensor_col,
                mask,
                score_mod,
                batch_idx,
                sink_weights,
            )
        else:
            mha_single_batch[
                config=config,
                group=group,
                use_score_mod=use_score_mod,
                sink=sink,
            ](
                q_ptr.offset(q_batch_offset),
                k,
                v,
                output_ptr.offset(q_batch_offset),
                scale,
                seq_len,
                max_seq_len,
                start_pos,
                num_keys,
                mask_tensor_col,
                mask,
                score_mod,
                batch_idx,
                sink_weights,
            )
    elif is_amd_gpu():
        constrained[
            use_score_mod == False,
            "use_score_mod must be False for AMD flash attention",
        ]()

        @parameter
        if (
            _cdna_4_or_newer()
            and env_get_bool["USE_EXPERIMENTAL_CDNA4_MHA_KERNEL", False]()
        ):
            mha_single_batch_gfx950[group=group, config=config, sink=sink](
                output_ptr.offset(q_batch_offset),
                q_ptr.offset(q_batch_offset),
                k,
                v,
                seq_len,
                num_keys,
                scale,
                batch_idx,
                Int(start_pos),
                mask,
                sink_weights,
            )
        else:
            mha_single_batch_gfx942[group=group, config=config, sink=sink](
                output_ptr.offset(q_batch_offset),
                q_ptr.offset(q_batch_offset),
                k,
                v,
                seq_len,
                num_keys,
                scale,
                batch_idx,
                Int(start_pos),
                mask,
                sink_weights,
            )
    else:
        return CompilationTarget.unsupported_target_error[operation="mha"]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
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
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias accum_type = get_accum_type[q_type]()
    alias k_type = k_t.dtype
    alias v_type = v_t.dtype
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simd_width_of[q_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias num_heads = config.num_heads
    alias depth = config.depth

    constrained[
        num_warps_m * num_warps_n == UInt(num_threads // UInt(WARP_SIZE)),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = thread_idx.x
    var warp_id: UInt32 = warp.broadcast(tid // WARP_SIZE)
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    # The entire query block (BM x depth) is tiled in shared memory.
    alias alignment = align_of[SIMD[q_type, simd_size]]()
    alias q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    alias k_smem_size = config.k_smem_size()
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    var k_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, k_smem_size)

    alias v_smem_size = config.v_smem_size()
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    var v_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, BN),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](v_smem, v_smem_size)

    var head_idx: UInt32 = block_idx.y
    var q_tile_idx: UInt32 = block_idx.x

    # Query global memory iterator
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var q_tile_num_rows = min(BM, UInt(seq_len) - q_tile_idx * BM)
    var q_offset = depth * (head_idx + num_heads * q_tile_idx * BM)
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
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    alias mma_shape = get_mma_shape[q_type, accum_type]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias WM = config.WM
    alias WN = config.WN
    alias num_m_mmas = WM // UInt(MMA_M)
    alias num_n_mmas = WN // UInt(MMA_N)

    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2
    alias p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0)
    )

    # Rowwise max and sum for online softmax
    alias row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

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
    var p_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](p_smem, BM * BN)

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space = AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # Mask global memory iterator.
    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_y * WM
    var mask_warp_col = warp_x * WN

    # Account for group query.
    alias kv_num_heads = num_heads // UInt(group)

    alias num_pipeline_stages = config.num_pipeline_stages

    alias q_num_vecs = BM * BK // UInt(simd_size)

    alias async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * UInt(simd_size) // BK,
        BK // UInt(simd_size),
    )

    @parameter
    for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(q_id)[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
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
                    Int(q_tile_idx * BM + start_pos),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.FULL_MASK
        ):
            mask_warp_col += BN
            return

        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )
        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[kv_gmem_layout](
            {kv_tile_num_rows, depth},
            {kv_num_heads * depth, 1},
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        @always_inline
        @parameter
        fn _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: __type_of(tensor)
        ):
            return {
                tensor.ptr,
                __type_of(tensor.runtime_layout)(
                    __type_of(tensor.runtime_layout.shape)(
                        num_rows, tensor.dim[1]()
                    ),
                    tensor.runtime_layout.stride,
                ),
            }

        alias kv_num_vecs = BN * BK // UInt(simd_size)
        alias async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * UInt(simd_size)
            // UInt(k_smem_iter.layout.stride[0].value()),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        @parameter
        for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(k_id)[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
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
            static_num_iters = Int(depth // BK),
            k_group_size = config.k_group_size,
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            depth // BK,
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
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + m_mma * UInt(MMA_M)
                    var mask_frag_col = mask_warp_col + n_mma * UInt(MMA_N)

                    # Offset to current thread's fragment
                    mask_frag_row += lane // (MMA_N // p_frag_simdwidth)
                    mask_frag_col += lane * p_frag_simdwidth % MMA_N

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row + mask_frag_row + i * MMA_M // 2
                        )
                        var score_col = mask_frag_col

                        score_row_with_start_pos = score_row + start_pos

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32,](
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
                                    IndexList[4, element_type = DType.uint32,](
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
                    Int(q_tile_idx * BM + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        alias reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
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
            output_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        alias async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * UInt(simd_size)
            // UInt(v_smem_iter.layout.stride[0].value()),
            v_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load V tile into smem
        @parameter
        for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(v_id)[]

            @parameter
            if not not_last_iter:
                var num_rows_bound = min(
                    Int(BK), end - (kv_tile_start_row + v_id * BK)
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle = v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
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
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
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
                BN // BK,
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
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
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
                BN // BK,
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    alias output_gmem_layout = Layout(
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
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
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
                num_threads * UInt(simd_size) // depth, depth // UInt(simd_size)
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
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
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
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias accum_type = get_accum_type[q_type]()
    alias k_type = k_t.dtype
    alias v_type = v_t.dtype
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simd_width_of[q_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias num_heads = config.num_heads
    alias depth = config.depth

    constrained[
        num_warps_m * num_warps_n == UInt(num_threads // UInt(WARP_SIZE)),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = thread_idx.x
    var warp_id: UInt32 = warp.broadcast(tid // WARP_SIZE)
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    # The entire query block (BM x depth) is tiled in shared memory.
    alias alignment = align_of[SIMD[q_type, simd_size]]()
    alias q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    alias k_smem_size = config.kv_smem_size()
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    var k_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, k_smem_size)

    var head_idx: UInt32 = block_idx.y
    var q_tile_idx: UInt32 = block_idx.x

    # Query global memory iterator
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var q_tile_num_rows = min(BM, UInt(seq_len) - q_tile_idx * BM)
    var q_offset = depth * (head_idx + num_heads * q_tile_idx * BM)
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
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    alias mma_shape = get_mma_shape[q_type, accum_type]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias WM = config.WM
    alias WN = config.WN
    alias num_m_mmas = WM // UInt(MMA_M)
    alias num_n_mmas = WN // UInt(MMA_N)

    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2
    alias p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0)
    )

    # Rowwise max and sum for online softmax
    alias row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

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
    var p_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](p_smem, BM * BN)

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(p_frag_simdwidth * num_warps_n, BM),
        address_space = AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # Mask global memory iterator.
    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_y * WM
    var mask_warp_col = warp_x * WN

    # Account for group query.
    alias kv_num_heads = num_heads // UInt(group)

    alias num_pipeline_stages = config.num_pipeline_stages
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
                    Int(q_tile_idx * BM + start_pos), Int(kv_tile_start_row)
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.FULL_MASK
        ):
            mask_warp_col += BN
            return

        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )
        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type = DType.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type = DType.int32](
                kv_num_heads * depth, 1
            ),
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        var num_b_rows = None if not_last_iter else OptionalReg[Int](
            kv_tile_num_rows
        )

        # First iteration load q from global memory to shared memory.
        if is_first_iter:
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
                continue_prefetch_b=True,
                b_next_smem_layout = Layout.row_major(BK, BN),
                next_op_b_iter_masked = __type_of(v_gmem_iter).masked,
                next_op_b_layout_int_type = __type_of(
                    v_gmem_iter
                ).layout_int_type,
                next_op_b_linear_idx_type = __type_of(
                    v_gmem_iter
                ).linear_idx_type,
                k_group_size = config.k_group_size,
            ](
                p_reg_tile,
                q_gmem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                depth // BK,
                next_op_b_iter=v_gmem_iter.bitcast[k_type](),
                num_b_rows=num_b_rows,
            )
            is_first_iter = False
        # Subsequent iterations just use q in share memory.
        # TODO: Figure out a better function interface instead of passing in
        # shared memory iterator twice.
        else:
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
                continue_prefetch_b=True,
                b_next_smem_layout = Layout.row_major(BK, BN),
                next_op_b_iter_masked = __type_of(v_gmem_iter).masked,
                next_op_b_layout_int_type = __type_of(
                    v_gmem_iter
                ).layout_int_type,
                next_op_b_linear_idx_type = __type_of(
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
                depth // BK,
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
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + m_mma * UInt(MMA_M)
                    var mask_frag_col = mask_warp_col + n_mma * UInt(MMA_N)

                    mask_frag_row += lane // (MMA_N // p_frag_simdwidth)
                    mask_frag_col += lane * p_frag_simdwidth % MMA_N

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row + mask_frag_row + (i * MMA_M // 2)
                        )
                        var score_col = mask_frag_col

                        var score_row_with_start_pos = score_row + start_pos

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32,](
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
                                    IndexList[4, element_type = DType.uint32,](
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
                    Int(q_tile_idx * BM + start_pos), kv_tile_start_row
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        alias reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
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
            output_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, p_frag_simdwidth
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, p_frag_simdwidth
            ](),
            warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        # V reuse K's smem iterator. They has same smem footage expect for different layouts.
        var v_smem_iter = k_smem_iter.reshape[
            Layout.row_major(BK, BN)
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
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
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
                BN // BK,
                num_b_rows=num_b_rows,
            )
        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
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
                BN // BK,
                num_b_rows=num_b_rows,
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    alias output_gmem_layout = Layout(
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
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
            Int(warp_y), Int(warp_x)
        )

        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
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
                num_threads * UInt(simd_size) // depth, depth // UInt(simd_size)
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
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    scale: Float32,
    batch_size: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        MutableAnyOrigin,
    ],  # valid length per batch
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
):
    alias accum_type = get_accum_type[q_type]()
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
    var qk_max_batch_ptr = __type_of(qk_max_ptr)()
    if qk_max_ptr:
        qk_max_batch_ptr = qk_max_ptr.offset(qk_max_offset)

    var exp_sum_batch_ptr = __type_of(exp_sum_ptr)()
    if exp_sum_ptr:
        exp_sum_batch_ptr = exp_sum_ptr.offset(exp_sum_offset)

    var seq_len: Int
    var q_batch_offset: Int
    var start_pos: UInt32 = 0

    @parameter
    if ragged:
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
                q_ptr.offset(q_batch_offset),
                k,
                v,
                output_ptr.offset(output_batch_offset),
                exp_sum_batch_ptr,
                qk_max_batch_ptr,
                scale,
                UInt(num_keys),
                UInt(num_partitions),
                UInt(max_cache_valid_length),
                sink_weights,
                mask,
                score_mod,
                batch_idx,
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
                q_ptr.offset(q_batch_offset),
                k,
                v,
                output_ptr.offset(output_batch_offset),
                exp_sum_batch_ptr,
                qk_max_batch_ptr,
                scale,
                UInt(num_keys),
                UInt(num_partitions),
                UInt(max_cache_valid_length),
                mask,
                score_mod,
                batch_idx,
                sink_weights,
            )
    elif is_amd_gpu():
        alias config = MHAConfig(
            q_type,
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
        constrained[
            use_score_mod == False,
            "use_score_mod must be False for AMD flash attention",
        ]()
        var sink_weights_lt: OptionalReg[
            LayoutTensor[
                q_ptr.type.dtype,
                Layout.row_major(UNKNOWN_VALUE),
                MutableAnyOrigin,
            ]
        ] = None
        if sink_weights:
            sink_weights_lt = LayoutTensor[
                q_ptr.type.dtype,
                Layout.row_major(UNKNOWN_VALUE),
                MutableAnyOrigin,
            ](
                sink_weights.value().ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    IndexList[1](sink_weights.value().size())
                ),
            )
        mha_decoding_single_batch_gfx942[
            group=group,
            config=config,
            sink=sink,
        ](
            output_ptr.offset(output_batch_offset),
            q_ptr.offset(q_batch_offset),
            k,
            v,
            exp_sum_batch_ptr,
            qk_max_batch_ptr,
            1,
            num_keys,
            num_partitions,
            scale,
            batch_idx,
            Int(0),
            mask,
            sink_weights_lt,
        )
    else:
        return CompilationTarget.unsupported_target_error[
            operation="mha_decoding"
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
        p_type, p_layout, address_space = AddressSpace.LOCAL
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
    alias num_groups_per_thread = ceildiv(group, 8)

    @parameter
    for n_mma in range(num_n_mmas):
        # offset in fragment
        var frag_offset = n_mma * MMA_N
        # Current thread's offset mapped in num_keys dim
        var key_offset = warp_offset + UInt(frag_offset)
        # Current thread's index in current mma tile, e.g. T1 and T5 are 1 in 16x8 mma output.
        var frag_lane_col = Int((lane % 4) * UInt(simd_width))

        @parameter
        for i_group in range(num_groups_per_thread):
            group_idx = i_group * 8 + lane // 4
            q_head_idx = block_idx.y * UInt(group) + UInt(group_idx)

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
                        Int(score_col),
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
                                Int(score_col),
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
                        kv_tile_start_row + bound,
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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    scale: Float32,
    num_keys: UInt,
    num_partitions: UInt,
    max_cache_valid_length: UInt,  # longest KV cache entry
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
):
    """Flash attention v2 algorithm."""
    alias accum_type = get_accum_type[q_type]()
    alias k_type = k_t.dtype
    alias v_type = v_t.dtype
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simd_width_of[q_type]()

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    constrained[
        num_warps_m * num_warps_n == UInt(num_threads // UInt(WARP_SIZE)),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    # It's because in online-softmax we only use the top 8x4 sub-matrix
    # in the 16x8 mma output for Nvidia GPU. It shouldn't matter for AMD
    constrained[
        group <= 16,
        String(
            "Only support GQA with group <= 16 for Nvidia, but got a group = '",
            group,
            "'.",
        ),
    ]()

    var tid = thread_idx.x
    var warp_id = warp.broadcast(tid // UInt(WARP_SIZE))
    var lane = lane_id()

    # Coordinates of the current warp.
    var warp_y, warp_x = divmod(warp_id, UInt(num_warps_n))

    # The entire query block (BM x depth) is tiled in shared memory.
    alias alignment = align_of[SIMD[q_type, simd_size]]()
    alias q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )

    alias k_smem_size = BN * depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    var k_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, k_smem_size)

    alias v_smem_size = BN * BN
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    var v_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, BN),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](v_smem, v_smem_size)

    var kv_head_idx = block_idx.y
    var q_head_idx = kv_head_idx * group + thread_idx.x // 4
    var partition_idx = block_idx.x

    alias mma_shape = get_mma_shape[q_type, accum_type]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_m_mmas = WM // UInt(MMA_M)
    alias num_n_mmas = WN // UInt(MMA_N)

    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2
    alias p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    # Note that
    # num_warps_n * num_n_mmas == BN // WN * num_n_mmas
    # so we can use multistage_mma
    alias num_output_rows = num_m_mmas * num_n_mmas
    alias num_output_rows_full = num_warps_n * num_output_rows if decoding_warp_split_k else num_output_rows
    # alias num_output_rows = num_warps_n * num_m_mmas * num_n_mmas if decoding_warp_split_k else num_m_mmas * num_n_mmas
    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_output_rows_full, p_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0.0)
    )

    # Rowwise max and sum for online softmax
    alias row_align = align_of[SIMD[accum_type, simd_width_of[accum_type]()]]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_align]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_align]()

    @parameter
    for i in range(WM):

        @parameter
        if sink:
            debug_assert(
                Bool(sink_weights),
                "expect sink_weights to be non-null when sink=true",
            )
            if thread_idx.x < UInt(4 * group):
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
    alias p_smem_size = BM * BN
    var p_smem_iter = LayoutTensorIter[
        v_type, Layout.row_major(BM, BK), address_space = AddressSpace.SHARED
    ](p_smem, BM * BN)

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space = AddressSpace.SHARED,
    ]((p_smem + BM * BN).bitcast[Scalar[accum_type]]())

    # Mask global memory iterator
    var stride = max_cache_valid_length

    # Account for group query.
    alias kv_num_heads = num_heads // group

    var q_offset = depth * kv_head_idx * group

    alias q_gmem_layout = Layout.row_major(BM, depth)
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
                group, depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    start, end = get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
    )

    alias q_num_vecs = BM * BK // UInt(simd_size)

    alias async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * UInt(simd_size) // BK,
        BK // UInt(simd_size),
    )

    @always_inline
    @parameter
    fn _mask_tensor_row(
        tensor: LayoutTensor, num_rows: Int
    ) -> __type_of(tensor):
        return {
            tensor.ptr,
            {{num_rows, tensor.dim[1]()}, tensor.runtime_layout.stride},
        }

    @parameter
    for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(q_id)[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
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
        var k_ptr = k.block_paged_ptr[BN](
            batch_idx, kv_tile_start_row, kv_head_idx, 0
        )
        var k_gmem_block = LayoutTensor[
            k_type,
            Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            ),
            masked = not not_last_iter,
        ](k_ptr)
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var kv_tile_num_rows = min(Int(BN), end - kv_tile_start_row)

        _ = p_reg_tile.fill(0)

        alias kv_num_vecs = BN * BK // UInt(simd_size)
        alias async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * UInt(simd_size)
            // UInt(k_smem_iter.layout.stride[0].value()),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        @parameter
        for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(k_id)[]

            @parameter
            if not not_last_iter:
                k_tensor = _mask_tensor_row(k_gmem_iter[], kv_tile_num_rows)
            else:
                k_tensor = k_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_tensor.vectorize[1, simd_size](),
            )

            k_gmem_iter._incr()

        async_copy_commit_group()

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
            static_num_iters = Int(depth // BK),
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            depth // BK,
        )

        scale_and_mask_helper[
            num_n_mmas=num_n_mmas,
            WN=WN,
            MMA_N=MMA_N,
            simd_width=p_frag_simdwidth,
            use_score_mod=use_score_mod,
            group=group,
        ](
            p_reg_tile,
            scale_log2e,
            num_keys,
            UInt(kv_tile_num_rows),
            lane,
            UInt(warp_id),
            mask,
            score_mod,
            kv_tile_start_row,
            stride,
            max_cache_valid_length,
        )

        # For 16x8 mma output, group <= 8 only uses the first 8x8 matrix
        # each thread only has one fragment vector of size 2.
        @parameter
        if group <= 8:
            var output_reg_vecs = output_reg_tile.tile[
                num_output_rows_full, p_frag_size // 2
            ](0, 0).vectorize[1, p_frag_simdwidth]()
            var p_reg_vecs = p_reg_tile.tile[
                num_m_mmas * num_n_mmas, p_frag_size // 2
            ](0, 0).vectorize[1, p_frag_simdwidth]()

            _online_softmax_iter_for_mma_output[
                accum_type,
                Layout.row_major(num_m_mmas, num_n_mmas),
                Layout.row_major(num_warps_m, num_warps_n),
                Layout.row_major(8, 4),
                warp_split_k=decoding_warp_split_k,
                use_exp2=True,
            ](
                output_reg_vecs,
                p_reg_vecs,
                warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
                rowmax,
                rowsum,
            )
        else:
            var output_reg_vecs = output_reg_tile.reshape[
                Layout.row_major(2 * num_output_rows_full, p_frag_simdwidth)
            ]().vectorize[1, p_frag_simdwidth]()
            var p_reg_vecs = p_reg_tile.reshape[
                Layout.row_major(2 * num_m_mmas * num_n_mmas, p_frag_simdwidth)
            ]().vectorize[1, p_frag_simdwidth]()

            _online_softmax_iter_for_mma_output[
                accum_type,
                Layout.row_major(2 * num_m_mmas, num_n_mmas),
                Layout.row_major(num_warps_m, num_warps_n),
                Layout.row_major(8, 4),
                warp_split_k=decoding_warp_split_k,
                use_exp2=True,
            ](
                output_reg_vecs,
                p_reg_vecs,
                warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
                rowmax,
                rowsum,
            )

        var v_ptr = v.block_paged_ptr[BN](
            batch_idx, kv_tile_start_row, kv_head_idx, 0
        )
        var v_gmem_block = LayoutTensor[
            v_type,
            Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            ),
            masked = not not_last_iter,
        ](v_ptr)
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        alias async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs) * UInt(simd_size) // BN,
            BN // UInt(simd_size),
        )

        # load V tile into smem
        @parameter
        for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(v_id)[]

            @parameter
            if not not_last_iter:
                var num_rows_bound = max(
                    0, end - (kv_tile_start_row + v_id * BK)
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle = v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
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
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)

        async_copy_wait_all()
        barrier()

        # if decoding_warp_split_k:
        #   S[m, (0:WN) + n*WN] @ V[(0:WN) + n*WN, :]
        # else:
        #   S[m, :] @ V[:, (0:WN) + n*WN]
        @parameter
        if decoding_warp_split_k:
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)
            var v_smem_sub = LayoutTensorIter[
                v_type,
                Layout.row_major(WN, BN),
                address_space = AddressSpace.SHARED,
                circular=True,
            ](v_smem + BN * WN * warp_x, v_smem_size)
            multistage_mma[
                BM,
                BN,
                WN,  # BK
                WM,
                BN,  # WN
                num_threads,
                num_pipeline_stages,
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
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
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
                BN // BK,
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](start, end)

    @parameter
    if decoding_warp_split_k:
        var output_reg_vecs = output_reg_tile.tile[
            num_warps_n * num_m_mmas * num_n_mmas, p_frag_size // 2
        ](0, 0).vectorize[1, p_frag_size // 2]()
        # offset on the pointer is to avoid possible races
        # with `accum_smem_warp_tile`.
        var o_smem_ptr = q_smem.bitcast[Scalar[accum_type]]()
        var scratch = LayoutTensor[
            accum_type,
            Layout.row_major(2 * num_warps_n, BM),
            address_space = AddressSpace.SHARED,
        ](o_smem_ptr + num_warps_n * (num_warps_n - 1) * WM * WN)

        # Note: Sink handling is done after warp reduction in partition-specific logic below.
        # The warp reduction just combines warps; sink contribution is added to rowsum later.
        _online_softmax_iter_for_mma_output_split_warp_reduce[
            accum_type,
            Layout.row_major(num_m_mmas, num_n_mmas),
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            WM,
            WN,
            use_exp2=True,
        ](
            output_reg_vecs,
            scratch.tile[2 * num_warps_n, WM](0, Int(warp_y)),
            o_smem_ptr,
            rowmax,
            rowsum,
        )

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        if m_mma * UInt(MMA_M) < group:
            var rowsum_inv = Scalar[accum_type](recip(rowsum[2 * m_mma]))

            @parameter
            for n_mma in range(num_n_mmas):
                output_reg_tile[n_mma * num_m_mmas + m_mma, 0] *= rowsum_inv
                output_reg_tile[n_mma * num_m_mmas + m_mma, 1] *= rowsum_inv

        @parameter
        if m_mma * UInt(MMA_M) + UInt(MMA_M // 2) < group:
            var rowsum_inv = Scalar[accum_type](recip(rowsum[2 * m_mma + 1]))

            @parameter
            for n_mma in range(num_n_mmas):
                output_reg_tile[n_mma * num_m_mmas + m_mma, 2] *= rowsum_inv
                output_reg_tile[n_mma * num_m_mmas + m_mma, 3] *= rowsum_inv

    if num_partitions > 1:
        if thread_idx.x % 4 == 0 and thread_idx.x < UInt(4 * group):
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
        Layout.row_major(WM, WN),
        address_space = AddressSpace.SHARED,
    ](accum_smem_warp_ptr)

    alias swizzle = make_swizzle[
        num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
    ]()

    @parameter
    if decoding_warp_split_k:
        copy_local_to_shared[
            thread_layout = Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.tile[num_output_rows, p_frag_size](0, 0)
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
    alias output_gmem_layout = Layout.row_major(BM, depth)
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
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )

    copy_sram_to_dram[
        thread_layout = Layout.row_major(
            WARP_SIZE * simd_size // WN, WN // UInt(simd_size)
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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    scale: Float32,
    num_keys: UInt,
    num_partitions: UInt,
    max_cache_valid_length: UInt,  # longest KV cache entry
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
):
    """Flash attention v2 algorithm."""
    alias accum_type = get_accum_type[q_type]()
    alias k_type = k_t.dtype
    alias v_type = v_t.dtype
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simd_width_of[q_type]()

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    constrained[
        num_warps_m * num_warps_n == UInt(num_threads // UInt(WARP_SIZE)),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    constrained[
        group <= 8,
        String(
            "Only support GQA with group <= 8 for Nvidia, but got a group = '",
            group,
            "'.",
        ),
    ]()

    var tid = thread_idx.x
    var warp_id = warp.broadcast(tid // UInt(WARP_SIZE))
    var lane = lane_id()

    # Coordinates of the current warp.
    warp_y, warp_x = divmod(warp_id, UInt(num_warps_n))

    # The entire query block (BM x depth) is tiled in shared memory.
    alias alignment = align_of[SIMD[q_type, simd_size]]()
    alias q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )

    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    alias k_smem_size = num_pipeline_stages * BN * BK
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    var k_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, k_smem_size)

    var kv_head_idx = block_idx.y

    alias mma_shape = get_mma_shape[q_type, accum_type]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_m_mmas = WM // UInt(MMA_M)
    alias num_n_mmas = WN // UInt(MMA_N)

    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2
    alias p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0.0)
    )

    # Account for group query.
    alias kv_num_heads = num_heads // group
    var q_head_idx = kv_head_idx * group + thread_idx.x // 4

    # Rowwise max and sum for online softmax
    alias row_align = align_of[SIMD[accum_type, simd_width_of[accum_type]()]]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_align]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_align]()

    var partition_idx = block_idx.x

    @parameter
    for i in range(WM):

        @parameter
        if sink:
            debug_assert(
                Bool(sink_weights),
                "expect sink_weights to be non-null when sink=true",
            )
            if thread_idx.x < UInt(4 * group):
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
    alias v_smem_size = num_pipeline_stages * BN * BK
    var v_smem = k_smem.bitcast[Scalar[v_type]]()
    var v_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, BN),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ](v_smem, v_smem_size)

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    alias p_smem_size = BM * BN
    var p_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](p_smem, BM * BN)

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(p_frag_simdwidth * num_warps_n, BM),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]((p_smem + BM * BN).bitcast[Scalar[accum_type]]())

    # Mask global memory iterator, seq_len = 1
    var stride = max_cache_valid_length

    var q_offset = depth * kv_head_idx * group

    alias q_gmem_layout = Layout.row_major(BM, depth)
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
                group, depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    # Loop over Key and Value tiles
    start, end = get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
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
        var k_ptr = k.block_paged_ptr[BN](
            batch_idx, kv_tile_start_row, kv_head_idx, 0
        )
        var k_gmem_block = LayoutTensor[
            k_type,
            Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            ),
            masked = not not_last_iter,
        ](k_ptr)
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var kv_tile_num_rows = min(Int(BN), end - kv_tile_start_row)

        _ = p_reg_tile.fill(0)

        if kv_tile_start_row == start:
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
            ](
                p_reg_tile,
                q_gmem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                depth // BK,
                num_b_rows=Int(kv_tile_num_rows),
            )
        else:
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
            ](
                p_reg_tile,
                q_smem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                depth // BK,
                num_b_rows=Int(kv_tile_num_rows),
            )

        scale_and_mask_helper[
            num_n_mmas=num_n_mmas,
            WN=WN,
            MMA_N=MMA_N,
            simd_width=p_frag_simdwidth,
            use_score_mod=use_score_mod,
            group=group,
        ](
            p_reg_tile,
            scale_log2e,
            num_keys,
            UInt(kv_tile_num_rows),
            lane,
            UInt(warp_id),
            mask,
            score_mod,
            kv_tile_start_row,
            stride,
            max_cache_valid_length,
        )

        # For 16x8 mma output, only the top 8x4 matrix matters for GQA since
        # G <= 8 typically holds
        var output_reg_vecs = output_reg_tile.tile[
            num_m_mmas * num_n_mmas, p_frag_size // 2
        ](0, 0).vectorize[1, p_frag_size // 2]()
        var p_reg_vecs = p_reg_tile.tile[
            num_m_mmas * num_n_mmas, p_frag_size // 2
        ](0, 0).vectorize[1, p_frag_size // 2]()

        _online_softmax_iter_for_mma_output[
            accum_type,
            Layout.row_major(num_m_mmas, num_n_mmas),
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_vecs,
            p_reg_vecs,
            warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        var v_ptr = v.block_paged_ptr[BN](
            batch_idx, kv_tile_start_row, kv_head_idx, 0
        )
        var v_gmem_block = LayoutTensor[
            v_type,
            Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            ),
            masked = not not_last_iter,
        ](v_ptr)
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # Copy score fragments to shared memory with swizzling to resolve bank
        # conflicts for ldmatrix in the 2nd matmul.
        _copy_frag_to_smem[
            BM, BN, BK, WM, WN, UInt(MMA_M), UInt(MMA_N), UInt(p_frag_simdwidth)
        ](p_smem_iter, p_reg_tile, warp_x, warp_y)
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            False,  # transpose_b
            swizzle_a=True,
        ](
            output_reg_tile,
            p_smem_iter,
            v_gmem_iter,
            p_smem_iter,
            v_smem_iter,
            BN // BK,
            num_b_rows=Int(kv_tile_num_rows),
        )

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](start, end)

    # Apply softmax denumerator.

    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = 1.0 / rowsum[2 * m_mma]

        @parameter
        for n_mma in range(num_n_mmas):
            output_reg_tile[n_mma, 0] *= rowsum_inv0
            output_reg_tile[n_mma, 1] *= rowsum_inv0

    if num_partitions > 1:
        if thread_idx.x % 4 == 0 and thread_idx.x < UInt(4 * group):
            var row_sum = rowsum[0]
            var row_max = rowmax[0]
            var q_head_idx = kv_head_idx * group + thread_idx.x // 4
            exp_sum_ptr[q_head_idx] = row_sum
            qk_max_ptr[q_head_idx] = row_max

    # Pack results in shared memory for wider simd width.
    var accum_smem_warp_tile = LayoutTensor[
        output_type,
        Layout.row_major(WM, WN),
        address_space = AddressSpace.SHARED,
    ](q_smem.bitcast[Scalar[output_type]]() + warp_id * WM * WN)

    alias swizzle = make_swizzle[
        num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
    ]()
    copy_local_to_shared[
        thread_layout = Layout.row_major(8, 4), swizzle=swizzle
    ](
        accum_smem_warp_tile.vectorize[1, 2](),
        output_reg_tile.vectorize[1, 2]().transpose(),
    )
    # Guard writing to shared memory.
    barrier()
    alias output_gmem_layout = Layout.row_major(BM, depth)
    var output_gmem_runtime_layout = RuntimeLayout[
        element_type = DType.int32, linear_idx_type = DType.int32
    ](
        RuntimeTuple[output_gmem_layout.shape, element_type = DType.int32](
            group, depth
        ),
        RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
            depth, 1
        ),
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        Layout.row_major(BM, depth),
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](output_ptr + q_offset, output_gmem_runtime_layout)
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )
    copy_sram_to_dram[
        thread_layout = Layout.row_major(
            WARP_SIZE * simd_size // WN, WN // UInt(simd_size)
        ),
        swizzle=swizzle,
    ](
        output_gmem_warp_tile.vectorize[1, simd_size](),
        accum_smem_warp_tile.vectorize[1, simd_size](),
    )


fn mha_splitk_reduce[
    output_type: DType,
    depth: UInt,
    num_heads: UInt,
    num_threads: UInt,
    group: UInt = 1,
    use_exp2: Bool = False,
](
    intermediate_ptr: UnsafePointer[Scalar[output_type]],
    output_ptr: UnsafePointer[Scalar[output_type]],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[output_type]()]],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[output_type]()]],
    batch_size: Int,
    num_partitions: Int,
):
    # we only reduce over a warp so limit number of warps to 1
    constrained[
        num_threads == UInt(WARP_SIZE),
        "num_threads: "
        + String(num_threads)
        + " should be equal to the warp_size:"
        + String(WARP_SIZE),
    ]()
    debug_assert(
        block_dim.x == UInt(WARP_SIZE),
        "block_dim.x should be equal to the warp_size",
    )

    alias accum_type = get_accum_type[output_type]()
    var batch_idx = block_idx.z
    var q_head_idx = block_idx.y

    debug_assert(
        num_partitions <= WARP_SIZE,
        "number of partitions should be less than or equal to the warp_size",
    )
    var partition_idx = lane_id()
    var l = min_or_neg_inf[accum_type]()
    if partition_idx < UInt(num_partitions):
        var qk_max_offset = (
            num_heads * batch_idx
            + num_heads * UInt(batch_size) * partition_idx
            + q_head_idx
        )
        l = qk_max_ptr[qk_max_offset]

    # TODO: use warp.lane_group_max since partition is going to be much smaller than WARP_SIZE
    var qk_max = warp.shuffle_idx(warp.max(l), 0)

    # since num_partitions <= WARP_SIZE, allocate buffer using WARP_SIZE
    var exp_sums = LayoutTensor[
        accum_type,
        Layout(WARP_SIZE),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    alias intermediate_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    var intermediate_output = LayoutTensor[output_type, intermediate_layout](
        intermediate_ptr,
        RuntimeLayout[intermediate_layout].row_major(
            Index(num_partitions, batch_size, num_heads, depth)
        ),
    )
    alias output_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, depth)
    var output = LayoutTensor[output_type, output_layout](
        output_ptr,
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, num_heads, depth)
        ),
    )

    var rescaled_exp_sum: Scalar[accum_type] = 0
    alias exp_fn = _exp2_concrete if use_exp2 else _exp_concrete
    if partition_idx < UInt(num_partitions):
        var qk_max_offset = (
            num_heads * batch_idx
            + num_heads * UInt(batch_size) * partition_idx
            + q_head_idx
        )
        rescaled_exp_sum = exp_sum_ptr[qk_max_offset] * exp_fn(l - qk_max)
        exp_sums[partition_idx] = rescaled_exp_sum

    # ensure exp_sums is written to before reading
    barrier()

    # TODO: use warp.lane_group_sum since partition is going to be much smaller than WARP_SIZE
    var exp_sum = warp.shuffle_idx(warp.sum(rescaled_exp_sum), 0)

    var inv_global_exp_sum = 1.0 / exp_sum
    # TODO: vectorize load and store operations
    alias width = Int(ceildiv(depth, num_threads))
    acc = (
        LayoutTensor[
            accum_type,
            Layout(width),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )
    for partition_idx in range(num_partitions):
        var partition_exp_sum = exp_sums[partition_idx]
        if partition_exp_sum > 0:

            @parameter
            for w in range(width):
                d = thread_idx.x + UInt(w * num_threads)
                if d < depth:
                    var x = (
                        intermediate_output[
                            partition_idx, batch_idx, q_head_idx, d
                        ].cast[accum_type]()
                        * inv_global_exp_sum
                        * partition_exp_sum
                    )
                    acc[w] += x[0]

    @parameter
    for w in range(width):
        d = thread_idx.x + UInt(w * num_threads)
        if d < depth:
            output[batch_idx, q_head_idx, d] = acc[w].cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead attention supporting flexible dimensions and
# batch_size > 1.
# ===-----------------------------------------------------------------------===#

alias _NAIVE_BMM_BLOCK_DIM = LaunchDim(32, 16, 1)
alias _NAIVE_BMM_BLOCK_TUPLE = StaticTuple[Int32, 1](
    _NAIVE_BMM_BLOCK_DIM.x()
    * _NAIVE_BMM_BLOCK_DIM.y()
    * _NAIVE_BMM_BLOCK_DIM.z()
)


fn mha_gpu_naive[
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask, //,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    q: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    output: LayoutTensor[
        output_type, address_space = AddressSpace.GENERIC, **_
    ],
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q.dtype, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ] = None,
) raises:
    alias q_type = q.dtype
    alias k_type = k_t.dtype
    alias v_type = k_type

    var num_keys = max_cache_size

    if batch_size == 0 or num_keys == 0 or max_prompt_len == 0:
        return

    alias p_type = get_accum_type[q_type]()
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
    alias kernel = _bmm0_bs[
        q_type,
        k_t,
        mask_t,
        p_type,
        __type_of(valid_length.to_layout_tensor()).layout,
        ragged=ragged,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
    ]

    ctx.enqueue_function_checked[kernel, kernel](
        p_device,
        q_device,
        k,
        valid_length.to_layout_tensor(),
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
    alias kernel_1 = _bmm1_bs[
        output_type,
        p_type,
        v_t,
        __type_of(valid_length.to_layout_tensor()).layout,
        ragged=ragged,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
    ]
    ctx.enqueue_function_checked[kernel_1, kernel_1](
        output_device,
        p_device,
        v,
        valid_length.to_layout_tensor(),
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
    p_ptr: UnsafePointer[Scalar[p_type]],
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        MutableAnyOrigin,
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

    alias k_type = k_t.dtype

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
            start_pos = k.cache_length(batch)

        seq_start = Int(valid_length[batch])
        seq_end = Int(valid_length[batch + 1])
        cur_query_len = seq_end - seq_start
        q_offset = Int((seq_start * num_heads + head) * depth)
        cur_cache_len = Int(start_pos) + cur_query_len
    elif _use_valid_length:
        cur_query_len = Int(valid_length[batch])
        q_offset = Int(
            depth * (head + UInt(num_heads * max_prompt_len * batch))
        )
        cur_cache_len = k.cache_length(batch) + cur_query_len
    # When inputs are all NDBuffers i.e. all sequences in batch have the same
    # length and same cache length
    else:
        cur_query_len = max_prompt_len
        q_offset = Int(
            depth * (head + UInt(num_heads * max_prompt_len * batch))
        )
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
        var k_ptr = k.block_paged_ptr[1](batch, x, kv_head, 0)

        # TODO: The AMD-specific path is to handle Llama shapes, similar
        #       to how things were before #53433. Once flash attention is
        #       supported on AMD, this stopgap AMD path should be eliminated to
        #       function as a generic fall-back (i.e., without vectorization).
        #       REL: KERN-1343.
        @parameter
        if is_amd_gpu():
            var accum_vec = SIMD[p_type, simd_width_of[p_type]()](0)

            @parameter
            fn accum_fn[width: Int](offset: Int):
                alias alignment = align_of[SIMD[p_type, width]]()
                var q_val = q.load[width=width, alignment=alignment](
                    y * UInt(num_heads) * UInt(depth) + UInt(offset)
                ).cast[k_type]()
                var k_val = k_ptr.load[width=width, alignment=alignment](offset)
                var qk_val = (q_val * k_val).cast[p_type]()

                @parameter
                if width == 1:
                    accum += rebind[__type_of(accum)](qk_val)
                else:
                    accum_vec += rebind[__type_of(accum_vec)](qk_val)

            vectorize[accum_fn, simd_width_of[p_type]()](depth)
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
    output_ptr: UnsafePointer[Scalar[output_type]],
    p_ptr: UnsafePointer[Scalar[p_type]],
    v: v_t,
    valid_length: LayoutTensor[
        DType.uint32,
        valid_length_layout,
        MutableAnyOrigin,
    ],
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
):
    alias v_type = v_t.dtype

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
            start_pos = v.cache_length(batch)

        seq_start = Int(valid_length[batch])
        seq_end = Int(valid_length[batch + 1])
        cur_query_len = seq_end - seq_start
        output_offset = Int((seq_start * num_heads + head) * depth)
        cur_cache_len = cur_query_len + Int(start_pos)
    elif _use_valid_length:
        cur_query_len = Int(valid_length[batch])
        output_offset = depth * (
            head + UInt(num_heads * max_prompt_len * batch)
        )
        cur_cache_len = cur_query_len + v.cache_length(batch)
    # When inputs are all NDBuffers i.e. all sequences in batch have the same
    # length and same cache length
    else:
        cur_query_len = max_prompt_len
        output_offset = depth * (
            head + UInt(num_heads * max_prompt_len * batch)
        )
        cur_cache_len = max_cache_size
        p_offset = batch_head * UInt(max_prompt_len) * UInt(max_cache_size)

    debug_assert(cur_query_len <= max_prompt_len, "Invalid cur_query_len")

    if x >= UInt(depth) or y >= UInt(cur_query_len):
        return

    var p = p_ptr + p_offset

    var kv_head = Int(head // UInt(group))
    var output = output_ptr + Int(output_offset)

    var accum = Float32(0.0)

    for i in range(cur_cache_len):
        var v_ptr = v.block_paged_ptr[1](batch, i, kv_head, x)
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
    mask_type: DType, //,
    sink: Bool = False,
](
    q: LayoutTensor[q_type, address_space = AddressSpace.GENERIC, **_],
    k: LayoutTensor[k_type, address_space = AddressSpace.GENERIC, **_],
    v: LayoutTensor[v_type, address_space = AddressSpace.GENERIC, **_],
    mask: LayoutTensor[mask_type, address_space = AddressSpace.GENERIC, **_],
    output: LayoutTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, **_
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
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ] = None,
) raises:
    var k_operand = LayoutTensorMHAOperand(
        LayoutTensor[
            k.dtype, Layout.row_major(k.layout.shape), MutableAnyOrigin
        ](
            k.ptr,
            RuntimeLayout[Layout.row_major(k.layout.shape)].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        )
    )
    var v_operand = LayoutTensorMHAOperand(
        LayoutTensor[
            v.dtype, Layout.row_major(v.layout.shape), MutableAnyOrigin
        ](
            v.ptr,
            RuntimeLayout[Layout.row_major(v.layout.shape)].row_major(
                v.runtime_layout.shape.value.canonicalize()
            ),
        )
    )
    var null_valid_length = ManagedTensorSlice[
        IOUnknown,
        static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
    ](UnsafePointer[UInt32](), IndexList[1](0), IndexList[1](0))

    mha_gpu_naive[_is_cache_length_accurate=True, sink=sink](
        q,
        k_operand,
        v_operand,
        MaterializedMask(
            LayoutTensor[
                mask_type,
                Layout.row_major(mask.layout.shape),
                MutableAnyOrigin,
            ](
                mask.ptr,
                RuntimeLayout[Layout.row_major(mask.layout.shape)].row_major(
                    mask.runtime_layout.shape.value.canonicalize()
                ),
            )
        ),
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
    mask_t: MHAMask, //,
    ragged: Bool = False,
    sink: Bool = False,
](
    q: LayoutTensor[q_type, address_space = AddressSpace.GENERIC, **_],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    output: LayoutTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, **_
    ],
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
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
        mut=True, dtype, address_space = AddressSpace.GENERIC, **_
    ],
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    k: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    v: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    mask: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
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
    alias simd_size = simd_width_of[dtype]()

    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()
    var num_heads = q.dim[2]()
    var depth = q.dim[3]()

    # Q, K, V transposed
    var qt_ptr = UnsafePointer[Scalar[dtype]].alloc(q.size())
    var kt_ptr = UnsafePointer[Scalar[dtype]].alloc(k.size())
    var vt_ptr = UnsafePointer[Scalar[dtype]].alloc(v.size())
    # Score = softmax(Q * K)
    var score_size = batch_size * num_heads * seq_len * num_keys
    var score_ptr = UnsafePointer[Scalar[dtype]].alloc(score_size)
    # O = Score * V. It's transposed and will be transposed back to output.
    var ot_ptr = UnsafePointer[Scalar[dtype]].alloc(output.size())

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

    alias layout_4d = Layout.row_major[4]()
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
        mut=True, dtype, address_space = AddressSpace.GENERIC, **_
    ],
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    k: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    v: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    mask: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    scale: Float32,
) raises:
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    """
    alias simd_size = simd_width_of[dtype]()

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var seq_len = q.dim[2]()
    var num_keys = v.dim[2]()

    # Allocate intermediate memory buffer.
    var score_size = batch_size * num_heads * seq_len * num_keys
    var score_ptr = UnsafePointer[Scalar[dtype]].alloc(score_size)
    var score = NDBuffer[dtype, 4](
        score_ptr, Index(batch_size, num_heads, seq_len, num_keys)
    )
    alias layout_4d = Layout.row_major[4]()
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
