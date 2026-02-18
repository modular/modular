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

from collections import OptionalReg
from math import align_up, ceildiv, recip
from nn.mha_utils import DynamicInt
from math.constants import log2e
from sys import (
    align_of,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    simd_width_of,
    size_of,
    is_nvidia_gpu,
    is_amd_gpu,
    env_get_int,
    CompilationTarget,
)

from nn.mha import q_num_matrix_view_rows
import gpu.primitives.warp as warp
from algorithm.functional import (
    _elementwise_impl_gpu,
    tile_and_unswitch,
    unswitch,
    vectorize,
)
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
from gpu.host import (
    DeviceContext,
    FuncAttribute,
    get_gpu_target,
    DeviceBuffer,
    Dim as LaunchDim,
)
from gpu.host.info import A100, H100, B200
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from kv_cache.types import KVCacheT
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    ThreadScope,
    copy_dram_to_local,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from memory import stack_allocation
from nn._ragged_utils import get_batch_from_row_offsets
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import (
    KVCacheMHAOperand,
    MHAOperand,
    LayoutTensorMHAOperand,
    RaggedMHAOperand,
)
from nn.mha_score_mod import ScoreModTrait
from nn.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    _copy_frag_to_smem,
    _kernel_mask,
    DynamicInt,
)
from nn.softmax import _exp2_concrete
from nn.mha_fa3_utils import NonNullPointer, NullPointer
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple

from .mha_utils import get_start_and_end_for_partitions
from .softmax import _online_softmax_iter_for_mma_output
from .attention.gpu.amd.mla import Attention, MLAAttentionConfig
from .mla_prefill_sm100 import mla_sm100_prefill
from gpu.host.info import B200, GPUInfo
from nn.mla_decode_sm100_dispatch import mla_decode_sm100_dispatch


# ===-----------------------------------------------------------------------===#
# GPU Multi-head Latent Attention (MLA) decoding implementations
# ===-----------------------------------------------------------------------===#


# entrypoint for MLA decoding kernels
@always_inline
fn flare_mla_decoding[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    q_layout: Layout,
    //,
    use_score_mod: Bool = False,
    config: MHAConfig[dtype] = {
        UInt(Int(q_layout.shape[rank - 2])),
        UInt(Int(q_layout.shape[rank - 1])),
    },
    ragged: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    output: LayoutTensor[
        mut=True, _, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, ...],
    k: cache_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
    num_partitions: Optional[Int] = None,
) raises:
    """MLA decoding kernel that would only be called in the optimized compute
    graph.

    The Q input has a shape of [seq_len, num_heads, depth].
    The K input has a shape of [seq_len, 1, depth].
    The V tensor is derived by reusing K, where V = K[:, :, :depth_v].

    Specifically, for DeepSeek V2/3, depth = 576 and depth_v = 512.

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
    comptime assert (
        q.dtype == output.dtype
    ), "Q, K, V, output should have same type."

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
        "flare_mla_decoding",
        Trace[
            TraceLevel.OP, target = ctx.default_device_info.api
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

        flare_mla_decoding_dispatch[
            kv_num_heads = Int(kv_num_heads),
            use_score_mod=use_score_mod,
            config=config,
            ragged=ragged,
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            output,
            q,
            k_operand,
            mask_functor,
            score_mod_functor,
            valid_length,
            max_prompt_len,
            num_keys,
            scale,
            ctx,
            kv_input_row_offsets,
            num_partitions,
        )


# entrypoint for LayoutTensor[mut=True, , Layout.row_major[3](), MutAnyOrigin]as K input, used by tests.
fn flare_mla_decoding[
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
](
    output: LayoutTensor[
        mut=True, _, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, ...],
    k: LayoutTensor[address_space = AddressSpace.GENERIC, ...],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: Optional[Int] = None,
) raises:
    comptime assert q.rank == 4, "only support rank 4 inputs."

    comptime kv_num_heads = Int(k.layout.shape[2])

    # Runtime dimensions.
    var num_keys = k.dim[1]()

    var k_operand = LayoutTensorMHAOperand(
        LayoutTensor[k.dtype, k.layout, MutAnyOrigin](
            k.ptr,
            RuntimeLayout[k.layout].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        )
    )

    var valid_length = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        UnsafePointer[UInt32, MutExternalOrigin](),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    flare_mla_decoding_dispatch[
        kv_num_heads=kv_num_heads,
        use_score_mod=use_score_mod,
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
        score_mod_functor,
        valid_length,
        q.dim[1](),
        num_keys,
        scale,
        ctx,
        None,
        num_partitions,
    )


@always_inline
fn flare_mla_decoding_dispatch[
    k_t: MHAOperand,
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
    # Work arounds to unify KVCache and LayoutTensor[mut=True, , Layout.row_major[3](), MutAnyOrigin]inputs:
    # Differentiate two cases, KV cache's length is before adding the latest
    # tokens e.g. zero for CE, and KV LayoutTensor's length is the latest length
    # e.g. prompt length for CE.
    _is_cache_length_accurate: Bool = False,
    # valid_length is needed for KV cache inputs and is empty for LayoutTensor[mut=True, , Layout.row_major[3](), MutAnyOrigin]inputs
    # to avoid overhead in benchmark.
    _use_valid_length: Bool = True,
    decoding_warp_split_k: Bool = False,
](
    output: LayoutTensor[address_space = AddressSpace.GENERIC, ...],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, ...],
    k: k_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    max_prompt_len: Int,
    max_cache_valid_length: Int,
    scale: Float32,
    ctx: DeviceContext,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
    num_partitions: Optional[Int] = None,
) raises:
    comptime num_heads = config.num_heads
    comptime depth = config.depth
    comptime group = config.num_heads // UInt(kv_num_heads)
    comptime assert num_heads == UInt(Int(q.layout.shape[q.rank - 2]))

    # only A100 or H100 have the enough smem to store the full BM * head_dim Q tensor.
    comptime has_enough_smem = ctx.default_device_info == A100 or ctx.default_device_info == H100

    comptime assert (
        depth == UInt(Int(q.layout.shape[q.rank - 1])) == 576
    ), "flareMLA_decoding only supports head_dim == 576."
    comptime assert (
        kv_num_heads == 1
    ), "flareMLA_decoding only supports kv_num_heads == 1."
    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "flareMLA_decoding currently only supports Nvidia and AMD GPUs."

    comptime assert q.dtype.is_half_float(), "Only support half precision."

    # Whether head and depth are static. With BSHD, B and S are dynamic.
    # H and D are always known for opaque KVCache types, we only check Q.
    comptime assert q.layout.shape.all_known[
        q.rank - 2, q.rank
    ](), "Need num_heads and head_dim to be static for Q."

    var batch_size: Int

    @parameter
    if ragged:
        batch_size = valid_length.dim[0]() - 1
    # This branch holds for both KVCache and LayoutTensor[mut=True, , Layout.row_major[3](), MutAnyOrigin]inputs.
    # Q is BSHD, S is either homogeneous or padded to same length.
    else:
        batch_size = q.dim[0]()

    if batch_size == 0:
        return

    @parameter
    if ctx.default_device_info == B200:
        mla_decode_sm100_dispatch[
            q.dtype,
            q.layout,
            k_t,
            output.dtype,
            output.layout,
            mask_t,
            score_mod_t,
            valid_length.layout,
            config=config,
            depth = Int(depth),
            num_heads = Int(num_heads),
            group = Int(group),
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            q,
            k,
            output,
            scale,
            batch_size,
            max_cache_valid_length,
            max_prompt_len,
            valid_length,
            mask_functor,
            score_mod_functor,
            ctx,
        )

    else:
        # only A100 or H100 have the enough smem to store the full BM * head_dim Q tensor.
        comptime has_enough_smem = ctx.default_device_info == A100 or ctx.default_device_info == H100

        comptime preferred_BM = 16 if (
            not has_enough_smem or has_amd_gpu_accelerator()
        ) else 32
        comptime BM = preferred_BM if UInt(preferred_BM) <= num_heads else Int(
            num_heads
        )
        comptime BN = 64 if has_nvidia_gpu_accelerator() else 128
        comptime BK = 64 if has_nvidia_gpu_accelerator() else 32  # need 8 mma_tile per row to resolve the bank conflict on nvidia
        comptime WM = BM
        comptime WN = 16 if has_nvidia_gpu_accelerator() else 32
        # num warps in M and N, multiplied by warp size.
        comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

        comptime accum_type = get_accum_type[q.dtype]()
        comptime num_pipeline_stages = 6
        # smem for q
        var shared_mem_bytes = BM * Int(depth) * size_of[q.dtype]()

        shared_mem_bytes += BN * Int(depth) * size_of[k_t.dtype]()

        comptime num_warps = ceildiv(num_threads, WARP_SIZE)

        # smem for p and warp_scratch
        shared_mem_bytes += (
            BM * BN * size_of[k_t.dtype]()
            + 2 * num_warps * BM * size_of[accum_type]()
        )

        shared_mem_bytes = (
            shared_mem_bytes if has_nvidia_gpu_accelerator() else 0
        )

        comptime num_blocks_y = num_heads // UInt(BM)

        comptime kernel = mla_decoding[
            q.dtype,
            k_t,
            output.dtype,
            mask_t,
            score_mod_t,
            valid_length.layout,
            BM = UInt(BM),
            BN = UInt(BN),
            BK = UInt(BK),
            WM = UInt(WM),
            WN = UInt(WN),
            depth=depth,
            num_heads=num_heads,
            num_threads = UInt(num_threads),
            num_pipeline_stages = UInt(num_pipeline_stages),
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _use_valid_length=_use_valid_length,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=decoding_warp_split_k,
        ]

        comptime nullptr = UnsafePointer[
            Scalar[accum_type], MutExternalOrigin
        ]()

        var num_partitions_value: Int = 1
        var q_device = DeviceBuffer[q.dtype](ctx, q.ptr, q.size(), owning=False)
        var output_device = DeviceBuffer[output.dtype](
            ctx, output.ptr, output.size(), owning=False
        )
        var nullptr_device = DeviceBuffer[accum_type](
            ctx, nullptr, 0, owning=False
        )

        ctx.enqueue_function[kernel, kernel](
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
            score_mod_functor,
            grid_dim=(1, Int(num_blocks_y), batch_size),
            block_dim=(num_threads, 1, 1),
            shared_mem_bytes=shared_mem_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(
                    ctx.default_device_info.shared_memory_per_multiprocessor
                    - 4096
                )
            ),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn mla_decoding[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    valid_layout: Layout,
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
    valid_length: LayoutTensor[
        DType.uint32,
        valid_layout,
        MutAnyOrigin,
    ],  # valid length per batch
    mask: mask_t,
    score_mod: score_mod_t,
):
    var batch_idx = block_idx.z

    comptime depth_v = depth - 64

    # split-k offsets
    var partition_idx = block_idx.x
    var output_batch_offset = (
        depth_v * num_heads * batch_idx
        + depth_v * num_heads * UInt(batch_size) * partition_idx
    )
    var qk_max_offset = (
        num_heads * batch_idx + num_heads * UInt(batch_size) * partition_idx
    )
    var exp_sum_offset = qk_max_offset

    # split-k intermediate buffers
    var qk_max_batch_ptr: type_of(qk_max_ptr) = {}
    if qk_max_ptr:
        qk_max_batch_ptr = qk_max_ptr + qk_max_offset

    var exp_sum_batch_ptr: type_of(exp_sum_ptr) = {}
    if exp_sum_ptr:
        exp_sum_batch_ptr = exp_sum_ptr + exp_sum_offset

    var seq_len: Int
    var q_batch_offset: Int

    @parameter
    if ragged:
        # treat valid_lengths as a input_row_offsets
        start_of_seq = Int(valid_length[batch_idx])
        end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq
        q_batch_offset = start_of_seq * Int(depth) * Int(num_heads)
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
        mla_decoding_single_batch[
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
            use_score_mod=use_score_mod,
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            q_ptr + q_batch_offset,
            k,
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

        comptime attention_config = MLAAttentionConfig[True, config]()

        var attention = Attention[
            config,
            Int(group),
            True,
            False,
            q_depth = Int(depth),
            output_depth = Int(depth_v),
        ](
            attention_config,
            output_ptr + output_batch_offset,
            q_ptr + q_batch_offset,
            k,
            k,
            mask,
            None,
            Int(batch_idx),
            scale,
            seq_len,
            num_keys,
            0,
        )
        attention.mla_decoding(
            exp_sum_batch_ptr,
            qk_max_batch_ptr,
            num_partitions,
        )
    else:
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@always_inline
fn mla_decoding_single_batch[
    q_type: DType,
    k_t: MHAOperand,
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
    depth_v: UInt,
    num_heads: UInt,
    num_threads: UInt,
    num_pipeline_stages: UInt,
    group: UInt = 1,
    use_score_mod: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
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
        num_threads // UInt(WARP_SIZE)
    ), "Number of warps doesn't match warp tile sizes."

    comptime assert (
        not decoding_warp_split_k
    ), "mla_decoding doesn't support warp split-k."

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
        Layout(IntTuple(Int(BN), Int(BK)), IntTuple(Int(nope_dim), 1)),
        address_space = AddressSpace.SHARED,
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
        Layout.row_major(Int(BK), Int(nope_dim)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        k_smem, IteratorTypeV.layout_uint_type(BN * nope_dim)
    )

    # smem for the last rope_dim of each head, will only be used during
    # Q @ K calculation.
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(Int(BN), Int(BK)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ]
    var k_rope_smem_iter = IteratorTypeK(
        k_smem + BN * nope_dim, IteratorTypeK.layout_uint_type(BN * rope_dim)
    )

    comptime mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = WM // UInt(MMA_M)
    comptime num_n_mmas = WN // UInt(MMA_N)

    comptime accum_type = get_accum_type[q_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    comptime num_output_rows = num_m_mmas * UInt(WN_O // MMA_N)  # num_n_mmas
    comptime num_output_rows_full = num_output_rows
    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(Int(num_output_rows_full), p_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
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
    for i in range(WM):
        rowmax[i] = min_or_neg_inf[accum_type]()
        rowsum[i] = 0.0

    # Shared memory for P = Q * K^t
    var p_smem = (k_smem + kv_smem_size).bitcast[Scalar[k_type]]()
    comptime p_smem_size = BM * BN
    comptime IteratorTypeP = LayoutTensorIter[
        k_type,
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

    comptime kv_num_heads = 1
    comptime kv_head_idx = 0
    var q_head_group = block_idx.y

    var q_offset = depth * BM * q_head_group

    comptime q_gmem_layout = Layout.row_major(Int(BM), Int(depth))
    var q_gmem_block = LayoutTensor[q_type, q_gmem_layout](q_ptr + q_offset)
    var q_gmem_iter = q_gmem_block.tiled_iterator[Int(BM), Int(BK), axis=1](
        0, 0
    )

    start, end = get_start_and_end_for_partitions[Int(BN)](
        Int(num_keys), Int(num_partitions), Int(block_idx.x)
    )

    # Mask global memory iterator, seq_len = 1
    comptime seq_len = 1
    var stride = max_cache_valid_length
    var mask_warp_col = warp_x * WN + UInt(start)

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

        async_copy_commit_group()

        q_gmem_iter._incr()

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        var k_ptr = k.block_paged_ptr[Int(BN)](
            UInt32(batch_idx), UInt32(kv_tile_start_row), kv_head_idx, 0
        )

        comptime kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(kv_num_heads * Int(depth), 1),
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
                kv_num_heads * Int(depth), 1
            ),
        )

        _ = p_reg_tile.fill(0)

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k_ptr,
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[Int(BN), Int(BK), axis=1](
            0, 0
        )

        # load K[:, nope_dim:(nope_dim+rope_dim)], this would be used later
        comptime k_rope_num_ves = BN * rope_dim // UInt(simd_size)
        comptime async_copy_k_rope_layout = Layout.row_major(
            Int(
                min(num_threads, k_rope_num_ves)
                * UInt(simd_size)
                // UInt(k_rope_smem_iter.layout.shape[1].value())
            ),
            k_rope_smem_iter.layout.shape[1].value() // simd_size,
        )

        @parameter
        for k_id in range(rope_dim // BK):
            var k_rope_smem_tile = k_rope_smem_iter.next_unsafe(
                k_rope_smem_iter.layout_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_rope_layout,
                swizzle=True,
                num_threads = Int(num_threads),
            ](
                k_rope_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter.next(Int(nope_dim // BK) + Int(k_id))[].vectorize[
                    1, simd_size
                ](),
            )

        # Calculate Q[:, :nope_dim] @ K[:, :nope_dim] (K transposed)
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
            prefetch_init=True,
            static_num_iters = Int(nope_dim // BK),
        ](
            p_reg_tile,
            q_smem_iter,
            k_gmem_iter,
            q_smem_iter,
            kv_nope_smem_iter,
            Int(nope_dim // BK),
        )

        # Calculate the last `rope_dim` part of Q @ K
        multistage_mma[
            Int(BM),
            Int(BN),
            Int(BK),
            Int(WM),
            Int(WN),
            Int(num_threads),
            1,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters = Int(rope_dim // BK),
        ](
            p_reg_tile,
            q_smem_iter.next_unsafe(
                q_smem_iter.linear_uint_type(Int(nope_dim // BK))
            ),
            k_rope_smem_iter,
            q_smem_iter.next_unsafe(
                q_smem_iter.linear_uint_type(Int(nope_dim // BK))
            ),
            k_rope_smem_iter,
            Int(nope_dim // BK),
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
                    var q_head_idx = q_head_group * BM + m_mma * UInt(MMA_M)
                    var mask_frag_col = mask_warp_col + n_mma * UInt(MMA_N)

                    # Offset to current thread's fragment
                    mask_frag_col += lane * UInt(p_frag_simdwidth) % UInt(MMA_N)

                    # Offset to current thread's head idx
                    q_head_idx += lane // UInt(MMA_N // p_frag_simdwidth)

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_col = mask_frag_col

                        var score_head_idx = q_head_idx + UInt(i * MMA_M // 2)

                        var score_row_with_start_pos = num_keys - 1
                        var score_row = (
                            0  # this is a decoding kernel with seq_len = 1
                        )

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32](
                                    Int(block_idx.z),
                                    Int(score_head_idx),
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
                                        Int(score_head_idx),
                                        Int(score_row_with_start_pos),
                                        Int(score_col),
                                    ),
                                    p_reg_vec2[mma_id, i],
                                    1,
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
                                    score_row, Int(score_col)
                                ),
                                IndexList[2, element_type = DType.uint32](
                                    seq_len,
                                    Int(num_keys),
                                ),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch[_apply_mask](
            mask.status(
                Index[dtype = DType.uint32](
                    num_keys,
                    kv_tile_start_row,
                ),
                Index[dtype = DType.uint32](1, BN),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * Int(num_m_mmas) * Int(num_n_mmas), 2
        )

        comptime output_layout_by_mma_unit = Layout.row_major(
            2 * Int(num_m_mmas) * (WN_O // MMA_N), 2
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
            output_reg_tile.reshape[output_layout_by_mma_unit]().vectorize[
                1, 2
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[Int(num_warps_n), Int(WM)](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        # Copy score fragments to shared memory with swizzling to resolve bank
        # conflicts for ldmatrix in the 2nd matmul.
        # warp_split_k does not need the copy as warps don't perform reduction
        # iterating across tiles, but use extra registers to perform MMAs
        # with warp-local data.
        _copy_frag_to_smem[
            BM, BN, BK, WM, WN, UInt(MMA_M), UInt(MMA_N), UInt(p_frag_simdwidth)
        ](p_smem_iter, p_reg_tile, UInt32(warp_x), UInt32(warp_y))

        async_copy_wait_all()
        barrier()

        # S[m, :] @ V[:, (0:WN) + n*WN]
        multistage_mma[
            Int(BM),
            Int(nope_dim),
            Int(BK),
            Int(WM),
            WN_O,
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

        barrier()

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](Int(BN))](start, end)

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * Int(m_mma)])
        var rowsum_inv1 = recip(rowsum[2 * Int(m_mma) + 1])

        @parameter
        for n_mma in range(WN_O // 8):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[
                    n_mma * Int(num_m_mmas) + Int(m_mma), i
                ] *= rowsum_inv0
                output_reg_tile[
                    n_mma * Int(num_m_mmas) + Int(m_mma), i + p_frag_size // 2
                ] *= rowsum_inv1

    var o_offset = nope_dim * BM * q_head_group

    comptime output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(nope_dim)), IntTuple(Int(nope_dim), 1)
    )
    var output_gmem_tile = LayoutTensor[output_type, output_gmem_layout](
        output_ptr + Int(o_offset),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[Int(WM), WN_O](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size = Int(nope_dim), access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(Int(BM), Int(nope_dim)),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[Int(WM), WN_O](
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
                WARP_SIZE * simd_size // WN_O, WN_O // simd_size
            ),
            swizzle=swizzle,
        ](
            output_gmem_warp_tile.vectorize[1, simd_size](),
            accum_smem_warp_tile.vectorize[1, simd_size](),
        )

    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


# ===-----------------------------------------------------------------------===#
# GPU Multi-head Latent Attention (MLA) prefill implementations
# ===-----------------------------------------------------------------------===#


# entrypoint for MLA prefill kernels
@always_inline
fn flare_mla_prefill[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    output_type: DType,
    q_layout: Layout,
    //,
    use_score_mod: Bool = False,
    use_fa4: Bool = False,
](
    output: LayoutTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[
        mut=False, dtype, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[mut=False, _, address_space = AddressSpace.GENERIC, ...],
    v: LayoutTensor[mut=False, _, address_space = AddressSpace.GENERIC, ...],
    k_rope: cache_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
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
    comptime assert (
        q.dtype == output.dtype
    ), "Q, K, V, output should have same type."
    comptime assert (
        q.dtype == DType.float32 or q.dtype.is_half_float()
    ), "Only support single and half precision."

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

    with Trace[TraceLevel.OP, target = ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target = ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k_rope.max_prompt_length())

        var k_operand = RaggedMHAOperand(
            LayoutTensor[k.dtype, k.layout, MutAnyOrigin](
                k.ptr,
                RuntimeLayout[k.layout].row_major(
                    k.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[
                cache_row_offsets.dtype,
                cache_row_offsets.layout,
                MutAnyOrigin,
            ](
                cache_row_offsets.ptr,
                RuntimeLayout[cache_row_offsets.layout].row_major(
                    cache_row_offsets.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        var v_operand = RaggedMHAOperand(
            LayoutTensor[v.dtype, v.layout, MutAnyOrigin](
                v.ptr,
                RuntimeLayout[v.layout].row_major(
                    v.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[
                cache_row_offsets.dtype,
                cache_row_offsets.layout,
                MutAnyOrigin,
            ](
                cache_row_offsets.ptr,
                RuntimeLayout[cache_row_offsets.layout].row_major(
                    cache_row_offsets.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        var k_rope_operand = KVCacheMHAOperand(k_rope)

        comptime kv_num_heads = cache_t.kv_params.num_heads
        comptime cache_depth = cache_t.kv_params.head_size
        comptime q_depth = Int(q_layout.shape[rank - 1])

        comptime num_keys_per_block = UInt(
            64
        ) if has_nvidia_gpu_accelerator() else UInt(
            128
        )  # BN = 64 for nvidia, 128 in the only supported BN for amd

        comptime mha_config = MHAConfig[dtype](
            UInt(Int(q_layout.shape[rank - 2])),  # num_heads
            UInt(Int(k.layout.shape[rank - 1])),  # depth
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )

        flare_mla_prefill_dispatch[
            kv_num_heads = Int(kv_num_heads),
            q_depth=q_depth,
            cache_depth = Int(cache_depth),
            config=mha_config,
            use_fa4=use_fa4,
        ](
            output,
            q,
            k_operand,
            v_operand,
            k_rope_operand,
            mask_functor,
            score_mod_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
            cache_offsets,
        )


# entrypoint for LayoutTensor[mut=True, , Layout.row_major[3](), MutAnyOrigin]as K_rope input, used by tests.
@always_inline
fn flare_mla_prefill[
    rank: Int,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    q_layout: Layout,
    //,
    use_score_mod: Bool = False,
    use_fa4: Bool = False,
](
    output: LayoutTensor[
        mut=True, _, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[
        mut=False, dtype, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[mut=False, _, address_space = AddressSpace.GENERIC, ...],
    v: LayoutTensor[mut=False, _, address_space = AddressSpace.GENERIC, ...],
    k_rope: LayoutTensor[
        mut=False, _, address_space = AddressSpace.GENERIC, ...
    ],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: LayoutTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
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
    ), "Q, K, V, output should have same type."
    comptime assert (
        q.dtype == DType.float32 or q.dtype.is_half_float()
    ), "Only support single and half precision."

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

    with Trace[TraceLevel.OP, target = ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target = ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int = q.dim[0]()

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        var cache_row_offsets_lt = LayoutTensor[
            cache_row_offsets.dtype,
            cache_row_offsets.layout,
            MutAnyOrigin,
        ](
            cache_row_offsets.ptr,
            RuntimeLayout[cache_row_offsets.layout].row_major(
                cache_row_offsets.runtime_layout.shape.value.canonicalize()
            ),
        )
        var k_operand = RaggedMHAOperand(
            LayoutTensor[k.dtype, k.layout, MutAnyOrigin](
                k.ptr,
                RuntimeLayout[k.layout].row_major(
                    k.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var v_operand = RaggedMHAOperand(
            LayoutTensor[v.dtype, v.layout, MutAnyOrigin](
                v.ptr,
                RuntimeLayout[v.layout].row_major(
                    v.runtime_layout.shape.value.canonicalize()
                ),
            ),
            cache_row_offsets_lt,
        )
        var k_rope_operand = LayoutTensorMHAOperand(
            LayoutTensor[k_rope.dtype, k_rope.layout, MutAnyOrigin](
                k_rope.ptr,
                RuntimeLayout[k_rope.layout].row_major(
                    k_rope.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )

        comptime output_type = output.dtype
        comptime kv_num_heads = Int(k_rope.layout.shape[2])
        comptime cache_depth = Int(k_rope.layout.shape[3])
        comptime q_depth = Int(q.layout.shape[q.rank - 1])  # hard code for now
        comptime num_keys_per_block = UInt(
            64
        ) if has_nvidia_gpu_accelerator() else UInt(
            128
        )  # BN = 64 for nvidia, 128 in the only supported BN for amd
        comptime mha_config = MHAConfig[dtype](
            UInt(Int(q_layout.shape[rank - 2])),
            UInt(Int(k.layout.shape[rank - 1])),
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
            use_fa4=use_fa4,
        ](
            output,
            q,
            k_operand,
            v_operand,
            k_rope_operand,
            mask_functor,
            score_mod_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
            cache_offsets,
        )


@always_inline
fn flare_mla_prefill_dispatch[
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    dtype: DType,
    output_type: DType,
    q_layout: Layout,
    //,
    kv_num_heads: Int,
    use_score_mod: Bool = False,
    q_depth: Int = 192,
    cache_depth: Int = 576,
    config: MHAConfig[dtype] = {
        UInt(Int(q_layout.shape[q_layout.rank() - 2])),
        UInt(Int(q_layout.shape[q_layout.rank() - 1])),
    },
    _ndbuffer_mha_operand: Bool = False,
    use_fa4: Bool = False,
](
    output: LayoutTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[
        mut=False, dtype, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
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
    comptime group = config.num_heads // UInt(kv_num_heads)
    comptime rank = output.layout.rank()

    comptime assert q_depth == Int(q.layout.shape[rank - 1])
    comptime assert num_heads == UInt(Int(q.layout.shape[rank - 2]))
    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "flareMLA_prefill currently only supports Nvidia and AMD GPUs."

    var batch_size: Int = valid_length.dim[0]() - 1

    if batch_size == 0 or max_prompt_len == 0:
        return

    comptime q_half_float = dtype in (DType.float16, DType.bfloat16)

    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()

    comptime q_smem = BM * UInt(q_depth)
    comptime k_smem = BN * UInt(q_depth)
    comptime v_smem = BN * depth

    comptime smem_use = (q_smem + k_smem + v_smem) * UInt(
        size_of[config.dtype]()
    ) if has_nvidia_gpu_accelerator() else 0

    var q_device = DeviceBuffer[q.dtype](ctx, q.ptr, q.size(), owning=False)
    var output_device = DeviceBuffer[output.dtype](
        ctx, output.ptr, output.size(), owning=False
    )

    comptime fa4_enabled = ctx.default_device_info == B200 and use_fa4

    @parameter
    if fa4_enabled:
        mla_sm100_prefill[
            config=config,
            group = Int(group),
            q_depth=q_depth,
            cache_depth=cache_depth,
            use_score_mod=use_score_mod,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](
            output,
            q,
            k,
            rebind[type_of(k)](v),
            k_rope,
            mask_functor,
            score_mod_functor,
            valid_length,
            DynamicInt(max_prompt_len),
            scale,
            batch_size,
            ctx,
        )

    else:
        comptime kernel = mla_prefill[
            config.dtype,
            k_t,
            v_t,
            k_rope_t,
            output.dtype,
            mask_t,
            score_mod_t,
            valid_length.layout,
            config,
            group = Int(group),
            use_score_mod=use_score_mod,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
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
            k_rope,
            output_device,
            scale,
            batch_size,
            max_prompt_len,
            valid_length,
            cache_offsets,
            mask_functor,
            score_mod_functor,
            grid_dim=grid_dim,
            block_dim=(Int(config.num_threads()), 1, 1),
            shared_mem_bytes=Int(smem_use),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
fn mla_prefill[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    valid_layout: Layout,
    config: MHAConfig,
    group: Int = 128,
    q_depth: Int = 192,
    cache_depth: Int = 576,
    use_score_mod: Bool = False,
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
    valid_length: LayoutTensor[
        DType.uint32,
        valid_layout,
        ImmutAnyOrigin,
    ],
    cache_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
):
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
    fn q_block_idx() -> UInt:
        return block_idx.x if is_nvidia_gpu() else block_idx.y

    @always_inline
    fn head_idx() -> UInt:
        return block_idx.y if is_nvidia_gpu() else block_idx.x

    if seq_len < Int(q_block_idx() * config.block_m()):
        return

    @parameter
    if _ndbuffer_mha_operand:
        num_keys = k_rope.cache_length(Int(batch_idx))
        start_pos = UInt32(num_keys - seq_len)

    else:
        start_pos = UInt32(k_rope.cache_length(Int(batch_idx)))
        num_keys = Int(start_pos) + seq_len

    if cache_offsets:
        var cache_offsets_nd = cache_offsets.value()
        cache_start_pos = cache_offsets_nd[batch_idx][0]

    q_batch_offset = start_of_seq * q_depth * Int(config.num_heads)
    o_batch_offset = start_of_seq * Int(depth) * Int(config.num_heads)

    @parameter
    if is_nvidia_gpu():
        mla_prefill_single_batch[
            config=config,
            group=group,
            q_depth=q_depth,
            cache_depth=cache_depth,
            use_score_mod=use_score_mod,
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
            score_mod,
            Int(batch_idx),
        )
    elif is_amd_gpu():
        comptime attention_config = MLAAttentionConfig[False, config]()
        var attention = Attention[config, 1, False, False, q_depth=q_depth](
            attention_config,
            output_ptr + o_batch_offset,
            q_ptr + q_batch_offset,
            k,
            v,
            mask,
            None,
            Int(batch_idx),
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
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@always_inline
fn mla_prefill_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    *,
    config: MHAConfig,
    group: Int = 1,
    q_depth: Int = 192,
    cache_depth: Int = 576,
    use_score_mod: Bool = False,
](
    q_ptr: UnsafePointer[mut=True, Scalar[q_type]],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    cache_start_pos: UInt32,
    num_keys: Int,
    mask: mask_t,
    score_mod: score_mod_t,
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

    comptime rope_depth = q_depth - Int(depth)

    comptime cache_num_heads = num_heads // UInt(group)

    comptime assert num_warps_m * num_warps_n == (
        num_threads // UInt(WARP_SIZE)
    ), "Number of warps doesn't match warp tile sizes."

    var tid: Int = Int(thread_idx.x)
    var warp_id = UInt32(warp.broadcast(tid // WARP_SIZE))
    var lane = UInt32(lane_id())

    # Coordinates of the current warp.
    var warp_y = warp_id // UInt32(num_warps_n)
    var warp_x = warp_id % UInt32(num_warps_n)

    # The entire query block (BM x q_depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * UInt(q_depth)
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
    comptime k_smem_size = BN * UInt(q_depth)
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

    comptime v_smem_size = BN * depth
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(Int(BK), Int(depth)),
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
        IntTuple(Int(BM), q_depth),
        IntTuple(Int(num_heads * UInt(q_depth)), 1),
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
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[q_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), q_depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                Int(num_heads * UInt(q_depth)), 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[Int(BM), Int(BK), axis=1](
        0, 0
    )
    # q tile has valid shape q_tile_num_rows x q_depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    comptime mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime WM = config.WM
    comptime WN = config.WN
    comptime WN_O = depth
    comptime num_m_mmas = WM // UInt(MMA_M)
    comptime num_n_mmas = WN // UInt(MMA_N)
    comptime num_n_mmas_output = WN_O // UInt(MMA_N)

    comptime accum_type = get_accum_type[q_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(Int(num_m_mmas * num_n_mmas), p_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(Int(num_m_mmas * num_n_mmas_output), p_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
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

    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime q_num_vecs = BM * BK // UInt(simd_size)

    comptime async_copy_q_layout = Layout.row_major(
        Int(min(num_threads, q_num_vecs) * UInt(simd_size) // BK),
        Int(BK // UInt(simd_size)),
    )

    @parameter
    for q_id in range(q_depth // Int(BK)):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.linear_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads = Int(num_threads),
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
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        if (
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    Int(UInt32(kv_tile_start_row) + cache_start_pos),
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.FULL_MASK
        ):
            return

        comptime kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(num_heads * depth), 1),
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
                Int(num_heads * depth), 1
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
                UInt32(Int(head_idx)),
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
                UInt32(Int(head_idx)),
                0,
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[
            Int(BK), Int(depth), axis=0
        ](0, 0)

        # here we set up variables for k_rope tensor
        comptime k_rope_gmem_layout = Layout(
            IntTuple(Int(BN), cache_depth),
            IntTuple(Int(cache_num_heads * UInt(cache_depth)), 1),
        )

        var k_rope_runtime_layout = RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[k_rope_gmem_layout.shape, element_type = DType.int32](
                kv_tile_num_rows, cache_depth
            ),
            RuntimeTuple[k_rope_gmem_layout.stride, element_type = DType.int32](
                Int(cache_num_heads * UInt(cache_depth)), 1
            ),
        )

        var k_rope_gmem_block = LayoutTensor[
            k_rope_type,
            k_rope_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k_rope.block_paged_ptr[Int(BN)](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row) + cache_start_pos,
                UInt32(Int(head_idx // UInt32(group))),
                UInt32(cache_depth - rope_depth),
            ),
            k_rope_runtime_layout,
        )
        var k_rope_gmem_iter = k_rope_gmem_block.tiled_iterator[
            Int(BN), Int(BK), axis=1
        ](0, 0)

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        @always_inline
        @parameter
        fn _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: type_of(tensor)
        ):
            return {
                tensor.ptr,
                {{num_rows, tensor.dim[1]()}, tensor.runtime_layout.stride},
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

            async_copy_commit_group()

            k_gmem_iter._incr()

        @parameter
        for k_id in range(depth // BK, q_depth // Int(BK)):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.linear_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads = Int(num_threads),
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
            static_num_iters = q_depth // Int(BK),
            k_group_size = config.k_group_size,
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            q_depth // Int(BK),
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
                        score_col_with_cache_start_pos = (
                            score_col + cache_start_pos
                        )

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32](
                                    Int(block_idx.z),
                                    Int(block_idx.y),
                                    Int(score_row_with_start_pos),
                                    Int(score_col_with_cache_start_pos),
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
                                        Int(score_col_with_cache_start_pos),
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
                    UInt32(kv_tile_start_row) + cache_start_pos,
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += UInt32(BN)

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * Int(num_m_mmas) * Int(num_n_mmas), 2
        )
        comptime reg_output_layout_by_mma_unit = Layout.row_major(
            2 * Int(num_m_mmas) * Int(num_n_mmas_output), 2
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
            output_reg_tile.reshape[reg_output_layout_by_mma_unit]().vectorize[
                1, 2
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[Int(num_warps_n), Int(WM)](0, Int(warp_y)),
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

            async_copy_commit_group()

            v_gmem_iter._incr()

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
                Int(depth),
                Int(BK),
                Int(WM),
                Int(depth),
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
                Int(depth),
                Int(BK),
                Int(WM),
                Int(depth),
                Int(num_threads),
                2,
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

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](Int(BN))](
        0, num_keys
    )

    comptime output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )

    var output_offset = UInt32(depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )
    var output_gemm_runtime_layout = RuntimeLayout[
        element_type = DType.int32, linear_idx_type = DType.int32
    ](
        RuntimeTuple[output_gmem_layout.shape, element_type = DType.int32](
            Int(q_tile_num_rows), Int(depth)
        ),
        RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
            Int(num_heads * depth), 1
        ),
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        output_ptr + Int(output_offset),
        output_gemm_runtime_layout,
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[Int(WM), Int(WN_O)](
        Int(warp_y), Int(warp_x)
    )

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * Int(m_mma)])
        var rowsum_inv1 = recip(rowsum[2 * Int(m_mma) + 1])

        @parameter
        for n_mma in range(num_n_mmas_output):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size = Int(depth), access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(Int(BM), Int(depth)),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[Int(WM), Int(depth)](
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


# ===-----------------------------------------------------------------------===#
# Helper function that creates cache_row_offsets for the MLA prefill kernel
# ===-----------------------------------------------------------------------===#


fn set_buffer_lengths_to_zero[
    buffer_lengths_layout: Layout
](
    buffer_lengths: LayoutTensor[
        DType.int32, buffer_lengths_layout, MutAnyOrigin
    ],
):
    comptime MAX_CHUNKS = Int(buffer_lengths_layout.shape[0])

    @parameter
    for chunk_idx in range(MAX_CHUNKS):
        buffer_lengths[chunk_idx] = 0


@always_inline
fn mla_prefill_plan[
    cache_t: KVCacheT,
](
    buffer_row_offsets: LayoutTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: LayoutTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_lengths: LayoutTensor[
        mut=True, DType.int32, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
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
    var batch_size: Int = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        # Fill buffer lengths with 0
        comptime kernel = set_buffer_lengths_to_zero[buffer_lengths.layout]
        ctx.enqueue_function[kernel, kernel](
            buffer_lengths, grid_dim=1, block_dim=1
        )
    else:
        comptime kernel = mla_prefill_plan_kernel[
            buffer_row_offsets.layout,
            cache_offsets.layout,
            buffer_lengths.layout,
            input_row_offsets.layout,
            cache_t,
        ]

        ctx.enqueue_function[kernel, kernel](
            buffer_row_offsets,
            cache_offsets,
            buffer_lengths,
            input_row_offsets.get_immutable(),
            k_cache,
            buffer_token_size,
            grid_dim=(ceildiv(batch_size, 128), 1, 1),
            block_dim=(128, 1, 1),
        )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
fn mla_prefill_plan_kernel[
    buffer_row_offsets_layout: Layout,
    cache_offsets_layout: Layout,
    buffer_lengths_layout: Layout,
    input_row_offsets_layout: Layout,
    cache_t: KVCacheT,
](
    buffer_row_offsets: LayoutTensor[
        DType.uint32,
        buffer_row_offsets_layout,
        MutAnyOrigin,
    ],
    cache_offsets: LayoutTensor[
        DType.uint32,
        cache_offsets_layout,
        MutAnyOrigin,
    ],
    buffer_lengths: LayoutTensor[
        DType.int32,
        buffer_lengths_layout,
        MutAnyOrigin,
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32,
        input_row_offsets_layout,
        ImmutExternalOrigin,
    ],
    k_cache: cache_t,
    buffer_token_size: UInt32,
):
    var seq_idx = global_idx.x
    var seq_start_pos = 0
    var seq_end_pos = 0
    var batch_size: Int = input_row_offsets.dim[0]() - 1
    var buffer_size: Int = Int(buffer_token_size)

    comptime MAX_CHUNKS = Int(buffer_lengths.layout.shape[0])
    comptime page_size = cache_t.page_size_
    comptime assert page_size != 0, "Only PagedKVCache is supported."

    if seq_idx >= UInt(batch_size):
        return

    # Calculate starting position for this sequence.
    # Note: the total cache length of each sequence is aligned to the page size.
    var prev_row_offset = Int(input_row_offsets[0])
    for i in range(seq_idx):
        # The cache length that has been prefilled in previous forward passes.
        var cache_length = k_cache.cache_length(Int(i))

        # account for the new input tokens.
        var row_offset_i = Int(input_row_offsets[i + 1])
        cache_length += row_offset_i - prev_row_offset
        prev_row_offset = row_offset_i

        seq_start_pos += align_up(cache_length, page_size)

    var curr_seq_len = align_up(
        k_cache.cache_length(Int(seq_idx))
        + Int(input_row_offsets[seq_idx + 1])
        - prev_row_offset,
        page_size,
    )

    # which chunk this sequence starts in
    var start_chunk = seq_start_pos // buffer_size
    var processed_seq_len = UInt32(0)
    var seq_len_left = curr_seq_len

    # Fill buffer offsets for this sequence
    @parameter
    for chunk_idx in range(MAX_CHUNKS):
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
    if seq_idx == UInt(batch_size - 1):
        seq_end_pos = seq_start_pos + curr_seq_len
        var end_chunk = (seq_end_pos + buffer_size - 1) // buffer_size - 1

        # Set buffer lengths for all chunks
        @parameter
        for chunk_idx in range(MAX_CHUNKS):
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
fn _k_cache_to_buffer[
    dtype: DType,
    cache_t: KVCacheT,
](
    buffer_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    k_cache: cache_t,
    length: Int32,
    buffer: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContext,
) raises:
    comptime num_heads = cache_t.kv_params.num_heads
    comptime assert num_heads == 1, "num_heads should be equal to 1"

    @always_inline
    @parameter
    @__copy_capture(k_cache, buffer_row_offsets, cache_offsets)
    fn copy_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 2, "rank should be equal to 2"

        var idx = rebind[IndexList[2]](idx_arg)
        var global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            buffer_row_offsets, global_token_idx
        )

        var token_idx = Int(
            UInt32(global_token_idx)
            - buffer_row_offsets[batch_idx][0]
            + cache_offsets[batch_idx][0]
        )

        var head_dim_idx = idx[1]

        var cache_val = rebind[SIMD[dtype, width]](
            k_cache.load[width=width](
                batch_idx, 0, token_idx, head_dim_idx
            ).cast[dtype]()
        )

        buffer.store(idx, cache_val)

    var launch_shape = IndexList[2](
        Int(length),
        buffer.dim[1](),
    )
    comptime target_simd_width = simd_width_of[
        dtype, target = get_gpu_target()
    ]()

    _elementwise_impl_gpu[func=copy_fn, simd_width = UInt(target_simd_width)](
        launch_shape, context
    )
