# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections import OptionalReg
from math import ceildiv, exp, recip
from math.constants import log2e
from sys import alignof, has_nvidia_gpu_accelerator, simdwidthof, sizeof

import gpu.warp as warp
from algorithm.functional import (
    _elementwise_impl_gpu,
    tile_and_unswitch,
    unswitch,
    vectorize,
)
from buffer import NDBuffer
from buffer.dimlist import DimList
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
from gpu.host import DeviceContext
from gpu.host import Dim as LaunchDim
from gpu.host import FuncAttribute
from gpu.host._compile import _get_gpu_target
from gpu.host.info import A100, H100
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
    copy,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import Swizzle, make_swizzle
from layout.tensor_builder import static
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg._multistage_gemm_gpu import multistage_mma
from linalg.transpose import transpose
from memory import UnsafePointer, stack_allocation
from memory.pointer import AddressSpace as _AddressSpace
from memory.unsafe import bitcast
from nn._ragged_utils import get_batch_from_row_offsets
from nn.mha_mask import MHAMask, NullMask, TileMaskStatus
from nn.mha_operand import (
    KVCacheMHAOperand,
    MHAOperand,
    NDBufferMHAOperand,
    RaggedMHAOperand,
)
from nn.mha_score_mod import AlibiScoreMod, IdentityScoreMod, ScoreModTrait
from nn.mha_utils import MHAConfig, _copy_frag_to_smem, _kernel_mask
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf, neg_inf
from utils.static_tuple import StaticTuple

from .mha import _get_start_and_end_for_partitions
from .softmax import _online_softmax_iter_for_mma_output

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
    type: DType,
    q_shape: DimList, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        type, q_shape.get[rank - 2](), q_shape.get[rank - 1]()
    ),
    ragged: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    output: NDBuffer[mut=True, _, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: cache_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: NDBuffer[DType.uint32, 1, *_],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ] = None,
    num_partitions: OptionalReg[Int] = None,
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
    constrained[
        ragged or rank == 4, "only support rank 4 inputs for non-ragged inputs."
    ]()
    constrained[
        not ragged or rank == 3, "only support rank 3 inputs for ragged inputs."
    ]()
    constrained[
        q.type == cache_t.type == output.type,
        "Q, K, V, output should have same type.",
    ]()

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("output", output),
        )

    with Trace[TraceLevel.OP, target = ctx.device_info.api](
        "flare_mla_decoding",
        Trace[TraceLevel.OP, target = ctx.device_info.api]._get_detail_str[
            description_fn
        ](),
    ):
        alias kv_num_heads = cache_t.kv_params.num_heads

        var max_prompt_len: Int
        var num_keys = Int(k.max_context_length())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k.max_prompt_length())

        var k_operand = KVCacheMHAOperand(k)

        flare_mla_decoding_dispatch[
            kv_num_heads=kv_num_heads,
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


# entrypoint for NDBuffer as K input, used by tests.
fn flare_mla_decoding[
    rank: Int,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_shape: DimList, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(type, q_shape.get[2](), q_shape.get[3]()),
    decoding_warp_split_k: Bool = False,
](
    output: NDBuffer[mut=True, _, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: NDBuffer[_, rank, *_],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: OptionalReg[Int] = None,
) raises:
    constrained[rank == 4, "only support rank 4 inputs."]()

    alias kv_num_heads = k.shape.get[2]()

    # Runtime dimensions.
    var num_keys = k.dim[1]()

    var k_operand = NDBufferMHAOperand(k)
    var valid_length = NDBuffer[DType.uint32, 1](
        UnsafePointer[UInt32](), Index(0)
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
    rank: Int,
    k_t: MHAOperand,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_shape: DimList, //,
    kv_num_heads: Int,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        type, q_shape.get[rank - 2](), q_shape.get[rank - 1]()
    ),
    ragged: Bool = False,
    # Work arounds to unify KVCache and NDBuffer inputs:
    # Differentiate two cases, KV cache's length is before adding the latest
    # tokens e.g. zero for CE, and KV NDBuffer's length is the latest length
    # e.g. prompt length for CE.
    _is_cache_length_accurate: Bool = False,
    # valid_length is needed for KV cache inputs and is empty for NDBuffer inputs
    # to avoid overhead in benchmark.
    _use_valid_length: Bool = True,
    decoding_warp_split_k: Bool = False,
](
    output: NDBuffer[_, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: k_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: NDBuffer[DType.uint32, 1, *_],
    max_prompt_len: Int,
    max_cache_valid_length: Int,
    scale: Float32,
    ctx: DeviceContext,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ] = None,
    num_partitions: OptionalReg[Int] = None,
) raises:
    alias num_heads = config.num_heads
    alias depth = config.depth
    alias group = config.num_heads // kv_num_heads
    constrained[num_heads == q.shape.get[rank - 2]()]()

    # only A100 or H100 have the enough smem to store the full BM * head_dim Q tensor.
    alias has_enough_smem = ctx.device_info is A100 or ctx.device_info is H100

    constrained[
        depth == q.shape.get[rank - 1]() == 576,
        "flareMLA_decoding only supports head_dim == 576.",
    ]()
    constrained[
        kv_num_heads == 1, "flareMLA_decoding only supports kv_num_heads == 1."
    ]()
    constrained[
        has_nvidia_gpu_accelerator(),
        "flareMLA_decoding currently only supports Nvidia GPUs.",
    ]()

    constrained[
        q.type.is_half_float(),
        "Only support half precision.",
    ]()

    # Whether head and depth are static. With BSHD, B and S are dynamic.
    # H and D are always known for opaque KVCache types, we only check Q.
    constrained[
        q.shape.all_known[rank - 2, rank](),
        "Need num_heads and head_dim to be static for Q.",
    ]()

    var batch_size: Int

    @parameter
    if ragged:
        batch_size = valid_length.dim[0]() - 1
    # This branch holds for both KVCache and NDBuffer inputs.
    # Q is BSHD, S is either homogeneous or padded to same length.
    else:
        batch_size = q.dim[0]()

    alias BM = 16 if num_heads == 16 or not has_enough_smem else 32  # for deepseek-v2 lite
    alias BN = 64
    alias BK = 64  # need 8 mma_tile per row the resolve the bank conflict
    alias WM = BM
    alias WN = 16
    # num warps in M and N, multipled by warp size.
    alias num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

    alias accum_type = get_accum_type[q.type]()
    alias num_pipeline_stages = 6
    # smem for q
    var shared_mem_bytes = BM * depth * sizeof[q.type]()

    shared_mem_bytes += BN * depth * sizeof[k_t.type]()

    alias num_warps = ceildiv(num_threads, WARP_SIZE)

    # smem for p and warp_scratch
    shared_mem_bytes += (
        BM * BN * sizeof[k_t.type]() + 2 * num_warps * BM * sizeof[accum_type]()
    )
    alias num_blocks_y = num_heads // BM

    alias kernel = mla_decoding[
        q.type,
        k_t,
        output.type,
        mask_t,
        score_mod_t,
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
        ragged=ragged,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
        decoding_warp_split_k=decoding_warp_split_k,
    ]

    alias nullptr = UnsafePointer[Scalar[accum_type]]()

    var num_partitions_value: Int = 1

    ctx.enqueue_function[kernel](
        q.data,
        k,
        output.data,
        nullptr,
        nullptr,
        scale,
        batch_size,
        num_partitions_value,
        max_cache_valid_length,
        valid_length,
        mask_functor,
        score_mod_functor,
        grid_dim=(1, Int(num_blocks_y), Int(batch_size)),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=shared_mem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            ctx.device_info.shared_memory_per_multiprocessor - 4096
        ),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn mla_decoding[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()]],
    scale: Float32,
    batch_size: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    valid_length: NDBuffer[
        DType.uint32, 1, MutableAnyOrigin
    ],  # valid length per batch
    mask: mask_t,
    score_mod: score_mod_t,
):
    var batch_idx = block_idx.z

    alias depth_v = depth - 64

    # split-k offsets
    var partition_idx = block_idx.x
    var output_batch_offset = depth_v * num_heads * batch_idx + depth_v * num_heads * batch_size * partition_idx
    var qk_max_offset = num_heads * batch_idx + num_heads * batch_size * partition_idx
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
        q_ptr.offset(q_batch_offset),
        k,
        output_ptr.offset(output_batch_offset),
        exp_sum_batch_ptr,
        qk_max_batch_ptr,
        scale,
        num_keys,
        num_partitions,
        max_cache_valid_length,
        mask,
        score_mod,
        batch_idx,
    )


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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
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
):
    """Flash attention v2 algorithm."""
    alias k_type = k_t.type
    constrained[q_type == k_type]()

    alias simd_size = simdwidthof[q_type]()

    alias WN_O = 128
    alias nope_dim = depth_v
    alias rope_dim = depth - depth_v

    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN

    constrained[
        num_warps_m * num_warps_n == (num_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    constrained[
        not decoding_warp_split_k,
        "mla_decoding doesn't support warp split-k.",
    ]()

    var tid = thread_idx.x
    var warp_id = warp.broadcast(tid // WARP_SIZE)
    var lane = lane_id()

    # Coordinates of the current warp.
    warp_y, warp_x = divmod(warp_id, UInt(num_warps_n))

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[q_type, simd_size]](),
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )

    alias kv_smem_size = BN * depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()

    # For MLA, We define V = K[:, :nope_dim], thus we spilt the K tensor
    # in two parts when storing it in the smem: K[:, :nope_dim] and
    # K[:, nope_dim:(nope_dim+rope_dim)].
    # Instead of intializing the tiled iterator with a row-major layout
    # (BN, BK) like standard mha kernels, we manully set the following
    # layout. This ensures that once Q @ K calculation is complete, the
    # K[:, :nope_dim] tensor stored continously in the smem.
    var kv_nope_smem_iter = LayoutTensorIter[
        k_type,
        Layout(IntTuple(BN, BK), IntTuple(nope_dim, 1)),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, nope_dim, stride=BK)

    # view the K[:, :nope_dim] as V tensor.
    var v_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BK, nope_dim),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, BN * nope_dim)

    # smem for the last rope_dim of each head, will only be used during
    # Q @ K calculation.
    var k_rope_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem + BN * nope_dim, BN * rope_dim)

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[q_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    alias num_output_rows = num_m_mmas * (WN_O // MMA_N)  # num_n_mmas
    alias num_output_rows_full = num_output_rows
    var output_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_output_rows_full, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().fill(0.0)

    # Rowwise max and sum for online softmax
    alias row_alignment = alignof[SIMD[accum_type, simdwidthof[accum_type]()]]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

    @parameter
    for i in range(Int(WM)):
        rowmax[i] = min_or_neg_inf[accum_type]()
        rowsum[i] = 0.0

    # Shared memory for P = Q * K^t
    var p_smem = (k_smem + kv_smem_size).bitcast[Scalar[k_type]]()
    alias p_smem_size = BM * BN
    var p_smem_iter = LayoutTensorIter[
        k_type, Layout.row_major(BM, BK), address_space = AddressSpace.SHARED
    ](p_smem, BM * BN)

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space = AddressSpace.SHARED,
    ]((p_smem + BM * BN).bitcast[Scalar[accum_type]]())

    alias kv_num_heads = 1
    alias kv_head_idx = 0
    var q_head_group = block_idx.y

    var q_offset = depth * BM * q_head_group

    alias q_gmem_layout = Layout.row_major(BM, depth)
    var q_gmem_block = LayoutTensor[q_type, q_gmem_layout](q_ptr + q_offset)
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    start, end = _get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
    )

    # Mask global memory iterator, seq_len = 1
    alias seq_len = 1
    var stride = max_cache_valid_length
    var mask_warp_col = warp_x * WN + start

    alias q_num_vecs = BM * BK // simd_size

    alias async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK, BK // simd_size
    )

    @parameter
    for q_id in range(Int(depth // BK)):
        var q_smem_tile = q_smem_iter.next_unsafe(q_id)[]

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
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        var k_ptr = k.block_paged_ptr[BN](
            batch_idx, kv_tile_start_row, kv_head_idx, 0
        )

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
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        # load K[:, nope_dim:(nope_dim+rope_dim)], this would be used later
        alias k_rope_num_ves = BN * rope_dim // simd_size
        alias async_copy_k_rope_layout = Layout.row_major(
            min(num_threads, k_rope_num_ves)
            * simd_size
            // k_rope_smem_iter.layout.shape[1].value(),
            k_rope_smem_iter.layout.shape[1].value() // simd_size,
        )

        @parameter
        for k_id in range(Int(rope_dim // BK)):
            var k_rope_smem_tile = k_rope_smem_iter.next_unsafe(k_id)[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_rope_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_rope_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter.next(Int(nope_dim // BK) + k_id)[].vectorize[
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
            static_num_iters = Int(nope_dim // BK),
        ](
            p_reg_tile,
            q_smem_iter,
            k_gmem_iter,
            q_smem_iter,
            kv_nope_smem_iter,
            nope_dim // BK,
        )

        # Cacluate the last `rope_dim` part of Q @ K
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
            static_num_iters = Int(rope_dim // BK),
        ](
            p_reg_tile,
            q_smem_iter.next_unsafe(Int(nope_dim // BK)),
            k_rope_smem_iter,
            q_smem_iter.next_unsafe(Int(nope_dim // BK)),
            k_rope_smem_iter,
            nope_dim // BK,
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
            for m_mma in range(Int(num_m_mmas)):

                @parameter
                for n_mma in range(Int(num_n_mmas)):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var q_head_idx = q_head_group * BM + m_mma * MMA_M
                    var mask_frag_col = mask_warp_col + n_mma * MMA_N

                    # Offset to current thread's fragment
                    mask_frag_col += lane * p_frag_simdwidth % MMA_N

                    # Offset to current thread's head idx
                    q_head_idx += lane // (MMA_N // p_frag_simdwidth)

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_col = mask_frag_col

                        var score_head_idx = q_head_idx + i * MMA_M // 2

                        var score_row_with_start_pos = num_keys - 1
                        var score_row = 0  # this is a decoding kernel with seq_len = 1

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32,](
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

                        @parameter
                        if use_score_mod:
                            p_reg_vec2[mma_id, i] = (
                                score_mod.score_mod(
                                    IndexList[4, element_type = DType.uint32,](
                                        block_idx.z,
                                        score_head_idx,
                                        score_row_with_start_pos,
                                        score_col,
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
                                    score_row, score_col
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
                    num_keys,
                    kv_tile_start_row,
                ),
                Index[dtype = DType.uint32](1, BN),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        alias reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )

        alias output_layout_by_mma_unit = Layout.row_major(
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
            warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        # Copy score fragments to shared memory with swizzling to resolve bank
        # conflicts for ldmatrix in the 2nd matmul.
        # warp_split_k does not need the copy as warps don't perform reduction
        # iterating across tiles, but use extra registers to perform MMAs
        # with warp-local data.
        _copy_frag_to_smem[BM, BN, BK, WM, WN, MMA_M, MMA_N, p_frag_simdwidth](
            p_smem_iter, p_reg_tile, warp_x, warp_y
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
            static_num_iters = Int(BN // BK),
        ](
            output_reg_tile,
            p_smem_iter,
            v_smem_iter,
            p_smem_iter,
            v_smem_iter,
            BN // BK,
        )

        barrier()

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](start, end)

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(Int(num_m_mmas)):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        @parameter
        for n_mma in range(Int(WN_O // 8)):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    var o_offset = nope_dim * BM * q_head_group

    alias output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(nope_dim)), IntTuple(Int(nope_dim), 1)
    )
    var output_gmem_tile = LayoutTensor[output_type, output_gmem_layout](
        output_ptr + Int(o_offset),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN_O](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=nope_dim, access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, nope_dim),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN_O](
            Int(warp_y), Int(warp_x)
        )

        copy[thread_layout = Layout.row_major(8, 4), swizzle=swizzle](
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
    type: DType,
    q_shape: DimList, //,
    use_score_mod: Bool = False,
](
    output: NDBuffer[mut=True, _, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: NDBuffer[_, 3, *_],
    v: NDBuffer[_, 3, *_],
    k_rope: cache_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: NDBuffer[DType.uint32, 1, *_],
    cache_row_offsets: NDBuffer[DType.uint32, 1, *_],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
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
    constrained[rank == 3, "only support ragged inputs"]()
    constrained[
        q.type == cache_t.type == output.type,
        "Q, K, V, output should have same type.",
    ]()
    constrained[
        q.type == DType.float32 or q.type.is_half_float(),
        "Only support single and half precision.",
    ]()

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("k", k),
            trace_arg("v", v),
            trace_arg("output", output),
        )

    with Trace[TraceLevel.OP, target = ctx.device_info.api](
        "flare_mla_prefill",
        Trace[TraceLevel.OP, target = ctx.device_info.api]._get_detail_str[
            description_fn
        ](),
    ):
        var max_prompt_len: Int

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k_rope.max_prompt_length())

        var k_operand = RaggedMHAOperand(k, cache_row_offsets)
        var v_operand = RaggedMHAOperand(v, cache_row_offsets)
        var k_rope_operand = KVCacheMHAOperand(k_rope)

        alias kv_num_heads = cache_t.kv_params.num_heads
        alias cache_depth = cache_t.kv_params.head_size
        alias q_depth = q_shape.get[rank - 1]()

        alias mha_config = MHAConfig(
            type,
            q_shape.get[rank - 2](),  # num_heads
            k.shape.get[rank - 1](),  # depth
            num_keys_per_block=UInt(64),
            WN=UInt(64),
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
            score_mod_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
        )


# entrypoint for NDBuffer as K_rope input, used by tests.
@always_inline
fn flare_mla_prefill[
    rank: Int,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_shape: DimList, //,
    use_score_mod: Bool = False,
](
    output: NDBuffer[mut=True, _, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: NDBuffer[_, 3, *_],
    v: NDBuffer[_, 3, *_],
    k_rope: NDBuffer[_, 4, *_],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: NDBuffer[DType.uint32, 1, *_],
    cache_row_offsets: NDBuffer[DType.uint32, 1, *_],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
) raises:
    constrained[rank == 3, "only support ragged inputs"]()
    constrained[
        q.type == k.type == v.type == k_rope.type == output.type,
        "Q, K, V, output should have same type.",
    ]()
    constrained[
        q.type == DType.float32 or q.type.is_half_float(),
        "Only support single and half precision.",
    ]()

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("k", k),
            trace_arg("v", v),
            trace_arg("output", output),
        )

    with Trace[TraceLevel.OP, target = ctx.device_info.api](
        "flare_mla_prefill",
        Trace[TraceLevel.OP, target = ctx.device_info.api]._get_detail_str[
            description_fn
        ](),
    ):
        var max_prompt_len: Int = q.dim[0]()

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()

        var k_operand = RaggedMHAOperand(k, cache_row_offsets)
        var v_operand = RaggedMHAOperand(v, cache_row_offsets)
        var k_rope_operand = NDBufferMHAOperand(k_rope)

        alias kv_num_heads = k_rope.shape.get[2]()
        alias cache_depth = k_rope.shape.get[3]()
        alias q_depth = q_shape.get[rank - 1]()  # hard code for now

        alias mha_config = MHAConfig(
            type,
            q_shape.get[rank - 2](),
            k.shape.get[rank - 1](),
            num_keys_per_block=UInt(64),
            WN=UInt(64),
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
            score_mod_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
        )


@always_inline
fn flare_mla_prefill_dispatch[
    rank: Int,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_shape: DimList, //,
    kv_num_heads: Int,
    use_score_mod: Bool = False,
    q_depth: Int = 192,
    cache_depth: Int = 576,
    config: MHAConfig = MHAConfig(
        type, q_shape.get[rank - 2](), q_shape.get[rank - 1]()
    ),
](
    output: NDBuffer[_, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: NDBuffer[DType.uint32, 1, *_],
    max_prompt_len: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    alias num_heads = config.num_heads
    alias depth = config.depth
    alias group = config.num_heads // kv_num_heads

    constrained[q_depth == q.shape.get[rank - 1]()]()
    constrained[num_heads == q.shape.get[rank - 2]()]()
    constrained[
        has_nvidia_gpu_accelerator(),
        "flareMLA_prefill currently only supports Nvidia GPUs.",
    ]()

    var batch_size: Int = valid_length.dim[0]() - 1

    alias q_half_float = type in (DType.float16, DType.bfloat16)

    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()

    alias q_smem = BM * q_depth
    alias k_smem = BN * q_depth
    alias v_smem = BN * depth

    alias smem_use = (q_smem + k_smem + v_smem) * config.type.sizeof()

    alias kernel = mla_prefill[
        config.type,
        k_t,
        v_t,
        k_rope_t,
        output.type,
        mask_t,
        score_mod_t,
        config,
        group=group,
        use_score_mod=use_score_mod,
        q_depth=q_depth,
        cache_depth=cache_depth,
    ]
    ctx.enqueue_function[kernel](
        q.data,
        k,
        v,
        k_rope,
        output.data,
        scale,
        batch_size,
        max_prompt_len,
        valid_length,
        mask_functor,
        score_mod_functor,
        grid_dim=(
            Int(ceildiv(max_prompt_len, BM)),
            Int(config.num_heads),
            Int(batch_size),
        ),
        block_dim=(Int(config.num_threads()), 1, 1),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn mla_prefill[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    config: MHAConfig,
    group: Int = 128,
    q_depth: Int = 192,
    cache_depth: Int = 576,
    use_score_mod: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    batch_size: Int,
    seq_len_arg: Int,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    mask: mask_t,
    score_mod: score_mod_t,
):
    alias depth = config.depth
    var batch_idx = block_idx.z

    # mha inputs
    var seq_len: Int
    var max_seq_len = seq_len_arg
    var num_keys: Int
    var start_pos: UInt32 = 0

    # treat valid_lengths as a input_row_offsets
    start_of_seq = Int(valid_length[batch_idx])
    end_of_seq = Int(valid_length[batch_idx + 1])
    seq_len = end_of_seq - start_of_seq

    if seq_len < block_idx.x * config.block_m():
        return

    num_keys = k.cache_length(batch_idx)
    start_pos = num_keys - seq_len

    q_batch_offset = start_of_seq * q_depth * config.num_heads
    o_batch_offset = start_of_seq * depth * config.num_heads

    mla_prefill_single_batch[
        config=config,
        group=group,
        q_depth=q_depth,
        cache_depth=cache_depth,
        use_score_mod=use_score_mod,
    ](
        q_ptr.offset(q_batch_offset),
        k,
        v,
        k_rope,
        output_ptr.offset(o_batch_offset),
        scale,
        seq_len,
        max_seq_len,
        start_pos,
        num_keys,
        mask,
        score_mod,
        batch_idx,
    )


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
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    num_keys: Int,
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
):
    """MLA for encoding where seqlen > 1."""
    alias k_type = k_t.type
    alias v_type = v_t.type
    alias k_rope_type = k_rope_t.type
    constrained[
        q_type == k_type and k_type == v_type and k_type == k_rope_type
    ]()

    alias simd_size = simdwidthof[q_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias num_heads = config.num_heads
    alias depth = config.depth

    alias rope_depth = q_depth - depth

    alias cache_num_heads = num_heads // group

    constrained[
        num_warps_m * num_warps_n == (num_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = thread_idx.x
    var warp_id: UInt32 = warp.broadcast(tid // WARP_SIZE)
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    # The entire query block (BM x q_depth) is tiled in shared memory.
    alias q_smem_size = BM * q_depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[q_type, simd_size]](),
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    alias k_smem_size = BN * q_depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    var k_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, k_smem_size)

    alias v_smem_size = BN * depth
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    var v_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, depth),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](v_smem, v_smem_size)

    var head_idx: UInt32 = block_idx.y
    var q_tile_idx: UInt32 = block_idx.x

    # Query global memory iterator
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(q_depth)), IntTuple(Int(num_heads * q_depth), 1)
    )
    var q_tile_num_rows = min(BM, UInt(seq_len) - q_tile_idx * BM)
    var q_offset = q_depth * (head_idx + num_heads * q_tile_idx * BM)

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
                num_heads * q_depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x q_depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias WM = config.WM
    alias WN = config.WN
    alias WN_O = depth
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N
    alias num_n_mmas_output = WN_O // MMA_N

    alias accum_type = get_accum_type[q_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    var output_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas_output, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().fill(0)

    # Rowwise max and sum for online softmax
    alias row_alignment = alignof[SIMD[accum_type, simdwidthof[accum_type]()]]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

    @parameter
    for i in range(0, Int(WM), 2):
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

    alias num_pipeline_stages = config.num_pipeline_stages

    alias q_num_vecs = BM * BK // simd_size

    alias async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK, BK // simd_size
    )

    @parameter
    for q_id in range(Int(q_depth // BK)):
        var q_smem_tile = q_smem_iter.next_unsafe(q_id)[]

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
            return

        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(num_heads * depth), 1),
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
                num_heads * depth, 1
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
                batch_idx, kv_tile_start_row, Int(head_idx), 0
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
                batch_idx, kv_tile_start_row, Int(head_idx), 0
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, depth, axis=0](0, 0)

        # here we set up varibles for k_rope tensor
        alias k_rope_gmem_layout = Layout(
            IntTuple(Int(BN), Int(cache_depth)),
            IntTuple(Int(cache_num_heads * cache_depth), 1),
        )

        var k_rope_runtime_layout = RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[k_rope_gmem_layout.shape, element_type = DType.int32](
                kv_tile_num_rows, cache_depth
            ),
            RuntimeTuple[k_rope_gmem_layout.stride, element_type = DType.int32](
                cache_num_heads * cache_depth, 1
            ),
        )

        var k_rope_gmem_block = LayoutTensor[
            k_rope_type,
            k_rope_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k_rope.block_paged_ptr[BN](
                batch_idx,
                kv_tile_start_row,
                Int(head_idx // group),
                cache_depth - rope_depth,
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
        fn _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: __type_of(tensor)
        ):
            return __type_of(tensor)(
                tensor.ptr,
                __type_of(tensor.runtime_layout)(
                    __type_of(tensor.runtime_layout.shape)(
                        num_rows, tensor.dim(1)
                    ),
                    tensor.runtime_layout.stride,
                ),
            )

        alias kv_num_vecs = BN * BK // simd_size
        alias async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // k_smem_iter.layout.stride[0].value(),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        @parameter
        for k_id in range(Int(depth // BK)):
            var k_smem_tile = k_smem_iter.next_unsafe(k_id)[]

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

        @parameter
        for k_id in range(Int(depth // BK), Int(q_depth // BK), 1):
            var k_smem_tile = k_smem_iter.next_unsafe(k_id)[]

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
            static_num_iters = Int(q_depth // BK),
            k_group_size = config.k_group_size,
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
        fn _apply_mask[masked: Bool]():
            var scale_log2e: SIMD[accum_type, 1] = (
                scale.cast[accum_type]() if use_score_mod
                or mask_t.apply_log2e_after_mask else scale.cast[accum_type]()
                * log2e
            )

            @parameter
            for m_mma in range(Int(num_m_mmas)):

                @parameter
                for n_mma in range(Int(num_n_mmas)):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + m_mma * MMA_M
                    var mask_frag_col = mask_warp_col + n_mma * MMA_N

                    # Offset to current thread's fragment
                    mask_frag_row += lane // (MMA_N // p_frag_simdwidth)
                    mask_frag_col += lane * p_frag_simdwidth % MMA_N

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = mask_block_row + mask_frag_row + i * MMA_M // 2
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
        alias reg_output_layout_by_mma_unit = Layout.row_major(
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

        alias async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // v_smem_iter.layout.stride[0].value(),
            v_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load V tile into smem
        @parameter
        for v_id in range(Int(BN // BK)):
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

            async_copy_commit_group()

            v_gmem_iter._incr()

        @parameter
        if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM, BN, BK, WM, WN, MMA_M, MMA_N, p_frag_simdwidth
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
                depth,
                BK,
                WM,
                depth,
                num_threads,
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
                BN // BK,
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(Int(num_m_mmas)):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        @parameter
        for n_mma in range(Int(num_n_mmas_output)):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    alias output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )

    var output_offset = depth * (head_idx + num_heads * q_tile_idx * BM)
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        output_ptr + Int(output_offset),
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
        copy[thread_layout = Layout.row_major(8, 4), swizzle=swizzle](
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
                num_threads * simd_size // depth, depth // simd_size
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


@always_inline
fn mla_prefill_plan[
    cache_t: KVCacheT,
](
    buffer_row_offsets: NDBuffer[mut=True, DType.uint32, 2, *_],
    cache_offsets: NDBuffer[mut=True, DType.uint32, 2, *_],
    buffer_lengths: NDBuffer[mut=True, DType.int32, 1, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
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

    alias kernel = mla_prefill_plan_kernel[
        buffer_lengths.shape,
        cache_t,
    ]

    ctx.enqueue_function[kernel](
        buffer_row_offsets,
        cache_offsets,
        buffer_lengths,
        input_row_offsets,
        k_cache,
        buffer_token_size,
        grid_dim=(Int(ceildiv(batch_size, 128)), 1, 1),
        block_dim=(128, 1, 1),
    )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
fn mla_prefill_plan_kernel[
    buffer_lengths_shape: DimList,
    cache_t: KVCacheT,
](
    buffer_row_offsets: NDBuffer[mut=True, DType.uint32, 2, MutableAnyOrigin],
    cache_offsets: NDBuffer[mut=True, DType.uint32, 2, MutableAnyOrigin],
    buffer_lengths: NDBuffer[
        mut=True, DType.int32, 1, MutableAnyOrigin, buffer_lengths_shape
    ],
    input_row_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    k_cache: cache_t,
    buffer_token_size: UInt32,
):
    var seq_idx = global_idx.x
    var seq_start_pos = 0
    var seq_end_pos = 0
    var batch_size: Int = input_row_offsets.dim[0]() - 1
    var buffer_size: Int = Int(buffer_token_size)

    alias MAX_CHUNKS = buffer_lengths.shape.get[0]()

    if seq_idx >= batch_size:
        return

    # Calculate starting position for this sequence
    for i in range(seq_idx):
        seq_start_pos += k_cache.cache_length(i)
    seq_start_pos += Int(input_row_offsets[seq_idx])

    # which chunk this sequence starts in
    var start_chunk = seq_start_pos // buffer_size

    var processed_seq_len = UInt32(0)
    var curr_seq_len = k_cache.cache_length(seq_idx) + Int(
        input_row_offsets[seq_idx + 1] - input_row_offsets[seq_idx]
    )
    var seq_len_left = curr_seq_len

    # Fill buffer offsets for this sequence
    @parameter
    for chunk_idx in range(MAX_CHUNKS):
        if chunk_idx < start_chunk:
            buffer_row_offsets[chunk_idx, seq_idx] = buffer_size
        elif chunk_idx == start_chunk:
            buffer_row_offsets[chunk_idx, seq_idx] = seq_start_pos % buffer_size
        else:
            buffer_row_offsets[chunk_idx, seq_idx] = 0

        cache_offsets[chunk_idx, seq_idx] = processed_seq_len

        var chunk_len = min(
            seq_len_left,
            buffer_size - Int(buffer_row_offsets[chunk_idx, seq_idx]),
        )
        processed_seq_len += chunk_len
        seq_len_left -= chunk_len

    # If this is the last sequence in the batch
    if seq_idx == batch_size - 1:
        seq_end_pos = seq_start_pos + curr_seq_len
        var end_chunk = seq_end_pos // buffer_size

        # Set buffer lengths for all chunks
        @parameter
        for chunk_idx in range(MAX_CHUNKS):
            if chunk_idx < end_chunk:
                buffer_row_offsets[chunk_idx, seq_idx + 1] = buffer_size
                buffer_lengths[chunk_idx] = buffer_size
            elif chunk_idx == end_chunk:
                buffer_row_offsets[chunk_idx, seq_idx + 1] = (
                    seq_end_pos % buffer_size
                )
                buffer_lengths[chunk_idx] = seq_end_pos % buffer_size
            else:
                buffer_row_offsets[chunk_idx, seq_idx + 1] = 0
                buffer_lengths[chunk_idx] = -1


# ===-----------------------------------------------------------------------===#
# Helper fucntion that copies K cache to a contiguous buffer
# ===-----------------------------------------------------------------------===#


@always_inline
fn _k_cache_to_buffer[
    type: DType,
    cache_t: KVCacheT,
](
    buffer_row_offsets: NDBuffer[DType.uint32, 1, *_],
    cache_offsets: NDBuffer[DType.uint32, 1, *_],
    k_cache: cache_t,
    length: Int32,
    buffer: NDBuffer[mut=True, type, 2, *_],
    context: DeviceContext,
) raises:
    alias num_heads = cache_t.kv_params.num_heads
    constrained[num_heads == 1, "num_heads should be equal to 1"]()

    @always_inline
    @parameter
    @__copy_capture(k_cache, buffer_row_offsets, cache_offsets)
    fn copy_fn[width: Int, rank: Int](idx_arg: IndexList[rank]):
        constrained[rank == 2, "rank should be equal to 2"]()

        var idx = rebind[IndexList[2]](idx_arg)
        var global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            buffer_row_offsets, global_token_idx
        )

        var token_idx = Int(
            global_token_idx
            - buffer_row_offsets[batch_idx]
            + cache_offsets[batch_idx]
        )

        var head_dim_idx = idx[1]

        var cache_val = rebind[SIMD[type, width]](
            k_cache.load[width=width](batch_idx, 0, token_idx, head_dim_idx)
        )

        buffer.store(idx, cache_val)

    var launch_shape = IndexList[2](
        Int(length),
        buffer.dim[1](),
    )
    alias compile_target = _get_gpu_target()
    alias target_simd_width = simdwidthof[type, target=compile_target]()

    _elementwise_impl_gpu[func=copy_fn, simd_width=target_simd_width](
        launch_shape, context
    )
