# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from algorithm.functional import tile_and_unswitch, unswitch
from buffer import NDBuffer
from collections import OptionalReg
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
from gpu.cluster import elect_one_sync
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.intrinsics import warpgroup_reg_dealloc, warpgroup_reg_alloc
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from gpu.sync import async_copy_arrive, named_barrier
import gpu.warp as warp
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_sram,
    copy_sram_to_dram,
    cp_async_k_major,
    cp_async_mn_major,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import make_swizzle
from layout.tma_async import PipelineState, TMABarrier, create_mbarrier_array
from layout.tensor_core import get_fragment_size
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
)
from utils.numerics import get_accum_type
from linalg._multistage_gemm_gpu import multistage_mma
from math import recip
from math.constants import log2e
from memory import UnsafePointer, stack_allocation
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand, NDBufferMHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    _copy_frag_to_smem,
    _kernel_mask,
)
from nn.softmax import _online_softmax_iter_for_mma_output_sm90
from sys import alignof, simdwidthof
from utils.index import Index, IndexList
from utils.numerics import min_or_neg_inf, neg_inf
from utils.static_tuple import StaticTuple


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        config.num_threads[True]()
    )
)
fn mha_sm90[
    mask_rank: Int,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_type: DType,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    config: MHAConfig,
    group: Int = 1,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    is_shared_kv: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    batch_size: Int,
    seq_len_arg: Int,
    num_keys_arg: Int,
    valid_length: NDBuffer[DType.uint32, 1],
    kv_input_row_offsets: OptionalReg[NDBuffer[DType.uint32, 1]],
    mask: mask_t,
    score_mod: score_mod_t,
):
    alias depth = config.depth
    alias num_heads = config.num_heads
    batch_idx = block_idx.z

    # mha inputs
    var seq_len: Int
    var max_seq_len: Int
    var num_keys: Int
    var mask_tensor_col: Int
    var mask_batch_offset: Int
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
            var cache_length = k.cache_length(batch_idx)
            start_pos = cache_length

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        if kv_input_row_offsets:
            var kv_row_offsets = kv_input_row_offsets.value()
            kv_seq_start = Int(kv_row_offsets[batch_idx])
            kv_seq_end = Int(kv_row_offsets[batch_idx + 1])
            cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = cur_kv_len + k.cache_length(batch_idx)
        else:
            num_keys = seq_len + k.cache_length(batch_idx)

        max_seq_len = seq_len_arg
        mask_tensor_col = seq_len_arg
        q_batch_offset = start_of_seq * config.depth * config.num_heads
        mask_batch_offset = (
            batch_idx
            * max_seq_len
            * max_seq_len
            * (config.num_heads if mask_rank == 4 else 1)
        )
    # NDBuffer inputs, homogeneous batching.
    else:
        seq_len = seq_len_arg

        if seq_len < block_idx.x * config.block_m():
            return

        max_seq_len = seq_len_arg
        num_keys = num_keys_arg

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len

        mask_tensor_col = num_keys_arg
        q_batch_offset = (
            config.depth * config.num_heads * max_seq_len * batch_idx
        )
        mask_batch_offset = (
            batch_idx
            * max_seq_len
            * num_keys
            * (config.num_heads if mask_rank == 4 else 1)
        )

    @parameter
    if config.algorithm == FlashAttentionAlgorithm(3):
        _mha_single_batch_sm90_fa3[
            mask_rank,
            config=config,
            group=group,
            use_score_mod=use_score_mod,
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
        )
    else:
        constrained[config.algorithm == FlashAttentionAlgorithm(2)]()
        _mha_single_batch_sm90_fa2[
            mask_rank,
            config=config,
            group=group,
            use_score_mod=use_score_mod,
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
        )


@always_inline
fn _mha_single_batch_sm90_fa3[
    mask_rank: Int,
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
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 Frist bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias k_type = k_t.type
    alias v_type = v_t.type
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simdwidthof[q_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_consumer_threads = config.num_consumer_threads()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias num_heads = config.num_heads
    alias depth = config.depth
    # num_consumer_threads ignores the producers
    # actual number of threads is num_consumer_threads + 128
    alias num_consumer = num_consumer_threads // 128
    alias pipeline_stages = Int(config.num_pipeline_stages)
    var tid: UInt32 = thread_idx.x
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)
    # warp_group_tid = tid % 128

    constrained[
        num_warps_m * num_warps_n == (num_consumer_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var warp_id: UInt32 = warp.broadcast((tid - 128) // WARP_SIZE)
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    var warp_y: UInt32 = warp_id // num_warps_n
    var warp_x: UInt32 = warp_id % num_warps_n

    alias q_smem_layout = tile_layout_k_major[
        DType.bfloat16, BM, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    alias k_smem_layout = tile_layout_k_major[
        DType.bfloat16, BN, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    alias v_smem_layout = tile_layout_mn_major[
        DType.bfloat16, BN, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = config.q_smem_size()
    q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[q_type, simd_size]](),
    ]()
    q_smem_iter = LayoutTensorIter[
        q_type,
        q_smem_layout,
        # Layout.row_major(BM, BK),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    q_smem_layout,
                    # Layout.row_major(BM, BK),
                    MutableAnyOrigin,
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    # alias kv_smem_size = config.kv_smem_size(True)
    alias k_smem_size = config.kv_smem_size(True)
    alias v_smem_size = config.kv_smem_size(True)
    k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    k_smem_iter = LayoutTensorIter[
        k_type,
        k_smem_layout,
        # Layout.row_major(BN, BK),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        # circular=True,
    ](k_smem, k_smem_size)

    v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    v_smem_iter = LayoutTensorIter[
        v_type,
        # Layout.row_major(BK, BN),
        v_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        # circular=True,
    ](v_smem, v_smem_size)

    var head_idx: UInt32 = block_idx.y
    var q_tile_idx: UInt32 = block_idx.x

    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    constrained[BN == depth, "Block tile shape N doesn't match head dim"]()
    alias mma_shape = Index(64, depth, 16)
    alias MMA_M = mma_shape[0] // 4
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias WM = config.WM
    alias WN = config.WN
    alias num_m_mmas = WM // MMA_M
    constrained[num_m_mmas == 1, "FIXME: life this constraint"]()
    alias num_n_mmas = WN // MMA_N
    alias num_k_mmas = BK // MMA_K

    alias accum_type = get_accum_type[q_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = MMA_M * MMA_N // WARP_SIZE
    alias p_frag_simdwidth = 2

    alias a_frag_size = MMA_M * MMA_K // WARP_SIZE
    constrained[
        BN * num_k_mmas * a_frag_size == BK * num_n_mmas * p_frag_size
    ]()
    #
    alias frag_ratio = p_frag_size // a_frag_size

    alias q_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias k_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias v_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias wgmma_0 = TensorCoreAsync[
        accum_type,
        q_type,
        k_type,
        mma_shape,
        a_swizzle=q_swizzle,
        b_swizzle=k_swizzle,
        transpose_b=True,
    ]()
    alias wgmma_1 = TensorCoreAsync[
        accum_type,
        v_type,
        v_type,
        mma_shape,
        a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=v_swizzle,
        transpose_b=False,
    ]()

    alias reg_tile_layout = Layout.row_major(
        num_m_mmas * num_n_mmas, p_frag_size
    )
    alias num_row_blocks_per_mma = 2
    # a wgmma.m64n32k16 `D` fragment looks like
    #
    # 0,1  4,5   8, 9  12,13
    # 2,3  6,7  10,11  14,15
    #
    # Each row/column has `p_frag_simdwidth`-sized vectors
    # (e.g. `4,5` is of size 2 = p_frag_simdwidth)
    # We have `num_row_blocks_per_mma` rows.
    # The total number of elements (16) equals `p_frag_size`.
    # The number of columns equals
    # `p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth)`
    #
    # This gives us the layout:
    #
    # Note the ordering of strides:
    # ((1, 3), (0, 2, 4))
    # alias output_layout = Layout(
    #     IntTuple(
    #         IntTuple(num_row_blocks_per_mma, num_m_mmas),
    #         IntTuple(
    #             p_frag_simdwidth,
    #             p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth),
    #             num_n_mmas,
    #         ),
    #     ),
    #     IntTuple(
    #         IntTuple(p_frag_simdwidth, p_frag_size),
    #         IntTuple(1, 2 * p_frag_simdwidth, num_m_mmas * p_frag_size),
    #     ),
    # )
    # Vectorizing the layout:
    alias element_layout = Layout.row_major(1, p_frag_simdwidth)
    alias vec_output_layout = Layout(
        IntTuple(
            IntTuple(num_row_blocks_per_mma, num_m_mmas),
            IntTuple(
                p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(p_frag_simdwidth, p_frag_size),
            IntTuple(
                num_row_blocks_per_mma * p_frag_simdwidth,
                num_m_mmas * p_frag_size,
            ),
        ),
    )
    alias num_colwise_tiles = vec_output_layout[0].size()
    alias num_rowwise_tiles = vec_output_layout[1].size()

    @always_inline
    fn vectorize_output(
        out result: LayoutTensor[
            accum_type,
            vec_output_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=element_layout,
        ],
        x: LayoutTensor[
            accum_type, reg_tile_layout, address_space = AddressSpace.LOCAL
        ],
    ):
        result = __type_of(result)(x.ptr)

    # Rowwise max and sum for online softmax
    alias accum_simd_width = simdwidthof[accum_type]()
    alias row_alignment = alignof[SIMD[accum_type, accum_simd_width]]()
    alias num_rows_per_warp = vec_output_layout[0].size()
    alias num_cols_per_warp = vec_output_layout[1].size()
    # Account for group query.
    alias kv_num_heads = num_heads // group

    alias q_num_vecs = BM * BK // simd_size

    alias async_copy_q_layout = Layout.row_major(
        min(num_consumer_threads, q_num_vecs) * simd_size // BK, BK // simd_size
    )
    # var lane_predicate = elect_one_sync() # not needed with async_copy

    alias mma_thread_layout = Layout.row_major(8, 4)

    var barrier_ptr = (v_smem + v_smem_size + BM * depth).bitcast[Int64]()
    alias MbarPtrType = UnsafePointer[
        Int64, address_space = AddressSpace.SHARED, alignment=8
    ]
    produced_mbar_k_ptr = barrier_ptr
    consumed_mbar_k_ptr = barrier_ptr + pipeline_stages
    produced_mbar_v_ptr = barrier_ptr + 2 * pipeline_stages
    consumed_mbar_v_ptr = barrier_ptr + 3 * pipeline_stages

    produced_mbar_k = create_mbarrier_array[pipeline_stages](barrier_ptr)
    consumed_mbar_k = create_mbarrier_array[pipeline_stages](
        barrier_ptr + pipeline_stages
    )
    produced_mbar_v = create_mbarrier_array[pipeline_stages](
        barrier_ptr + 2 * pipeline_stages
    )
    consumed_mbar_v = create_mbarrier_array[pipeline_stages](
        barrier_ptr + 3 * pipeline_stages
    )

    if tid == 0:

        @parameter
        for i in range(pipeline_stages):
            # until we can use TMA, we need 128 producers working on async copies
            produced_mbar_k[i].init(128)
            consumed_mbar_k[i].init(num_consumer_threads)
            produced_mbar_v[i].init(128)
            consumed_mbar_v[i].init(num_consumer_threads)

    alias USE_TMA = False
    # https://github.com/Dao-AILab/flash-attention/blob/3b5047d2ce742848f45d44b143d511f211eba2d2/hopper/flash_fwd_kernel_sm90.h#L81-L82
    alias num_producer_regs = 56 if num_consumer == 1 else (
        (24 if USE_TMA else 40) if num_consumer == 2 else 32
    )
    alias num_consumer_regs = 256 if num_consumer == 1 else (
        (240 if USE_TMA else 232) if num_consumer == 2 else 160
    )
    alias num_k_iters_0 = Int(depth // BK)
    alias num_k_iters_1 = Int(BN // BK)
    var q_tile_num_rows: UInt32 = min(BM, UInt32(seq_len) - q_tile_idx * BM)
    var q_offset: UInt32 = depth * (head_idx + num_heads * q_tile_idx * BM)
    barrier()
    if warp_group_idx == 0:
        # producer
        warpgroup_reg_dealloc[num_producer_regs]()
        write_pipeline_states = PipelineState[pipeline_stages]()

        @always_inline
        @parameter
        fn _mask_tensor_row(
            tensor: LayoutTensor,
            num_rows: Int,
            out result: __type_of(tensor),
        ):
            return __type_of(tensor)(
                tensor.ptr,
                RuntimeLayout(
                    RuntimeTuple[tensor.layout.shape, unsigned=True](
                        num_rows, tensor.dim(1)
                    ),
                    tensor.runtime_layout.stride,
                ),
            )

        # Query global memory iterator
        alias q_gmem_layout = Layout(
            IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
        )
        q_gmem_block = LayoutTensor[q_type, q_gmem_layout, masked=True,](
            q_ptr + Int(q_offset),
            RuntimeLayout(
                RuntimeTuple[q_gmem_layout.shape, unsigned=True](
                    Int(q_tile_num_rows), depth
                ),
                RuntimeTuple[q_gmem_layout.stride, unsigned=True](
                    num_heads * depth, 1
                ),
            ),
        )
        q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

        @parameter
        for q_id in range(num_k_iters_0):
            cp_async_k_major(q_smem_iter.next_unsafe(q_id)[], q_gmem_iter[])

            q_gmem_iter._incr()

        # not needed with async_copy
        # if warp_group_tid == 0 and lane_predicate:
        @__copy_capture(seq_len, max_seq_len, num_keys, start_pos)
        @always_inline
        @parameter
        fn produce_kv[
            tile_size: Int, not_last_iter: Bool
        ](kv_tile_start_row: Int, end: Int):
            write_idx = write_pipeline_states.index()
            write_phase = write_pipeline_states.phase()
            if (
                mask.status(
                    Index[element_bitwidth=32, unsigned=True](
                        Int(q_tile_idx * BM + start_pos),
                        Int(kv_tile_start_row),
                    ),
                    Index[element_bitwidth=32, unsigned=True](Int(BM), Int(BN)),
                )
                == TileMaskStatus.FULL_MASK
            ):
                # wait to make sure we stay synchronized
                TMABarrier(MbarPtrType(consumed_mbar_k_ptr + write_idx)).wait(
                    write_phase
                )
                _ = TMABarrier(
                    MbarPtrType(produced_mbar_k_ptr + write_idx)
                ).arrive()
                TMABarrier(MbarPtrType(consumed_mbar_v_ptr + write_idx)).wait(
                    write_phase
                )
                _ = TMABarrier(
                    MbarPtrType(produced_mbar_v_ptr + write_idx)
                ).arrive()
                write_pipeline_states.step()
                return

            k_smem_subi = k_smem_iter.next_unsafe(write_idx * num_k_iters_0)
            alias kv_gmem_layout = Layout(
                IntTuple(Int(BN), Int(depth)),
                IntTuple(Int(kv_num_heads * depth), 1),
            )
            kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

            # kv cache gmem has to clip num rows as runtime layout
            kv_runtime_layout = RuntimeLayout(
                RuntimeTuple[kv_gmem_layout.shape, unsigned=True](
                    kv_tile_num_rows, depth
                ),
                RuntimeTuple[kv_gmem_layout.stride, unsigned=True](
                    kv_num_heads * depth, 1
                ),
            )

            k_gmem_block = LayoutTensor[
                k_type,
                kv_gmem_layout,
                masked = not not_last_iter,
            ](
                k.block_paged_ptr[BN](
                    batch_idx, kv_tile_start_row, Int(head_idx // group), 0
                ),
                kv_runtime_layout,
            )
            k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

            v_gmem_block = LayoutTensor[
                v_type,
                kv_gmem_layout,
                masked = not not_last_iter,
            ](
                v.block_paged_ptr[BN](
                    batch_idx, kv_tile_start_row, Int(head_idx // group), 0
                ),
                kv_runtime_layout,
            )
            v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

            TMABarrier(MbarPtrType(consumed_mbar_k_ptr + write_idx)).wait(
                write_phase
            )

            # load K tile into smem
            @parameter
            for k_id in range(num_k_iters_0):
                cp_async_k_major(k_smem_subi.next_unsafe(k_id)[], k_gmem_iter[])

                k_gmem_iter._incr()

            # synchronize here since we can overlap q tile and first k tile copy
            async_copy_arrive(produced_mbar_k_ptr + write_idx)
            _ = TMABarrier(
                MbarPtrType(produced_mbar_k_ptr + write_idx)
            ).arrive()

            TMABarrier(MbarPtrType(consumed_mbar_v_ptr + write_idx)).wait(
                write_phase
            )

            v_smem_subi = v_smem_iter.next_unsafe(write_idx * num_k_iters_1)

            # load V tile into smem
            @parameter
            for v_id in range(num_k_iters_1):
                v_smem_tile = v_smem_subi.next_unsafe(v_id)[]

                @parameter
                if not not_last_iter:
                    num_rows_bound = min(
                        Int(BK), end - (kv_tile_start_row + v_id * BK)
                    )
                    v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
                else:
                    v_tensor = v_gmem_iter[]

                cp_async_mn_major(v_smem_tile, v_tensor)

                v_gmem_iter._incr()

            async_copy_arrive(produced_mbar_v_ptr + write_idx)
            _ = TMABarrier(
                MbarPtrType(produced_mbar_v_ptr + write_idx)
            ).arrive()
            write_pipeline_states.step()

        tile_and_unswitch[produce_kv, VariadicList[Int](BN)](0, num_keys)

    else:
        warpgroup_reg_alloc[num_consumer_regs]()

        # order the `k`s first
        @parameter
        for i in range(pipeline_stages):
            _ = consumed_mbar_k[i].arrive()
            _ = consumed_mbar_v[i].arrive()

        var local_warp_group_idx: UInt32 = warp_group_idx - 1

        # layout is
        # shape  = (2, num_m_mmas) x (2, num_n_mmas)
        # stride = (2, 4*num_n_mmas) x (1, 4)
        p_reg_tile = LayoutTensor[
            accum_type,
            reg_tile_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        output_reg_tile = (
            LayoutTensor[
                accum_type,
                reg_tile_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ]
            .stack_allocation()
            .fill(0)
        )

        rowmax = LayoutTensor[
            accum_type,
            Layout.row_major(num_rows_per_warp),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        rowsum = LayoutTensor[
            accum_type,
            Layout.row_major(num_rows_per_warp),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        _ = rowmax.vectorize[accum_simd_width]().fill(
            min_or_neg_inf[accum_type]()
        )
        _ = rowsum.vectorize[accum_simd_width]().fill(0)

        # Mask global memory iterator.
        var mask_block_row: UInt32 = q_tile_idx * BM
        mask_warp_row = warp_y * WM
        mask_warp_col = warp_x * WN

        read_pipeline_states = PipelineState[pipeline_stages]()

        # consumer
        # Iterate over KV, equivalent to the following with if hoisted out.
        #   ```
        #   for i in range(kv_tile_start_row, seq_len, tile_size):
        #     if i + tile_size >= seq_len:
        #       consume_kv[tile_size, False]
        #     else:
        #       consume_kv[tile_size, True]
        #   ```
        # Only the last iteration is doing boundary check.
        @__copy_capture(seq_len, max_seq_len, num_keys, start_pos)
        @always_inline
        @parameter
        fn consume_kv[
            tile_size: Int, not_last_iter: Bool
        ](kv_tile_start_row: Int, end: Int):
            # for kv_tile_start_row in range(0, num_keys, BN):
            # var not_last_iter = kv_tile_start_row < num_keys - BN

            read_idx = read_pipeline_states.index()
            read_phase = read_pipeline_states.phase()
            if (
                mask.status(
                    Index[element_bitwidth=32, unsigned=True](
                        Int(q_tile_idx * BM + start_pos),
                        Int(kv_tile_start_row),
                    ),
                    Index[element_bitwidth=32, unsigned=True](Int(BM), Int(BN)),
                )
                == TileMaskStatus.FULL_MASK
            ):
                # wait to make sure we stay synchronized
                TMABarrier(MbarPtrType(produced_mbar_k_ptr + read_idx)).wait(
                    read_phase
                )
                _ = TMABarrier(
                    MbarPtrType(consumed_mbar_k_ptr + read_idx)
                ).arrive()
                TMABarrier(MbarPtrType(produced_mbar_v_ptr + read_idx)).wait(
                    read_phase
                )
                _ = TMABarrier(
                    MbarPtrType(consumed_mbar_v_ptr + read_idx)
                ).arrive()
                read_pipeline_states.step()
                return

            _ = p_reg_tile.fill(0)
            TMABarrier(MbarPtrType(produced_mbar_k_ptr + read_idx)).wait(
                read_phase
            )

            k_smem_subi = k_smem_iter.next_unsafe(read_idx * num_k_iters_0)
            wgmma_0.arrive()

            @parameter
            for k_iter in range(num_k_iters_0):
                wgmma_0.wgmma[num_consumer](
                    q_smem_iter.next_unsafe(k_iter)[],
                    k_smem_subi.next_unsafe(k_iter)[],
                    p_reg_tile,
                    Int(local_warp_group_idx),
                )
            wgmma_0.commit_group()
            wgmma_0.wait_for_all()

            # if warp_group_tid == 0:
            @parameter
            if not_last_iter:
                _ = TMABarrier(
                    MbarPtrType(consumed_mbar_k_ptr + read_idx)
                ).arrive()

            # Vectorize by 2.
            p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

            @always_inline
            @parameter
            fn _apply_mask[masked: Bool]():
                var scale_log2e: Scalar[accum_type] = (
                    scale.cast[accum_type]() if use_score_mod else scale.cast[
                        accum_type
                    ]()
                    * log2e
                )

                @parameter
                for m_mma in range(Int(num_m_mmas)):

                    @parameter
                    for n_mma in range(Int(num_n_mmas)):
                        alias mma_id = n_mma * num_m_mmas + m_mma
                        # Coordinates in mask for current mma tile.
                        mask_frag_row = mask_warp_row + m_mma * MMA_M
                        mask_frag_col = mask_warp_col + n_mma * MMA_N
                        # Offset to current thread's fragment
                        mask_frag_row += lane // 4
                        mask_frag_col += (lane * p_frag_simdwidth % MMA_N) % 8

                        @parameter
                        for i in range(2):
                            # The row in score matrix of shape seq_len x num_keys.
                            # Mask col is score col since we don't partition in col.
                            score_row = (
                                mask_block_row + mask_frag_row + i * MMA_M // 2
                            )
                            score_row_with_start_pos = score_row + start_pos

                            @parameter
                            for j in range(MMA_N // 8):
                                score_col = mask_frag_col + j * 8

                                @parameter
                                if masked:
                                    p_reg_vec2[mma_id, i + j * 2] = mask.mask(
                                        IndexList[
                                            4,
                                            element_bitwidth=32,
                                            unsigned=True,
                                        ](
                                            Int(block_idx.z),
                                            Int(block_idx.y),
                                            Int(score_row_with_start_pos),
                                            Int(score_col),
                                        ),
                                        p_reg_vec2[mma_id, i + j * 2]
                                        * scale_log2e,
                                    )
                                else:
                                    p_reg_vec2[mma_id, i + j * 2] = (
                                        p_reg_vec2[mma_id, i + j * 2]
                                        * scale_log2e
                                    )

                                @parameter
                                if use_score_mod:
                                    p_reg_vec2[mma_id, i + j * 2] = (
                                        score_mod.score_mod(
                                            IndexList[
                                                4,
                                                element_bitwidth=32,
                                                unsigned=True,
                                            ](
                                                Int(block_idx.z),
                                                Int(block_idx.y),
                                                Int(score_row_with_start_pos),
                                                Int(score_col),
                                            ),
                                            p_reg_vec2[mma_id, i + j * 2],
                                            max_seq_len,
                                        )
                                        * log2e
                                    )

                                @parameter
                                if not not_last_iter:
                                    p_reg_vec2[
                                        mma_id, i + j * 2
                                    ] = _kernel_mask(
                                        IndexList[
                                            2,
                                            element_bitwidth=32,
                                            unsigned=True,
                                        ](Int(score_row), Int(score_col)),
                                        IndexList[
                                            2,
                                            element_bitwidth=32,
                                            unsigned=True,
                                        ](
                                            seq_len,
                                            num_keys,
                                        ),
                                        p_reg_vec2[mma_id, i + j * 2],
                                    )

            unswitch[_apply_mask](
                mask.status(
                    Index[element_bitwidth=32, unsigned=True](
                        Int(q_tile_idx * BM + start_pos),
                        kv_tile_start_row,
                    ),
                    Index[element_bitwidth=32, unsigned=True](Int(BM), Int(BN)),
                )
                == TileMaskStatus.PARTIAL_MASK
            )

            # Increment mask to next BM x BN block.
            mask_warp_col += BN

            correction = _online_softmax_iter_for_mma_output_sm90[
                # threads layout by warp
                Layout.row_major(num_warps_m, num_warps_n),
                mma_thread_layout,
                use_exp2=True,
            ](
                vectorize_output(p_reg_tile),
                rowmax,
                rowsum,
            )
            vout = vectorize_output(output_reg_tile)

            # Correct previous result
            @parameter
            for col_tile in range(num_colwise_tiles):
                c = correction._get[col_tile, size = element_layout.size()]()

                @parameter
                for row_tile in range(num_rowwise_tiles):
                    vout._set[col_tile, row_tile](
                        vout._get[col_tile, row_tile]() * c
                    )

            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            p_frag = LayoutTensor[
                v_type,
                Layout.row_major(
                    num_m_mmas * num_n_mmas * frag_ratio, a_frag_size
                ),
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ].stack_allocation()
            # Convert 1st matmul's output FP32->BF16, layout are the same.
            p_frag.copy_from(
                p_reg_tile.reshape[
                    Layout.row_major(
                        num_m_mmas * num_n_mmas * frag_ratio, a_frag_size
                    )
                ](),
            )

            TMABarrier(MbarPtrType(produced_mbar_v_ptr + read_idx)).wait(
                read_phase
            )
            v_smem_subi = v_smem_iter.next_unsafe(read_idx * num_k_iters_1)

            wgmma_1.arrive()

            @parameter
            for k_iter in range(num_k_iters_1):
                wgmma_1.wgmma(
                    p_frag.tile[num_m_mmas * num_k_mmas, a_frag_size](
                        k_iter, 0
                    ),
                    v_smem_subi.next_unsafe(k_iter)[],
                    output_reg_tile,
                )
            wgmma_1.commit_group()
            wgmma_1.wait_for_all()

            @parameter
            if not_last_iter:
                _ = TMABarrier(
                    MbarPtrType(consumed_mbar_v_ptr + read_idx)
                ).arrive()
            read_pipeline_states.step()

        tile_and_unswitch[consume_kv, VariadicList[Int](BN)](0, num_keys)

        # Apply softmax denumerator.
        vout = vectorize_output(output_reg_tile)

        @parameter
        for row in range(num_rows_per_warp):
            rowsum_inv = vout.element_type(recip(rowsum._get[row]())[0])

            @parameter
            for col in range(num_cols_per_warp):
                vout._set[row, col](vout._get[row, col]() * rowsum_inv)

        alias output_gmem_layout = Layout(
            IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
        )
        output_gmem_tile = LayoutTensor[
            output_type, output_gmem_layout, masked=True
        ](
            output_ptr + Int(q_offset),
            RuntimeLayout(
                RuntimeTuple[output_gmem_layout.shape, unsigned=True](
                    Int(q_tile_num_rows), depth
                ),
                RuntimeTuple[output_gmem_layout.stride, unsigned=True](
                    num_heads * depth, 1
                ),
            ),
        )

        # Write to global memory.
        constrained[
            output_type.is_half_float(), "we don't support Float32 output"
        ]()
        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=WN, access_size=8
        ]()
        # Reuse a_smem for c tile in smem
        accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space = AddressSpace.SHARED,
        ]((v_smem + v_smem_size).bitcast[Scalar[output_type]]())
        accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
            Int(warp_y), Int(warp_x)
        )
        alias ortvsize = __type_of(
            output_reg_tile.vectorize[1, p_frag_size]()
        ).layout.size()
        # named_barrier[num_consumer_threads]()
        copy_local_to_sram[thread_layout=mma_thread_layout, swizzle=swizzle](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )
        # Guard writing to shared memory.
        named_barrier[num_consumer_threads]()
        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout = Layout.row_major(
                num_consumer_threads * simd_size // depth, depth // simd_size
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )


@always_inline
fn _mha_single_batch_sm90_fa2[
    mask_rank: Int,
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
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 Frist bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias k_type = k_t.type
    alias v_type = v_t.type
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simdwidthof[q_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()
    alias num_consumer = num_threads // 128
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias num_heads = config.num_heads
    alias depth = config.depth

    constrained[
        num_warps_m * num_warps_n == (num_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()
    constrained[num_warps_m == 4, "num_warps_m != 4 currently unsupported."]()
    constrained[num_warps_n == 1, "num_warps_n != 1 currently unsupported."]()

    var tid: UInt32 = thread_idx.x
    var warp_id: UInt32 = warp.broadcast(tid // WARP_SIZE)
    var warp_group_idx: UInt32 = warp.broadcast(
        tid // 128
    ) if num_consumer > 1 else 0
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    warp_y = warp_id // num_warps_n
    warp_x = warp_id % num_warps_n

    alias q_smem_layout = tile_layout_k_major[
        DType.bfloat16, BM, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    alias k_smem_layout = tile_layout_k_major[
        DType.bfloat16, BN, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    alias v_smem_layout = tile_layout_mn_major[
        DType.bfloat16, BN, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = config.q_smem_size()
    q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[q_type, simd_size]](),
    ]()
    q_smem_iter = LayoutTensorIter[
        q_type,
        q_smem_layout,
        # Layout.row_major(BM, BK),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    q_smem_layout,
                    # Layout.row_major(BM, BK),
                    MutableAnyOrigin,
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    alias k_smem_size = config.k_smem_size(False)
    k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    k_smem_iter = LayoutTensorIter[
        k_type,
        k_smem_layout,
        # Layout.row_major(BN, BK),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        # circular=True,
    ](k_smem, k_smem_size)

    alias v_smem_size = config.v_smem_size(False)
    v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    v_smem_iter = LayoutTensorIter[
        v_type,
        # Layout.row_major(BK, BN),
        v_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        # circular=True,
    ](v_smem, v_smem_size)

    var head_idx: UInt32 = block_idx.y
    var q_tile_idx: UInt32 = block_idx.x

    # Query global memory iterator
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    q_tile_num_rows = min(BM, UInt(seq_len) - q_tile_idx * BM)
    q_offset = depth * (head_idx + num_heads * q_tile_idx * BM)
    q_gmem_block = LayoutTensor[q_type, q_gmem_layout, masked=True,](
        q_ptr + Int(q_offset),
        RuntimeLayout(
            RuntimeTuple[q_gmem_layout.shape, unsigned=True](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[q_gmem_layout.stride, unsigned=True](
                num_heads * depth, 1
            ),
        ),
    )
    q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    constrained[BN == depth, "Block tile shape N doesn't match head dim"]()
    alias mma_shape = Index(64, depth, 16)
    alias MMA_M = mma_shape[0] // 4
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias WM = config.WM
    alias WN = config.WN
    alias num_m_mmas = WM // MMA_M
    constrained[num_m_mmas == 1, "FIXME: life this constraint"]()
    alias num_n_mmas = WN // MMA_N
    alias num_k_mmas = BK // MMA_K

    alias accum_type = get_accum_type[q_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = MMA_M * MMA_N // WARP_SIZE
    alias p_frag_simdwidth = 2

    alias a_frag_size = MMA_M * MMA_K // WARP_SIZE
    constrained[
        BN * num_k_mmas * a_frag_size == BK * num_n_mmas * p_frag_size
    ]()
    #
    alias frag_ratio = p_frag_size // a_frag_size

    alias q_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias k_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias v_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias wgmma_0 = TensorCoreAsync[
        accum_type,
        q_type,
        k_type,
        mma_shape,
        a_swizzle=q_swizzle,
        b_swizzle=k_swizzle,
        transpose_b=True,
    ]()
    alias wgmma_1 = TensorCoreAsync[
        accum_type,
        v_type,
        v_type,
        mma_shape,
        a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=v_swizzle,
        transpose_b=False,
    ]()

    alias reg_tile_layout = Layout.row_major(
        num_m_mmas * num_n_mmas, p_frag_size
    )
    # layout is
    # shape  = (2, num_m_mmas) x (2, num_n_mmas)
    # stride = (2, 4*num_n_mmas) x (1, 4)
    p_reg_tile = LayoutTensor[
        accum_type,
        reg_tile_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    output_reg_tile = (
        LayoutTensor[
            accum_type,
            reg_tile_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )
    alias num_row_blocks_per_mma = 2
    # a wgmma.m64n32k16 `D` fragment looks like
    #
    # 0,1  4,5   8, 9  12,13
    # 2,3  6,7  10,11  14,15
    #
    # Each row/column has `p_frag_simdwidth`-sized vectors
    # (e.g. `4,5` is of size 2 = p_frag_simdwidth)
    # We have `num_row_blocks_per_mma` rows.
    # The total number of elements (16) equals `p_frag_size`.
    # The number of columns equals
    # `p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth)`
    #
    # This gives us the layout:
    #
    # Note the ordering of strides:
    # ((1, 3), (0, 2, 4))
    # alias output_layout = Layout(
    #     IntTuple(
    #         IntTuple(num_row_blocks_per_mma, num_m_mmas),
    #         IntTuple(
    #             p_frag_simdwidth,
    #             p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth),
    #             num_n_mmas,
    #         ),
    #     ),
    #     IntTuple(
    #         IntTuple(p_frag_simdwidth, p_frag_size),
    #         IntTuple(1, 2 * p_frag_simdwidth, num_m_mmas * p_frag_size),
    #     ),
    # )
    # Vectorizing the layout:
    alias element_layout = Layout.row_major(1, p_frag_simdwidth)
    alias vec_output_layout = Layout(
        IntTuple(
            IntTuple(num_row_blocks_per_mma, num_m_mmas),
            IntTuple(
                p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(p_frag_simdwidth, p_frag_size),
            IntTuple(
                num_row_blocks_per_mma * p_frag_simdwidth,
                num_m_mmas * p_frag_size,
            ),
        ),
    )
    alias num_colwise_tiles = vec_output_layout[0].size()
    alias num_rowwise_tiles = vec_output_layout[1].size()

    @always_inline
    fn vectorize_output(
        out result: LayoutTensor[
            accum_type,
            vec_output_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
            element_layout=element_layout,
        ],
        x: LayoutTensor[
            accum_type, reg_tile_layout, address_space = AddressSpace.LOCAL
        ],
    ):
        result = __type_of(result)(x.ptr)

    # Rowwise max and sum for online softmax
    alias accum_simd_width = simdwidthof[accum_type]()
    alias row_alignment = alignof[SIMD[accum_type, accum_simd_width]]()
    alias num_rows_per_warp = vec_output_layout[0].size()
    alias num_cols_per_warp = vec_output_layout[1].size()
    rowmax = LayoutTensor[
        accum_type,
        Layout.row_major(num_rows_per_warp),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    rowsum = LayoutTensor[
        accum_type,
        Layout.row_major(num_rows_per_warp),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    _ = rowmax.vectorize[accum_simd_width]().fill(min_or_neg_inf[accum_type]())
    _ = rowsum.vectorize[accum_simd_width]().fill(0)

    # Mask global memory iterator.
    var mask_block_row: UInt32 = q_tile_idx * BM
    mask_warp_row = warp_y * WM
    mask_warp_col = warp_x * WN

    # Account for group query.
    alias kv_num_heads = num_heads // group

    alias num_pipeline_stages = config.num_pipeline_stages

    alias q_num_vecs = BM * BK // simd_size

    alias async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK, BK // simd_size
    )

    alias mma_thread_layout = Layout.row_major(8, 4)

    alias num_k_iters_0 = Int(depth // BK)
    alias num_k_iters_1 = Int(BN // BK)

    @parameter
    for q_id in range(num_k_iters_0):
        q_smem_tile = q_smem_iter.next_unsafe(q_id)[]

        cp_async_k_major(q_smem_tile, q_gmem_iter[])

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
                Index[element_bitwidth=32, unsigned=True](
                    Int(q_tile_idx * BM + start_pos),
                    Int(kv_tile_start_row),
                ),
                Index[element_bitwidth=32, unsigned=True](Int(BM), Int(BN)),
            )
            == TileMaskStatus.FULL_MASK
        ):
            return

        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )
        kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        kv_runtime_layout = RuntimeLayout(
            RuntimeTuple[kv_gmem_layout.shape, unsigned=True](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, unsigned=True](
                kv_num_heads * depth, 1
            ),
        )

        k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            masked = not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            masked = not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        @always_inline
        @parameter
        fn _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: __type_of(tensor)
        ):
            return __type_of(tensor)(
                tensor.ptr,
                RuntimeLayout(
                    RuntimeTuple[tensor.layout.shape, unsigned=True](
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
        for k_id in range(num_k_iters_0):
            k_smem_tile = k_smem_iter.next_unsafe(k_id)[]

            cp_async_k_major(k_smem_tile, k_gmem_iter[])

            async_copy_commit_group()

            k_gmem_iter._incr()

        # synchronize here since we can overlap q tile and first k tile copy
        async_copy_wait_all()
        barrier()

        wgmma_0.arrive()

        @parameter
        for k_iter in range(num_k_iters_0):
            wgmma_0.wgmma[num_consumer](
                q_smem_iter.next_unsafe(k_iter)[],
                k_smem_iter.next_unsafe(k_iter)[],
                p_reg_tile,
                Int(warp_group_idx),
            )
        wgmma_0.commit_group()
        wgmma_0.wait_for_all()

        # Vectorize by 2.
        p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        @always_inline
        @parameter
        fn _apply_mask[masked: Bool]():
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[accum_type]() if use_score_mod else scale.cast[
                    accum_type
                ]()
                * log2e
            )

            @parameter
            for m_mma in range(Int(num_m_mmas)):

                @parameter
                for n_mma in range(Int(num_n_mmas)):
                    alias mma_id = n_mma * num_m_mmas + m_mma
                    # Coordinates in mask for current mma tile.
                    mask_frag_row = mask_warp_row + m_mma * MMA_M
                    mask_frag_col = mask_warp_col + n_mma * MMA_N
                    # Offset to current thread's fragment
                    mask_frag_row += lane // 4
                    mask_frag_col += (lane * p_frag_simdwidth % MMA_N) % 8

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        score_row = (
                            mask_block_row + mask_frag_row + i * MMA_M // 2
                        )
                        score_row_with_start_pos = score_row + start_pos

                        @parameter
                        for j in range(MMA_N // 8):
                            score_col = mask_frag_col + j * 8

                            @parameter
                            if masked:
                                p_reg_vec2[mma_id, i + j * 2] = mask.mask(
                                    IndexList[
                                        4,
                                        element_bitwidth=32,
                                        unsigned=True,
                                    ](
                                        Int(block_idx.z),
                                        Int(block_idx.y),
                                        Int(score_row_with_start_pos),
                                        Int(score_col),
                                    ),
                                    p_reg_vec2[mma_id, i + j * 2] * scale_log2e,
                                )
                            else:
                                p_reg_vec2[mma_id, i + j * 2] = (
                                    p_reg_vec2[mma_id, i + j * 2] * scale_log2e
                                )

                            @parameter
                            if use_score_mod:
                                p_reg_vec2[mma_id, i + j * 2] = (
                                    score_mod.score_mod(
                                        IndexList[
                                            4,
                                            element_bitwidth=32,
                                            unsigned=True,
                                        ](
                                            Int(block_idx.z),
                                            Int(block_idx.y),
                                            Int(score_row_with_start_pos),
                                            Int(score_col),
                                        ),
                                        p_reg_vec2[mma_id, i + j * 2],
                                        max_seq_len,
                                    )
                                    * log2e
                                )
                            if not not_last_iter:
                                p_reg_vec2[mma_id, i + j * 2] = _kernel_mask(
                                    IndexList[
                                        2, element_bitwidth=32, unsigned=True
                                    ](Int(score_row), Int(score_col)),
                                    IndexList[
                                        2, element_bitwidth=32, unsigned=True
                                    ](
                                        seq_len,
                                        num_keys,
                                    ),
                                    p_reg_vec2[mma_id, i + j * 2],
                                )

        unswitch[_apply_mask](
            mask.status(
                Index[element_bitwidth=32, unsigned=True](
                    Int(q_tile_idx * BM + start_pos),
                    kv_tile_start_row,
                ),
                Index[element_bitwidth=32, unsigned=True](Int(BM), Int(BN)),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        correction = _online_softmax_iter_for_mma_output_sm90[
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            mma_thread_layout,
            use_exp2=True,
        ](
            vectorize_output(p_reg_tile),
            rowmax,
            rowsum,
        )
        vout = vectorize_output(output_reg_tile)

        # Correct previous result
        @parameter
        for col_tile in range(num_colwise_tiles):
            c = correction._get[col_tile, size = element_layout.size()]()

            @parameter
            for row_tile in range(num_rowwise_tiles):
                vout._set[col_tile, row_tile](
                    vout._get[col_tile, row_tile]() * c
                )

        # load V tile into smem
        @parameter
        for v_id in range(num_k_iters_1):
            v_smem_tile = v_smem_iter.next_unsafe(v_id)[]

            @parameter
            if not not_last_iter:
                num_rows_bound = min(
                    Int(BK), end - (kv_tile_start_row + v_id * BK)
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            cp_async_mn_major(v_smem_tile, v_tensor)

            async_copy_commit_group()

            v_gmem_iter._incr()

        # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
        # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
        p_reg_iter = p_reg_tile.reshape[
            Layout.row_major(
                num_m_mmas * num_n_mmas * frag_ratio,
                a_frag_size,
            )
        ]().tiled_iterator[num_m_mmas * num_k_mmas, a_frag_size](0, 0)
        async_copy_wait_all()
        barrier()
        p_frag = LayoutTensor[
            v_type,
            Layout.row_major(num_m_mmas * num_k_mmas, a_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        wgmma_1.arrive()

        @parameter
        for k_iter in range(num_k_iters_1):
            p_frag.copy_from(p_reg_iter.next_unsafe(k_iter)[])

            wgmma_1.wgmma(
                p_frag,
                v_smem_iter.next_unsafe(k_iter)[],
                output_reg_tile,
            )
        wgmma_1.commit_group()
        wgmma_1.wait_for_all()

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    # Apply softmax denumerator.
    vout = vectorize_output(output_reg_tile)

    @parameter
    for row in range(num_rows_per_warp):
        rowsum_inv = vout.element_type(recip(rowsum._get[row]())[0])

        @parameter
        for col in range(num_cols_per_warp):
            vout._set[row, col](vout._get[row, col]() * rowsum_inv)

    alias output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    output_gmem_tile = LayoutTensor[
        output_type, output_gmem_layout, masked=True
    ](
        output_ptr + Int(q_offset),
        RuntimeLayout(
            RuntimeTuple[output_gmem_layout.shape, unsigned=True](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[output_gmem_layout.stride, unsigned=True](
                num_heads * depth, 1
            ),
        ),
    )

    # Write to global memory.
    constrained[
        output_type.is_half_float(), "we don't support Float32 output"
    ]()
    alias swizzle = make_swizzle[
        num_rows = MMA_M // 2, row_size=WN, access_size=8
    ]()
    # Reuse a_smem for c tile in smem
    accum_smem_tile = LayoutTensor[
        output_type,
        Layout.row_major(BM, depth),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ](q_smem.bitcast[Scalar[output_type]]())
    accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )
    alias ortvsize = __type_of(
        output_reg_tile.vectorize[1, p_frag_size]()
    ).layout.size()
    copy_local_to_sram[thread_layout=mma_thread_layout, swizzle=swizzle](
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
