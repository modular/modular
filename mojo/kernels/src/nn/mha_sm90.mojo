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
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
import gpu.warp as warp
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_local,
    copy_local_to_sram,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import make_swizzle
from layout.tensor_core import get_accum_type, get_fragment_size, get_mma_shape
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
)
from linalg._multistage_gemm_gpu import multistage_mma
from math import recip
from math.constants import log2e
from memory import UnsafePointer, stack_allocation
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand, NDBufferMHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_utils import MHAConfig, _copy_frag_to_smem, _kernel_mask
from nn.softmax import _online_softmax_iter_for_mma_output
from sys import alignof, simdwidthof
from utils.index import Index, IndexList
from utils.numerics import min_or_neg_inf, neg_inf
from utils.static_tuple import StaticTuple


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
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
    var batch_idx = block_idx.z

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

    mha_single_batch_sm90[
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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn mha_single_batch_sm90[
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
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias num_heads = config.num_heads
    alias depth = config.depth

    constrained[
        num_warps_m * num_warps_n <= 4,
        "Kernel requires single warp group, but we have:\nnum_warps_m = "
        + String(num_warps_m)
        + "\nnum_warps_n = "
        + String(num_warps_n),
    ]()
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

    alias q_smem_layout = tile_layout_k_major[
        DType.bfloat16, BM, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    alias k_smem_layout = tile_layout_k_major[
        DType.bfloat16, BN, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    # alias v_smem_layout = tile_layout_mn_major[DType.bfloat16, BN, BK, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B]()

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[q_type, simd_size]](),
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        q_smem_layout,
        # Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    q_smem_layout,
                    # Layout.row_major(BM, BK),
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
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
        k_smem_layout,
        # Layout.row_major(BN, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, k_smem_size)

    alias v_smem_size = config.kv_smem_size()
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
    var q_gmem_block = LayoutTensor[q_type, q_gmem_layout, masked=True,](
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
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias WM = config.WM
    alias WN = config.WN
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[q_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2

    alias wgmma_0 = TensorCoreAsync[
        accum_type,
        q_type,
        k_type,
        Index(64, 8, 16),
        a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
        b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
        transpose_b=True,
    ]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    var output_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
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

    # Account for group query.
    alias kv_num_heads = num_heads // group

    alias num_pipeline_stages = config.num_pipeline_stages

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
        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout(
            RuntimeTuple[kv_gmem_layout.shape, unsigned=True](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, unsigned=True](
                kv_num_heads * depth, 1
            ),
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
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

        # synchronize here since we can overlap q tile and first k tile copy
        async_copy_wait_all()
        barrier()

        alias static_num_iters = Int(depth // BK)

        wgmma_0.arrive()

        @parameter
        for k_iter in range(static_num_iters):
            wgmma_0.wgmma(
                q_smem_iter.next_unsafe(k_iter)[],
                k_smem_iter.next_unsafe(k_iter)[],
                p_reg_tile,
            )
        wgmma_0.commit_group()
        wgmma_0.wait_for_all()

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        @parameter
        fn _apply_mask[masked: Bool]():
            var scale_log2e: SIMD[accum_type, 1] = scale.cast[
                accum_type
            ]() if use_score_mod else scale.cast[accum_type]() * log2e

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
                        var score_row_with_start_pos = score_row + start_pos

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
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
                                    p_reg_vec2[mma_id, i],
                                    max_seq_len,
                                )
                                * log2e
                            )
                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[
                                    2, element_bitwidth=32, unsigned=True
                                ](Int(score_row), Int(score_col)),
                                IndexList[
                                    2, element_bitwidth=32, unsigned=True
                                ](
                                    seq_len,
                                    num_keys,
                                ),
                                p_reg_vec2[mma_id, i],
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
    for m_mma in range(Int(num_m_mmas)):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        @parameter
        for n_mma in range(Int(num_n_mmas)):

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
        copy_local_to_sram[
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
