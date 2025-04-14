# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import ceildiv, recip
from math.constants import log2e
from sys import alignof, simdwidthof, sizeof

import gpu.warp as warp
from algorithm.functional import tile_and_unswitch, unswitch
from buffer import NDBuffer
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
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import (
    AddressSpace,
    CacheEviction,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from gpu.sync import async_copy_arrive, named_barrier
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_sram_to_dram,
    cp_async_k_major,
    cp_async_mn_major,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import PipelineState, SharedMemBarrier
from linalg._multistage_gemm_gpu import multistage_mma
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
from nn.softmax import (
    _online_softmax_correction,
    _online_softmax_iter_for_mma_output_sm90,
    _rowmax_online_softmax,
    _rowsum,
)
from nn.mha_tile_scheduler import (
    MHASchedule,
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    WorkInfo,
)

from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf, neg_inf
from utils.static_tuple import StaticTuple

from gpu.id import sm_id


@value
@register_passable("trivial")
struct MHAPosition[BM: Int, BN: Int, depth: Int, num_heads: Int]:
    var q_out_offset: Int
    var seq_len: UInt32
    var num_keys: Int
    var start_pos: UInt32
    var prompt_offset: UInt32
    var head_idx: UInt32
    var prompt_idx: UInt32

    alias q_output_gmem_layout = Layout(
        IntTuple(Int(Self.BM), Int(Self.depth)),
        IntTuple(Int(Self.num_heads * Self.depth), 1),
    )

    @always_inline
    fn __init__(
        out self,
        q_out_offset: Int,
        num_keys: Int,
        start_pos: UInt32,
        seq_info: SeqInfo,
    ):
        self.q_out_offset = q_out_offset
        self.seq_len = seq_info.seq_len
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.prompt_offset = seq_info.prompt_offset
        self.head_idx = seq_info.head_idx
        self.prompt_idx = seq_info.prompt_idx  # batch idx

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "(",
            self.q_out_offset,
            ", ",
            self.seq_len,
            ", ",
            self.num_keys,
            ", ",
            self.start_pos,
            ", ",
            self.prompt_offset,
            ", ",
            self.head_idx,
            ", ",
            self.prompt_idx,
            ")",
        )

    @always_inline
    fn q_tile_num_rows(self) -> UInt32:
        return min(BM, self.seq_len - self.prompt_offset)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.q_out_offset == other.q_out_offset

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self.q_out_offset != other.q_out_offset

    @always_inline
    fn q_out_gmem_tensor[
        dtype: DType
    ](
        self,
        ptr: UnsafePointer[Scalar[dtype]],
        out gmem_block: LayoutTensor[
            dtype,
            Self.q_output_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        gmem_block = __type_of(gmem_block)(
            ptr + self.q_out_offset,
            __type_of(gmem_block.runtime_layout)(
                __type_of(gmem_block.runtime_layout.shape)(
                    Int(self.q_tile_num_rows()), depth
                ),
                __type_of(gmem_block.runtime_layout.stride)(
                    num_heads * depth, 1
                ),
            ),
        )

    @always_inline
    fn mask_status[
        mask_t: MHAMask
    ](self, mask: mask_t, kv_tile_start_row: UInt32) -> TileMaskStatus:
        return mask.status(
            Index[type = DType.int32](
                Int(self.prompt_offset + self.start_pos),
                Int(kv_tile_start_row),
            ),
            Index[type = DType.int32](Int(Self.BM), Int(Self.BN)),
        )


@always_inline
fn _get_position[
    k_t: MHAOperand, //,
    config: MHAConfig,
    ragged: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    seq_info: SeqInfo,
    k: k_t,
    max_seq_len: UInt32,
    num_keys_arg: UInt32,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
) -> MHAPosition[
    config.block_m(), config.block_n(), config.depth, config.num_heads
]:
    alias depth = config.depth
    alias num_heads = config.num_heads

    var batch_idx: UInt32 = seq_info.prompt_idx
    # mha inputs
    var seq_len: UInt32 = seq_info.seq_len
    var num_keys: UInt32
    var start_pos: UInt32
    var q_batch_offset: Int

    @parameter
    if ragged:
        cache_len = k.cache_length(Int(batch_idx))

        @parameter
        if not _is_cache_length_accurate:
            start_pos = cache_len
        else:
            start_pos = 0

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        if kv_input_row_offsets:
            var kv_row_offsets = kv_input_row_offsets.value()
            kv_seq_start = Int(kv_row_offsets[Int(batch_idx)])
            kv_seq_end = Int(kv_row_offsets[Int(batch_idx) + 1])
            cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = cur_kv_len + cache_len
        else:
            num_keys = Int(seq_len) + cache_len
        q_batch_offset = Int(seq_info.start_of_seq) * Int(
            config.depth * config.num_heads
        )

    # NDBuffer inputs, homogeneous batching.
    else:
        num_keys = num_keys_arg

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len
        q_batch_offset = Int(depth * num_heads * batch_idx) * Int(max_seq_len)

    q_out_offset = q_batch_offset + Int(depth) * (
        Int(seq_info.head_idx) + Int(seq_info.prompt_offset) * Int(num_heads)
    )
    return MHAPosition[config.block_m(), config.block_n(), depth, num_heads](
        q_out_offset,
        Int(num_keys),
        start_pos,
        seq_info,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        config.num_threads[True]()
    )
)
fn mha_sm90[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    scheduler_t: MHATileScheduler,
    config: MHAConfig,
    group: Int = 1,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    is_shared_kv: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    scheduler: scheduler_t,
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    batch_size: Int,
    seq_len_arg: Int,
    num_keys_arg: Int,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
):
    # mha inputs
    var max_seq_len: Int = seq_len_arg

    constrained[config.algorithm == FlashAttentionAlgorithm(3)]()
    _mha_single_batch_sm90_fa3[
        config=config,
        group=group,
        use_score_mod=use_score_mod,
        ragged=ragged,
        _is_cache_length_accurate=_is_cache_length_accurate,
    ](
        q_ptr,
        k,
        v,
        output_ptr,
        scale,
        max_seq_len,
        num_keys_arg,
        valid_length,
        kv_input_row_offsets,
        mask,
        score_mod,
        scheduler,
        batch_size,
    )


@always_inline
fn _mask_tensor_row(
    tensor: LayoutTensor,
    num_rows: Int,
    out result: __type_of(tensor),
):
    return __type_of(tensor)(
        tensor.ptr,
        __type_of(tensor.runtime_layout)(
            __type_of(tensor.runtime_layout.shape)(num_rows, tensor.dim(1)),
            tensor.runtime_layout.stride,
        ),
    )


@always_inline("nodebug")
fn _produce[
    smem_layout: Layout,
    kv_t: MHAOperand,
    BM: Int,
    BN: Int,
    depth: Int,
    num_heads: Int, //,
    kv_num_heads: Int,
    BK: Int,
    group: Int,
    kv_gmem_layout: Layout,
    num_k_iters: Int,
    *,
    axis: Int,
    wait: Bool,
](
    write_idx: UInt32,
    write_phase: UInt32,
    kv_tile_start_row: UInt32,
    position: MHAPosition[BM, BN, depth, num_heads],
    consumed_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    produced_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    kv: kv_t,
    smem_iter: LayoutTensorIter[
        kv_t.type,
        smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
):
    kv_tile_num_rows = min(
        Int(BN), Int(position.num_keys) - Int(kv_tile_start_row)
    )

    alias kv_runtime_layout_type = RuntimeLayout[
        kv_gmem_layout,
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    kv_runtime_layout = kv_runtime_layout_type(
        RuntimeTuple[
            kv_gmem_layout.shape,
            element_type = kv_runtime_layout_type.element_type,
        ](kv_tile_num_rows, depth),
        RuntimeTuple[
            kv_gmem_layout.stride,
            element_type = kv_runtime_layout_type.linear_idx_type,
        ](kv_num_heads * depth, 1),
    )

    smem_subi = smem_iter.next_unsafe(Int(write_idx * num_k_iters))

    @parameter
    if wait:
        consumed_mbar[write_idx].wait(write_phase)

    @parameter
    @always_inline
    fn copy_gmem_to_smem[masked: Bool]():
        gmem_block = LayoutTensor[
            kv_t.type,
            kv_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=masked,
        ](
            kv.block_paged_ptr[BN](
                position.prompt_idx,
                Int(kv_tile_start_row),
                Int(position.head_idx // group),
                0,
            ),
            kv_runtime_layout,
        )
        gmem_iter = gmem_block.tiled_iterator[
            BK if axis == 0 else BN, BN if axis == 0 else BK, axis=axis
        ](0, 0)

        # load V tile into smem
        @parameter
        for k_id in range(num_k_iters):
            smem_tile = smem_subi.next_unsafe(k_id)[]

            @parameter
            if axis == 0 and masked:
                num_rows_bound = min(
                    Int(BK),
                    Int(position.num_keys)
                    - (Int(kv_tile_start_row) + k_id * BK),
                )
                tensor = _mask_tensor_row(gmem_iter[], num_rows_bound)
            else:
                tensor = gmem_iter[]

            @parameter
            if axis == 0:
                cp_async_mn_major(smem_tile, tensor)
            else:
                cp_async_k_major(smem_tile, tensor)

            gmem_iter._incr()

    unswitch[copy_gmem_to_smem](kv_tile_num_rows < BN)

    @parameter
    if wait:
        async_copy_arrive(produced_mbar + write_idx)
        _ = produced_mbar[write_idx].arrive()


@always_inline
fn _apply_mask[
    accum_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    reg_tile_layout: Layout, //,
    # last_iter: Bool,
    MMA_M: Int,
    MMA_N: Int,
    BM: Int,
    BN: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    p_frag_simdwidth: Int,
    use_score_mod: Bool,
](
    mask_warp_row: UInt32,
    mask_warp_col: UInt32,
    start_pos: UInt32,
    lane: UInt32,
    num_keys: UInt32,
    seq_len: UInt32,
    max_seq_len: Int,
    mask_block_row: UInt32,
    scale_log2e: Scalar[accum_type],
    mask: mask_t,
    mask_status: TileMaskStatus,
    score_mod: score_mod_t,
    p_reg_tile: LayoutTensor[
        accum_type,
        reg_tile_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ],
):
    # Vectorize by 2.
    p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

    @parameter
    @always_inline
    fn _apply_mask_capture[masked: Bool]():
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
                    score_row = mask_block_row + mask_frag_row + i * MMA_M // 2
                    score_row_with_start_pos = score_row + start_pos

                    @parameter
                    for j in range(MMA_N // 8):
                        score_col = mask_frag_col + j * 8

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i + j * 2] = mask.mask(
                                IndexList[4, element_type = DType.uint32](
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
                                    IndexList[4, element_type = DType.uint32,](
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
                        elif mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i + j * 2] = (
                                p_reg_vec2[mma_id, i + j * 2] * log2e
                            )

                        @parameter
                        if masked:
                            # if last_iter:
                            p_reg_vec2[mma_id, i + j * 2] = _kernel_mask(
                                IndexList[2, element_type = DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                IndexList[2, element_type = DType.uint32](
                                    Int(seq_len),
                                    Int(num_keys),
                                ),
                                p_reg_vec2[mma_id, i + j * 2],
                            )

    unswitch[_apply_mask_capture](mask_status == TileMaskStatus.PARTIAL_MASK)


@always_inline
fn _mha_single_batch_sm90_fa3[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    scheduler_t: MHATileScheduler,
    *,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    q_ptr_arg: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr_arg: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    max_seq_len: Int,  # sequence length after padding.
    num_keys_arg: Int,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    scheduler: scheduler_t,
    batch_size: UInt32,
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
    alias q_smem_size = config.q_smem_size(True)
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

    # var head_idx: UInt32 = block_idx.y
    # var q_tile_idx: UInt32 = block_idx.x

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

    produced_mbar_k = (v_smem + v_smem_size).bitcast[SharedMemBarrier]()
    consumed_mbar_k = produced_mbar_k + pipeline_stages
    produced_mbar_v = consumed_mbar_k + pipeline_stages
    consumed_mbar_v = produced_mbar_v + pipeline_stages
    produced_mbar_q = consumed_mbar_v + pipeline_stages
    consumed_mbar_q = produced_mbar_q + 2
    block_idx_ptr = (consumed_mbar_q + 2).bitcast[UInt32]()

    alias USE_TMA = False
    # https://github.com/Dao-AILab/flash-attention/blob/3b5047d2ce742848f45d44b143d511f211eba2d2/hopper/flash_fwd_kernel_sm90.h#L81-L82
    # alias num_producer_regs = 56 if num_consumer == 1 else (
    #     (24 if USE_TMA else 40) if num_consumer == 2 else 32
    # )
    # alias num_consumer_regs = 256 if num_consumer == 1 else (
    #     (240 if USE_TMA else 232) if num_consumer == 2 else 160
    # )
    alias num_producer_regs = 56
    alias num_consumer_regs = 224

    alias num_k_iters_0 = Int(depth // BK)
    alias num_k_iters_1 = Int(BN // BK)

    # constructing calls barrier()
    var tile_summary = MHATileSummary(
        batch_size, ceildiv(max_seq_len, BM), valid_length, max_seq_len
    )
    var state: MHATileState = scheduler.initial_state(
        block_idx_ptr, tile_summary
    )

    # returns `true` if we are done
    @parameter
    @always_inline
    fn advance[
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](pipeline_idx: UInt32) -> OptionalReg[SeqInfo]:
        return scheduler.advance[ragged, producer, sync](
            tile_summary, state, pipeline_idx
        )

    # The persistent kernels limit the grid size.
    # initial_seq_info = scheduler.unsafe_get_current_work_info(tile_summary, state)

    initial_seq_info = scheduler.unsafe_seq_info[ragged](tile_summary, state)

    if not initial_seq_info.is_valid():

        @parameter
        if scheduler_t.may_advance:
            seq_info = advance[True, MHASchedulerSynchronization.ALL](1)
            if seq_info:
                initial_seq_info = seq_info.value()
            else:
                return
        else:
            return

    if tid == 0:

        @parameter
        for i in range(pipeline_stages):
            # until we can use TMA, we need 128 producers working on async copies
            produced_mbar_k[i].init(128)
            consumed_mbar_k[i].init(num_consumer_threads)
            produced_mbar_v[i].init(128)
            consumed_mbar_v[i].init(num_consumer_threads)

        @parameter
        if scheduler_t.may_advance:

            @parameter
            for i in range(2):
                produced_mbar_q[i].init(128)
                consumed_mbar_q[i].init(num_consumer_threads)

    alias position_t = MHAPosition[BM, BN, depth, num_heads]

    @parameter
    @always_inline
    fn get_position(seq_info: SeqInfo) -> position_t:
        return _get_position[config, ragged, _is_cache_length_accurate](
            seq_info,
            k,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_input_row_offsets,
        )

    var position: position_t = get_position(initial_seq_info)

    q_pipeline_state = PipelineState[2]()

    barrier()
    # For intra-warp overlap, we initiate wgmmas as
    # Q @ K_0, Q @ K_1, P_0 @ V_0, Q @ K_2, P_1 @ V_1, ...
    # ..., Q @ K_{N-1}, P_{N-2} @ V_{N-2}, P_{N-1} @ V_{N-1}
    #
    # Due to this, we can overlap wgmmas and softmax calculations.
    if warp_group_idx == 0:
        # producer
        warpgroup_reg_dealloc[num_producer_regs]()
        write_pipeline_states = PipelineState[pipeline_stages]()

        # note that Q does not wait or arrive...
        # it assumes you can use K's
        @parameter
        @always_inline("nodebug")
        fn produce_q(
            position: position_t,
            pipeline_idx: UInt32,
        ):
            # Query global memory iterator
            q_gmem_block = position.q_out_gmem_tensor(q_ptr_arg)
            q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
            q_smem_subi = q_smem_iter.next_unsafe(
                Int(num_k_iters_0 * pipeline_idx)
            )

            # these copies get commited with the first `K`
            @parameter
            for q_id in range(num_k_iters_0):
                cp_async_k_major(q_smem_subi.next_unsafe(q_id)[], q_gmem_iter[])
                # cp_async_k_major[eviction_policy=CacheEviction.EVICT_FIRST](q_smem_subi.next_unsafe(q_id)[], q_gmem_iter[])

                q_gmem_iter._incr()

        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )

        @parameter
        @always_inline("nodebug")
        fn produce_k[
            wait: Bool
        ](
            write_idx: UInt32,
            write_phase: UInt32,
            kv_tile_start_row: UInt32,
            position: position_t,
        ):
            _produce[
                kv_num_heads,
                BK,
                group,
                kv_gmem_layout,
                num_k_iters_0,
                axis=1,
                wait=wait,
            ](
                write_idx,
                write_phase,
                kv_tile_start_row,
                position,
                consumed_mbar_k,
                produced_mbar_k,
                k,
                k_smem_iter,
            )

        @parameter
        @always_inline("nodebug")
        fn produce_v(
            write_idx: UInt32,
            write_phase: UInt32,
            kv_tile_start_row: UInt32,
            position: position_t,
        ):
            _produce[
                kv_num_heads,
                BK,
                group,
                kv_gmem_layout,
                num_k_iters_1,
                axis=0,
                wait=True,
            ](
                write_idx,
                write_phase,
                kv_tile_start_row,
                position,
                consumed_mbar_v,
                produced_mbar_v,
                v,
                v_smem_iter,
            )

        produce_q(position, q_pipeline_state.index())

        var kv_tile_start_row: UInt32 = 0

        while (
            position.mask_status(mask, kv_tile_start_row)
            == TileMaskStatus.FULL_MASK
        ):
            kv_tile_start_row += BN

        var write_idx: UInt32 = write_pipeline_states.index()
        var write_phase: UInt32 = write_pipeline_states.phase()

        var write_idx_prev: UInt32 = write_idx
        var write_phase_prev: UInt32 = write_phase
        var kv_tile_start_row_prev = kv_tile_start_row
        var position_prev: position_t = position

        produce_k[False](write_idx, write_phase, kv_tile_start_row, position)

        # wait to flip phase, but only bother after producing
        # there isn't any memory we can throttle
        async_copy_arrive(produced_mbar_k + write_idx)
        _ = produced_mbar_k[write_idx].arrive()
        # the order of the consumer's arrivals determines the
        # order of the producer's waits.
        # few_keys = num_keys <= BN

        # Process work with the tile size until there's not enough remaining work
        # to fit in a tile.
        # Production order:
        # Preheader: Q0, K0
        # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
        # Exit: V{-1}
        while True:
            # this loops over num_keys
            kv_tile_start_row += BN
            if kv_tile_start_row >= position.num_keys:

                @parameter
                if scheduler_t.may_advance:
                    kv_tile_start_row = 0
                    var q_idx_old: UInt32 = q_pipeline_state.index()
                    var q_phase_old: UInt32 = q_pipeline_state.phase()
                    q_pipeline_state.step()
                    consumed_mbar_q[q_idx_old].wait(q_phase_old)
                    # we must wait before advancing, as this mbar
                    # is for both `q_smem` and `sidx_ptr`
                    var q_idx: UInt32 = q_pipeline_state.index()
                    docontinue = advance[True](q_idx_old)
                    # commit writes to `q_smem` and to `sidx_ptr`
                    async_copy_arrive(produced_mbar_q + q_idx_old)
                    _ = produced_mbar_q[q_idx_old].arrive()
                    if not docontinue:
                        break
                    position = get_position(docontinue.value())
                    # merge with following produce_k
                    produce_q(position, q_idx)
                else:
                    break

            if (
                position.mask_status(mask, kv_tile_start_row)
                == TileMaskStatus.FULL_MASK
            ):
                continue
            # new pipeline states
            write_pipeline_states.step()
            write_idx = write_pipeline_states.index()
            write_phase = write_pipeline_states.phase()
            consumed_mbar_k[write_idx_prev].wait(write_phase_prev)
            produce_k[False](
                write_idx,
                write_phase,
                kv_tile_start_row,
                position,
            )
            async_copy_arrive(produced_mbar_k + write_idx)
            _ = produced_mbar_k[write_idx].arrive()
            produce_v(
                write_idx_prev,
                write_phase_prev,
                kv_tile_start_row_prev,
                position_prev,
            )
            # cache old
            write_idx_prev = write_idx
            write_phase_prev = write_phase
            kv_tile_start_row_prev = kv_tile_start_row
            position_prev = position

        produce_v(
            write_idx_prev,
            write_phase_prev,
            kv_tile_start_row_prev,
            position_prev,
        )

    else:
        warpgroup_reg_alloc[num_consumer_regs]()

        # arrive in order that that they're to be fetched.
        @parameter
        for i in range(pipeline_stages - 1):
            _ = consumed_mbar_k[i].arrive()
            _ = consumed_mbar_v[i].arrive()
        _ = consumed_mbar_v[pipeline_stages - 1].arrive()

        @parameter
        if scheduler_t.may_advance:
            _ = consumed_mbar_q[0].arrive()
        var local_warp_group_idx: UInt32 = warp_group_idx - 1

        # layout is
        # shape  = (2, num_m_mmas) x (2, num_n_mmas)
        # stride = (2, 4*num_n_mmas) x (1, 4)
        alias reg_tensor = LayoutTensor[
            accum_type,
            reg_tile_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        p_reg_tile = reg_tensor.stack_allocation()

        output_reg_tile = reg_tensor.stack_allocation().fill(0)

        p_frag = LayoutTensor[
            v_type,
            Layout.row_major(num_m_mmas * num_n_mmas * frag_ratio, a_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        @always_inline
        fn vectorize_output(
            out result: LayoutTensor[
                accum_type,
                vec_output_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
                element_layout=element_layout,
            ],
            x: reg_tensor,
        ):
            result = __type_of(result)(x.ptr)

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

        # Mask global memory iterator.
        mask_warp_row = warp_y * WM
        mask_warp_col = warp_x * WN
        var scale_log2e: Scalar[accum_type] = (
            scale.cast[accum_type]() if use_score_mod
            or mask_t.apply_log2e_after_mask else scale.cast[accum_type]()
            * log2e
        )
        read_pipeline_states = PipelineState[pipeline_stages]()

        @parameter
        @always_inline
        fn q_mul_k(read_idx: UInt32, read_phase: UInt32, q_idx: UInt32):
            k_smem_subi = k_smem_iter.next_unsafe(Int(read_idx * num_k_iters_0))
            q_smem_subi = q_smem_iter.next_unsafe(Int(q_idx * num_k_iters_0))
            produced_mbar_k[read_idx].wait(read_phase)
            wgmma_0.arrive()

            @parameter
            for k_iter in range(num_k_iters_0):
                alias scale_c = 0 if k_iter == 0 else 1
                wgmma_0.wgmma[num_consumer, scale_c=scale_c](
                    q_smem_subi.next_unsafe(k_iter)[],
                    k_smem_subi.next_unsafe(k_iter)[],
                    p_reg_tile,
                    Int(local_warp_group_idx),
                )
            wgmma_0.commit_group()

        @parameter
        @always_inline
        fn p_mul_v(read_idx: UInt32, read_phase: UInt32):
            v_smem_subi = v_smem_iter.next_unsafe(Int(read_idx * num_k_iters_1))
            produced_mbar_v[read_idx].wait(read_phase)
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

        @parameter
        @always_inline
        fn wait_for_q_mul_k[wgmma_left_in_flight: Int](read_idx: UInt32):
            wgmma_0.wait_group[wgmma_left_in_flight]()  # P is available
            _ = consumed_mbar_k[read_idx].arrive()

        @parameter
        @always_inline
        fn wait_for_p_mul_v(read_idx: UInt32):
            wgmma_1.wait_group[0]()  # output is available
            _ = consumed_mbar_v[read_idx].arrive()

        @parameter
        @always_inline
        fn apply_mask(
            position: position_t,
            mask_status: TileMaskStatus,
        ):
            _apply_mask[
                MMA_M,
                MMA_N,
                BM,
                BN,
                num_m_mmas,
                num_n_mmas,
                p_frag_simdwidth,
                use_score_mod,
            ](
                mask_warp_row,
                mask_warp_col,
                position.start_pos,
                lane,
                position.num_keys,
                position.seq_len,
                max_seq_len,
                position.prompt_offset,
                scale_log2e,
                mask,
                mask_status,
                score_mod,
                p_reg_tile,
            )

            # Increment mask to next BM x BN block.
            mask_warp_col += BN

        @parameter
        @always_inline
        fn scale_output(correction: __type_of(rowmax)):
            # we are now able to read/modify `output_reg_tile` and modify `p_frag`
            vout = vectorize_output(output_reg_tile)

            # Correct output
            # We could avoid this on the first iter
            # if we specialize and unswitch on `first_iter`
            # otherwise, the branch requires synchronization
            @parameter
            for col_tile in range(num_colwise_tiles):
                c = correction._get[col_tile, size = element_layout.size()]()

                @parameter
                for row_tile in range(num_rowwise_tiles):
                    vout._set[col_tile, row_tile](
                        vout._get[col_tile, row_tile]() * c
                    )

        @always_inline
        fn elementwise_reciprocal(
            old_rowsum: __type_of(rowsum), new_rowsum: __type_of(rowsum)
        ):
            # new_rowsum, old_rowsum = 1/old_rowsum, new_rowsum
            @parameter
            for row in range(num_rows_per_warp):
                old = old_rowsum._get[row]()
                new = new_rowsum._get[row]()
                new_rowsum._set[row](recip(old)[0])
                old_rowsum._set[row](new)

        @parameter
        @always_inline
        fn write_output(
            position: position_t,
            q_idx: UInt32,
            rowsum_inv: __type_of(rowsum),
        ):
            # Apply softmax denumerator.
            vout = vectorize_output(output_reg_tile)

            @parameter
            for row in range(num_rows_per_warp):
                rs_inv = vout.element_type(rowsum_inv._get[row]()[0])

                @parameter
                for col in range(num_cols_per_warp):
                    vout._set[row, col](vout._get[row, col]() * rs_inv)

            output_gmem_tile = position.q_out_gmem_tensor(output_ptr_arg)

            # Write to global memory.
            constrained[
                output_type.is_half_float(), "we don't support Float32 output"
            ]()
            constrained[sizeof[q_type]() == sizeof[output_type]()]()
            alias swizzle = make_swizzle[
                num_rows = MMA_M // 2, row_size=WN, access_size=8
            ]()
            # Reuse a_smem for c tile in smem
            alias q_tile_size: UInt32 = q_smem_size // 2
            accum_smem_tile = LayoutTensor[
                output_type,
                Layout.row_major(BM, depth),
                address_space = AddressSpace.SHARED,
            ]((q_smem + q_idx * q_tile_size).bitcast[Scalar[output_type]]())
            accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
                Int(warp_y), Int(warp_x)
            )

            # @parameter
            # if num_heads_per_block > 1:

            #     @parameter
            #     for i in range(reg_tile_layout.size()):
            #         output_reg_tile.ptr.store(
            #             i, output_reg_tile.ptr.load(i) * 1024
            #         )
            # ensure all threads have finished reading `q_smem`
            named_barrier[num_consumer_threads]()
            copy[thread_layout=mma_thread_layout, swizzle=swizzle](
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
                    num_consumer_threads * simd_size // depth,
                    depth // simd_size,
                ),
                swizzle=swizzle,
            ](
                output_gmem_tile.vectorize[1, simd_size](),
                accum_smem_tile.vectorize[1, simd_size](),
            )

        var read_idx: UInt32 = read_pipeline_states.index()
        var read_phase: UInt32 = read_pipeline_states.phase()
        var kv_tile_start_row: UInt32 = 0
        var mask_status: TileMaskStatus
        while True:
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status != TileMaskStatus.FULL_MASK:
                break
            kv_tile_start_row += BN
            mask_warp_col += BN

        # q_mul_k must wait on fetching q and k
        # therefore, we find `kv_tile_start_row` first.
        q_mul_k(read_idx, read_phase, q_pipeline_state.index())
        wait_for_q_mul_k[0](pipeline_stages - 1)
        # few_keys = num_keys <= BN

        apply_mask(position, mask_status)
        rowmax.copy_from(
            _rowmax_online_softmax[
                # threads layout by warp
                Layout.row_major(num_warps_m, num_warps_n),
                mma_thread_layout,
                use_exp2=True,
            ](vectorize_output(p_reg_tile), rowmax, init_rowmax=True)
        )
        rowsum.copy_from(
            _rowsum[mma_thread_layout](vectorize_output(p_reg_tile))
        )

        var read_idx_prev: UInt32 = read_idx
        var read_phase_prev: UInt32 = read_phase
        var position_prev: position_t = position
        var q_idx_old: UInt32 = q_pipeline_state.index()
        var q_phase_old: UInt32 = q_pipeline_state.phase()
        # Consumption order:
        # Preheader: Q0, K0
        # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
        # Exit: V{-1}
        while True:
            # this loops over num_keys
            kv_tile_start_row += BN
            if kv_tile_start_row >= position.num_keys:

                @parameter
                if scheduler_t.may_advance:
                    kv_tile_start_row = 0
                    mask_warp_col = warp_x * WN  # reset?
                    var q_idx_old: UInt32 = q_pipeline_state.index()
                    var q_phase_old: UInt32 = q_pipeline_state.phase()
                    q_pipeline_state.step()
                    produced_mbar_q[q_idx_old].wait(q_phase_old)
                    docontinue = advance[False](q_idx_old)
                    if not docontinue:
                        break
                    position = get_position(docontinue.value())
                else:
                    break

            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status == TileMaskStatus.FULL_MASK:
                mask_warp_col += BN
                continue
            # new pipeline states
            read_pipeline_states.step()
            read_idx = read_pipeline_states.index()
            read_phase = read_pipeline_states.phase()
            p_frag.vectorize[
                1, a_frag_size
            ]().copy_from(  # copy new pfrag, used by `p_mul_v` on next iter
                p_reg_tile.reshape[
                    Layout.row_major(
                        num_m_mmas * num_n_mmas * frag_ratio, a_frag_size
                    )
                ]().vectorize[1, a_frag_size](),
            )

            # start wgmmas
            q_mul_k(
                read_idx, read_phase, q_pipeline_state.index()
            )  # can't rw `p_reg_tile`
            p_mul_v(read_idx_prev, read_phase_prev)  # can't rw output or pfrag
            wait_for_q_mul_k[1](read_idx_prev)  # can rw `p_reg_tile`

            apply_mask(position, mask_status)
            new_q = (
                scheduler_t.may_advance
                and q_idx_old != q_pipeline_state.index()
            )
            score_frag_rowmax = _rowmax_online_softmax[
                # threads layout by warp
                Layout.row_major(num_warps_m, num_warps_n),
                mma_thread_layout,
                use_exp2=True,
            ](
                vectorize_output(p_reg_tile),
                rowmax,
                new_q,
            )
            if new_q:
                score_frag_rowsum = rebind[__type_of(rowsum)](
                    _rowsum[mma_thread_layout](vectorize_output(p_reg_tile))
                )
                rowmax.copy_from(score_frag_rowmax)
                elementwise_reciprocal(rowsum, score_frag_rowsum)
                wait_for_p_mul_v(read_idx_prev)  # can rw output and pfrag
                # we `^ 1` to access the previous
                # Two separate issues:
                # 0. Which q do we use for `accum_smem`?
                # 1. Which qs, if any, do we `arrive` at?
                #
                # If the next q_idx != the current q_idx (i.e. q_idx_n != q_idx)
                # then we can use the current q for writing smem.
                # If `q_idx_n == q_idx`, then we use the old q_idx (i.e. q_idx_o).
                # This means we were not allowed to `arrive` at `q_idx_o`.
                #
                # Letting `0` indicate inequality, and `1` equality,
                # let x = q_idx == q_idx_n
                # let y = q_idx_n == q_idx_n_n
                # We thus have 4 states `xy`:
                # 0. 00: We use q_idx and arrive
                # 1. 01: We use q_idx, but do not arrive on q_idx
                # 2. 10: We use q_idx_o, do not arrive on q_idx
                # 3. 11: We use q_idx_o, do not arrive on q_idx
                #
                # Only in `00` do we get to arrive on `q_idx` early.
                # Given `BN < num_keys`, it won't often be the case
                # that we can arrive at Q early; we need a series
                # of q_tile_idx and head_idx that have a lot of
                # `FULL_MASK`s, which our iteration scheme is supposed
                # to make unlikely.
                # Thus, we're going to simplify the problem by assuming
                # scenario `0.` is unlikely unless `BN >= num_keys`,
                # in which case it is guaranteed.
                # var q_idx: UInt32 = q_pipeline_state.index() if few_keys else q_idx_old
                write_output(position_prev, q_idx_old, score_frag_rowsum)
                var q_idx_new: UInt32 = q_pipeline_state.index()

                _ = consumed_mbar_q[q_idx_new].arrive()
                _ = output_reg_tile.vectorize[accum_simd_width]().fill(0)
                position_prev = position
                q_idx_old = q_idx_new
                q_phase_old = q_pipeline_state.phase()
            else:
                score_frag_rowsum = rebind[__type_of(rowsum)](
                    _rowsum[mma_thread_layout](vectorize_output(p_reg_tile))
                )
                _online_softmax_correction[use_exp2=True](
                    rowmax, score_frag_rowmax
                )
                # rowmax now holds score_frag_rowmax
                # score_frag_rowmax now holds the correction

                @parameter
                for i in range(num_rows_per_warp):
                    rowsum._set[i](
                        rowsum._get[i]() * score_frag_rowmax._get[i]()
                        + rebind[Scalar[accum_type]](
                            score_frag_rowsum._get[i]()
                        )
                    )

                wait_for_p_mul_v(read_idx_prev)  # can rw output and pfrag
                scale_output(score_frag_rowmax)  # scale output
            read_idx_prev = read_idx
            read_phase_prev = read_phase

        p_frag.vectorize[1, a_frag_size]().copy_from(
            p_reg_tile.reshape[
                Layout.row_major(
                    num_m_mmas * num_n_mmas * frag_ratio, a_frag_size
                )
            ]().vectorize[1, a_frag_size](),
        )
        p_mul_v(read_idx_prev, read_phase_prev)

        @parameter
        for row in range(num_rows_per_warp):
            rowsum._set[row](recip(rowsum._get[row]())[0])
        wgmma_1.wait_group()
        write_output(position, q_pipeline_state.index(), rowsum)
        # don't arrive
