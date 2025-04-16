# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray, OptionalReg
from math import align_down, align_up, ceildiv, exp, recip
from math.constants import log2e
from pathlib import Path
from sys import alignof, simdwidthof, sizeof
from sys.intrinsics import readfirstlane

import gpu.warp as warp
from algorithm.functional import tile_and_unswitch, unswitch, vectorize
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    grid_dim,
    lane_id,
    warp_id as get_warp_id,
    thread_idx,
)
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from gpu.mma import mma as mma_simd
from gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import hash, idx2crd
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    copy,
    copy_dram_to_local,
    copy_dram_to_sram,
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_builder import static
from layout.tensor_core import TensorCore, get_mma_shape, num_matrix_reg
from linalg.utils import GemmShape, apply_epilogue, elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig
from memory import UnsafePointer, stack_allocation
from nn.mha_mask import MHAMask, NullMask, TileMaskStatus
from nn.mha_operand import KVCacheMHAOperand, MHAOperand, NDBufferMHAOperand
from nn.mha_utils import MHAConfig, _kernel_mask
from nn.softmax import (
    _online_softmax_iter_for_mma_output,
    _online_softmax_iter_for_mma_output_split_warp_reduce,
    _softmax_gpu,
    softmax,
)

from utils import Index, IndexList, StaticTuple
from utils.numerics import get_accum_type, min_or_neg_inf, neg_inf


@always_inline
fn tensor_core_mma[
    swap_operands: Bool = False
](a_frag: LayoutTensor, b_frag: LayoutTensor, c_frag: LayoutTensor):
    alias num_m_mmas = a_frag.shape[0]()
    alias num_n_mmas = b_frag.shape[0]()
    constrained[
        c_frag.shape[0]() == num_m_mmas * num_n_mmas,
        "Fragments size mismatch. Expected c_frag shape[0] to be num_m_mmas"
        " * num_n_mmas = "
        + String(num_m_mmas * num_n_mmas)
        + ", got "
        + String(c_frag.shape[0]()),
    ]()

    @always_inline
    fn flatten_index(m_mma: Int, n_mma: Int) -> Int:
        @parameter
        if not swap_operands:
            return n_mma * num_m_mmas + m_mma
        else:
            return m_mma * num_n_mmas + n_mma

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            mma_simd(
                c_frag[flatten_index(m_mma, n_mma), 0],
                a_frag[m_mma, 0],
                b_frag[n_mma, 0],
                c_frag[flatten_index(m_mma, n_mma), 0],
            )


@always_inline
fn mma[
    transpose_b: Bool,
    k_group_size: Int,
    config: MHAConfig,
    swizzle: OptionalReg[Swizzle] = None,
    swap_operands: Bool = False,
    num_iters: Int = 1,
    token_gen: Bool = False,
](
    c: LayoutTensor,
    mut a_iter: LayoutTensorIter,
    a_smem_iter: LayoutTensorIter,
    mut b_iter: LayoutTensorIter,
    b_smem_iter: LayoutTensorIter,
    num_b_rows: OptionalReg[Int] = None,
):
    alias BK = config.block_k()
    # a can be either bfloat16 or float32 but b is always the same type as mma_input_type
    alias mma_input_type = b_iter.type
    alias simd_width = simdwidthof[mma_input_type]()
    alias mma_shape = get_mma_shape[
        mma_input_type, get_accum_type[mma_input_type]()
    ]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias accum_type = get_accum_type[mma_input_type]()
    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    var warp_id = get_warp_id()
    alias num_warps = config.num_threads() // WARP_SIZE
    alias num_threads = config.num_threads()
    alias num_warps_n = BN // WN

    var warp_row = warp_id // num_warps_n
    var warp_col = warp_id % num_warps_n

    alias thread_layout_b = Layout.row_major(
        min(num_threads, BN * BK // simd_width)
        * simd_width
        // b_smem_iter.layout.stride[0].value(),
        b_smem_iter.layout.stride[0].value() // simd_width,
    )

    @parameter
    if b_iter.address_space != AddressSpace.SHARED:
        alias b_stride = b_iter.layout.stride[0].value()

        @parameter
        for i in range(num_iters):
            var b_smem_tile = b_smem_iter.next_unsafe(i)[]

            copy_dram_to_sram[thread_layout=thread_layout_b, swizzle=swizzle,](
                b_smem_tile.vectorize[1, simd_width](),
                b_iter,
                num_b_rows.value() * b_stride if num_b_rows else Int.MAX,
            )
            b_iter._incr()

            @parameter
            if token_gen:
                # this is a workaround to ensure good register allocation
                # otherwise each gmem->smem pair was using the same register
                # leading to unnecessary syncs. I am pretty sure this has some
                # negative performance impact so we should hopefully remove it
                # in the future
                schedule_barrier()
    barrier()

    alias mma_op = TensorCore[
        accum_type,
        mma_input_type,
        (MMA_M, MMA_N, MMA_K),
        transpose_b=transpose_b,
    ]()

    alias num_m_mmas = ceildiv(WM, MMA_M)
    alias num_n_mmas = ceildiv(WN, MMA_N)
    alias num_k_mmas2 = ceildiv(BK, (MMA_K * k_group_size))

    alias a_frag_size = num_matrix_reg[MMA_M, MMA_K]()
    alias b_frag_size = num_matrix_reg[MMA_N, MMA_K]()
    alias c_frag_size = num_matrix_reg[MMA_M, MMA_N]()

    var a_reg_tile = tb[mma_input_type]().row_major[
        num_m_mmas, a_frag_size * k_group_size
    ]().local().alloc()
    var b_reg_tile = tb[mma_input_type]().row_major[
        num_n_mmas, b_frag_size * k_group_size
    ]().local().alloc()

    @parameter
    for i in range(num_iters):
        var a_warp_tile = a_smem_iter.next_unsafe(i)[].tile[WM, BK](warp_row, 0)
        alias b_wtile_dim0 = WN if transpose_b else BK
        alias b_wtile_dim1 = BK if transpose_b else WN
        var b_wtile_coord0 = Int(warp_col) if transpose_b else 0
        var b_wtile_coord1 = 0 if transpose_b else Int(warp_col)
        var b_warp_tile = b_smem_iter.next_unsafe(i)[].tile[
            b_wtile_dim0, b_wtile_dim1
        ](b_wtile_coord0, b_wtile_coord1)

        @parameter
        for k_mma in range(Int(num_k_mmas2)):

            @parameter
            if a_iter.address_space != AddressSpace.LOCAL:
                mma_op.load_a[swizzle=swizzle](
                    a_warp_tile,
                    a_reg_tile.vectorize[1, a_frag_size * k_group_size](),
                    k_mma,
                )
            else:
                var a_reg_tile_input = a_iter.next_unsafe(i)[]
                a_reg_tile.vectorize[1, a_frag_size]().copy_from(
                    a_reg_tile_input.vectorize[1, a_frag_size]()
                )

            mma_op.load_b[swizzle=swizzle](
                b_warp_tile,
                b_reg_tile.vectorize[1, b_frag_size * k_group_size](),
                k_mma,
            )

            @parameter
            for k in range(k_group_size):
                var a_reg_k = a_reg_tile.tile[num_m_mmas, a_frag_size](
                    0, k
                ).vectorize[1, a_frag_size]()
                var b_reg_k = b_reg_tile.tile[num_n_mmas, b_frag_size](
                    0, k
                ).vectorize[1, b_frag_size]()

                # TODO (KERN-1702): this would not work if num_m_mmas > 1 as inside mma function
                # we will have to change the layout of c from col_major to row_major
                @parameter
                if swap_operands:
                    tensor_core_mma[True](
                        b_reg_k, a_reg_k, c.vectorize[1, c_frag_size]()
                    )
                else:
                    tensor_core_mma(
                        a_reg_k, b_reg_k, c.vectorize[1, c_frag_size]()
                    )


@always_inline
fn _apply_mask[
    masked: Bool,
    accum_type: DType,
    token_gen: Bool,
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    mask_t: MHAMask,
    group: Int,
    fragment_layout: Layout,
    warp_layout: Layout,
    not_last_iter: Bool,
    use_exp2: Bool = False,
](
    kv_tile_start_row: Int,
    kv_tile_num_rows: Int,
    start_pos: Int,
    seq_len: Int,
    num_keys: Int,
    mask_block_row: Int,
    mask_warp_row: Int,
    mask_warp_col: Int,
    scale: Float32,
    mask: mask_t,
    p_reg_vectorized: LayoutTensor[accum_type, **_],
):
    var scale_log2e: SIMD[accum_type, 1] = scale.cast[accum_type]() * (
        log2e if use_exp2
        and not mask_t.apply_log2e_after_mask else Scalar[accum_type](1)
    )
    var lane = lane_id()
    alias output_frag_size = fragment_layout.size()

    alias frag_num_rows = fragment_layout.shape[0].value()
    alias frag_num_cols = fragment_layout.shape[1].value()

    alias frag_is_row_vector = frag_num_rows == 1

    var coords = idx2crd[warp_layout](lane)
    var lane_row = coords[0] * frag_num_rows
    var lane_col = coords[1] * frag_num_cols

    @parameter
    if token_gen:
        if lane_row >= group:
            return

    @parameter
    for m_mma in range(Int(num_m_mmas)):

        @parameter
        for n_mma in range(Int(num_n_mmas)):
            alias mma_id = n_mma * num_m_mmas + m_mma
            # Coordinates in mask for current mma tile.
            var mask_frag_row = mask_warp_row + m_mma * MMA_M
            var mask_frag_col = mask_warp_col + n_mma * MMA_N
            mask_frag_row += Int(lane_row)
            mask_frag_col += Int(lane_col)
            # The row in score matrix of shape seq_len x num_keys.
            # Mask col is score col since we don't partition in col.
            var score_row = (
                num_keys - 1
            ) if token_gen else mask_block_row + mask_frag_row
            var score_col = mask_frag_col
            var score_row_with_start_pos = score_row + start_pos

            @parameter
            if masked:

                @parameter
                for j in range(output_frag_size):
                    var fragment_row = idx2crd[fragment_layout](j)[0]
                    var fragment_col = idx2crd[fragment_layout](j)[1]
                    var group_idx = lane_row + fragment_row
                    var q_head_idx = (
                        block_idx.y * group + group_idx
                    ) if token_gen else block_idx.y
                    p_reg_vectorized[mma_id, 0][j] = mask.mask(
                        IndexList[4, element_type = DType.uint32](
                            Int(block_idx.z),
                            Int(q_head_idx),
                            Int(score_row_with_start_pos + fragment_row),
                            Int(score_col + fragment_col),
                        ),
                        p_reg_vectorized[mma_id, 0][j] * scale_log2e,
                    )
            else:
                p_reg_vectorized[mma_id, 0] = (
                    p_reg_vectorized[mma_id, 0] * scale_log2e
                )

            @parameter
            if mask_t.apply_log2e_after_mask:
                p_reg_vectorized[mma_id, 0] = (
                    p_reg_vectorized[mma_id, 0] * log2e
                )

            @parameter
            if (not not_last_iter or token_gen) and mask_t.mask_out_of_bound:
                var bound_y = kv_tile_start_row + kv_tile_num_rows if token_gen else num_keys

                @parameter
                for j in range(output_frag_size):
                    var fragment_row = idx2crd[fragment_layout](j)[0]
                    var fragment_col = idx2crd[fragment_layout](j)[1]
                    var bound_x = num_keys if token_gen else seq_len

                    p_reg_vectorized[mma_id, 0][j] = _kernel_mask(
                        IndexList[2, element_type = DType.uint32](
                            Int(
                                score_row
                                + (fragment_row if not token_gen else 0)
                            ),
                            Int(score_col + fragment_col),
                        ),
                        IndexList[2, element_type = DType.uint32](
                            bound_x, bound_y
                        ),
                        p_reg_vectorized[mma_id, 0][j],
                    )


@always_inline
fn apply_softmax_denominator[
    accum_type: DType, //,
    num_m_mmas: Int,
    num_n_mmas: Int,
    fragment_layout: Layout,
](
    out_reg_tile: LayoutTensor[accum_type, **_],
    rowsum: LayoutTensor[accum_type, **_],
):
    @parameter
    for m_mma in range(Int(num_m_mmas)):
        var rowsum_inv = recip(rowsum[m_mma, 0])

        @parameter
        for n_mma in range(Int(num_n_mmas)):

            @parameter
            for i in range(fragment_layout.size()):

                @parameter
                if fragment_layout.shape[0].value() > 1:
                    rowsum_inv = recip(rowsum[m_mma, i])
                out_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rebind[
                    out_reg_tile.element_type
                ](rowsum_inv)


struct SharedMemoryManager[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    num_rowwise_warps: Int,
    token_gen: Bool,
]:
    var p_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    # p_smem is used for p
    var k_v_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    # k_v_smem is used for k, v, and scratch
    alias _alignment = alignof[SIMD[dtype, simdwidthof[dtype]()]]()
    alias _accum_type = get_accum_type[dtype]()
    alias _p_smem_size = BM * BN if token_gen else 0
    alias _k_v_smem_size = max(BM, BN) * depth

    @always_inline
    fn __init__(out self):
        self.p_smem = stack_allocation[
            Self._p_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self._alignment,
        ]()
        self.k_v_smem = stack_allocation[
            Self._k_v_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self._alignment,
        ]()

    @always_inline
    fn get_k_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BN, BK),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        return __type_of(result)(self.k_v_smem, BN * depth)

    @always_inline
    fn get_v_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BK, BN),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        return __type_of(result)(self.k_v_smem, BN * depth)

    @always_inline
    fn get_p_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BM, BK),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            circular=True,
        ],
    ):
        return __type_of(result)(
            self.p_smem,
            BM * BN,
        )

    @always_inline
    fn get_warp_scratch_tensor(
        self,
        out result: LayoutTensor[
            Self._accum_type,
            Layout.row_major(2 * num_rowwise_warps, BM),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        constrained[
            result.layout.size()
            * (sizeof[Self._accum_type]() // sizeof[dtype]())
            <= Self._k_v_smem_size,
            "warp_scratch_tile is too large",
        ]()
        return __type_of(result)(
            self.k_v_smem.bitcast[Scalar[Self._accum_type]]()
        )


struct GlobalMemoryManager[
    dtype: DType,
    BM: UInt32,
    BN: UInt32,
    BK: UInt32,
    depth: UInt32,
    num_heads: UInt32,
    group: UInt32,
    token_gen: Bool,
]:
    alias _kv_num_heads = num_heads // group
    alias _q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)),
        IntTuple(Int(num_heads * depth), 1),
    ) if not token_gen else Layout.row_major(Int(BM), Int(depth))

    alias _kv_gmem_layout = Layout(
        IntTuple(Int(BN), Int(depth)),
        IntTuple(Int(Self._kv_num_heads * depth), 1),
    )

    var q_offset: UInt32
    var q_runtime_layout: RuntimeLayout[
        Self._q_gmem_layout,
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    @always_inline
    fn __init__(
        out self, q_tile_idx: UInt32, kv_head_idx: UInt32, seq_len: Int
    ):
        var q_tile_num_rows = min(
            BM, UInt(seq_len) - q_tile_idx * BM
        ) if not token_gen else group

        self.q_offset = depth * (
            (kv_head_idx * group if token_gen else block_idx.y)
            + num_heads * q_tile_idx * BM
        )

        self.q_runtime_layout = __type_of(self.q_runtime_layout)(
            RuntimeTuple[
                Self._q_gmem_layout.shape,
                element_type = __type_of(self.q_runtime_layout).element_type,
            ](Int(q_tile_num_rows), Int(depth)),
            RuntimeTuple[
                Self._q_gmem_layout.stride,
                element_type = __type_of(self.q_runtime_layout).linear_idx_type,
            ](Int(num_heads * depth if not token_gen else depth), 1),
        )

    @always_inline
    fn get_q_tensor[
        qtype: DType,
    ](
        self,
        ptr: UnsafePointer[Scalar[qtype]],
        out result: LayoutTensor[
            qtype,
            Self._q_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return __type_of(result)(
            ptr + Int(self.q_offset),
            self.q_runtime_layout,
        )

    @always_inline
    fn get_output_tensor[
        out_type: DType,
    ](
        self,
        ptr: UnsafePointer[Scalar[out_type]],
        out result: LayoutTensor[
            out_type,
            Self._q_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        return self.get_q_tensor(ptr)

    @always_inline
    fn get_kv_tensor[
        kvtype: DType, //,
    ](
        self,
        ptr: UnsafePointer[Scalar[kvtype], **_],
        kv_tile_num_rows: Int,
        out result: LayoutTensor[
            kvtype,
            Self._kv_gmem_layout,
            ptr.origin,
            masked=True,
            address_space = ptr.address_space,
            alignment = ptr.alignment,
        ],
    ):
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = __type_of(result.runtime_layout)(
            __type_of(result.runtime_layout.shape)(
                kv_tile_num_rows, Int(depth)
            ),
            __type_of(result.runtime_layout.stride)(
                Int(Self._kv_num_heads * depth), 1
            ),
        )

        return __type_of(result)(
            ptr,
            kv_runtime_layout,
        )


@always_inline
fn mha_single_batch[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    group: Int,
    config: MHAConfig,
](
    output: UnsafePointer[Scalar[output_type],],
    q: UnsafePointer[Scalar[q_type],],
    k: k_t,
    v: v_t,
    seq_len: Int,
    num_keys: Int,
    scale: Float32,
    batch_idx: Int,
    start_pos: Int,
    mask: mask_t,
):
    alias token_gen = False
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    alias num_heads = config.num_heads
    alias kv_num_heads = num_heads // group
    alias BK = config.block_k()
    constrained[BN == depth, "BN must be equal to depth"]()
    alias simd_width = simdwidthof[q_type]()

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias use_transposed_layout = True
    alias fragment_layout = Layout.row_major(
        1, 4
    ) if use_transposed_layout else Layout.row_major(4, 1)
    alias warp_layout = Layout.col_major(
        16, 4
    ) if use_transposed_layout else Layout.row_major(4, 16)
    alias swap_mma_operands = use_transposed_layout

    alias output_frag_size = fragment_layout.size()
    alias accum_type = get_accum_type[q_type]()

    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias num_m_mmas = ceildiv(WM, MMA_M)
    alias num_n_mmas = ceildiv(WN, MMA_N)
    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var out_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, output_frag_size
    ]().local().alloc().fill(0)

    var warp_id = get_warp_id()

    var warp_row = warp_id // num_warps_n
    var warp_col = warp_id % num_warps_n

    var kv_head_idx = block_idx.y // group

    var q_tile_idx = block_idx.x

    var gmem_manager = GlobalMemoryManager[
        q_type, BM, BN, BK, depth, num_heads, group, token_gen
    ](q_tile_idx, kv_head_idx, seq_len)

    var q_tile = gmem_manager.get_q_tensor(q)

    var output_tile = gmem_manager.get_output_tensor(output)

    var rowmax = tb[accum_type]().row_major[
        num_m_mmas, fragment_layout.shape[0].value()
    ]().local().alloc().fill(min_or_neg_inf[accum_type]())
    var rowsum = tb[accum_type]().row_major[
        num_m_mmas, fragment_layout.shape[0].value()
    ]().local().alloc().fill(0)

    var smem_manager = SharedMemoryManager[
        q_type, BM, BN, BK, depth, num_warps_n, token_gen
    ]()

    var k_smem_iter = smem_manager.get_k_iter()

    var warp_scratch = smem_manager.get_warp_scratch_tensor()

    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_row * WM
    var mask_warp_col = warp_col * WN

    constrained[BK == 32, "BK must be 32"]()

    # the following assumes BK == 32, i.e. simd_width = 2*frag_size
    alias q_reg_size = (depth // BK) * num_m_mmas * simd_width

    var q_reg_data = stack_allocation[
        q_reg_size,
        q_type,
        address_space = AddressSpace.LOCAL,
    ]()

    var q_reg_tile_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(num_m_mmas, simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ](q_reg_data, q_reg_size)

    var q_gmem_warp_iter = q_tile.tiled_iterator[WM, BK, axis=1](warp_row, 0)

    # TODO: This is expensive, dereferencing q_gmem_warp_iter[] is expensive and
    # using its dim() is also expensive. Need to find a better way to do this.
    var q_bounds = max(
        min(Int32(WM), Int32(q_tile.dim(0) - WM * warp_row))
        * q_tile.stride[0](),
        0,
    )

    @parameter
    for i in range(Int(depth // BK)):
        var q_reg_tile = q_reg_tile_iter.next_unsafe(i)[]
        copy_dram_to_local[
            src_thread_layout = Layout.col_major(16, 4),
            thread_scope = ThreadScope.WARP,
        ](
            q_reg_tile.vectorize[1, simd_width](),
            q_gmem_warp_iter,
            Int(readfirstlane(Int32(q_bounds))),
        )
        q_gmem_warp_iter._incr()

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        var mask_status = mask.status(
            Index[dtype = DType.uint32](
                Int(q_tile_idx * BM + start_pos),
                Int(kv_tile_start_row),
            ),
            Index[dtype = DType.uint32](Int(BM), Int(BN)),
        )

        @parameter
        if not token_gen:
            if mask_status == TileMaskStatus.FULL_MASK:
                mask_warp_col += BN
                return

        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        var k_tile = gmem_manager.get_kv_tensor(
            k.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )
        var k_gmem_iter = k_tile.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_tile = gmem_manager.get_kv_tensor(
            v.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )

        var v_gmem_iter = v_tile.tiled_iterator[BK, BN, axis=0](0, 0)

        var p_reg_tile = tb[accum_type]().row_major[
            num_m_mmas * num_n_mmas, output_frag_size
        ]().local().alloc().fill(0)

        alias swizzle = Swizzle(2, 0, 2)

        var num_b_rows = OptionalReg[Int](kv_tile_num_rows)

        # TODO (KERN-1708):this is just a dummy iterator to satisfy the interface
        # will fix it with better interface later
        var q_smem_iter = LayoutTensorIter[
            q_type,
            Layout.row_major(num_m_mmas, simd_width),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
        ](
            UnsafePointer[
                Scalar[q_type], address_space = AddressSpace.SHARED
            ](),
            q_reg_size,
        )

        mma[
            transpose_b=True,
            k_group_size=2,
            config=config,
            swizzle=swizzle,
            swap_operands=swap_mma_operands,
            num_iters = Int(depth // BK),
            token_gen=token_gen,
        ](
            p_reg_tile,
            q_reg_tile_iter,
            q_smem_iter,
            k_gmem_iter,
            k_smem_iter,
            num_b_rows,
        )

        var p_reg_vectorized = p_reg_tile.vectorize[1, output_frag_size]()

        alias use_exp2 = True

        @always_inline
        @parameter
        fn _apply_mask_impl[masked: Bool]():
            _apply_mask[
                masked=masked,
                accum_type=accum_type,
                token_gen=token_gen,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                mask_t=mask_t,
                group=group,
                fragment_layout=fragment_layout,
                warp_layout=warp_layout,
                not_last_iter=not_last_iter,
                use_exp2=use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                start_pos,
                seq_len,
                num_keys,
                Int(mask_block_row),
                Int(mask_warp_row),
                mask_warp_col,
                scale,
                mask,
                p_reg_vectorized,
            )

        unswitch[_apply_mask_impl](mask_status == TileMaskStatus.PARTIAL_MASK)

        mask_warp_col += BN
        alias reg_layout_by_mma_unit = Layout.row_major(
            num_m_mmas * num_n_mmas, output_frag_size
        )

        # Not sure why we need this barrier here, but the code hangs without it
        barrier()

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            warp_layout,
            use_exp2=use_exp2,
            fragment_layout=fragment_layout,
        ](
            out_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_row)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        # warp scratch and p_smem are using the same smem space
        barrier()

        @parameter
        for i in range(Int(BN // BK)):
            # we multiply v^T x p^T instead of p x v
            # here all threads work to load 16xdepth tile at a time
            # with each warp loading 4xdepth tile
            # each thread loads v_reg_tile is therefore BK//MMA_N 16B elements
            var v_reg_tile = tb[q_type]().row_major[
                BK // MMA_N, simd_width
            ]().local().alloc()

            var v_gmem_tile = v_gmem_iter[]

            @parameter
            for i in range(Int(BK // MMA_N)):
                var v_warp_tile = v_gmem_tile.tile[BK // 2, depth](i, 0).tile[
                    4, depth
                ](warp_id, 0)
                copy_dram_to_local[
                    src_thread_layout = Layout.row_major(4, 16),
                    thread_scope = ThreadScope.WARP,
                ](
                    v_reg_tile.tile[1, simd_width](i, 0).vectorize[
                        1, simd_width
                    ](),
                    v_warp_tile.vectorize[1, simd_width](),
                    v_tile,
                )
            v_gmem_iter._incr()

            # transpose v_gmem_tile to v_smem
            # each thread writes 8x2 elements to smem using 4x4B writes
            # shared memory layout is row_major(depth, BK // num_warps) repeated num_warps times
            # and each warp writes to a different tile in smem

            alias num_warps = config.num_threads() // WARP_SIZE
            constrained[
                BK // num_warps == simd_width,
                "BK//num_warps must be equal to simd_width",
            ]()

            var lane_coords = idx2crd[Layout.col_major(16, 4)](lane_id())
            var lane_row = lane_coords[0]
            var lane_col = lane_coords[1]

            var v_smem_iter = LayoutTensorIter[
                q_type,
                Layout.row_major(depth, BK // num_warps),
                MutableAnyOrigin,
                address_space = AddressSpace.SHARED,
                circular=True,
            ](
                smem_manager.get_v_iter().ptr.offset(depth * BK * i),
                depth * BK,
            )

            var v_smem_warp_tile = v_smem_iter.next_unsafe(warp_id)[]

            var v_lane_tile = v_smem_warp_tile.tile[simd_width, 2](
                lane_row, lane_col
            ).vectorize[1, 2]()

            @parameter
            for j in range(simd_width):
                # each thread loads 2x8 elements from gmem
                # they are interleaved and written to smem
                var v_reg_tile_0 = v_reg_tile[0, j][0]
                var v_reg_tile_1 = v_reg_tile[1, j][0]
                var v_01 = SIMD[q_type, 2](v_reg_tile_0, v_reg_tile_1)
                v_lane_tile[j, 0] = rebind[v_lane_tile.element_type](v_01)

            # ensure that shared memory is filled
            barrier()

            # MMA
            # threads in 16x4 layout
            # each column loads depth x 8 elements from smem
            var v_smem_warp_tile_mma = v_smem_iter.next_unsafe(
                lane_col
            )[].vectorize[1, simd_width]()
            var v_smem_warp_tile_mma_tile = v_smem_warp_tile_mma.distribute[
                Layout.row_major(16, 1)
            ](lane_row)

            alias mma_op = TensorCore[
                accum_type,
                q_type,
                (MMA_M, MMA_N, MMA_K),
                transpose_b=False,
            ]()

            var v_reg_tile_mma = tb[q_type]().row_major[
                depth // MMA_M, simd_width
            ]().local().alloc()
            v_reg_tile_mma.vectorize[1, simd_width]().copy_from(
                v_smem_warp_tile_mma_tile
            )

            var p_mma_tile_interleaved = tb[q_type]().row_major[
                WM // MMA_M, simd_width
            ]().local().alloc()

            constrained[BK // MMA_N == 2, "BK//MMA_N must be 2"]()

            var p_mma_tile = tb[q_type]().row_major[
                num_m_mmas * 2, output_frag_size
            ]().local().alloc().split[num_m_mmas]()

            @parameter
            for m_mma in range(Int(num_m_mmas)):
                p_mma_tile[m_mma].copy_from(
                    p_reg_tile.tile[2, output_frag_size](2 * i + m_mma, 0)
                )

            # interleave p_reg_tile to match the pattern in v_smem_warp_tile_mma_tile
            @parameter
            for j in range(output_frag_size):

                @parameter
                for k in range(2):

                    @parameter
                    for m_mma in range(Int(num_m_mmas)):
                        p_mma_tile_interleaved[k, 2 * j + m_mma] = p_mma_tile[
                            m_mma
                        ][k, j].cast[q_type]()

            @parameter
            for k in range(2):
                var a_reg_k = v_reg_tile_mma.tile[
                    depth // MMA_M, output_frag_size
                ](0, k).vectorize[1, output_frag_size]()
                var b_reg_k = p_mma_tile_interleaved.tile[
                    WM // MMA_N, output_frag_size
                ](0, k).vectorize[1, output_frag_size]()
                tensor_core_mma[True](
                    a_reg_k,
                    b_reg_k,
                    out_reg_tile.vectorize[1, output_frag_size](),
                )
        barrier()

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    # Apply softmax denominator.
    apply_softmax_denominator[
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        fragment_layout=fragment_layout,
    ](out_reg_tile, rowsum)

    var output_warp_tile = output_tile.tile[WM, WN](warp_row, warp_col)
    copy_local_to_dram[
        dst_thread_layout=warp_layout,
        thread_scope = ThreadScope.WARP,
    ](
        output_warp_tile.vectorize[
            fragment_layout.shape[0].value(),
            fragment_layout.shape[1].value(),
        ](),
        out_reg_tile.vectorize[1, output_frag_size](),
        output_tile,
    )


@always_inline
fn mha_decoding_single_batch[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    group: Int,
    config: MHAConfig,
](
    output: UnsafePointer[Scalar[output_type],],
    q: UnsafePointer[Scalar[q_type],],
    k: k_t,
    v: v_t,
    seq_len: Int,
    num_keys: Int,
    scale: Float32,
    batch_idx: Int,
    start_pos: Int,
    mask: mask_t,
):
    alias token_gen = True

    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    alias num_heads = config.num_heads
    alias kv_num_heads = num_heads // group
    alias BK = config.block_k()
    constrained[BN == depth, "BN must be equal to depth"]()
    alias simd_width = simdwidthof[q_type]()

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias use_transposed_layout = True
    alias fragment_layout = Layout.row_major(
        1, 4
    ) if use_transposed_layout else Layout.row_major(4, 1)
    alias warp_layout = Layout.col_major(
        16, 4
    ) if use_transposed_layout else Layout.row_major(4, 16)
    alias swap_mma_operands = use_transposed_layout

    alias output_frag_size = fragment_layout.size()
    alias accum_type = get_accum_type[q_type]()

    alias WM = config.warp_m()
    alias WN = config.warp_n()
    alias num_m_mmas = ceildiv(WM, MMA_M)
    alias num_n_mmas = ceildiv(WN, MMA_N)
    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    var out_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, output_frag_size
    ]().local().alloc().fill(0)

    var warp_id = get_warp_id()

    var warp_row = warp_id // num_warps_n
    var warp_col = warp_id % num_warps_n

    var kv_head_idx = block_idx.y

    var q_tile_idx = block_idx.x

    var gmem_manager = GlobalMemoryManager[
        q_type, BM, BN, BK, depth, num_heads, group, token_gen
    ](q_tile_idx, kv_head_idx, seq_len)

    var q_tile = gmem_manager.get_q_tensor(q)

    var output_tile = gmem_manager.get_output_tensor(output)

    var rowmax = tb[accum_type]().row_major[
        num_m_mmas, fragment_layout.shape[0].value()
    ]().local().alloc().fill(min_or_neg_inf[accum_type]())
    var rowsum = tb[accum_type]().row_major[
        num_m_mmas, fragment_layout.shape[0].value()
    ]().local().alloc().fill(0)

    var smem_manager = SharedMemoryManager[
        q_type, BM, BN, BK, depth, num_warps_n, token_gen
    ]()

    var p_smem_iter = smem_manager.get_p_iter()
    var k_smem_iter = smem_manager.get_k_iter()
    var v_smem_iter = smem_manager.get_v_iter()

    var warp_scratch = smem_manager.get_warp_scratch_tensor()

    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_row * WM
    var mask_warp_col = warp_col * WN

    constrained[BK == 32, "BK must be 32"]()

    # the following assumes BK == 32, i.e. simd_width = 2*frag_size
    alias q_reg_size = (depth // BK) * num_m_mmas * simd_width

    var q_reg_data = stack_allocation[
        q_reg_size,
        q_type,
        address_space = AddressSpace.LOCAL,
    ]()

    var q_reg_tile_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(num_m_mmas, simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ](q_reg_data, q_reg_size)

    var q_gmem_warp_iter = q_tile.tiled_iterator[WM, BK, axis=1](warp_row, 0)

    @parameter
    for i in range(Int(depth // BK)):
        var q_reg_tile = q_reg_tile_iter.next_unsafe(i)[]
        copy_dram_to_local[
            src_thread_layout = Layout.col_major(16, 4),
            thread_scope = ThreadScope.WARP,
        ](
            q_reg_tile.vectorize[1, simd_width](),
            q_gmem_warp_iter,
            q_tile.dim(0) * q_tile.stride[0](),
        )
        q_gmem_warp_iter._incr()

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        var mask_status = mask.status(
            Index[dtype = DType.uint32](
                Int(q_tile_idx * BM + start_pos),
                Int(kv_tile_start_row),
            ),
            Index[dtype = DType.uint32](Int(BM), Int(BN)),
        )

        @parameter
        if not token_gen:
            if mask_status == TileMaskStatus.FULL_MASK:
                mask_warp_col += BN
                return

        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        var k_tile = gmem_manager.get_kv_tensor(
            k.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )
        var k_gmem_iter = k_tile.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_tile = gmem_manager.get_kv_tensor(
            v.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )

        var v_gmem_iter = v_tile.tiled_iterator[BK, BN, axis=0](0, 0)

        var p_reg_tile = tb[accum_type]().row_major[
            num_m_mmas * num_n_mmas, output_frag_size
        ]().local().alloc().fill(0)

        alias swizzle = Swizzle(2, 0, 2)

        var num_b_rows = OptionalReg[Int](
            kv_tile_num_rows
        ) if not not_last_iter else None

        # TODO (KERN-1708):this is just a dummy iterator to satisfy the interface
        # will fix it with better interface later
        var q_smem_iter = LayoutTensorIter[
            q_type,
            Layout.row_major(num_m_mmas, simd_width),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
        ](
            UnsafePointer[
                Scalar[q_type], address_space = AddressSpace.SHARED
            ](),
            q_reg_size,
        )

        mma[
            transpose_b=True,
            k_group_size=2,
            config=config,
            swizzle=swizzle,
            swap_operands=swap_mma_operands,
            num_iters = Int(depth // BK),
            token_gen=token_gen,
        ](
            p_reg_tile,
            q_reg_tile_iter,
            q_smem_iter,
            k_gmem_iter,
            k_smem_iter,
            num_b_rows,
        )

        var p_reg_vectorized = p_reg_tile.vectorize[1, output_frag_size]()

        alias use_exp2 = True

        @always_inline
        @parameter
        fn _apply_mask_impl[masked: Bool]():
            _apply_mask[
                masked=masked,
                accum_type=accum_type,
                token_gen=token_gen,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                mask_t=mask_t,
                group=group,
                fragment_layout=fragment_layout,
                warp_layout=warp_layout,
                not_last_iter=not_last_iter,
                use_exp2=use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                start_pos,
                seq_len,
                num_keys,
                Int(mask_block_row),
                Int(mask_warp_row),
                mask_warp_col,
                scale,
                mask,
                p_reg_vectorized,
            )

        @parameter
        if not token_gen:
            unswitch[_apply_mask_impl](
                mask_status == TileMaskStatus.PARTIAL_MASK
            )
        else:
            _apply_mask_impl[masked=True]()

        mask_warp_col += BN
        alias reg_layout_by_mma_unit = Layout.row_major(
            num_m_mmas * num_n_mmas, output_frag_size
        )

        # Not sure why we need this barrier here, but the code hangs without it
        barrier()

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            warp_layout,
            use_exp2=use_exp2,
            fragment_layout=fragment_layout,
        ](
            out_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, output_frag_size
            ](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_row)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        # warp scratch and p_smem are using the same smem space
        barrier()

        copy_fragment_to_smem[
            BM,
            BN,
            BK,
            WM,
            WN,
            MMA_M,
            MMA_N,
            num_m_mmas,
            num_n_mmas,
            fragment_layout,
            warp_layout,
        ](
            p_smem_iter,
            p_reg_vectorized,
            warp_row,
            warp_col,
        )

        barrier()

        mma[
            transpose_b=False,
            k_group_size=2,
            config=config,
            swizzle=None,
            swap_operands=swap_mma_operands,
            num_iters = Int(BN // BK),
            token_gen=token_gen,
        ](
            out_reg_tile,
            p_smem_iter,
            p_smem_iter,
            v_gmem_iter,
            v_smem_iter,
            num_b_rows,
        )
        # ensure that smem for v is not required anymore
        barrier()

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    # Apply softmax denominator.
    apply_softmax_denominator[
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        fragment_layout=fragment_layout,
    ](out_reg_tile, rowsum)

    var output_warp_tile = output_tile.tile[WM, WN](warp_row, warp_col)
    copy_local_to_dram[
        dst_thread_layout=warp_layout,
        thread_scope = ThreadScope.WARP,
    ](
        output_warp_tile.vectorize[
            fragment_layout.shape[0].value(),
            fragment_layout.shape[1].value(),
        ](),
        out_reg_tile.vectorize[1, output_frag_size](),
        output_tile,
    )


@always_inline
fn copy_fragment_to_smem[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    fragment_layout: Layout,
    warp_layout: Layout,
](
    p_smem_iter: LayoutTensorIter[*_, address_space = AddressSpace.SHARED, **_],
    p_reg_vectorized: LayoutTensor[*_, address_space = AddressSpace.LOCAL, **_],
    warp_row: Int,
    warp_col: Int,
):
    alias num_n_mmas_per_bk = num_n_mmas // (WN // BK)

    # for the following indexing logic, WN must be equal to BN or BK
    constrained[WN == BK or WN == BN, "WN must be equal to BN or BK"]()

    @parameter
    for i in range(Int(WN // BK)):
        var p_smem_tile = p_smem_iter.next_unsafe(i + warp_col * (WN // BK))[]
        var p_smem_warp_tile = p_smem_tile.tile[WM, BK](warp_row, i)

        @parameter
        for m_mma in range(Int(num_m_mmas)):

            @parameter
            for n_mma in range(Int(num_n_mmas_per_bk)):
                var p_smem_mma_tile = p_smem_warp_tile.tile[MMA_M, MMA_N](
                    m_mma, n_mma
                )
                var p_reg_tile = p_reg_vectorized.tile[1, 1](
                    (n_mma + i * num_n_mmas_per_bk) * num_m_mmas + m_mma,
                    0,
                )
                copy[thread_layout=warp_layout](
                    p_smem_mma_tile.vectorize[
                        fragment_layout.shape[0].value(),
                        fragment_layout.shape[1].value(),
                    ](),
                    p_reg_tile,
                )
