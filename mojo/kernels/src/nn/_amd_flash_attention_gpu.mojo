# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray, OptionalReg
from gpu import (
    barrier,
    lane_id,
    block_dim,
    block_idx,
    global_idx,
    thread_idx,
    WARP_SIZE,
    grid_dim,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from nn.mha_operand import KVCacheMHAOperand, MHAOperand, NDBufferMHAOperand

from math.constants import log2e

from math import align_down, align_up, ceildiv, exp, recip

from memory import UnsafePointer, stack_allocation

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.sync import (
    schedule_barrier,
    schedule_group_barrier,
    AMDScheduleBarrierMask,
)
import gpu.warp as warp
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, IntTuple
from layout.layout_tensor import (
    copy_dram_to_sram,
    copy_local_to_dram,
    copy_dram_to_local,
    copy,
    copy_sram_to_dram,
    ThreadScope,
)
from nn.softmax import (
    _online_softmax_iter_for_mma_output,
    _online_softmax_iter_for_mma_output_split_warp_reduce,
    _softmax_gpu,
    softmax,
)
from nn.mha_utils import _kernel_mask

from nn.mha_mask import MHAMask, NullMask, TileMaskStatus

from layout.runtime_tuple import RuntimeTuple
from utils.numerics import min_or_neg_inf, neg_inf
from pathlib import Path
from algorithm.functional import tile_and_unswitch, unswitch, vectorize

from layout.runtime_layout import RuntimeLayout
from layout.tensor_builder import LayoutTensorBuild as tb, static
from layout.tensor_core import TensorCore, get_mma_shape, num_matrix_reg
from linalg.utils import GemmShape
from math import align_down, ceildiv, align_up
from memory import UnsafePointer
from sys import simdwidthof, alignof, sizeof
from utils import Index, IndexList, StaticTuple
from utils.numerics import get_accum_type
from linalg.utils_gpu import MatmulConfig
from linalg.utils import apply_epilogue, elementwise_epilogue_type
from layout.swizzle import Swizzle
from math import exp
from nn.mha_utils import MHAConfig
from layout._utils import idx2crd, hash
from layout.layout_tensor import LayoutTensorIter


# @always_inline
# fn mma[
#     transpose_b: Bool,
#     k_group_size: Int,
#     config: MHAConfig,
#     swizzle: OptionalReg[Swizzle] = None,
#     swap_operands: Bool = False,
# ](
#     c: LayoutTensor,
#     a: LayoutTensor,
#     a_smem: LayoutTensor,
#     b: LayoutTensor,
#     b_smem: LayoutTensor,
# ):
#     alias mma_input_type = b.dtype
#     alias mma_input_type = b.dtype
#     alias simd_width = (simdwidthof[mma_input_type]() // 2) * k_group_size
#     alias mma_shape = get_mma_shape[mma_input_type, get_accum_type[mma_input_type]()]()
#     alias MMA_M = mma_shape[0]
#     alias MMA_N = mma_shape[1]
#     alias MMA_K = mma_shape[2]
#     alias accum_type = get_accum_type[mma_input_type]()
#     alias WM = config.warp_m()
#     alias BM = config.block_m()
#     alias BN = config.block_n()
#     alias depth = config.depth
#     var warp_id = thread_idx.x // WARP_SIZE
#     alias num_warps = config.num_threads() // WARP_SIZE
#     alias num_threads = config.num_threads()

#     @parameter
#     if a.address_space in (AddressSpace.GLOBAL, AddressSpace.GENERIC):
#         copy_dram_to_sram[
#             thread_layout = Layout.row_major(num_threads // 16, 16),
#             swizzle=swizzle,
#         ](
#             a_smem.vectorize[1, 8](),
#             a.vectorize[1, 8](),
#             a,
#         )

#     @parameter
#     if b.address_space != AddressSpace.SHARED:
#         copy_dram_to_sram[
#             thread_layout = Layout.row_major(num_threads // 16, 16),
#             swizzle=swizzle,
#         ](
#             b_smem.vectorize[1, 8](),
#             b.vectorize[1, 8](),
#             b,
#         )

#     barrier()
#     var mma_op = TensorCore[
#         accum_type,
#         mma_input_type,
#         (MMA_M, MMA_N, MMA_K),
#         transpose_b=transpose_b,
#     ]()

#     var a_warp_tile = a_smem.tile[WM, depth](warp_id, 0)

#     alias num_m_mmas = ceildiv(WM, MMA_M)
#     alias num_n_mmas = ceildiv(BN, MMA_N)
#     alias num_k_mmas2 = ceildiv(depth, (MMA_K * k_group_size))

#     var a_reg_tile = tb[mma_input_type]().row_major[
#         num_m_mmas * num_k_mmas2, simd_width
#     ]().local().alloc()
#     var b_reg_tile = tb[mma_input_type]().row_major[
#         num_n_mmas * num_k_mmas2, simd_width
#     ]().local().alloc()

#     @parameter
#     for k_mma in range(Int(num_k_mmas2)):

#         @parameter
#         if a.address_space != AddressSpace.LOCAL:
#             mma_op.load_a[swizzle= True if swizzle else False](
#                 a_warp_tile,
#                 a_reg_tile.tile[num_m_mmas, simd_width](k_mma, 0).vectorize[
#                     1, simd_width
#                 ](),
#                 k_mma,
#             )
#         mma_op.load_b[swizzle=swizzle](
#             b_smem,
#             b_reg_tile.tile[num_n_mmas, simd_width](k_mma, 0).vectorize[
#                 1, simd_width
#             ](),
#             k_mma,
#         )

#     # var p_reg_tile = tb[accum_type]().row_major[
#     #     num_m_mmas * num_n_mmas, output_frag_size
#     # ]().local().alloc().fill(0)

#     @parameter
#     if a.address_space == AddressSpace.LOCAL:
#         a_reg_tile.copy_from(a)

#     @parameter
#     for k_mma in range(Int(num_k_mmas2)):

#         @parameter
#         for k in range(k_group_size):
#             alias elements_per_thread = simd_width // k_group_size
#             var a_reg_k = a_reg_tile.tile[num_m_mmas, elements_per_thread](
#                 k_mma, k
#             ).vectorize[1, elements_per_thread]()
#             var b_reg_k = b_reg_tile.tile[num_n_mmas, elements_per_thread](
#                 k_mma, k
#             ).vectorize[1, elements_per_thread]()

#             @parameter
#             if swap_operands:
#                 mma_op.mma(b_reg_k, a_reg_k, c.vectorize[1, 4]())
#             else:
#                 mma_op.mma(a_reg_k, b_reg_k, c.vectorize[1, 4]())


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
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias depth = config.depth
    var warp_id = thread_idx.x // WARP_SIZE
    alias num_warps = config.num_threads() // WARP_SIZE
    alias num_threads = config.num_threads()

    @always_inline
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

    alias thread_layout_a = Layout.row_major(
        min(Int(num_threads), Int(BM * BK) // simd_width) // (BK // simd_width),
        BK // simd_width,
    )

    alias thread_layout_b = Layout.row_major(
        min(num_threads, BN * BK // simd_width)
        * simd_width
        // b_smem_iter.layout.stride[0].value(),
        b_smem_iter.layout.stride[0].value() // simd_width,
    )

    @parameter
    if a_iter.address_space in (AddressSpace.GLOBAL, AddressSpace.GENERIC):
        alias a_stride = a_iter.layout.stride[0].value()

        @parameter
        for i in range(num_iters):
            var a_tile = a_iter[]
            var a_smem_tile = a_smem_iter.next_unsafe(i)[]
            copy_dram_to_sram[
                thread_layout=thread_layout_a,
                swizzle=swizzle,
                num_threads=num_threads,
            ](
                a_smem_tile.vectorize[1, simd_width](),
                a_iter,
                a_tile.dim(0) * a_stride,
            )
            a_iter._incr()

            @parameter
            if token_gen:
                # this is a workaround to ensure good register allocation
                # otherwise each gmem->smem pair was using the same register
                # leading to unnecessary syncs. I am pretty sure this has some
                # negative performance impact so we should hopefully remove it
                # in the future
                schedule_barrier()

    @parameter
    if b_iter.address_space != AddressSpace.SHARED:
        alias b_stride = b_iter.layout.stride[0].value()

        @parameter
        for i in range(num_iters):
            var b_smem_tile = b_smem_iter.next_unsafe(i)[]

            if num_b_rows:
                copy_dram_to_sram[
                    thread_layout=thread_layout_b,
                    swizzle=swizzle,
                ](
                    b_smem_tile.vectorize[1, simd_width](),
                    b_iter,
                    num_b_rows.value() * b_stride,
                )
            else:
                copy_dram_to_sram[
                    thread_layout=thread_layout_b,
                    swizzle=swizzle,
                ](
                    b_smem_tile.vectorize[1, simd_width](),
                    b_iter,
                    # bounds don't matter for this case as we will never be out of bounds
                    # if num_b_rows is not provided
                    Int.MAX,
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
    alias num_n_mmas = ceildiv(BN, MMA_N)
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
        var a_warp_tile = a_smem_iter.next_unsafe(i)[].tile[WM, BK](warp_id, 0)
        var b_warp_tile = b_smem_iter.next_unsafe(i)[]

        @parameter
        for k_mma in range(Int(num_k_mmas2)):

            @parameter
            if a_iter.address_space != AddressSpace.LOCAL:
                mma_op.load_a[swizzle= True if swizzle else False](
                    a_warp_tile,
                    a_reg_tile.vectorize[1, a_frag_size * k_group_size](),
                    k_mma,
                )
            else:
                constrained[False, " not supported"]()
                constrained[k_group_size == 1, "k_group_size must be 1"]()
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

                @parameter
                if swap_operands:
                    mma_op.mma(b_reg_k, a_reg_k, c.vectorize[1, c_frag_size]())
                else:
                    mma_op.mma(a_reg_k, b_reg_k, c.vectorize[1, c_frag_size]())


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
    not_last_iter: Bool,
    group: Int,
    fragment_layout: Layout,
    warp_layout: Layout,
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
        log2e if use_exp2 else Scalar[accum_type](1)
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
                        IndexList[4, element_bitwidth=32, unsigned=True,](
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

            if not not_last_iter or token_gen:
                var bound_y = kv_tile_start_row + kv_tile_num_rows if token_gen else num_keys

                @parameter
                for j in range(output_frag_size):
                    var fragment_row = idx2crd[fragment_layout](j)[0]
                    var fragment_col = idx2crd[fragment_layout](j)[1]
                    var bound_x = num_keys if token_gen else seq_len

                    p_reg_vectorized[mma_id, 0][j] = _kernel_mask(
                        IndexList[2, element_bitwidth=32, unsigned=True,](
                            Int(
                                score_row
                                + (fragment_row if not token_gen else 0)
                            ),
                            Int(score_col + fragment_col),
                        ),
                        IndexList[
                            2,
                            element_bitwidth=32,
                            unsigned=True,
                        ](bound_x, bound_y),
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
    dtype: DType, BM: Int, BN: Int, BK: Int, depth: Int, num_rowwise_warps: Int
]:
    var q_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    # q_smem is used for q and output
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
    alias _q_smem_size = BM * depth
    alias _p_smem_size = BM * BN
    alias _k_v_smem_size = max(BM, BN) * depth

    @always_inline
    fn __init__(out self):
        self.q_smem = stack_allocation[
            Self._q_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
            alignment = Self._alignment,
        ]()

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
    fn get_q_iter(
        self,
        out result: LayoutTensorIter[
            dtype,
            Layout.row_major(BM, BK),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment = Self._alignment,
        ],
    ):
        return __type_of(result)(
            self.q_smem,
            BM * depth,
        )

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
    fn get_output_tensor(
        self,
        out result: LayoutTensor[
            dtype,
            Layout.row_major(BM, depth),
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        return __type_of(result)(self.q_smem)

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
        Self._q_gmem_layout, linear_idx_type = DType.int32
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

        self.q_runtime_layout = RuntimeLayout[linear_idx_type = DType.int32](
            RuntimeTuple[Self._q_gmem_layout.shape, unsigned=True](
                Int(q_tile_num_rows), Int(depth)
            ),
            RuntimeTuple[Self._q_gmem_layout.stride, unsigned=True](
                Int(num_heads * depth if not token_gen else depth), 1
            ),
        )

    @always_inline
    fn get_q_tensor[
        qtype: DType,
    ](
        self,
        ptr: UnsafePointer[Scalar[qtype]],
        out result: LayoutTensor[
            qtype, Self._q_gmem_layout, MutableAnyOrigin, masked=True
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
            out_type, Self._q_gmem_layout, MutableAnyOrigin, masked=True
        ],
    ):
        return self.get_q_tensor(ptr)

    @always_inline
    fn get_kv_tensor[
        kvtype: DType, //,
        not_last_iter: Bool,
    ](
        self,
        ptr: UnsafePointer[Scalar[kvtype], **_],
        kv_tile_num_rows: Int,
        out result: LayoutTensor[
            kvtype,
            Self._kv_gmem_layout,
            ptr.origin,
            masked = not not_last_iter,
            address_space = ptr.address_space,
            alignment = ptr.alignment,
        ],
    ):
        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[linear_idx_type = DType.int32](
            RuntimeTuple[Self._kv_gmem_layout.shape, unsigned=True](
                kv_tile_num_rows, Int(depth)
            ),
            RuntimeTuple[Self._kv_gmem_layout.stride, unsigned=True](
                Int(Self._kv_num_heads * depth), 1
            ),
        )

        return __type_of(result)(
            ptr,
            kv_runtime_layout,
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
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
    token_gen: Bool = False,
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
    alias use_transposed_layout = False  # not token_gen
    alias fragment_layout = Layout.row_major(
        1, 4
    ) if use_transposed_layout else Layout.row_major(4, 1)
    alias warp_layout = Layout.col_major(
        16, 4
    ) if use_transposed_layout else Layout.row_major(4, 16)
    alias swap_mma_operands = use_transposed_layout

    alias output_frag_size = fragment_layout.size()
    alias accum_type = get_accum_type[q_type]()

    alias WM = config.WM
    alias num_m_mmas = ceildiv(WM, MMA_M)
    alias num_n_mmas = ceildiv(BN, MMA_N)
    var out_reg_tile = tb[accum_type]().row_major[
        num_m_mmas * num_n_mmas, output_frag_size
    ]().local().alloc().fill(0)

    var warp_id = thread_idx.x // WARP_SIZE

    var kv_head_idx = block_idx.y if token_gen else block_idx.y // group

    var q_tile_idx = block_idx.x

    var gmem_manager = GlobalMemoryManager[
        q_type, BM, BN, BK, depth, num_heads, group, token_gen
    ](q_tile_idx, kv_head_idx, seq_len)

    var q_tile = gmem_manager.get_q_tensor(q)

    var q_gmem_iter = q_tile.tiled_iterator[BM, BK, axis=1](0, 0)

    var output_tile = gmem_manager.get_output_tensor(output)

    var rowmax = tb[accum_type]().row_major[
        num_m_mmas, fragment_layout.shape[0].value()
    ]().local().alloc().fill(min_or_neg_inf[accum_type]())
    var rowsum = tb[accum_type]().row_major[
        num_m_mmas, fragment_layout.shape[0].value()
    ]().local().alloc().fill(0)

    alias num_rowwise_warps = 1  # BN // WN

    var smem_manager = SharedMemoryManager[
        q_type, BM, BN, BK, depth, num_rowwise_warps
    ]()
    var q_smem_iter = smem_manager.get_q_iter()
    var p_smem_iter = smem_manager.get_p_iter()
    var k_smem_iter = smem_manager.get_k_iter()
    var v_smem_iter = smem_manager.get_v_iter()

    var warp_scratch = smem_manager.get_warp_scratch_tensor()

    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_id * WM
    var mask_warp_col = 0

    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        var k_tile = gmem_manager.get_kv_tensor[not_last_iter=not_last_iter](
            k.block_paged_ptr[BN](batch_idx, kv_tile_start_row, kv_head_idx, 0),
            kv_tile_num_rows,
        )
        var k_gmem_iter = k_tile.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_tile = gmem_manager.get_kv_tensor[not_last_iter=not_last_iter](
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
        if kv_tile_start_row == 0:
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
                q_gmem_iter,
                q_smem_iter,
                k_gmem_iter,
                k_smem_iter,
                num_b_rows,
            )
        else:
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
                q_smem_iter,
                q_smem_iter,
                k_gmem_iter,
                k_smem_iter,
                num_b_rows,
            )

        var p_reg_vectorized = p_reg_tile.vectorize[1, output_frag_size]()

        # unswitch[_apply_mask](
        #     mask.status(
        #         Index[element_bitwidth=32, unsigned=True](
        #             Int(q_tile_idx * BM + start_pos), kv_tile_start_row
        #         ),
        #         Index[element_bitwidth=32, unsigned=True](Int(BM), Int(BN)),
        #     )
        #     == TileMaskStatus.PARTIAL_MASK
        # )
        # ^ this somehow does not give correct results

        alias use_exp2 = True
        _apply_mask[
            masked=True,
            accum_type=accum_type,
            token_gen=token_gen,
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            mask_t=mask_t,
            not_last_iter=not_last_iter,
            group=group,
            fragment_layout=fragment_layout,
            warp_layout=warp_layout,
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
            Layout.row_major(BM // WM, 1),
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
            warp_scratch.tile[2, WM](0, Int(warp_id)),
            rowmax.ptr.address_space_cast[AddressSpace.GENERIC](),
            rowsum.ptr.address_space_cast[AddressSpace.GENERIC](),
        )

        # warp scratch and p_smem are using the same smem space
        barrier()

        @parameter
        if not use_transposed_layout:
            copy_fragment_to_smem[
                BM,
                BN,
                BK,
                WM,
                MMA_M,
                MMA_N,
                num_m_mmas,
                num_n_mmas,
                fragment_layout,
                warp_layout,
            ](
                p_smem_iter,
                p_reg_vectorized,
                warp_id,
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
        else:
            alias num_n_mmas_per_bk = num_n_mmas // (BN // BK)
            var p_reg_tile_iter = p_reg_tile.tiled_iterator[
                num_m_mmas * num_n_mmas_per_bk, output_frag_size, axis=0
            ](0, 0)

            mma[
                transpose_b=False,
                k_group_size=1,
                config=config,
                swizzle=None,
                swap_operands=swap_mma_operands,
                num_iters = Int(BN // BK),
                token_gen=token_gen,
            ](
                out_reg_tile,
                p_reg_tile_iter,
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

    alias use_smem_when_writing_output = False

    @parameter
    if use_smem_when_writing_output:
        var output_smem = smem_manager.get_output_tensor()
        var output_smem_warp_tile = output_smem.tile[WM, depth](warp_id, 0)

        copy[thread_layout=warp_layout, thread_scope = ThreadScope.WARP,](
            output_smem_warp_tile.vectorize[
                fragment_layout.shape[0].value(),
                fragment_layout.shape[1].value(),
            ](),
            out_reg_tile.vectorize[1, output_frag_size](),
        )

        barrier()

        alias num_threads = config.num_threads()

        alias output_thread_layout = Layout.row_major(
            num_threads * simd_width // depth,
            depth // simd_width,
        )

        copy_sram_to_dram[
            thread_layout=output_thread_layout,
            num_threads=num_threads,
        ](
            output_tile.vectorize[1, simd_width](),
            output_smem.vectorize[1, simd_width](),
        )
    else:
        var output_warp_tile = output_tile.tile[WM, depth](warp_id, 0)
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
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    fragment_layout: Layout,
    warp_layout: Layout,
](
    p_smem_iter: LayoutTensorIter[*_, address_space = AddressSpace.SHARED, **_],
    p_reg_vectorized: LayoutTensor[*_, address_space = AddressSpace.LOCAL, **_],
    warp_id: Int,
):
    alias num_n_mmas_per_bk = num_n_mmas // (BN // BK)

    @parameter
    for i in range(Int(BN // BK)):
        var p_smem_tile = p_smem_iter.next_unsafe(i)[]
        var p_smem_warp_tile = p_smem_tile.tile[WM, BK](warp_id, 0)

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
