# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides abstractions for using Async Tensor Cores to perform asynchronous
matrix multiplication operations.
"""
from layout import IntTuple, Layout, LayoutTensor
from layout.layout import is_row_major

from gpu import WARP_SIZE
from gpu.memory import AddressSpace
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import thread_idx
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)

from sys import sizeof
from utils import Index, IndexList

# ===-----------------------------------------------------------------------===#
# WGMMA shared memory layout                                                   #
# ===-----------------------------------------------------------------------===#
#
# TODO: add more context for WGMMA, core matrix. Assuming reader know them for now
#
#
# -------------------------------
# M/N x K, K-major, w/o swizzling
# -------------------------------
#
# Consider core matrix cm_M x cm_K, where cm_M = 8 and cm_K = 16 // sizeof[type]()
#
#    !!!! core matrix one contiguous chunk in row-major(cm_M, cm_K) !!!!
#
# E.g.
#                 | core matrix 00 | core matrix 01 |
#                 | core matrix 10 | core matrix 11 |
#
# The elements are stored in shared memory as
#
#       | core matrix 00 | core matrix 10 | core matrix 01 | core matrix 11 |
#
# and each core matrix ij is a contiguous 128B with row_major(cm_M, cn_K) layout.
#
# The share memory tile is logically mapped to a BM x BK sub-matrix in global memory,
# Without swizzling, the tile layout is
#
#     ((cm_M,  BM // cm_M), (cm_K, BK // cm_K))
#   : ((cm_K, cm_M * cm_K), (   1, BM  * cm_K))
#
# coalesceable to (but we use the above like cutlass)
#
#     (  BM, (cm_K, BK // cm_K))
#   : (cm_K, (   1, BM  * cm_K))
#

alias supported_mma_shape = (
    Index(64, 8, 8),
    Index(64, 8, 16),
    Index(64, 16, 16),
    Index(64, 32, 16),
    Index(64, 64, 16),
    Index(64, 128, 16),
    Index(64, 256, 16),
)

# Core matrix dimensions
alias _CM_M = 8
alias _CM_N = 8
alias _CM_K_BYTES = 16
# TODO: unify by the following
alias _CM_NUM_ROWS = 8
alias _CM_ROW_BYTES = 16

alias WGMMA_K_BYTES = 32


# TODO(KERN-1301): Layouts are calculated for 64x8x8 instruction
fn _lhs_layout[mma_shape: IndexList[3]]() -> Layout:
    @parameter
    if mma_shape == Index(64, 8, 8):
        return Layout(
            IntTuple(IntTuple(8, 8), IntTuple(4, 2)),
            IntTuple(IntTuple(4, 32), IntTuple(1, 256)),
        )
    elif mma_shape == Index(64, 8, 16):
        return Layout(
            IntTuple(IntTuple(8, 8), IntTuple(8, 2)),
            IntTuple(IntTuple(8, 64), IntTuple(1, 512)),
        )

    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()

    return Layout()


fn tile_layout_k_major[
    type: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> Layout:
    alias _CM_K = _CM_K_BYTES // sizeof[type]()

    @parameter
    if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        alias k_bytes = BK * sizeof[type]()
        constrained[
            k_bytes == swizzle_mode.bytes(),
            "K dim "
            + String(k_bytes)
            + " doesn't match "
            + String(swizzle_mode),
        ]()

        return Layout(
            IntTuple(
                IntTuple(_CM_M, BM // _CM_M), IntTuple(_CM_K, BK // _CM_K)
            ),
            IntTuple(IntTuple(BK, _CM_M * BK), IntTuple(1, _CM_K)),
        )

    return Layout(
        IntTuple(IntTuple(_CM_M, BM // _CM_M), IntTuple(_CM_K, BK // _CM_K)),
        IntTuple(IntTuple(_CM_K, _CM_M * _CM_K), IntTuple(1, BM * _CM_K)),
    )


fn tile_layout_mn_major[
    type: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> Layout:
    """Return the shared memory layout for mn-major input.

    This is the "unit" layout, the actual shared memory layout can be multiple
    of this unit.
    """

    constrained[
        swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE, "Only support no swizzle"
    ]()

    # Number of elements per row in core matrix
    alias _CM_ROW_LEN = _CM_ROW_BYTES // sizeof[type]()

    return Layout(
        IntTuple(
            IntTuple(_CM_ROW_LEN, mn_dim // _CM_ROW_LEN),
            IntTuple(_CM_NUM_ROWS, k_dim // _CM_NUM_ROWS),
        ),
        IntTuple(
            IntTuple(1, _CM_ROW_LEN * k_dim),
            IntTuple(_CM_ROW_LEN, _CM_NUM_ROWS * _CM_ROW_LEN),
        ),
    )


fn _rhs_layout[mma_shape: IndexList[3]]() -> Layout:
    @parameter
    if mma_shape == Index(64, 8, 8):
        return Layout(IntTuple(IntTuple(4, 2), 8), IntTuple(IntTuple(1, 32), 4))
    elif mma_shape == Index(64, 8, 16):
        return Layout(IntTuple(IntTuple(8, 2), 8), IntTuple(IntTuple(1, 64), 8))

    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()

    return Layout()


fn _lhs_descriptor[
    mma_shape: IndexList[3],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    tensor: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_]
) -> WGMMADescriptor[tensor.dtype]:
    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()

    alias layout = __type_of(tensor).layout
    constrained[
        layout.rank() == 2 and layout[0].rank() == 2 and layout[1].rank() == 2,
        "shared memory tile layout should have structure (rank-2, rank-2).",
    ]()

    alias shape00 = layout[0].shape[0].value()
    alias shape11 = layout[1].shape[1].value()

    # General constraints for all swizzle types.
    constrained[
        shape00 == 8 and shape11 % 2 == 0,
        "Tile shape must be ((8, _), (_, multiple of 2)).",
    ]()

    alias type = __type_of(tensor).dtype
    alias stride10 = layout[1].stride[0].value()

    # Constraints for swizzle
    @parameter
    if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        alias stride_bytes = stride10 * sizeof[type]()
        constrained[
            stride_bytes != swizzle_mode.bytes(),
            "Stride dim bytes "
            + String(stride_bytes)
            + " doesn't match "
            + String(swizzle_mode),
        ]()

    # Ingore 4 LSB.
    alias SBO = (layout[0].stride[1].value() * sizeof[type]()) >> 4
    alias LBO = (layout[1].stride[1].value() * sizeof[type]()) >> 4

    return WGMMADescriptor.create[SBO, LBO, swizzle_mode](tensor.ptr)


fn _rhs_descriptor[
    mma_shape: IndexList[3],
    transposed: Bool = False,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    tensor: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_]
) -> WGMMADescriptor[tensor.dtype]:
    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()

    # Transposed case is same to K-major A matrix.
    @parameter
    if transposed:
        return _lhs_descriptor[mma_shape, swizzle_mode](tensor)

    # Non-Transposed case is MN-major
    alias layout = tensor.layout
    alias stride01 = layout[0].stride[1].value()
    alias stride11 = layout[1].stride[1].value()

    alias type = tensor.dtype
    alias no_swizzle = swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE

    # Swizzle and non-swizzle modes switch SBO and LBO based on
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=bar%2520sync#asynchronous-warpgroup-level-majorness-supported-by-strides
    alias SBO = ((stride01 if no_swizzle else stride11) * sizeof[type]()) >> 4
    alias LBO = ((stride11 if no_swizzle else stride01) * sizeof[type]()) >> 4

    return WGMMADescriptor.create[SBO, LBO](tensor.ptr)


# TODO(KERN-1301): Layouts are calculated for 64x8x8 instruction
fn _output_register_size[mma_shape: IndexList[3]]() -> Int:
    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()
    return mma_shape[0] * mma_shape[1] // 128


fn _dtype(dtype: DType) -> DType:
    if dtype is DType.float32:
        return DType.tensor_float32
    return dtype


struct TensorCoreAsync[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    mma_shape: IndexList[3],
    /,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = False,
]:
    alias lhs_operand_type = LayoutTensor[
        a_type,
        _lhs_layout[mma_shape](),
        address_space = AddressSpace.SHARED,
    ]

    alias rhs_operand_type = LayoutTensor[
        b_type,
        _rhs_layout[mma_shape](),
        address_space = AddressSpace.SHARED,
    ]

    alias result_operand_type = SIMD[c_type, _output_register_size[mma_shape]()]

    @always_inline
    fn __init__(out self):
        constrained[
            mma_shape in supported_mma_shape,
            String(
                "WGMMA operation of shape '", mma_shape, "' is not supported"
            ),
        ]()

    @staticmethod
    @always_inline
    fn __call__(
        lhs: Self.lhs_operand_type,
        rhs: Self.rhs_operand_type,
        c_reg: Self.result_operand_type,
    ) -> Self.result_operand_type:
        lhs_descriptor = _lhs_descriptor[mma_shape](lhs)
        rhs_descriptor = _rhs_descriptor[
            mma_shape, transposed = Self.transpose_b
        ](rhs)
        r_reg = wgmma_async[
            mma_shape[0],
            mma_shape[1],
            mma_shape[2],
            a_type = _dtype(a_type),
            b_type = _dtype(b_type),
        ](lhs_descriptor, rhs_descriptor, c_reg)
        return r_reg

    @staticmethod
    @always_inline
    fn wgmma(
        a_smem_tile: LayoutTensor[
            a_type, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        b_smem_tile: LayoutTensor[
            b_type, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        c_reg_tile: LayoutTensor[
            c_type, _, address_space = AddressSpace.LOCAL, *_, **_
        ],
    ):
        a_desc = _lhs_descriptor[mma_shape, a_swizzle](a_smem_tile)
        b_desc = _rhs_descriptor[mma_shape, transpose_b, b_swizzle](b_smem_tile)

        alias a_smem_layout = a_smem_tile.layout
        alias b_smem_layout = b_smem_tile.layout

        # Number of core matrices in stride dim
        alias a_num_CMs_m = mma_shape[0] // _CM_M
        alias b_num_CMs_n = mma_shape[1] // _CM_N

        # Layout modes are always (MN, K) transpose or not.
        alias a_stride01 = a_smem_layout[0].stride[1].value()
        alias a_stride11 = a_smem_layout[1].stride[1].value()
        alias b_stride01 = b_smem_layout[0].stride[1].value()
        alias b_stride11 = b_smem_layout[1].stride[1].value()
        # Strides between WGMMA tiles
        alias a_m_stride = a_stride01 * a_num_CMs_m * sizeof[a_type]()
        alias b_n_stride = b_stride01 * b_num_CMs_n * sizeof[b_type]()
        # K dim is stepped by 2 core matrices.
        alias a_k_stride = a_stride11 * 2 * sizeof[a_type]()
        alias b_k_stride = b_stride11 * 2 * sizeof[b_type]()

        alias num_m_mmas = a_smem_layout[0].size() // mma_shape[0]
        alias num_n_mmas = b_smem_layout[0].size() // mma_shape[1]
        alias num_k_mmas = a_smem_layout[1].size() // mma_shape[2]

        # Vectorize each wgmma's fragment size.
        alias c_frag_size = mma_shape[0] * mma_shape[1] // 128
        c_frags = c_reg_tile.vectorize[1, c_frag_size]()
        constrained[
            __type_of(c_frags).layout.size() == num_m_mmas * num_n_mmas,
            "C fragments' size doesn't match the total number of wgmma.",
        ]()

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma

                @parameter
                for k_mma in range(num_k_mmas):
                    a_desc_m = a_desc + m_mma * a_m_stride + k_mma * a_k_stride
                    b_desc_n = b_desc + n_mma * b_n_stride + k_mma * b_k_stride

                    c_frags[mma_id, 0] = wgmma_async[
                        mma_shape[0],
                        mma_shape[1],
                        mma_shape[2],
                        a_type = _dtype(a_type),
                        b_type = _dtype(b_type),
                    ](a_desc_m, b_desc_n, c_frags[mma_id, 0])

    @staticmethod
    @always_inline
    fn arrive():
        wgmma_fence_aligned()

    @staticmethod
    @always_inline
    fn commit_group():
        wgmma_commit_group_sync()

    @staticmethod
    @always_inline
    fn wait_for_all():
        wgmma_wait_group_sync()

    # TODO(KERN-1301): Output layout is calculated for 64x8x8 instruction
    # Stores the result into the corresponding warp group tile fragment.
    @staticmethod
    @always_inline
    fn store_result(
        warp_group_tile: LayoutTensor[c_type, _, *_, **_],
        res_reg_tile: Self.result_operand_type,
    ):
        constrained[
            mma_shape in supported_mma_shape,
            String(
                "WGMMA operation of shape '", mma_shape, "' is not supported"
            ),
        ]()
        warp_id, lan_id = divmod(thread_idx.x, UInt(WARP_SIZE))
        alias warp_row_major_layout = Layout.row_major(8, 4)

        var th_local_res = (
            warp_group_tile.tile[16, 8](warp_id, 0)
            .vectorize[1, 2]()
            .distribute[warp_row_major_layout](lan_id)
        )

        th_local_res[0][0] = res_reg_tile[0]
        th_local_res[0][1] = res_reg_tile[1]
        th_local_res[1][0] = res_reg_tile[2]
        th_local_res[1][1] = res_reg_tile[3]

    @staticmethod
    @always_inline
    fn allocate_lhs() -> Self.lhs_operand_type:
        return Self.lhs_operand_type.stack_allocation()

    @staticmethod
    @always_inline
    fn allocate_rhs() -> Self.rhs_operand_type:
        return Self.rhs_operand_type.stack_allocation()

    @staticmethod
    @always_inline
    fn allocate_result(
        initial_val: Scalar[c_type],
    ) -> Self.result_operand_type:
        return Self.result_operand_type(initial_val)
