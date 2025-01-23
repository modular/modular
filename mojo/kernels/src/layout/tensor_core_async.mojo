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
from gpu.id import thread_idx
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
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
# M/N x K, K-major, w/o siwzzling
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
)

# Core matrix dimensions
alias _CM_M = 8
alias _CM_N = 8
alias _CM_K_BYTES = 16

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


fn tile_layout_k_major[type: DType, BM: Int, BK: Int]() -> Layout:
    alias _CM_K = _CM_K_BYTES // sizeof[type]()

    return Layout(
        IntTuple(IntTuple(_CM_M, BM // _CM_M), IntTuple(_CM_K, BK // _CM_K)),
        IntTuple(IntTuple(_CM_K, _CM_M * _CM_K), IntTuple(1, BM * _CM_K)),
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
    mma_shape: IndexList[3]
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

    constrained[
        layout[0].shape[0].value() == 8 and layout[1].shape[1].value() == 2,
        "shared memory tile layout should have shape ((8, _), (_, 2)).",
    ]()

    # Ingore 4 LSB.
    alias T = __type_of(tensor).dtype
    alias SBO = (layout[0].stride[1].value() * sizeof[T]()) >> 4
    alias LBO = (layout[1].stride[1].value() * sizeof[T]()) >> 4

    return WGMMADescriptor.create[SBO, LBO](tensor.ptr)


fn _rhs_descriptor[
    mma_shape: IndexList[3], transposed: Bool = False
](
    tensor: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_]
) -> WGMMADescriptor[tensor.dtype]:
    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()
    return WGMMADescriptor.create[1, 8](tensor.ptr)


# TODO(KERN-1301): Layouts are calculated for 64x8x8 instruction
fn _output_register_size[
    mma_shape: IndexList[3]
](in_dtype: DType, out_dtype: DType) -> Int:
    constrained[
        mma_shape in supported_mma_shape,
        String("WGMMA operation of shape '", mma_shape, "' is not supported"),
    ]()
    return 4


fn _dtype(dtype: DType) -> DType:
    if dtype is DType.float32:
        return DType.tensor_float32
    return dtype


struct TensorCoreAsync[
    out_type: DType,
    in_type: DType,
    mma_shape: IndexList[3],
]:
    alias lhs_operand_type = LayoutTensor[
        in_type,
        _lhs_layout[mma_shape](),
        address_space = AddressSpace.SHARED,
    ]

    alias rhs_operand_type = LayoutTensor[
        in_type,
        _rhs_layout[mma_shape](),
        address_space = AddressSpace.SHARED,
    ]

    alias result_operand_type = SIMD[
        out_type, _output_register_size[mma_shape](out_type, in_type)
    ]

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
        rhs_descriptor = _rhs_descriptor[mma_shape](rhs)
        r_reg = wgmma_async[
            mma_shape[0],
            mma_shape[1],
            mma_shape[2],
            a_type = _dtype(in_type),
            b_type = _dtype(in_type),
        ](lhs_descriptor, rhs_descriptor, c_reg)
        return r_reg

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
        warp_group_tile: LayoutTensor[out_type, _, *_, **_],
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
        initial_val: Scalar[out_type],
    ) -> Self.result_operand_type:
        return Self.result_operand_type(initial_val)
