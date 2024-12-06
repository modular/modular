# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides abstractions for using Async Tensor Cores to perform asynchronous 
matrix multiplication operations.
"""
from layout import IntTuple, Layout, LayoutTensor

from gpu.memory import AddressSpace
from gpu.id import ThreadIdx
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_wait_group_sync,
)

from utils import Index, IndexList


# TODO: Remove when fix stdlib bug!
fn _to_str(mma_shape: IndexList[3]) -> String:
    return (
        "["
        + str(mma_shape[1])
        + ", "
        + str(mma_shape[1])
        + ", "
        + str(mma_shape[2])
        + "]"
    )


# TODO(KERN-1301): Layouts are calculated for 64x8x8 instruction
fn _lhs_layout[mma_shape: IndexList[3]]() -> Layout:
    constrained[
        mma_shape == Index(64, 8, 8),
        "WGMMA instruction shape `" + _to_str(mma_shape) + "` is not supported",
    ]()
    return Layout(
        IntTuple(IntTuple(8, 8), IntTuple(4, 2)),
        IntTuple(IntTuple(4, 32), IntTuple(1, 256)),
    )


fn _rhs_layout[mma_shape: IndexList[3]]() -> Layout:
    constrained[
        mma_shape == Index(64, 8, 8),
        "WGMMA instruction shape `" + _to_str(mma_shape) + "` is not supported",
    ]()
    return Layout(IntTuple(IntTuple(4, 2), 8), IntTuple(IntTuple(1, 32), 4))


fn _lhs_descriptor[
    mma_shape: IndexList[3]
](
    tensor: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_]
) -> WGMMADescriptor[tensor.dtype]:
    constrained[
        mma_shape == Index(64, 8, 8),
        "WGMMA instruction shape`" + _to_str(mma_shape) + "` is not supported",
    ]()
    return WGMMADescriptor.create[8, 64](tensor.ptr)


fn _rhs_descriptor[
    mma_shape: IndexList[3]
](
    tensor: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_]
) -> WGMMADescriptor[tensor.dtype]:
    constrained[
        mma_shape == Index(64, 8, 8),
        "WGMMA instruction shape`" + _to_str(mma_shape) + "` is not supported",
    ]()
    return WGMMADescriptor.create[1, 8](tensor.ptr)


# TODO(KERN-1301): Layouts are calculated for 64x8x8 instruction
fn _output_register_size[
    mma_shape: IndexList[3]
](in_dtype: DType, out_dtype: DType) -> Int:
    constrained[
        mma_shape == Index(64, 8, 8),
        "WGMMA instruction shape`" + _to_str(mma_shape) + "` is not supported",
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
            mma_shape == Index(64, 8, 8),
            "WGMMA instruction shape`"
            + _to_str(mma_shape)
            + "` is not supported",
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
            mma_shape == Index(64, 8, 8),
            "WGMMA instruction shape`"
            + _to_str(mma_shape)
            + "` is not supported",
        ]()
        warp_id = ThreadIdx.x // 32
        lan_id = ThreadIdx.x % 32
        alias warp_row_major_layout = Layout.row_major(8, 4)

        th_local_res = (
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
