# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg

from buffer.dimlist import DimList

from utils import IndexList, StaticTuple


fn __mogg_intrinsic_attr(intrin: StringLiteral):
    return


# Register a DPS Kernel
@__mogg_intrinsic_attr("mogg.intrinsic_register")
fn register(name: StringLiteral, num_dps_outputs: Int = 1):
    pass


# Indicates that a DPS Kernel is elementwise
@__mogg_intrinsic_attr("mogg.elementwise")
fn elementwise():
    return


# Indicates which I/Os of the DPS kernel support fusion
@__mogg_intrinsic_attr("mogg.enable_fusion_for")
fn enable_fusion_for(*names: StringLiteral):
    return


# Indicates that a DPS Kernel is a view operation
@__mogg_intrinsic_attr("mogg.view_kernel")
fn view_kernel():
    return


@__mogg_intrinsic_attr("mogg.mutable")
fn mutable(*names: StringLiteral):
    return


# Compile time Tensor informations
@value
@register_passable("trivial")
struct StaticTensorSpec[
    type: DType,
    rank: Int,
]:
    # Represents the DimList type (not accessible from KGEN tests).
    alias in_lambda_t = fn[simd_width: Int] (IndexList[rank]) capturing -> SIMD[
        type, simd_width
    ]
    alias out_lambda_t = fn[simd_width: Int] (
        IndexList[rank], SIMD[type, simd_width]
    ) capturing -> None

    var shape: DimList
    var strides: DimList

    var alignment: Int
    var address_space: AddressSpace
    var exclusive: Bool

    var in_lambda: OptionalReg[Self.in_lambda_t]
    var out_lambda: OptionalReg[Self.out_lambda_t]

    fn __init__(
        out self,
        shape: DimList,
        strides: DimList,
        alignment: Int,
        address_space: AddressSpace,
        exclusive: Bool,
        in_lambda: OptionalReg[Self.in_lambda_t],
        out_lambda: OptionalReg[Self.out_lambda_t],
    ):
        self.shape = shape
        self.strides = strides
        self.alignment = alignment
        self.address_space = address_space
        self.exclusive = exclusive
        self.in_lambda = in_lambda
        self.out_lambda = out_lambda

    fn __init__(out self):
        self = Self(
            DimList.create_unknown[rank](),
            DimList.create_unknown[rank](),
            1,
            AddressSpace.GENERIC,
            True,
            OptionalReg[Self.in_lambda_t](None),
            OptionalReg[Self.out_lambda_t](None),
        )


fn create_none_spec[type: DType, rank: Int]() -> StaticTensorSpec[type, rank]:
    return StaticTensorSpec[type, rank]()


# Returns the compile time informations of a DPS kernel I/O.
# fn foo(x: UnsafeTensorSlice):
#   alias specs = specsof[x.type, x.rank]("x")
@__mogg_intrinsic_attr("mogg.intrinsic_tensor_spec_hook")
fn specsof[
    type: DType, rank: Int
](name: StringLiteral) -> StaticTensorSpec[type, rank]:
    alias TENSOR_SPEC_NONE = create_none_spec[type, rank]()
    return TENSOR_SPEC_NONE


@__mogg_intrinsic_attr("mogg.intrinsic_tensor_spec_tuple_hook")
fn specsof[
    type: DType,
    rank: Int,
    tensors_count: Int,
](name: StringLiteral) -> StaticTuple[
    StaticTensorSpec[type, rank], tensors_count
]:
    alias TENSOR_SPEC_NONE = create_none_spec[type, rank]()
    return TENSOR_SPEC_NONE
