# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import simdwidthof
from tensor_utils_internal import ManagedTensorSlice
from buffer.dimlist import DimList
from utils import StaticIntTuple
from collections import Optional, OptionalReg
import algorithm.functional


fn __mogg_intrinsic_attr(intrin: StringLiteral):
    return


@__mogg_intrinsic_attr("mogg.intrinsic_register")
fn register(name: StringLiteral):
    pass


fn elementwise():
    return


@no_inline
fn shapeof(tensor_name: StringLiteral) -> Optional[DimList]:
    return None


@no_inline
fn stridesof(tensor_name: StringLiteral) -> Optional[DimList]:
    return None


@no_inline
fn output_lambda[
    type: DType = DType.invalid,
    rank: Int = 0,
](tensor_name: StringLiteral) -> Optional[
    fn[width: Int] (StaticIntTuple[rank], SIMD[type, width]) capturing -> None
]:
    return None


@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (StaticIntTuple[rank]) capturing -> SIMD[type, width],
](tensor: ManagedTensorSlice[type, rank]):
    alias simd_width = simdwidthof[tensor.type]()

    @parameter
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](index: StaticIntTuple[rank]) capturing:
        constrained[rank == tensor.rank]()
        var val = func[width](rebind[StaticIntTuple[tensor.rank]](index))
        tensor.store(index, val)

    algorithm.functional.elementwise[elementwise_fn_wrapper, simd_width](
        tensor.get_static_spec().shape
    )


struct StaticTensorSpec[type: DType, rank: Int]:
    # Represents the DimList type (not accessible from KGEN tests).
    alias in_lambda_t = fn[simd_width: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width]
    alias out_lambda_t = fn[simd_width: Int] (
        StaticIntTuple[rank], SIMD[type, simd_width]
    ) capturing -> None

    var shape: DimList
    var strides: DimList
    var in_lambda: OptionalReg[Self.in_lambda_t]
    var out_lambda: OptionalReg[Self.out_lambda_t]

    fn __init__(
        inout self,
        shape: DimList,
        strides: DimList,
        in_lambda: OptionalReg[Self.in_lambda_t],
        out_lambda: OptionalReg[Self.out_lambda_t],
    ):
        self.shape = shape
        self.strides = strides
        self.in_lambda = in_lambda
        self.out_lambda = out_lambda

    fn __init__(inout self):
        self = Self(
            DimList(),
            DimList(),
            OptionalReg[Self.in_lambda_t](None),
            OptionalReg[Self.out_lambda_t](None),
        )


fn create_none_spec[type: DType, rank: Int]() -> StaticTensorSpec[type, rank]:
    return StaticTensorSpec[type, rank]()


@__mogg_intrinsic_attr("mogg.intrinsic_tensor_spec_hook")
@export
fn specsof[
    type: DType, rank: Int
](name: StringLiteral) -> StaticTensorSpec[type, rank]:
    alias TENSOR_SPEC_NONE = create_none_spec[type, rank]()
    return TENSOR_SPEC_NONE
