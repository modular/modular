# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import simdwidthof
from tensor_utils_internal import UnsafeTensorSlice
from buffer.dimlist import DimList
from utils import StaticIntTuple
from collections import Optional
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
](tensor: UnsafeTensorSlice[type, rank]):
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


@register_passable("trivial")
struct OptionalReg[T: AnyTrivialRegType]:
    alias _mlir_type = __mlir_type[`!kgen.variant<`, T, `, i1>`]
    var _value: Self._mlir_type

    fn __init__(inout self):
        self = Self(None)

    fn __init__(inout self, value: NoneType):
        self._value = __mlir_op.`kgen.variant.create`[
            _type = Self._mlir_type, index = Int(1).value
        ](__mlir_attr.false)

    fn __bool__(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Int(0).value](self._value)

    fn value(self) -> T:
        return __mlir_op.`kgen.variant.get`[index = Int(0).value](self._value)


struct StaticTensorSpec[type: DType, rank: Int]:
    # Represents the DimList type (not accessible from KGEN tests).
    alias _dims_type = __mlir_type[`!kgen.variadic<index>`]
    alias in_lambda_t = fn[simd_width: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width]
    alias out_lambda_t = fn[simd_width: Int] (
        StaticIntTuple[rank], SIMD[type, simd_width]
    ) capturing -> None

    var shape: Self._dims_type
    var strides: Self._dims_type
    var in_lambda: OptionalReg[Self.in_lambda_t]
    var out_lambda: OptionalReg[Self.out_lambda_t]

    fn __init__(
        inout self,
        shape: Self._dims_type,
        strides: Self._dims_type,
        in_lambda: OptionalReg[Self.in_lambda_t],
        out_lambda: OptionalReg[Self.out_lambda_t],
    ):
        self.shape = shape
        self.strides = strides
        self.in_lambda = in_lambda
        self.out_lambda = out_lambda

    fn __init__(inout self):
        var shape = __mlir_op.`pop.variadic.create`[_type = Self._dims_type,]()
        var strides = __mlir_op.`pop.variadic.create`[
            _type = Self._dims_type,
        ]()
        self = Self(
            shape,
            strides,
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
