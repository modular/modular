# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor_utils.tensor_spec import StaticTensorSpec
from tensor_utils.managed_tensor_slice import ManagedTensorSlice
from utils import StaticIntTuple


fn __mogg_intrinsic_attr(intrin: StringLiteral):
    return


@__mogg_intrinsic_attr("mogg.intrinsic_register")
fn register(name: StringLiteral):
    pass


@__mogg_intrinsic_attr("mogg.elementwise")
fn elementwise():
    pass


fn create_none_spec[type: DType, rank: Int]() -> StaticTensorSpec[type, rank]:
    return StaticTensorSpec[type, rank]()


@__mogg_intrinsic_attr("mogg.intrinsic_tensor_spec_hook")
@export
fn specsof[
    type: DType, rank: Int
](name: StringLiteral) -> StaticTensorSpec[type, rank]:
    alias TENSOR_SPEC_NONE = create_none_spec[type, rank]()
    return TENSOR_SPEC_NONE


@__mogg_intrinsic_attr("mogg.for_each")
fn for_each[
    type: DType,
    rank: Int,
    func: fn[_width: Int] (StaticIntTuple[rank]) capturing -> SIMD[
        type, _width
    ],
](arr: ManagedTensorSlice[type, rank]):
    pass
