# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor_utils.tensor_spec import StaticTensorSpec


fn __mogg_intrinsic_attr(intrin: StringLiteral):
    return


@__mogg_intrinsic_attr("mogg.intrinsic_register")
fn register(name: StringLiteral):
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
