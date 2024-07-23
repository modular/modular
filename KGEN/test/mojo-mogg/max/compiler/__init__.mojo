# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.driver.tensor_spec import StaticTensorSpec


fn register(name: StringLiteral):
    pass


fn mogg_intrinsic_attr(intrin: StringLiteral):
    return


fn create_none_spec() -> StaticTensorSpec:
    return StaticTensorSpec()


@mogg_intrinsic_attr("mogg.tensor_spec_hook")
@export
fn specsof(name: StringLiteral) -> StaticTensorSpec:
    alias TENSOR_SPEC_NONE = create_none_spec()
    return TENSOR_SPEC_NONE
