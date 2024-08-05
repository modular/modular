# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor_utils.tensor_spec import StaticTensorSpec
from tensor_utils.unsafe_tensor_slice import _StaticTensorType


fn register(name: StringLiteral):
    pass


fn mogg_intrinsic_attr(intrin: StringLiteral):
    return


fn create_none_spec[type: DType, rank: Int]() -> StaticTensorSpec[type, rank]:
    return StaticTensorSpec[type, rank]()


struct KernelSpec:
    pass


@mogg_intrinsic_attr("mogg.tensor_spec_hook")
@export
fn specsof[
    T: _StaticTensorType
](name: StringLiteral) -> StaticTensorSpec[
    T._get_dtype(), T._get_static_rank()
]:
    alias TENSOR_SPEC_NONE = create_none_spec[
        T._get_dtype(), T._get_static_rank()
    ]()
    return TENSOR_SPEC_NONE
