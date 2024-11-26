# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor_utils.tensor_spec import StaticTensorSpec
from tensor_utils.managed_tensor_slice import ManagedTensorSlice
from utils import IndexList


fn __mogg_intrinsic_attr(intrin: StringLiteral):
    return


@__mogg_intrinsic_attr("mogg.intrinsic_register")
fn register(name: StringLiteral, num_dps_outputs: Int = 1):
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
    func: fn[_width: Int] (IndexList[rank]) capturing -> SIMD[type, _width],
](arr: ManagedTensorSlice[type, rank]):
    pass


fn uses_opaque():
    """
    Dummy registration decorator for testing.

    TODO(GEX-1145): Remove the need for this.
    """
    return


@__mogg_intrinsic_attr("mogg.mutable")
fn mutable(*names: StringLiteral):
    return
