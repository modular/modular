# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from MOGGExperimental import empty_tensor
from MOGGIntList import IntList
from MOGGTensor import Tensor
from register import *

from buffer.list import DimList

# Dummy kernels to test fundamental mechanisms without overwriting the normal
# kernels.


@mogg_register_override("mo.matmul", 1000)
@export
fn mo_matmul[
    out_shape: DimList,
    transpose_in_1: Bool,
    packed_in_1: Bool,
    lambdas_have_fusion: Bool = False,
](x: Tensor, y: Tensor) -> Tensor[x.type, out_shape]:
    var shape = IntList[out_shape](x.shape[0], y.shape[1])
    var out = empty_tensor[x.type](shape)

    print(transpose_in_1)
    print(packed_in_1)
    print(lambdas_have_fusion)

    x.shape.print()
    y.shape.print()
    out.shape.print()

    return out
