# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from String import String
from Index import StaticIntTuple  # skip
from StaticTuple import StaticTuple  # skip
import DType


fn functionWithNestedType(x: DType.DType):
    pass


fn functionWithBuiltins(x: Bool) -> Bool:
    let copy: Bool = x
    return copy


fn functionWithParametrizedArgument(x: StaticIntTuple[2]) -> StaticIntTuple[2]:
    let copy: StaticIntTuple[2] = x
    return copy


fn parametrizedFunction[
    size: Int
](x: StaticTuple[size, Int]) -> StaticTuple[size, Int]:
    let copy: StaticTuple[size, Int] = x
    return copy
