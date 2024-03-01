# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils.index import StaticIntTuple  # skip
from utils.static_tuple import StaticTuple  # skip
import builtin
import builtin.dtype


fn functionWithNestedType(x: dtype.DType):
    var y: builtin.int.Int = 12
    pass


fn functionWithBuiltins(x: Bool) -> Bool:
    var copy: Bool = x
    return copy


fn functionWithParametrizedArgument(x: StaticIntTuple[2]) -> StaticIntTuple[2]:
    var copy: StaticIntTuple[2] = x
    return copy


fn parametrizedFunction[
    size: Int
](x: StaticTuple[Int, size]) -> StaticTuple[Int, size]:
    var copy: StaticTuple[Int, size] = x
    return copy
