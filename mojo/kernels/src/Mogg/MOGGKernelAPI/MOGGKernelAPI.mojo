# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from tensor_utils import UnsafeTensorSlice
from utils.index import StaticIntTuple


@compiler.register("imposter_add")
struct Foo:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: UnsafeTensorSlice, x: UnsafeTensorSlice, y: UnsafeTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x.load[width](idx)) + rebind[
                SIMD[z.type, width]
            ](y.load[width](idx))

        compiler.foreach[func](z)

    @staticmethod
    fn shape(
        x: UnsafeTensorSlice, y: UnsafeTensorSlice
    ) -> StaticIntTuple[x.rank]:
        return x.get_static_spec().shape
