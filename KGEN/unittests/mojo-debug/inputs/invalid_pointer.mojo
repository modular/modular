# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive
from std.memory import dealloc

# The invalid dtype, built directly from the underlying MLIR attribute so this
# debugger test does not depend on any public `DType` alias. This exercises the
# type system's rendering of `scalar<invalid>` (see PrimitiveTypesTest.cpp).
comptime _invalid_dtype = DType(
    mlir_value=__mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
)


def main():
    var base_alloc = alloc[Float32]({count = 1})
    var base: UnsafePointer[
        Float32, origin_of(base_alloc._alloc)
    ] = base_alloc.unsafe_ptr()
    var ptr = base.bitcast[Scalar[_invalid_dtype]]()
    keep_alive(ptr)  # breakpoint
    dealloc(base_alloc^)
