# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from register import *


@mogg_register_override("mo.add", 1)
@mogg_elementwise
@export
fn my_add[
    type: DType, simd_width: Int
](value1: SIMD[type, simd_width], value2: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return value1 + value2 + 100
