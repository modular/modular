# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from register import *


@register_internal_override("mo.isqrt", 1)
@mogg_elementwise
fn my_isqrt_failing_constraint[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    constrained[
        1 == 2,
        "Expected constraint failure for error message testing",
    ]()
    print("My isqrt")
    return value
