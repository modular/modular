# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from register import *


@mogg_register_override("mo.max", 1)
@mogg_elementwise
@always_inline
@export
fn mogg_max[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return max(x, y)


@mogg_register_override("mo.min", 1)
@mogg_elementwise
@always_inline
@export
fn mogg_min[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return mogg_max(x, y)


@mogg_register("test_override_dummy_op")
@mogg_elementwise
@always_inline
@export
fn my_func[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return mogg_min(x, y)
