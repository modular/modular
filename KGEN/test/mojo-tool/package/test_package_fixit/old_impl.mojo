# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: fn old_origin_of
fn old_origin_of[T: AnyType](b: T):
    # CHECK-NEXT: _ = origin_of(b)
    # CHECK-NOT: _ = __origin_of(b)
    _ = __origin_of(b)
