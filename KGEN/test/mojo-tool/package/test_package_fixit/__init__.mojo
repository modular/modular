# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .old_impl import *


# CHECK-LABEL: fn old_origin_of_2
fn old_origin_of_2[T: AnyType](c: T):
    # CHECK-NEXT: _ = origin_of(c)
    # CHECK-NOT: _ = __origin_of(c)
    _ = __origin_of(c)
