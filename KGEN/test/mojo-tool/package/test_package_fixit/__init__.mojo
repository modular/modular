# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .old_impl import *


# CHECK-LABEL: def old_origin_of_2
def old_origin_of_2[T: AnyType](c: T):
    # CHECK-NEXT: _ = origin_of(c)
    # CHECK-NOT: _ = __origin_of(c)
    _ = __origin_of(c)
