# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from register import *
from buffer import NDBuffer


@mogg_register("fails_to_elaborate")
fn fails_to_elaborate(out: NDBuffer[DType.int32, 1]):
    constrained[False, "oops"]()
