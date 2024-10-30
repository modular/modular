# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from register import mogg_register


@mogg_register("mutable_test_op")
fn mutable_test_op[
    type: DType,
    rank: Int,
](output: NDBuffer[type, rank]):
    """
    Increment the first buffer value by 1.

    Used for testing in-place custom ops.
    """
    x = output.data.load(0)
    x += 1
    output.data.store(0, x)
