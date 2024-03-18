# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from register import *
from utils.index import StaticIntTuple


@mogg_register("test_indices_deduction")
@export
fn _test_indices_deduction[
    num_indices: Int
](indices: StaticIntTuple[num_indices]):
    """
    Used as a test to make sure we correctly deduce the size of indices.
    """
    print("Indices size: ")
    print(num_indices)
    print("Indices: ")
    print(indices)


@mogg_register("test_make_indices")
@export
fn _test_make_indices[num_indices: Int]() -> StaticIntTuple[num_indices]:
    """
    Used to return indices which we can use as a target for tests.
    """
    var out = StaticIntTuple[num_indices]()
    for i in range(num_indices):
        out[i] = i
    return out
