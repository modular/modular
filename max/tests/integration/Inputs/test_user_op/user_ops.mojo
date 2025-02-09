# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from register import *

from utils.index import IndexList


@register_internal("test_indices_deduction")
fn _test_indices_deduction[num_indices: Int](indices: IndexList[num_indices]):
    """
    Used as a test to make sure we correctly deduce the size of indices.
    """
    print("Indices size: ")
    print(num_indices)
    print("Indices: ")
    print(indices)


@register_internal("test_make_indices")
fn _test_make_indices[num_indices: Int]() -> IndexList[num_indices]:
    """
    Used to return indices which we can use as a target for tests.
    """
    var out = IndexList[num_indices]()
    for i in range(num_indices):
        out[i] = i
    return out
