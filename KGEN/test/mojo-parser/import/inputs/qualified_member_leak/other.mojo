# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Deliberately defines NEITHER `shared_fn` nor `helpers`, so qualified accesses
# `other.shared_fn` and `other.helpers` have no real member to resolve to.
def other_fn() -> Int:
    return 3
