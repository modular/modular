# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# `other` is imported explicitly, but `helpers` is NOT a member of `other` (it
# is a sibling module of the package). A qualified `other.helpers` must be a
# hard error - the intra-package sibling-module fallback must not fire for a
# member access whose base is some other module.
from . import other


def consume() -> Int:
    return other.helpers.helper_fn()
