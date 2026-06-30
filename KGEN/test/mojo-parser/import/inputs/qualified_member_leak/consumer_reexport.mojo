# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# `other` is imported explicitly, but `shared_fn` is NOT a member of `other`
# (it lives in the package's __init__). A qualified `other.shared_fn` must be a
# hard error - the intra-package __init__ fallback must not fire for a member
# access whose base is some other module.
from . import other


def consume() -> Int:
    return other.shared_fn()
