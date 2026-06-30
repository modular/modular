# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Does NOT import reexported_fn; the bare reference resolves only via the
# package's __init__ re-export (the deprecated flavor-2 path).
def consume() -> Int:
    return reexported_fn()
