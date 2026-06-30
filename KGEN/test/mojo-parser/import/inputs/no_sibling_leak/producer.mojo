# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

def producer_fn() -> Int:
    return 1


# Re-exported by the package's __init__ (unlike producer_fn), so a sibling that
# names it bare exercises the flavor-2 deprecation path.
def reexported_fn() -> Int:
    return 2
