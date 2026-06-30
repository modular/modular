# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Does NOT `from .producer import producer_fn`; the bare reference must not resolve.
def consume() -> Int:
    return producer_fn()
