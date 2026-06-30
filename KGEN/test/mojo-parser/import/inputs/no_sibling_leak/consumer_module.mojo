# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Does NOT `from . import producer`; the bare reference must not resolve.
def consume() -> Int:
    return producer.producer_fn()
