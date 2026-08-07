# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Does NOT `from .producer import producer_fn`; the bare reference must not resolve.
def consume() -> Int:
# expected-error @below {{use of unknown declaration 'producer_fn'}}
    return producer_fn()
