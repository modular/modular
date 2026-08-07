# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Does NOT `from . import producer`; the bare reference must not resolve.
def consume() -> Int:
# expected-error @below {{use of unknown declaration 'producer'}}
    return producer.producer_fn()
