# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Does NOT import reexported_fn; the bare reference must not resolve.
def consume() -> Int:
# expected-error @below {{use of unknown declaration 'reexported_fn'}}
    return reexported_fn()
