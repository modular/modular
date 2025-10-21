# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Generic struct for testing extensions on parametric types


struct Container[T: ImplicitlyCopyable]:
    var value: T

    fn __init__(out self, value: T):
        self.value = value

    fn get(self) -> T:
        return self.value
