# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Generic struct for testing extensions on parametric types


struct Container[T: ImplicitlyCopyable]:
    var value: Self.T

    fn __init__(out self, value: Self.T):
        self.value = value

    fn get(self) -> Self.T:
        return self.value
