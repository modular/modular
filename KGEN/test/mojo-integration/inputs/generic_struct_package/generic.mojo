# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Generic struct for testing extensions on parametric types


struct Container[T: ImplicitlyCopyable & ImplicitlyDestructible]:
    var value: Self.T

    def __init__(out self, value: Self.T):
        self.value = value

    def get(self) -> Self.T:
        return self.value
