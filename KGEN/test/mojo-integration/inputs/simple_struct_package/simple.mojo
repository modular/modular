# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Minimal struct for testing extensions


@fieldwise_init
struct PlainStruct:
    pass


# Generic struct for testing extensions on parametric types
struct GenericBox[T: ImplicitlyCopyable]:
    var value: T

    fn __init__(out self, value: T):
        self.value = value

    fn get(self) -> T:
        return self.value
