# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Generic struct with a conditional `Deinitable` conformance, used to
# check that destructor discharge works when this struct is imported and left
# signature-resolved only.


struct ConditionalHelper[T: Movable](
    Deinitable where conforms_to(T, Deinitable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        pass
