# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Generic struct with a conditional `ImplicitlyDeletable` conformance, used to
# check that destructor discharge works when this struct is imported and left
# signature-resolved only.


struct ConditionalHelper[T: Movable](
    ImplicitlyDeletable where conforms_to(T, ImplicitlyDeletable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def __del__(deinit self) where conforms_to(Self.T, ImplicitlyDeletable):
        pass
