# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct PlainStruct:
    pass


# TODO(MOCO-522): Simplify generic_struct_package, struct_only_package,
# and simple_struct_package into this one package
struct GenericBox[T: ImplicitlyCopyable & ImplicitlyDestructible]:
    var value: Self.T

    def __init__(out self, value: Self.T):
        self.value = value

    def get(self) -> Self.T:
        return self.value


struct MyStruct(Copyable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self, *, copy: Self):
        self.value = copy.value
