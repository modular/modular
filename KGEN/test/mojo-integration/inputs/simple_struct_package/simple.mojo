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
struct GenericBox[T: ImplicitlyCopyable]:
    var value: T

    fn __init__(out self, value: T):
        self.value = value

    fn get(self) -> T:
        return self.value


struct MyStruct(Copyable):
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    fn __copyinit__(out self, existing: Self):
        self.value = existing.value
