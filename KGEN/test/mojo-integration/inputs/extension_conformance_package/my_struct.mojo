# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias int = __mlir_type.index


trait Convertible:
    fn convert(self) -> int:
        ...


struct MyStruct:
    var value: int

    fn __init__(out self, value: int):
        self.value = value


__extension MyStruct(Convertible):
    alias ExtensionAlias = int

    fn convert(self) -> int:
        return self.value
