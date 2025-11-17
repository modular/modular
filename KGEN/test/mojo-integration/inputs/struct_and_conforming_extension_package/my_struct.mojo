# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait Convertible:
    fn convert(self) -> Int:
        ...


struct MyStruct:
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value


__extension MyStruct(Convertible):
    comptime ExtensionAlias = Int

    fn convert(self) -> Int:
        return self.value
