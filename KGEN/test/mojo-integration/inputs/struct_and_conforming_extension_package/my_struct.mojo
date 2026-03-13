# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


trait Convertible:
    def convert(self) -> Int:
        ...


struct MyStruct:
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


__extension MyStruct(Convertible):
    comptime ExtensionAlias = Int

    def convert(self) -> Int:
        return self.value
