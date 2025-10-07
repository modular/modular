# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias int = __mlir_type.index

struct MyStruct:
    var value: int

    fn __init__(out self, value: int):
        self.value = value


__extension MyStruct:
    fn extended_method(self):
        pass