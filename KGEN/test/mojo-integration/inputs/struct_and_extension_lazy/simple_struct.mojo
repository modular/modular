# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias int = __mlir_type.index


struct BaseType:
    var value: int

    fn __init__(out self, value: int):
        self.value = value

    fn __init__(out self):
        self.value = __mlir_attr.`42 : index`
