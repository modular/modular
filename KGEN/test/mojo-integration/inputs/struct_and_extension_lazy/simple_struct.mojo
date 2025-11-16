# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct BaseType:
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    fn __init__(out self):
        self.value = 42
