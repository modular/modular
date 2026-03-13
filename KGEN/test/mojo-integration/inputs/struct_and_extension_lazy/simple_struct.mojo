# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct BaseType:
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self):
        self.value = 42
