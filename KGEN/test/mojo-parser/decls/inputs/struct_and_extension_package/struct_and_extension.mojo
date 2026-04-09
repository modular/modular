# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct MyStruct:
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


__extension MyStruct:
    def extended_method(self):
        pass
