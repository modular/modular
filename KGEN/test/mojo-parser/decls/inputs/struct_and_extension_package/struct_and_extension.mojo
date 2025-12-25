# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

struct MyStruct:
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value


__extension MyStruct:
    fn extended_method(self):
        pass
