# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct ZInt:
    def __init__(out self):
        pass


# Struct with a basic constructor
struct MyType:
    var value: ZInt

    def __init__(out self):
        self.value = ZInt()


# Extension that adds an alternate constructor
__extension MyType:
    def __init__(out self, initial_value: ZInt):
        self.value = ZInt()
