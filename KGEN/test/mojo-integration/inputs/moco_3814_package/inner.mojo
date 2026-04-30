# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct Inner:
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def get(self) -> Int:
        return self.value
