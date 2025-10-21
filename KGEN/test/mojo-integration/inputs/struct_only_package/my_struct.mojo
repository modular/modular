# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct MyStruct:
    var speed: Int

    fn __init__(out self):
        self.speed = 0

    fn accelerate(mut self):
        self.speed = 10
