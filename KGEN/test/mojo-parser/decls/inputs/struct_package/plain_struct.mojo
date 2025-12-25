# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

struct PlainStruct:
    var location: Int

    fn __init__(out self):
        self.location = 0

    fn set_location(mut self, new_location: Int):
        self.location = new_location
