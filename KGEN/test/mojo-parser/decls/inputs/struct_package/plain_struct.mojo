# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct PlainStruct(Movable where False):
    var location: Int

    def __init__(out self):
        self.location = 0

    def set_location(mut self, new_location: Int):
        self.location = new_location
