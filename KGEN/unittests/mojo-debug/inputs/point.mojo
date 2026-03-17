# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct Point(TrivialRegisterPassable):
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x  # breakpoint
        self.y = y
        return


def main():
    var p1 = Point(1, -1)
    var p2 = Point(2, -2)
    print(p1.x, p2.y)
