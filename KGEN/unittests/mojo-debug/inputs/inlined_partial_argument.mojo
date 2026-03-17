# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
def modify(mut x: Int):
    x = 42


def use_ints(x: Int, y: Int):
    pass


@fieldwise_init
struct MyPair(TrivialRegisterPassable):
    var x: Int
    var y: Int


def main():
    var p = MyPair(3, 4)
    modify(p.x)
    use_ints(p.x, p.y)  # breakpoint
