# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

struct Foo(Movable where False):
    def __init__(out self):
        pass


__extension Foo:
    def x_inst(self) -> Int:
        return 1

    @staticmethod
    def x_static() -> Int:
        return 10
