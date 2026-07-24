# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

struct Foo(Movable where False):
    def __init__(out self):
        pass


__extension Foo:
    def y_inst(self) -> Int:
        return 2

    @staticmethod
    def y_static() -> Int:
        return 20
