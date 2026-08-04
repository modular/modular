# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from other.pkg.m import Nameable, Thing


# A parameter constrained by an imported trait: the constraint reaches the
# artifact as rendered text inside a debug-info source name, not as a symbol
# reference.
struct Wrap[T: Nameable]:
    def get(self) -> Int:
        return 0


def make() -> Int:
    var thing = Thing(42)
    return thing.get()
