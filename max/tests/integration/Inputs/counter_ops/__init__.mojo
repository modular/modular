# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from register import mogg_register


struct Counter(Movable):
    var a: Int
    var b: Int

    fn __init__(inout self):
        self.a = 0
        self.b = 0

    fn __moveinit__(inout self, owned other: Self):
        self.a = other.a
        self.b = other.b

    fn bump(inout self):
        self.a += 1
        self.b += self.a
        print("bumped", self.a, self.b)


@mogg_register("make_counter")
fn make_counter() -> Counter:
    print("making")
    return Counter()


@mogg_register("bump_counter")
fn bump_counter(inout c: Counter):
    print("bumping")
    c.bump()


@mogg_register("drop_counter")
fn drop_counter(owned c: Counter):
    print("dropping")
    _ = c^
