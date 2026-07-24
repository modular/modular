# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# @expected-note @below {{'Foo' also declared here}}
struct Foo(Movable where False):
    pass
