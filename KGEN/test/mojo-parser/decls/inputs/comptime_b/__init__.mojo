# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct _Impl(Movable where False):
    pass


# @expected-note @below {{'Foo' also declared here}}
comptime Foo = _Impl
