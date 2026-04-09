# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct _Impl:
    pass


# @expected-note @below {{'Foo' also declared here}}
comptime Foo = _Impl
