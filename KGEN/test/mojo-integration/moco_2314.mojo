# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo -debug-level full %s -verify-diagnostics


# expected-note  {{no instantiation for trait {{.*}}Foo, get witness table failed}}
trait Foo:
    comptime x: Bool


fn main():
    var b = Foo.x
    print(b)
