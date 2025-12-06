# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -elaborate %s --verify-diagnostics


# This creates recursive cycles: foo[D] -> bar[D] -> foo[D] and foo[D] -> baz[D] -> foo[D]
# expected-note @below {{recursively instantiated through here}}
fn bar[D: Int]() -> Int:
    comptime x = foo[D]()
    return x


# expected-note @below {{function instantiation failed}}
# expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
fn baz[D: Int]() -> Int:
    comptime x = foo[D]()
    return x


# expected-note @below {{function instantiation failed}}
fn foo[D: Int]() -> Int:
    # expected-note @below {{recursively instantiated through here}}
    var x = bar[D]()
    # expected-note @below {{call expansion failed with parameter value(s): ("D": 2)}}
    var y = baz[D]()
    _ = x
    _ = y
    return y


# expected-error @below {{function instantiation failed}}
fn main():
    # expected-note @below {{call expansion failed with parameter value(s): ("D": 2)}}
    _ = foo[2]()
