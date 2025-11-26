# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -elaborate %s --verify-diagnostics

# This creates a recursive cycle: foo[D] -> bar[D] -> foo[D]


# expected-note @below {{function instantiation failed}}
# expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
# expected-note @below {{back to parameter domain function call here}}
fn bar[D: Int]() -> Int:
    alias x = foo[D]()
    return x


# expected-note @below {{function instantiation failed}}
fn foo[D: Int]() -> Int:
    # expected-note @below {{recursively instantiated through here}}
    # expected-note @below {{call expansion failed with parameter value(s): ("D": 1)}}
    var x = bar[D]()
    return x


# expected-error @below {{function instantiation failed}}
fn main():
    # expected-note @below {{call expansion failed with parameter value(s): ("D": 1)}}
    _ = foo[1]()
