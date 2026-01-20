# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics

# Test that direct access of trait members produces a parser error.

trait Foo:
    comptime x: Bool


fn main():
    # expected-error @+1 {{Direct access of trait members is not supported}}
    var b = Foo.x
    _ = b
