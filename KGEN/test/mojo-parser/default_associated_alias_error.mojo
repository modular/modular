# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --verify-diagnostics %s

# Basic tests for when multiple parent traits define a associated alias with
# conflicting signature and no user-provided value in the child struct.


trait B:
    # expected-note@below{{conflicting implementation from trait B here}}
    comptime a: Int = 1


trait A:
    # expected-note@below{{original default implementation from trait A here}}
    comptime a: Int = 2


# expected-error@below{{trait member 'a' has conflicting default implementations in B and A; you must implement it manually}}
struct Foo(A, B):
    pass
