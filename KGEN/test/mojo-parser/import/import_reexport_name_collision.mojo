# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that a function re-exported from a package's __init__.mojo is preferred
# over the submodule of the same name during name resolution.
#
# When a package re-exports a function whose name collides with a submodule
# (e.g. `test_reexport_name_collision/foo.mojo` defines `fn foo`, and
# `__init__.mojo` does `from .foo import foo`), importing via the package
# should resolve to the function, not the module.
#
# This is a known bug: submodule names from the directory scan shadow
# re-exported symbols from __init__.mojo, so the module wins and the
# function becomes inaccessible through the re-export.

# RUN: %parse-mojo-isolated -verify-diagnostics -I=%S %s

# The direct import path works:
from test_reexport_name_collision.foo import foo as foo_direct

# The re-export path should also work, but currently the compiler resolves
# `foo` to the module instead of the function:
from test_reexport_name_collision import foo

fn main():
    # Direct import works fine.
    var a = foo_direct[42]()

    # Re-exported import hits the bug: the compiler thinks `foo` is the
    # module, not the parametric function.
    # expected-error @+1 {{'foo' is not subscriptable}}
    var b = foo[42]()
