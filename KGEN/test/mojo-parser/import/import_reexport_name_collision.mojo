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

# RUN: %parse-mojo-isolated -I=%S %s | FileCheck %s

# The direct import path works:
from test_reexport_name_collision.foo import foo as foo_direct
from test_reexport_name_collision.bar import bar as bar_direct

# The re-export path should also work:
from test_reexport_name_collision import foo
from test_reexport_name_collision import bar

# CHECK-LABEL: lit.fn @"main
fn main():
    # Parametric function: direct import works fine.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@foo::@"foo
    var a = foo_direct[42]()

    # Parametric function: re-exported import should also resolve.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@foo::@"foo
    var b = foo[42]()

    # Non-parametric function: direct import works fine.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@bar::@"bar
    var c = bar_direct()

    # Non-parametric function: re-exported import should also resolve.
    # CHECK: lit.call {{.*}}@test_reexport_name_collision::@bar::@"bar
    var d = bar()
