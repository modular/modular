# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-error @+1 {{@export requires a string specifying the name of the exported symbol}}
@export(1)
def export_me() raises:
    ...


# expected-note @+1 {{previous export here}}
@export("my_foo")
def foo() raises:
    ...


# expected-error @+1 {{invalid re-export of my_foo}}
@export("my_foo")
def bar() raises:
    ...


# expected-warning @+2 {{ABI="C" is deprecated; use abi("C") instead}}
# expected-error @+1 {{my+foo is not a valid C identifier}}
@export("my+foo", ABI="C")
def bad_name() raises:
    ...


# expected-note @+1 {{previous export here}}
@export
def func_overloaded(x: Int) raises:
    ...


# expected-error @+1 {{invalid re-export of func_overloaded}}
@export
def func_overloaded(x: Bool) raises:
    ...
