# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-warning @+1 {{@export requires an explicit 'abi()' effect on the function}}
@export("noabi")
def no_abi() raises:
    ...


# expected-error @+1 {{@export requires a string specifying the name of the exported symbol}}
@export(1)
def export_me() abi("C") raises:
    ...


# expected-note @+1 {{previous export here}}
@export("my_foo")
def foo() abi("C") raises:
    ...


# expected-error @+1 {{invalid re-export of my_foo}}
@export("my_foo")
def bar() abi("C") raises:
    ...


# expected-warning @+2 {{ABI="C" is deprecated; use abi("C") instead}}
# expected-error @+1 {{my+foo is not a valid C identifier}}
@export("my+foo", ABI="C")
def bad_name() raises:
    ...

# expected-note @+1 {{previous export here}}
@export
def func_overloaded(x: Int) abi("C") raises:
    ...


# expected-error @+1 {{invalid re-export of func_overloaded}}
@export
def func_overloaded(x: Bool) abi("C") raises:
    ...
