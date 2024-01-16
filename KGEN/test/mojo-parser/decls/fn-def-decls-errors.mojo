# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index

alias `1` = __mlir_attr.`1 : index`

struct object: pass

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

fn test_never_declared_fn():
    # expected-error @+1 {{use of unknown declaration 'never_declared_fn'}}
    never_declared_fn()

fn implicit_var_decl(a: Int):
    # expected-error @+1 {{use of unknown declaration 'c', 'fn' declarations require explicit variable declarations}}
    c = a

# expected-error @+1 {{special function '__add__' must have 2 operands}}
fn __add__():
    pass

# expected-error @+1 {{special function must be a method}}
fn __sub__(self: Int, a: Int):
    pass

fn missing_colon()  # expected-error {{expected ':' in function definition}}
    # Don't get confused by comments or blank lines!

    var x = `1`

# Missing colon after fn definition complains about function effects
# https://github.com/modularml/modular/issues/23359
# expected-error @+1 {{missing ':' at end of function signature}}
def missing_colon_2()
    test_never_declared_fn()

# expected-error @below {{expected parameter name}}
# expected-error @below {{unexpected token in expression}}
fn missing_argument_name(*: Int): pass

# expected-error @below {{expected parameter name}}
# expected-error @below {{unexpected token in expression}}
fn missing_parameter_name[: Int](): pass

# expected-error @+1 {{use of unknown declaration 'InvalidType'}}
fn test_unknown_arg_type(a: InvalidType):
    _ = a.value  # Should not produce a follow-on error.
    return
