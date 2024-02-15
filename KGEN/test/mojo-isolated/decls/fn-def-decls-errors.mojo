# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s -verify-diagnostics


fn test_never_declared_fn():
    # expected-error @+1 {{use of unknown declaration 'never_declared_fn'}}
    never_declared_fn()

fn implicit_var_decl(a: int):
    # expected-error @+1 {{use of unknown declaration 'c', 'fn' declarations require explicit variable declarations}}
    c = a

# expected-error @+1 {{'__add__' requires 2 operands}}
fn __add__():
    pass

# expected-error @+1 {{'__sub__' must be a method}}
fn __sub__(self: int, a: int):
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
fn missing_argument_name(*: int): pass

# expected-error @below {{expected parameter name}}
# expected-error @below {{unexpected token in expression}}
fn missing_parameter_name[: int](): pass

# expected-error @+1 {{use of unknown declaration 'InvalidType'}}
fn test_unknown_arg_type(a: InvalidType):
    _ = a.value  # Should not produce a follow-on error.
    return

# expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_stars(a: int, *, *, b: int):
    pass

# expected-error @+1 {{cannot have two '/' markers in the same argument list}}
fn two_slashes(a: int, /, /, b: int):
    pass

# expected-error @+1 {{cannot specify '/' marker after '*' marker}}
fn slash_after_start(a: int, *, /, b: int):
    pass

# expected-error @+1 {{'/' marker cannot be used at the start of the argument list}}
fn leading_slash(/, a: int):
    pass

# expected-error @+1 {{'*' marker is not allowed at end of argument list}}
fn trailing_star(a: int, *):
    pass

# TODO(#21950): fix how we model variadics to suppress this error
# expected-error @+3 {{keyword-only arguments after variadics not supported yet}}
# expected-error @+2 {{unexpected token in expression}}
# # expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_variadics(*a: int, *b: int):
    pass

# TODO(#21950): fix how we model variadics to suppress this error
# expected-error @+3 {{keyword-only arguments after variadics not supported yet}}
# expected-error @+2 {{unexpected token in expression}}
# expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_variadic_packs[*Ts: AnyRegType](*a: *Ts, *b: *Ts):
    pass

# TODO(#21950): fix how we model variadics to allow this
# expected-error @+2 {{unexpected token in expression}}
# expected-error @+1 {{keyword-only arguments after variadics not supported yet}}
fn variadic_and_kw_only(a: int, *b: int, c: int):
    pass

# expected-error @+1 {{parametric functions may not be used as arguments; consider passing as a parameter instead}}
fn foo(x: fn[a: int] () -> None):
    pass
