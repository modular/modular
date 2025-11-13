# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate=no-use-parametric-interpreter -verify-diagnostics


from collections.string.string_slice import StaticString, _get_kgen_string


# expected-error @+2{{function instantiation failed}}
@export
fn entry_method():
    foo()  # expected-note {{call expansion failed}}


@always_inline("nodebug")
fn foo():
    bar()  # expected-note {{call expansion failed}}


@always_inline("nodebug")
fn bar():
    baz()  # expected-note {{call expansion failed}}


@always_inline("nodebug")
fn baz():
    __mlir_op.`kgen.param.assert`[
        cond = __mlir_attr.`false`, message = "oops".value
    ]()  # expected-note {{constraint failed}}


# expected-error @+2{{function instantiation failed}}
@export
fn test_no_params():
    no_parameters()  # expected-note {{call expansion failed}}


# expected-note @+2{{function instantiation failed}}
@no_inline
fn no_parameters():
    parametric[1]()  # expected-note {{call expansion failed}}


@no_inline
fn parametric[param: Int]():  # expected-note {{function instantiation failed}}
    constrained[
        param == 2, "param must be 2"
    ]()  # expected-note {{call expansion failed}}


# This is copied so the note ends up in this file.
@always_inline("nodebug")
fn constrained[cond: Bool, msg: StaticString]():
    alias msg_literal = _get_kgen_string[msg]()
    __mlir_op.`kgen.param.assert`[
        cond = cond.__mlir_i1__(), message=msg_literal
    ]()  # expected-note {{constraint failed: param must be 2}}


# expected-error @+2{{function instantiation failed}}
@export
fn test_comptime_assert():
    parametric_assert[1]()  # expected-note {{call expansion failed}}


# expected-note @+2{{function instantiation failed}}
@no_inline
fn parametric_assert[param: Int]():
    # expected-note @below {{constraint failed: param must be 2}}
    __comptime_assert param == 2, "param must be 2"
