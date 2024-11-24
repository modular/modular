# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate -verify-diagnostics


@export
fn entry_method():
    foo()  # expected-error {{call expansion failed}}


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


@export
fn test_no_params():
    no_parameters()


@no_inline
fn no_parameters():
    parametric[1]()  # expected-error {{call expansion failed}}


@no_inline
fn parametric[param: Int]():  # expected-note {{function instantiation failed}}
    constrained[
        param == 2, "param must be 2 but is " + str("something else")
    ]()  # expected-note {{call expansion failed}}


@always_inline("nodebug")
fn constrained[cond: Bool, msg: String]():
    __mlir_op.`kgen.param.assert.ex`[
        cond = cond.__mlir_i1__(),
        messageStart = msg.unsafe_ptr().address,
        messageLength = msg.byte_length().value,
    ]()  # expected-note {{constraint failed: param must be 2 but is something else}}
