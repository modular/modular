# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate -verify-diagnostics


@export
fn entry_method():  # expected-error {{function instantiation failed}}
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
