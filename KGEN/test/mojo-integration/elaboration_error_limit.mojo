# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate -elaboration-error-limit=2 --num-threads=1 -verify-diagnostics


# expected-error @+2{{function instantiation failed}}
@export
fn entry_method0():
    foo()  # expected-note {{call expansion failed}}


# expected-error @+3{{function instantiation failed}}
# expected-note @+2{{too many errors emitted, stopping now}}
@export
fn entry_method1():
    bar()  # expected-note {{call expansion failed}}


@export
fn entry_method2():
    baz()


# expected-note @+2{{function instantiation failed}}
@no_inline
fn foo():
    __mlir_op.`kgen.param.assert`[
        cond = __mlir_attr.`false`, message = "oops".value
    ]()  # expected-note {{constraint failed}}


# expected-note @+2{{function instantiation failed}}
@no_inline
fn bar():
    __mlir_op.`kgen.param.assert`[
        cond = __mlir_attr.`false`, message = "oops".value
    ]()  # expected-note {{constraint failed}}


@no_inline
fn baz():
    __mlir_op.`kgen.param.assert`[
        cond = __mlir_attr.`false`, message = "oops".value
    ]()
