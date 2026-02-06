# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate -elaboration-error-limit=2 --num-threads=1 -verify-diagnostics
# RUN: not kgen %s -elaborate=use-parametric-interpreter -elaboration-error-limit=2 --num-threads=1 2>&1 | FileCheck %s --check-prefix=CHECK-PARAM

# COM: -elaborate=use-parametric-interpreter slight difference from -elaborate for error messages.
#      Using FileCheck instead to check those with CHECK-PRAMA prefix.
#      (TODO) error message for parametric interpreter is non-deterministic with limit=2, i.e.
#      either combination of 2 out of foo, bar and baz can show up, hence not checking with a label


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


# CHECK-PARAM: function instantiation failed
# CHECK-PARAM: constraint failed
# expected-note @+2 {{function instantiation failed}}
@no_inline
fn bar():
    __mlir_op.`kgen.param.assert`[
        cond = __mlir_attr.`false`, message = "oops".value
    ]()  # expected-note {{constraint failed}}


# CHECK-PARAM: function instantiation failed
# CHECK-PARAM: constraint failed
@no_inline
fn baz():
    __mlir_op.`kgen.param.assert`[
        cond = __mlir_attr.`false`, message = "oops".value
    ]()


# CHECK-PARAM: too many errors emitted, stopping now
