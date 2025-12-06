# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -elaborate=no-use-parametric-interpreter -verify-diagnostics
# RUN: not kgen -elaborate  -D TEST_RECURSION2=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-RECURSION2
# RUN: not kgen -elaborate  -D TEST_RECURSION3=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-RECURSION3

from collections.string.string_slice import StaticString, _get_kgen_string
from sys import env_get_bool


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
    comptime msg_literal = _get_kgen_string[msg]()
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


# This creates recursive cycles: foo[D] -> bar[D] -> foo[D] and foo[D] -> baz[D] -> foo[D]
fn bar[D: Int]() -> Int:
    comptime x = foo[D]()
    return x


fn baz[D: Int]() -> Int:
    comptime x = foo[D]()
    return x


fn foo[D: Int]() -> Int:
    var x = bar[D]()
    # CHECK-RECURSION2: call expansion failed with parameter value(s): ("D": 2)
    var y = baz[D]()
    _ = x
    _ = y
    return y


fn test_recursion2():
    comptime run_test = env_get_bool["TEST_RECURSION2", False]()

    @parameter
    if run_test:
        # CHECK-RECURSION2: call expansion failed with parameter value(s): ("D": 2)
        _ = foo[2]()

        # CHECK-RECURSION2: function instantiation in parameter domain that recursively requires itself
        # CHECK-RECURSION2: recursively instantiated through here


# This creates a recursive cycle: foo1[D] -> bar1[D] -> foo1[D]
fn bar1[D: Int]() -> Int:
    comptime x = foo1[D]()
    return x


fn foo1[D: Int]() -> Int:
    # CHECK-RECURSION3: call expansion failed with parameter value(s): ("D": 1)
    var x = bar1[D]()
    return x


fn test_recursion3():
    comptime run_test = env_get_bool["TEST_RECURSION3", False]()

    @parameter
    if run_test:
        # CHECK-RECURSION3: call expansion failed with parameter value(s): ("D": 1)
        _ = foo1[1]()

        # CHECK-RECURSION3: function instantiation in parameter domain that recursively requires itself
        # CHECK-RECURSION3: recursively instantiated through here


fn main():
    test_recursion2()
    test_recursion3()
