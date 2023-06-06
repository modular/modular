# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -verify-diagnostics -I %S/../mojo-examples/ | kgen-opt -verify-parameters | FileCheck %s


@adaptive
# CHECK: lit.func @"foo()"() -> !lit.none attributes {isAdaptive
fn foo():
    let b = 3
    return


@adaptive
# CHECK: lit.func @"foo()_0"() -> !lit.none attributes {isAdaptive
fn foo():
    let b = 5
    return


@adaptive
# CHECK: lit.func @"bar()"() -> !lit.none attributes {isAdaptive
fn bar():
    let b = 7
    return


@adaptive
# CHECK: lit.func @"bar()_0"() -> !lit.none attributes {isAdaptive
fn bar():
    let b = 9
    return


fn main():
    # CHECK: kgen.param.fork *"(adaptive)foo[[S0:.*]]": () -> !lit.none = <[@"$adaptive"::@"foo()", @"$adaptive"::@"foo()_0"]>
    # CHECK-NEXT: call_param[() -> !lit.none: *"(adaptive)foo[[S0]]"]()
    foo()
    # CHECK: kgen.param.fork *"(adaptive)foo[[S1:.*]]": () -> !lit.none =
    # CHECK-NEXT: call_param[() -> !lit.none: *"(adaptive)foo[[S1]]"]()
    foo()

    # CHECK: kgen.param.fork *"(adaptive)bar{{.*}}": () -> !lit.none = <[@"$adaptive"::@"bar()", @"$adaptive"::@"bar()_0"]>
    # CHECK-NEXT: call_param[() -> !lit.none: *"(adaptive)bar{{.*}}"]()
    bar()
    return
