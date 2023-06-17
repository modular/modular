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


@register_passable("trivial")
struct TrivialStuff[*size: Int]:
    pass


@adaptive
# CHECK: lit.func @"foobar{{.*}}"<width: @"$Int"::@Int>() ->
# CHECK-SAME: !kgen.declref<@{{.*}}::@TrivialStuff<size: variadic<@"$Int"::@Int> = [width]>> attributes {isAdaptive,
fn foobar[width: Int]() -> TrivialStuff[width]:
    pass


@adaptive
# CHECK: lit.func @"foobar{{.*}}"<w: @"$Int"::@Int>() ->
# CHECK-SAME: !kgen.declref<@{{.*}}::@TrivialStuff<size: variadic<@"$Int"::@Int> = [w]>> attributes {isAdaptive,
fn foobar[w: Int]() -> TrivialStuff[w]:
    pass


fn main[x: Int]():
    # CHECK: kgen.param.fork *"(adaptive)foo[[S0:.*]]": () -> !lit.none = <[@{{.*}}::@"foo()", @{{.*}}::@"foo()_0"]>
    # CHECK-NEXT: call_param[() -> !lit.none: *"(adaptive)foo[[S0]]"]()
    foo()
    # CHECK: kgen.param.fork *"(adaptive)foo[[S1:.*]]": () -> !lit.none =
    # CHECK-NEXT: call_param[() -> !lit.none: *"(adaptive)foo[[S1]]"]()
    foo()

    # CHECK: kgen.param.fork *"(adaptive)bar{{.*}}": () -> !lit.none = <[@{{.*}}::@"bar()", @{{.*}}::@"bar()_0"]>
    # CHECK-NEXT: call_param[() -> !lit.none: *"(adaptive)bar{{.*}}"]()
    bar()

    # CHECK: kgen.param.fork *"(adaptive)foobar{{.*}}": () -> !kgen.declref<@{{.*}}::@TrivialStuff<size: variadic<@"$Int"::@Int> = [x]>>
    # CHECK-SAME: = <[@{{.*}}::@"foobar[$Int::Int]()"<:@"$Int"::@Int x>, @{{.*}}::@"foobar[$Int::Int]()_0"<:@"$Int"::@Int x>]>
    # CHECK-NEXT: call_param[() -> !kgen.declref<@"{{.*}}"::@TrivialStuff<size: variadic<@"$Int"::@Int> = [x]>>: *"(adaptive)foobar{{.*}}]()
    _ = foobar[x]()

    return
