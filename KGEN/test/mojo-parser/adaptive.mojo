# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: kgen-translate -import-mojo %s | kgen-opt -verify-parameters | FileCheck %s


@adaptive
# CHECK: lit.func @"foo()"() -> !kgen.none attributes {isAdaptive
fn foo():
    let b = 3
    return


@adaptive
# CHECK: lit.func @"foo()_0"() -> !kgen.none attributes {isAdaptive
fn foo():
    let b = 5
    return


@adaptive
# CHECK: lit.func @"bar()"() -> !kgen.none attributes {isAdaptive
fn bar():
    let b = 7
    return


@adaptive
# CHECK: lit.func @"bar()_0"() -> !kgen.none attributes {isAdaptive
fn bar():
    let b = 9
    return


@register_passable("trivial")
# CHECK: lit.struct.decl @TrivialStuff<[[S:.*]][size]: variadic
struct TrivialStuff[*size: Int]:
    pass


@adaptive
# CHECK: lit.func @"foobar{{.*}}"<[[WIDTH:.*_width]][width]: !Int>() ->
# CHECK-SAME: !kgen.declref<@{{.*}}::@TrivialStuff<:variadic<!Int> [[[WIDTH]]]>{{.*}}> attributes {isAdaptive,
fn foobar[width: Int]() -> TrivialStuff[width]:
    pass


@adaptive
# CHECK: lit.func @"foobar{{.*}}"<[[W:.*_w]][w]: !Int>() ->
# CHECK-SAME: !kgen.declref<@{{.*}}::@TrivialStuff<:variadic<!Int> [[[W]]]>{{.*}}> attributes {isAdaptive,
fn foobar[w: Int]() -> TrivialStuff[w]:
    pass


# CHECK: lit.func @"main_func[{{.*}}$int::Int]()"<[[X:.*_x]][x]: !Int
fn main_func[x: Int]():
    # CHECK: kgen.param.fork *"(adaptive)foo[[S0:.*]]": !lit.signature<() -> !kgen.none> = <[@{{.*}}::@"foo()", @{{.*}}::@"foo()_0"]>
    # CHECK-NEXT: call_param[!lit.signature<() -> !kgen.none>: *"(adaptive)foo[[S0]]"]()
    foo()
    # CHECK: kgen.param.fork *"(adaptive)foo[[S1:.*]]": !lit.signature<() -> !kgen.none> =
    # CHECK-NEXT: call_param[!lit.signature<() -> !kgen.none>: *"(adaptive)foo[[S1]]"]()
    foo()

    # CHECK: kgen.param.fork *"(adaptive)bar{{.*}}": !lit.signature<() -> !kgen.none> = <[@{{.*}}::@"bar()", @{{.*}}::@"bar()_0"]>
    # CHECK-NEXT: call_param[!lit.signature<() -> !kgen.none>: *"(adaptive)bar{{.*}}"]()
    bar()

    # CHECK: kgen.param.fork *"(adaptive)foobar{{.*}}": !lit.signature<() -> !kgen.declref<@{{.*}}::@TrivialStuff<:variadic<!Int> [[[X]]]>
    # CHECK-SAME: = <[@{{.*}}::@"foobar[{{.*}}$int::Int]()"<:!Int [[X]]>, @{{.*}}::@"foobar[{{.*}}$int::Int]()_0"<:!Int [[X]]>]>
    # CHECK-NEXT: call_param[!lit.signature<() -> !kgen.declref<@"{{.*}}"::@TrivialStuff<:variadic<!Int> [[[X]]]>{{.*}}>>: *"(adaptive)foobar{{.*}}]()
    _ = foobar[x]()

    return
