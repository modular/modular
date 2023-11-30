# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

@adaptive
# CHECK: lit.func @"foo()"() -> !kgen.none attributes {isAdaptive
fn foo():
    return


@adaptive
# CHECK: lit.func @"foo()_0"() -> !kgen.none attributes {isAdaptive
fn foo():
    return


@adaptive
# CHECK: lit.func @"bar()"() -> !kgen.none attributes {isAdaptive
fn bar():
    return


@adaptive
# CHECK: lit.func @"bar()_0"() -> !kgen.none attributes {isAdaptive
fn bar():
    return


@register_passable("trivial")
# CHECK: lit.struct.decl @TrivialStuff<[[S:.*]][size]: variadic
struct TrivialStuff[*size: Int]:
    pass


@adaptive
# CHECK: lit.func @"foobar{{.*}}"<[[WIDTH:.*_width]][width]>() ->
# CHECK-SAME: !kgen.declref<@{{.*}}::@TrivialStuff<:variadic<index> [[[WIDTH]]]>{{.*}}> attributes {isAdaptive,
fn foobar[width: Int]() -> TrivialStuff[width]:
    pass


@adaptive
# CHECK: lit.func @"foobar{{.*}}"<[[W:.*_w]][w]>() ->
# CHECK-SAME: !kgen.declref<@{{.*}}::@TrivialStuff<:variadic<index> [[[W]]]>{{.*}}> attributes {isAdaptive,
fn foobar[w: Int]() -> TrivialStuff[w]:
    pass


# CHECK: lit.func @"main_func{{.*}}"<[[X:.*_x]][x]
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

    # CHECK: kgen.param.fork *"(adaptive)foobar{{.*}}": !lit.signature<() -> !kgen.declref<@{{.*}}::@TrivialStuff<:variadic<index> [[[X]]]>
    # CHECK-SAME: = <[@{{.*}}::@"foobar{{.*}}"<[[X]]>, @{{.*}}::@"foobar{{.*}}_0"<[[X]]>]>
    # CHECK-NEXT: call_param[!lit.signature<() -> !kgen.declref<@"{{.*}}"::@TrivialStuff<:variadic<index> [[[X]]]>{{.*}}>>: *"(adaptive)foobar{{.*}}]()
    _ = foobar[x]()
