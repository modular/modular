# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo -split-input-file | FileCheck %s


# CHECK: lit.struct.decl @"_CI_{{.*}}"<p0[p0]: !Int, |>
# CHECK: lit.func @"__init__{{.*}}_CW_{{.*}}"<[[p0:.*]][[[p0]]]: !Int, |>(%self[self]: !kgen.pointer<!escaping> init_self, %impl[impl]: !kgen.pointer<@"{{.*}}::@"_CI_{{.*}}"<:!Int [[p0]]>{{.*}}> owned_in_mem, |) -> !kgen.none {{.*}}specialFnKind = 2 : i8
# CHECK: lit.func @"{{.*}}_copyinit__CI_{{.*}}"<[[copyp0:.*]][[[copyp0]]]: !Int, |>(%arg[ptrToImpl]: !kgen.pointer<pointer<none>> borrow, %other[other]: !kgen.pointer<none> borrow_in_mem, |) -> !kgen.none {{.*}}specialFnKind = 0 : i8
# CHECK: lit.func @"{{.*}}_dtor__CI_{{.*}}"<[[delp0:.*]][[[delp0]]]: !Int, |>(%self[self]: !kgen.pointer<none>, |) -> !kgen.none {{.*}}specialFnKind = 0 : i8
# CHECK: lit.func @"{{.*}}_call__CI_{{.*}}"<[[callp0:.*]][[[callp0]]]: !Int, |>(%0[*""]: !kgen.pointer<none> borrow_in_mem, |, %x[x]: !Int borrow) -> !Int {{.*}}specialFnKind = 0 : i8}
fn parameter_capture[a: Int](c: Int) -> fn (x: Int) escaping -> Int:
    fn p_capture(x: Int) escaping -> Int:
        return c + a + x

    return p_capture


# // -----


@value
@register_passable
struct Foo[a: Int]:
    var b: Int


# CHECK:  lit.func @"__call__({{.*}}_CI_${{.*}}"
# CHECK-NEXT: [[VAR1:%.*]] = lit.struct.gep %0[field0]
# CHECK-NEXT: [[VAR2:%.*]] = pop.load [[VAR1]] : !kgen.pointer<!Int>
# CHECK-NEXT: lit.alias.decl *"[[XREF:.*X]]":
# CHECK-SAME: <apply(:!lit.signature<("b": !Int borrow) ownedresult -> {{.*}}Foo<:!Int p0>{{.*}}>{{.*}}> {{.*}}Foo::@"__init__{{.*}}"<:!Int p0>, #lit.struct<{value = 1}>)>
# CHECK-NEXT: kgen.param.constant: !Int = <#lit.struct.extract<:@"${{.*}}"::@Foo<:!Int p0> {{.*}} *"[[XREF]]", "b">>
fn parameter_capture[a: Int](c: Int) -> fn (x: Int) escaping -> Int:
    alias X = Foo[a](1)

    fn p_capture(x: Int) escaping -> Int:
        return X.b + c

    return p_capture


# // -----


@value
@register_passable
struct Foo[a: Int]:
    var b: Int

    fn get(self) -> Int:
        return a + self.b


fn bar[a: Int, b: Int]() -> Int:
    return b * a


# CHECK: lit.struct.decl @"_CI_{{.*}}"<p0[p0]: !Int, |>
# CHECK: lit.func @"__call__(${{.*}}::_CI_
# CHECK-NEXT: lit.struct.gep
# CHECK-NEXT: pop.load
# CHECK-NEXT: lit.alias.decl *"[[#LINE:]]_[[#OLINE:]]x[[#COL:]]_X"
# CHECK-NEXT: lit.alias.decl *"[[#LINE]]_[[#OLINE:]]x[[#COL:]]_Y"
fn parameter_capture_multiple_levels[
    a: Int
](c: Int) -> fn (x: Int) escaping -> Int:
    alias X = bar[a, a]
    alias Y = Foo[X()](2)

    fn p_capture(x: Int) escaping -> Int:
        return Y.b + c

    return p_capture


# // -----

# COM: Signature Capture


@value
@register_passable
struct Foo[a: Int]:
    var b: Int

    fn get(self) -> Int:
        return a + self.b


fn foo[Z: Int, W: Int]() -> Int:
    return Z * W


# COM: Closure Impl has correct input parameters and copied aliases
# CHECK: lit.struct.decl @"_CI_
# CHECK-SAME: <p0[p0]: !Int, p1[p1]: !Int, p2[p2]: !Int, p3[p3]: !Int, |>
# CHECK: lit.alias.decl *"[[#LINE:]]_[[#OLINE:]]x[[#OCOL:]]_Y": !Int = <apply(:!lit.signature<() -> !Int> @"${{.*}}"::@"foo[{{.*}}]()"<:!Int p0, :!Int p3>)>


# COM: Closure Wrapper has correct input parameters and initializer parameters
# CHECK: lit.struct.decl @"_CW_
# CHECK-SAME: <p0[p0]: !Int, p1[p1]: !Int, |>
# CHECK: lit.func @"__init__{{.*}}"<[[initP0:.*]][[[initP0]]]: !Int, [[initP3:.*]][[[initP3]]]: !Int, |>
# CHECK-SAME: (%self[self]: !kgen.pointer<@"${{.*}}"::@"_CW_{{.*}}"<:!Int p0, :!Int p1>
# CHECK-SAME: %impl[impl]: !kgen.pointer<@"${{.*}}"::@"_CI_{{.*}}"<:!Int [[initP0]], :!Int p0, :!Int p1, :!Int [[initP3]]>
fn test_captures_are_ordered_correctly[
    aa: Int, a: Int, b: Int, bb: Int
](c: Int) -> fn (x: Int, y: Foo[b]) escaping -> Foo[a]:
    alias Y = foo[aa, bb]()

    fn p_capture(x: Int, y: Foo[b]) escaping -> Foo[a]:
        return Foo[a](c + Y + b)

    return p_capture
