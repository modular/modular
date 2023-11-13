# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo -split-input-file | FileCheck %s


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[A:.*]]: !Int, |>
# CHECK: lit.struct.decl @"_CW_{{.*}}" attributes
# CHECK: lit.func @"__init__{{.*}}"<[[a:.*]][a]: !Int, |>(%self[self]: !kgen.pointer<!wrapper> init_self, %impl[impl]: !kgen.pointer<@"{{.*}}::@"`_CI_{{.*}}"<:!Int [[a]]>{{.*}}> owned_in_mem, |) -> !kgen.none {{.*}}specialFnKind = 2 : i8
# CHECK: lit.func @"{{.*}}_copyinit_`_CI_{{.*}}"<[[a:.*]][a]: !Int, |>(%arg[ptrToImpl]: !kgen.pointer<pointer<none>> borrow, %other[other]: !kgen.pointer<none> borrow, |) -> !kgen.none {{.*}}specialFnKind = 0 : i8
# CHECK: lit.func @"{{.*}}_dtor_`_CI_{{.*}}"<[[a:.*]][a]: !Int, |>(%self[self]: !kgen.pointer<none>, |) -> !kgen.none {{.*}}specialFnKind = 0 : i8
# CHECK: lit.func @"{{.*}}_call_`_CI_{{.*}}"<[[a:.*]][a]: !Int, |>(%0[*""]: !kgen.pointer<none> borrow, |, %x[x]: !Int borrow) -> !Int {{.*}}specialFnKind = 0 : i8}
fn parameter_capture[a: Int](c: Int) -> fn (x: Int) escaping -> Int:
    fn p_capture(x: Int) escaping -> Int:
        return c + a + x

    return p_capture


# // -----


@value
@register_passable
struct Foo[a: Int]:
    var b: Int


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[a:.*a]]: !Int, [[X:.*X]]: [[FOO:.*]]<:!Int [[a]]> : metatype<[[FOO]]<:!Int [[a]]>>, |>
# CHECK: lit.func @"__call__{{.*}}"({{.*}}<:!Int [[a]], :[[FOO]]<:!Int [[a]]>
# CHECK-NEXT: [[VAR1:%.*]] = lit.struct.gep %0[field0]
# CHECK-NEXT: [[VAR2:%.*]] = pop.load [[VAR1]] : !kgen.pointer<!Int>
# CHECK-NEXT: kgen.param.constant: !Int = <#lit.struct.extract<:[[FOO]]<:!Int [[a]]> {{.*}} [[X]], "b">>
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


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[X:.*]]: !lit.signature<() -> !Int>, [[Y:[0-9a-zA-Z_]+]]: {{.*}}Foo<:!Int apply(:!lit.signature<() -> !Int> [[X]])>
# CHECK: lit.func @"__call__{{.*}}"(
# CHECK: constant: !Int = <{{.*}} [[X]])> {{.*}} [[Y]], "b">
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


# COM: Closure Impl has correct input parameters.
# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[b:.*]]: !Int, [[a:.*]]: !Int, [[Y:.*]]: !Int, |>


# COM: Closure Wrapper has correct input parameters and initializer parameters
# CHECK: lit.struct.decl @"_CW_
# CHECK-SAME: <p0: !Int, p1: !Int, |>
# CHECK: lit.func @"__init__{{.*}}"<[[Y:.*]][Y]: !Int, |>
# CHECK-SAME: (%self[self]: !kgen.pointer<@"${{.*}}"::@"_CW_{{.*}}"<:!Int p0, :!Int p1>
# CHECK-SAME: %impl[impl]: !kgen.pointer<@"${{.*}}"::@"`_CI_{{.*}}"<:!Int p0, :!Int p1, :!Int [[Y]]>
fn test_captures_are_ordered_correctly[
    aa: Int, a: Int, b: Int, bb: Int
](c: Int) -> fn (x: Int, y: Foo[b]) escaping -> Foo[a]:
    alias Y = foo[aa, bb]()

    fn p_capture(x: Int, y: Foo[b]) escaping -> Foo[a]:
        return Foo[a](c + Y + b)

    return p_capture


# // -----

# COM: Check that the parameter is properly added to the ClosureImpl despite being defined two levels up.

# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[A:.*]]: !Int, |>

# COM: Check that the closure impl parameter is bound to the struct parameter:
# CHECK: lit.call @"${{.*}}"::@"`_CI_{{.*}}"::@"__init__{{.*}}"<:!Int [[ALoc:.*]]_A>(%0, %self) : !lit.signature<("self": !kgen.pointer<@"${{.*}}"::@"`_CI_{{.*}}"<:!Int [[ALoc]]_A>


@value
@register_passable
struct Foo[A: Int]:
    var b: Int

    fn get[C: Int](self) -> fn (y: Int) escaping -> Int:
        fn bar(y: Int) escaping -> Int:
            let w = A + self.b + y
            return w

        return bar


# // -----

# COM: Check that the parameter is properly added to the ClosureWrapper and ClosureImpl despite being defined two levels up.

# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[B:.*]]: !Int, [[A:.*]]: !Int, |>

# CHECK: lit.struct.decl @"_CW_
# CHECK-SAME: <p0: !Int, p1: !Int, |>

# COM: Check that the closure impl parameter is bound to the struct parameter:
# CHECK: lit.call @"${{.*}}"::@"`_CI_{{.*}}"::@"__init__{{.*}}"<:!Int [[BLoc:.*]]_B, :!Int [[ALoc:.*]]_A>(%0, %self) : !lit.signature<("self": !kgen.pointer<@"${{.*}}"::@"`_CI_{{.*}}"<:!Int [[BLoc]]_B, :!Int [[ALoc]]_A>

# COM: Check that the closure wrapper parameter is bound to the struct parameter:
# CHECK: lit.call @"${{.*}}"::@"_CW_{{.*}}"::@"__init__{{.*}}"<:!Int [[BLoc:.*]]_B, :!Int [[ALoc:.*]]_A>(%{{.*}}, %0) : !lit.signature<("self": !kgen.pointer<@"${{.*}}"::@"_CW_{{.*}}"<:!Int [[BLoc]]_B, :!Int [[ALoc]]_A>


@value
struct Foo[C: Int, D: Int]:
    var x: Int

    fn get(self) -> Int:
        return self.x + C


@value
@register_passable
struct Bat[A: Int]:
    var b: Int

    fn get[B: Int](self) -> fn (y: Int) escaping -> Foo[B, A]:
        fn bar(y: Int) escaping -> Foo[B, A]:
            let w = B + self.b + y
            return Foo[B, A](w + A)

        return bar


# // -----

# COM: Capture inside a nested escaping closure.


@value
struct MemType:
    var x: Int

    fn __add__(self, rhs: MemType) -> MemType:
        return MemType(rhs.x + self.x)

    fn __add__(self, rhs: Int) -> MemType:
        return MemType(self.x + rhs)


# COM: Check that the parameter capture "A" is forwarded to the outer escaping closure
# CHECK: lit.struct.decl @"`_CI_{{.*}}escaping1"<[[A:.*]]: !Int, |>


# COM: Check that the parameter capture "A" is forwarded to the outer escaping closure
# CHECK: lit.struct.decl @"`_CI_{{.*}}escaping0"<[[A]]: !Int, |>
fn makes_escaping_closure[
    A: Int
](m: MemType) -> fn (n: MemType) escaping -> MemType:
    fn myclosure(n: MemType) escaping -> MemType:
        fn nested_nested(k: MemType, l: MemType) escaping -> MemType:
            return n + k + A

        return nested_nested(n, m)

    return myclosure


# // -----


@value
@register_passable
struct Foo[A: Int, B: DType]:
    fn get(self) -> Int:
        return A


fn use(a: Int):
    pass


# COM: Ensure the captured parameter is added to the Closure Impl
# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[C_TYPE:.*]]: !DType, |>

# COM: Ensure the captured parameter is added to the Closure Wrapper
# CHECK: lit.struct.decl @"_CW_{{.*}}"<p0: !DType, |>


fn make_closure[
    c_type: DType
](w: Int) -> fn (z: Foo[2, c_type]) escaping -> None:
    fn foo(z: Foo[2, c_type]) escaping -> None:
        use(z.get())

    return foo
