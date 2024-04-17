# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full -O0 %s 2 3 | FileCheck %s

from sys import argv


@value
struct MemType:
    var member: Int

    fn __add__(self, other: MemType) -> MemType:
        return MemType(self.member + other.member)


fn makes_escaping_closure(m: Int) -> fn (n: Int) escaping -> Int:
    fn myclosure(n: Int) -> Int:
        return n + m

    return myclosure


@value
@register_passable
struct Foo[a: Int]:
    var b: Int

    fn get(self) -> Int:
        return a + self.b


fn legal_type_ref[a: Int](c: Int) -> fn (x: Int, y: Int) escaping -> Int:
    fn p_capture(x: Int, y: Int) -> Int:
        return Foo[a](x + c + y).get()

    return p_capture


fn parameter_capture[a: Int, b: Int](c: Int) -> fn (x: Int) escaping -> Int:
    alias X = Foo[a](1)

    fn p_capture(x: Int) -> Int:
        return X.get() + c + b

    return p_capture


@value
struct Bar[C: Int, D: Int]:
    var x: Int

    fn get(self) -> Int:
        return self.x + C


@value
@register_passable
struct Bat[A: Int]:
    var b: Int

    fn get[B: Int](self) -> fn (y: Int) escaping -> Bar[B, A]:
        fn bar(y: Int) -> Bar[B, A]:
            var w = B + self.b + y
            return Bar[B, A](w + A)

        return bar


fn makes_escaping_closure_position_only(
    m: MemType,
) -> fn (n: MemType, /) escaping -> MemType:
    fn myclosure(n: MemType, /) -> MemType:
        return n + m

    return myclosure


fn foo[Z: Int, W: Int]() -> Int:
    return Z * W


fn test_captures_are_ordered_correctly[
    aa: Int, a: Int, b: Int, bb: Int
](c: Int) -> fn (x: Int, y: Foo[b]) escaping -> Foo[a]:
    alias Y = foo[aa, bb]()

    fn p_capture(x: Int, y: Foo[b]) -> Foo[a]:
        return Foo[a](c + Y + b)

    return p_capture


fn bar[a: Int](x: Foo[a]) -> Int:
    return x.get() * x.get() + a


fn captureCallable[
    callable: fn[A: Int] (p: Foo[A]) -> Int
](a: Int) -> fn (x: Int) escaping -> Int:
    fn foo(x: Int) -> Int:
        var w = callable[3](Foo[3](a))
        return w + a

    return foo


@value
@register_passable
struct C[B: DType]:
    var b: Int

    fn get(self) -> Int:
        return self.b


fn take_closure[
    c_type: DType
](x: C[c_type], closure: fn (z: C[c_type]) escaping -> None):
    closure(x)


fn make_closure[c_type: DType]() -> fn (z: C[c_type]) escaping -> None:
    fn foo(z: C[c_type]) -> None:
        print(z.get())

    return foo


fn deep_runtime_capture(
    m: Int, flag: Bool
) raises -> fn (n: Int) escaping -> fn (o: Int) escaping -> Int:
    if flag:
        var q = m + m

        fn myclosure2(n: Int) -> fn (o: Int) escaping -> Int:
            fn my_inner_closure(o: Int) -> Int:
                var x = o + q
                return x + n

            return my_inner_closure

        return myclosure2
    else:
        raise Error("unreachable")


fn takeClosure(formatter: fn (v: Int) escaping -> Int, value: Int):
    print(formatter(value))


fn makeEscapingClosure[
    parametricClosure: fn (v: Int) capturing -> Int
](x: Int) -> fn (v: Int) escaping -> Int:
    fn formatter(v: Int) -> Int:
        return parametricClosure(x + v)

    return formatter


fn makeEscapingClosureWithUselessCopyDecorator(
    y: String,
) -> fn (x: String) escaping -> String:
    @__copy_capture(y)
    fn ec(x: String) -> String:
        return x + y

    return ec


fn main():
    var x = 2
    var c = makes_escaping_closure(x.value)
    # CHECK: 4
    print(c(x))

    var result: MemType = makes_escaping_closure_position_only(MemType(43))(42)
    # CHECK: 85
    print(result.member)

    # CHECK: 53
    print(legal_type_ref[45](1)(3, 4))

    # CHECK: 56
    print(parameter_capture[43, 7](5)(43))

    # CHECK: 37
    var foo = Foo[7](2)
    var closure2 = test_captures_are_ordered_correctly[1, 23, 7, 2](5)
    print(closure2(3, foo).get())

    # CHECK: 30
    print(captureCallable[bar](x)(x))

    var bat = Bat[3](4)
    var bat_closure = bat.get[5]()
    var bar = bat_closure(3)
    # CHECK: 20
    print(bar.get())

    alias a = DType.int8
    # CHECK: 3
    take_closure[a](C[a](3), make_closure[a]())

    var v1 = 2
    var v2 = 3
    var v3 = 7
    try:
        # CHECK: 14
        print(deep_runtime_capture(v1, True)(v2)(v3))
    except:
        pass

    try:
        var str = argv()[1]
        var x = atol(str)
        var y = atol(argv()[2])

        @__copy_capture(x)
        @parameter
        fn formatter(v: Int) -> Int:
            return x + v

        var f = makeEscapingClosure[formatter](y)
        # CHECK: 8
        takeClosure(f, y)

        @__copy_capture(y)
        @parameter
        fn formatter2(v: Int) -> Int:
            return y + formatter(v)

        var f2 = makeEscapingClosure[formatter2](y)
        # CHECK: 11
        takeClosure(f2, y)

        # CHECK: 22
        print(makeEscapingClosureWithUselessCopyDecorator(x)(x))
    except e:
        print(e)
