# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


@value
struct MemType:
    var member: Int

    fn __add__(self, other: MemType) -> MemType:
        return MemType(self.member + other.member)


fn makes_escaping_closure(
    m: __mlir_type.index,
) -> fn (n: __mlir_type.index) escaping -> __mlir_type.index:
    fn myclosure(n: __mlir_type.index) escaping -> __mlir_type.index:
        return __mlir_op.`index.add`(n, m)

    return myclosure


@value
@register_passable
struct Foo[a: Int]:
    var b: Int

    fn get(self) -> Int:
        return a + self.b


fn legal_type_ref[a: Int](c: Int) -> fn (x: Int, y: Int) escaping -> Int:
    fn p_capture(x: Int, y: Int) escaping -> Int:
        return Foo[a](x + c + y).get()

    return p_capture


fn parameter_capture[a: Int, b: Int](c: Int) -> fn (x: Int) escaping -> Int:
    alias X = Foo[a](1)

    fn p_capture(x: Int) escaping -> Int:
        return X.get() + c + b

    return p_capture


fn makes_escaping_closure_position_only(
    m: MemType,
) -> fn (n: MemType, /) escaping -> MemType:
    fn myclosure(n: MemType, /) escaping -> MemType:
        return n + m

    return myclosure


fn foo[Z: Int, W: Int]() -> Int:
    return Z * W


fn test_captures_are_ordered_correctly[
    aa: Int, a: Int, b: Int, bb: Int
](c: Int) -> fn (x: Int, y: Foo[b]) escaping -> Foo[a]:
    alias Y = foo[aa, bb]()

    fn p_capture(x: Int, y: Foo[b]) escaping -> Foo[a]:
        return Foo[a](c + Y + b)

    return p_capture


fn bar[a: Int](x: Foo[a]) -> Int:
    return x.get() * x.get() + a


fn captureCallable[
    callable: fn[A: Int] (p: Foo[A]) -> Int
](a: Int) -> fn (x: Int) escaping -> Int:
    fn foo(x: Int) escaping -> Int:
        let w = callable[3](Foo[3](a))
        return w + a

    return foo


fn main():
    let x = 2
    let c = makes_escaping_closure(x.value)
    # CHECK: 4
    print(c(x.value))

    let result: MemType = makes_escaping_closure_position_only(MemType(43))(42)
    # CHECK: 85
    print(result.member)

    # CHECK: 53
    print(legal_type_ref[45](1)(3, 4))

    # CHECK: 56
    print(parameter_capture[43, 7](5)(43))

    # CHECK: 37
    let foo = Foo[7](2)
    let closure2 = test_captures_are_ordered_correctly[1, 23, 7, 2](5)
    print(closure2(3, foo).get())

    # CHECK: 30
    print(captureCallable[bar](x)(x))
