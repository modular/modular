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
