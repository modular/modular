# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn getFloat() -> Float32:
    return 4.125


@value
@register_passable("trivial")
struct ARegisterPassableStruct:
    var int: Int
    var f32: Float32
    var another_int: Int
    var float16: Float16
    var uint8: UInt8
    var simd: SIMD[DType.float16, 4]
    var none: None
    var uint16: UInt16
    var int32: Int32

    fn __init__() -> Self:
        return Self {
            int: -101,
            f32: 24.125,
            another_int: 101,
            float16: 25.125,
            uint8: 123,
            simd: SIMD[DType.float16, 4](-0.125, -1.5, -1, 5.725),
            none: None,
            uint16: 123,
            int32: 485,
        }


struct AStruct:
    var int: UInt8
    var tuple: Tuple[Int, Int8, Float32]

    fn __init__(inout self):
        self.int = 12
        self.tuple = Tuple[Int, Int8, Float32](1, 87, 123.125)


struct ParamStruct[T: AnyRegType]:
    var t: T

    fn __init__(inout self, t: T):
        self.t = t


fn main():
    var a_var_index = __mlir_op.`index.constant`[
        value = __mlir_attr.`48:index`
    ]()

    let a_let_index = __mlir_op.`index.constant`[
        value = __mlir_attr.`48:index`
    ]()

    let a_register_passable_struct = ARegisterPassableStruct()

    let a_struct = AStruct()

    let p_struct_int = ParamStruct[Int](8)

    let p_struct_stringref = ParamStruct[StringRef]("hello")

    let an_int: Int = 123

    let a_literal_float = 3.125

    let a_float = Float32(3.125)

    let another_float = getFloat()

    let `^ uncommon name` = 1123123

    let a_string_literal = "fofofo"

    let a_list = [1, 2.125, 3]

    let a_simd = SIMD[DType.float16, 4](1.125, 2.5, 0, -3.725)

    # fmt: off
    let b_simd = SIMD[DType.int64, 32](
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
    )
    # fmt: on

    let c_simd = SIMD[DType.index, 2](5, 6)

    let none = None

    print("breakpoint")
