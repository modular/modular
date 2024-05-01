# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils.variant import Variant


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


fn keep_alive[*Ts: AnyType](*args: *Ts):
    pass


fn main():
    var a_var_index = __mlir_op.`index.constant`[
        value = __mlir_attr.`48:index`
    ]()

    var a_register_passable_struct = ARegisterPassableStruct()

    var a_struct = AStruct()

    var p_struct_int = ParamStruct[Int](8)

    var p_struct_stringref = ParamStruct[StringRef]("hello")

    var an_int: Int = 123

    var a_literal_float = 3.125

    var a_float = Float32(3.125)

    var another_float = getFloat()

    var `^ uncommon name` = 1123123

    var a_string_literal = "fofofo"

    var a_list = [1, 2.125, 3]

    var a_simd = SIMD[DType.float16, 4](1.125, 2.5, 0, -3.725)

    # fmt: off
    var b_simd = SIMD[DType.int64, 32](
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
    )
    # fmt: on

    var c_simd = SIMD[DType.index, 2](5, 6)

    var none = None

    print("breakpoint")  # breakpoint

    keep_alive(
        a_var_index,
        a_register_passable_struct,
        a_struct,
        p_struct_int,
        p_struct_stringref,
        an_int,
        a_literal_float,
        a_float,
        another_float,
        `^ uncommon name`,
        a_string_literal,
        a_list,
        a_simd,
        b_simd,
        c_simd,
        none,
    )
