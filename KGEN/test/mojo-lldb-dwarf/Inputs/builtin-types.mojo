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
    var uint16: UInt16
    var int32: Int32

    fn __init__() -> Self:
        return Self {
            int: -101,
            f32: 24.125,
            another_int: 101,
            float16: 25.125,
            uint8: 123,
            uint16: 123,
            int32: 485,
        }


struct AStruct:
    var int: UInt8
    var tuple: Tuple[Int, Int8, Float32]

    fn __init__(inout self):
        self.int = 12
        self.tuple = Tuple[Int, Int8, Float32](1, 87, 123.125)


fn main():
    var a_var_index = __mlir_op.`index.constant`[
        value = __mlir_attr.`48:index`
    ]()
    # FIXME(22592): We need to print the address, otherwise the var has wrong DI.
    print(Pointer(__get_lvalue_as_address(a_var_index)).__as_index())

    let a_let_index = __mlir_op.`index.constant`[
        value = __mlir_attr.`48:index`
    ]()

    let a_register_passable_struct = ARegisterPassableStruct()

    let a_struct = AStruct()

    let an_int: Int = 123

    let a_literal_float = 3.125

    let a_float = Float32(3.125)

    let another_float = getFloat()

    let `^ uncommon name` = 1123123

    let a_string_literal = "fofofo"

    let a_list = [1, 2.125, 3]

    print("end")
