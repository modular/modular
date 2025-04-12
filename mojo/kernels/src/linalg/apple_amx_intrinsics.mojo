# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file contains wrappers around Apple's AMX assembly instruction set.
# For information on the Apple AMX instruction set, see
# https://www.notion.so/modularai/Apple-AMX-Resources-2cc523b9c851498787dfloat946ebb09930e.
#
# ===-----------------------------------------------------------------------===#

from collections.string import StaticString
from sys._assembly import inlined_assembly
from sys.info import sizeof

from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import (
    AddressSpace,
    UnsafePointer,
    memcpy,
    memset_zero,
    stack_allocation,
)

# All AMX instructions are of the form
# `0x00201000 | ((op & 0x1F) << 5) | (operand & 0x1F)`
# where `op` is the operation and `operand` is the register to operate on.


@always_inline
fn _no_op_imms[op: Int32, imm: Int32]():
    # In Apple's Accelerate, instruction 17 is apparently always prefixed by
    # three nops.
    inlined_assembly[
        "nop\nnop\nnop\n.word (0x201000 + ($0 << 5) + $1)",
        NoneType,
        constraints="i,i,~{memory}",
        has_side_effect=True,
    ](op, imm)


@always_inline
fn _op_gpr[op: Int32](gpr: Int64):
    inlined_assembly[
        ".word (0x201000 + ($0 << 5) + 0$1 - ((0$1 >> 4) * 6))",
        NoneType,
        constraints="i,r,~{memory}",
        has_side_effect=True,
    ](op, gpr)


# The `set` and `clr` take no non-constant operands, and so we pass them as
# immediate values via meta parameters.
@always_inline
fn _set():
    _no_op_imms[17, 0]()


@always_inline
fn _clr():
    _no_op_imms[17, 1]()


@always_inline
fn ldx(gpr: Int):
    _op_gpr[0](gpr)


@always_inline
fn ldy(gpr: Int):
    _op_gpr[1](gpr)


@always_inline
fn stx(gpr: Int):
    _op_gpr[2](gpr)


@always_inline
fn sty(gpr: Int):
    _op_gpr[3](gpr)


@always_inline
fn ldz(gpr: Int):
    _op_gpr[4](gpr)


@always_inline
fn stz(gpr: Int):
    _op_gpr[5](gpr)


@always_inline
fn ldzi(gpr: Int):
    _op_gpr[6](gpr)


@always_inline
fn stzi(gpr: Int):
    _op_gpr[7](gpr)


@always_inline
fn extrx(gpr: Int):
    """
    Extracts a row or moves it to x, result in amx0.
    """
    _op_gpr[8](gpr)


@always_inline
fn extry(gpr: Int):
    """
    Extracts a row or moves it to y, result in amx0.
    """
    _op_gpr[9](gpr)


@always_inline
fn fma64(gpr: Int):
    """
    Float64 matrix multiply and add.
    """
    _op_gpr[10](gpr)


@always_inline
fn fsm64(gpr: Int):
    """
    Float64 matrix multiply and subtract.
    """
    _op_gpr[11](gpr)


@always_inline
fn fma32(gpr: Int):
    """
    Float32 matrix multiply and add.
    """
    _op_gpr[12](gpr)


@always_inline
fn fsm32(gpr: Int):
    """
    Float32 matrix multiply and subtract.
    """
    _op_gpr[13](gpr)


@always_inline
fn mac16(gpr: Int):
    """
    SI16 matrix multiply and add.
    """
    _op_gpr[14](gpr)


@always_inline
fn fma16(gpr: Int):
    """
    Float16 matrix multiply and subtract.
    """
    _op_gpr[15](gpr)


@always_inline
fn fms16(gpr: Int):
    """
    Float16 matrix multiply and add.
    """
    _op_gpr[16](gpr)


@always_inline
fn vec_int__(gpr: Int):
    """
    Horizontal ui16 multiply `z0[i] += x0[i] + y0[i]`.
    """
    _op_gpr[18](gpr)


@always_inline
fn vecfp(gpr: Int):
    """
    Horizontal float16 multiply `z0[i] += x0[i] + y0[i]`.
    """
    _op_gpr[19](gpr)


@always_inline
fn max_int__(gpr: Int):
    """
    UI16 matrix multiply.
    """
    _op_gpr[20](gpr)


@always_inline
fn matfp(gpr: Int):
    """
    Float16 matrix multiply.
    """
    _op_gpr[21](gpr)


@always_inline
fn genlut(gpr: Int):
    _op_gpr[22](gpr)


# Apple.amx.LoadStore is a set of utilities that are thin wrappers around
# the inline assembly calls, and they provide an easier interface to use
# the amx registers.
#
# The M1 AMX hardware has 3 dedicated register banks, in fp32 mode they
# can be described as:
#
#     float X[8][16], Y[8][16], Z[64][16];
#
#  All instructions reading and writing these AMX registers are memory
#  instructions. The ops defined here marks the direction into/out of amx
#  registers. e.g. :
#
#       load_store.store_x(ptr, idx),
#
#   will read a row of 16 fp32 elements from memory at `ptr`, and save the
#   data in X[idx][:].
#   while
#
#       load_store.load_x (ptr, idx),
#
#   is the opposite, taking X[idx][:] and write to the memory location `ptr`.


@always_inline
fn _encode_load_store[
    row_count: Int, type: DType
](src: UnsafePointer[Scalar[type]], start_index: Int) -> Int:
    """
    Utility to do the bit encoding for load and store ops.
    """
    var src_idx = Int(src) | (start_index << 56)

    @parameter
    if row_count == 2:
        src_idx |= 1 << 62
    return src_idx


@always_inline
fn store_x[
    row_count: Int, type: DType
](src: UnsafePointer[Scalar[type]], start_index: Int):
    ldx(_encode_load_store[row_count, type](src, start_index))


@always_inline
fn store_y[
    row_count: Int, type: DType
](src: UnsafePointer[Scalar[type]], start_index: Int):
    ldy(_encode_load_store[row_count, type](src, start_index))


@always_inline
fn store_z[
    row_count: Int, type: DType
](src: UnsafePointer[Scalar[type]], start_index: Int):
    ldz(_encode_load_store[row_count, type](src, start_index))


@always_inline
fn read_x[
    row_count: Int, type: DType
](src: UnsafePointer[Scalar[type]], start_index: Int):
    stx(_encode_load_store[row_count, type](src, start_index))


@always_inline
fn read_y[
    row_count: Int, type: DType
](src: UnsafePointer[Scalar[type]], start_index: Int):
    sty(_encode_load_store[row_count, type](src, start_index))


@always_inline
fn load_z[
    row_count: Int, type: DType
](src: UnsafePointer[Scalar[type]], start_index: Int):
    stz(_encode_load_store[row_count, type](src, start_index))


@always_inline
fn transpose_z_to_x_or_y[
    destination: StaticString, type: DType
](z_col_index: Int, xy_row_index: Int, z_row_suboffset: Int):
    # transpose_z_to_x_or_y is a thin wrapper around the fp32 transpose mode of
    # the amx instruction `extry`. This instruction takes a (sub) column of
    # register Z (see description above), and transposes it into a row in either
    # register X or register Y.
    #
    # Note that each column of Z has 64 element but each row of X or Y has only
    # 16 elements. The slightly strange part of this instruction is that the
    # value written into X/Y is actually a downsample (i.e. one in every four)
    # result of a column of Z.
    #
    # The instruction takes 1 static parameter dest and 3 dynamic parameters:
    # z_col_index, xy_row_index, and z_row_suboffset.
    # dest can be either `X` or `Y`.
    # With the X,Y,Z data layout described as
    #
    #    float X[8][16], Y[8][16], Z[64][16];
    #
    #  This instruction essentially takes:
    #
    #    extracted_column [16] = Z[z_row_suboffset : 64 : 4][z_col_index]
    #
    # and writes extracted_column[16] to X/Y[xy_row_index][:].
    #  Legal ranges for the parameters:
    #    z_col_index needs to be 0-15,
    #    xy_row_index needs to be 0-7,
    #    z_row_suboffset needs to be 0-4.

    # The destination must be either "X" or "Y".
    constrained[destination == "X" or destination == "Y"]()
    # The type must be Float32.
    constrained[type is DType.float32]()

    # make the y offset field
    #  shift left by 6 to make this an offset in rows,
    #    in fp32 mode, there are 16 elements / 64 byte per row.
    #  The offset field has to be given in bytes.
    var offset = ((z_col_index << 2) | z_row_suboffset) << 20 | (
        xy_row_index << 6
    )

    alias is_x_destination = destination == "X"

    var operand = offset | (
        0x8000000004004000 if is_x_destination else 0x8000000010004000
    )

    extry(operand)


@always_inline
fn fma[
    mode: StaticString, type: DType
](z_row_index: Int, x_row_index: Int, y_row_index: Int, clear_z: Bool):
    # Apple.amx.fma abstracts the fma operation on the amx hardware. Two modes of
    #  fma operations are supported in this instruction, referred to here as
    #  `RowMode` and `TileMode`.
    # `RowMode` is elementwise fma, for each set of given indices, the instruction
    #  computes z[z_row_index][:] += X[x_row_index][:] * Y[y_row_index][:].
    # `TileMode` is matrix fma, each op computes an outer product of:
    #   Y[y_row_index][:] X X[x_row_index][:], (generating a 16x16 matrix)
    #   and the resulting matrix is accumulated into Z[z_row_index::step 4][:].
    #  When clear_z is true, the existing value in Z will be ignored instead of
    #   being accumulated.
    #
    # Issues fma.fp32 instruction to AMX.
    #  Required input range (behavior for out of range is undefined):
    #  z_row_index : [0, 8) in row mode, [0, 4) in tile mode.
    #  x_row_index, y_row_index : always in [0, 8).

    # The mode must be either "TILE" or "ROW".
    constrained[mode == "TILE" or mode == "ROW"]()
    # The type must be Float32.
    constrained[type is DType.float32]()

    alias is_row_mode = mode == "ROW"

    var operand = (
        y_row_index << 6
        | x_row_index << 16
        | z_row_index << 20
        | ((1 << 27) if clear_z else 0)
        | ((1 << 63) if is_row_mode else 0)
    )

    fma32(operand)


@always_inline
fn dot_at_b_impl(
    c: NDBuffer[DType.float32, 2, shape= (16, 16)],
    a: NDBuffer[DType.float32, 2, shape= (16, 16)],
    b: NDBuffer[DType.float32, 2, shape= (16, 16)],
):
    # Performs a 16x16x16 matrix multiply on the given matrices storing the
    # result into the C matrix. The matrix multiplication is performed as:
    #
    #     C = A^T * B
    #
    # Where the dimensions of the matrices are all 16x16. The A matrix is
    # assumed to be transposed and all matrices are stored in row-major
    # order.

    var a_pointer = a.data
    var b_pointer = b.data
    var c_pointer = c.data

    alias num_elements = Int(c.shape.at[1]() * c.shape.at[0]())

    # TODO: We can elide the copy if the data is already is already aligned.
    var a_buffer = stack_allocation[num_elements, Float32, alignment=128]()
    var b_buffer = stack_allocation[num_elements, Float32, alignment=128]()
    var c_buffer = stack_allocation[num_elements, Float32, alignment=128]()

    memcpy(a_buffer, a_pointer, num_elements)
    memcpy(b_buffer, b_pointer, num_elements)
    memset_zero(c_buffer, num_elements)

    # _set() has the side effect of clearing the z tile
    _set()

    @parameter
    for j in range(2):

        @parameter
        for i in range(8):
            ldx((i << 56) | Int(b_buffer.offset((j * 8 + i) * b.dim[0]())))
            ldy((i << 56) | Int(a_buffer.offset((j * 8 + i) * a.dim[0]())))

        @parameter
        for i in range(8):
            fma32((i << 6 << 10) | (i << 6))

    @parameter
    for i in range(0, 64, 4):
        stz((i << 56) | Int(c_buffer.offset((i >> 2) * c.dim[0]())))

    _clr()

    memcpy(c_pointer, c_buffer, num_elements)


@always_inline
fn dot_at_b_impl(
    c: NDBuffer[DType.float16, 2, shape= (32, 32)],
    a: NDBuffer[DType.float16, 2, shape= (32, 32)],
    b: NDBuffer[DType.float16, 2, shape= (32, 32)],
):
    var a_pointer = a.data
    var b_pointer = b.data
    var c_pointer = c.data

    alias num_elements = Int(c.shape.at[1]() * c.shape.at[0]())

    var a_buffer = stack_allocation[num_elements, Float16, alignment=128]()
    var b_buffer = stack_allocation[num_elements, Float16, alignment=128]()
    var c_buffer = stack_allocation[num_elements, Float16, alignment=128]()

    memcpy(a_buffer, a_pointer, num_elements)
    memcpy(b_buffer, b_pointer, num_elements)
    memset_zero(c_buffer, num_elements)

    # _set() has the side effect of clearing the z tile
    _set()

    @parameter
    for j in range(4):

        @parameter
        for i in range(8):
            ldx((i << 56) | Int(b_buffer.offset((j * 8 + i) * b.dim[0]())))
            ldy((i << 56) | Int(a_buffer.offset((j * 8 + i) * a.dim[0]())))

        @parameter
        for i in range(8):
            fma16((i << 6 << 10) | (i << 6))

    @parameter
    for i in range(0, 64, 2):
        stz((i << 56) | Int(c_buffer.offset((i >> 1) * c.dim[0]())))

    _clr()

    memcpy(c_pointer, c_buffer, num_elements)


@always_inline
fn dot_at_b(c: NDBuffer, a: __type_of(c), b: __type_of(c)):
    constrained[
        c.type is DType.float32 or c.type is DType.float16,
        "the buffer dtype must be float32 or float16",
    ]()

    @parameter
    if c.type is DType.float32:
        dot_at_b_impl(
            NDBuffer[
                DType.float32,
                2,
                shape= (16, 16),
                address_space = AddressSpace.GENERIC,
            ](
                c.data.bitcast[Float32]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
            ),
            NDBuffer[
                DType.float32,
                2,
                shape= (16, 16),
                address_space = AddressSpace.GENERIC,
            ](
                a.data.bitcast[Float32]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
            ),
            NDBuffer[
                DType.float32,
                2,
                shape= (16, 16),
                address_space = AddressSpace.GENERIC,
            ](
                b.data.bitcast[Float32]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
            ),
        )
    elif c.type is DType.float16:
        dot_at_b_impl(
            NDBuffer[
                DType.float16,
                2,
                shape= (32, 32),
                address_space = AddressSpace.GENERIC,
            ](
                c.data.bitcast[Float16]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
            ),
            NDBuffer[
                DType.float16,
                2,
                shape= (32, 32),
                address_space = AddressSpace.GENERIC,
            ](
                a.data.bitcast[Float16]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
            ),
            NDBuffer[
                DType.float16,
                2,
                shape= (32, 32),
                address_space = AddressSpace.GENERIC,
            ](
                b.data.bitcast[Float16]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
            ),
        )
