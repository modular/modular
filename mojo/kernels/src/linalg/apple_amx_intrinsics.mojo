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
# ===----------------------------------------------------------------------===#

from sys._assembly import inlined_assembly
from sys.info import sizeof

from memory import memcpy, memset_zero, stack_allocation
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer

from utils.list import DimList


struct amx_detail:
    # All AMX instructions are of the form
    # `0x00201000 | ((op & 0x1F) << 5) | (operand & 0x1F)`
    # where `op` is the operation and `operand` is the register to operate on.

    @staticmethod
    fn _no_op_imms[op: Int32, imm: Int32]():
        # In Apple's Accelerate, instruction 17 is apparently always prefixed by
        # three nops.
        inlined_assembly[
            "nop\nnop\nnop\n.word (0x201000 + ($0 << 5) + $1)",
            NoneType,
            constraints="i,i,~{memory}",
            has_side_effect=True,
        ](op, imm)

    @staticmethod
    fn _op_gpr[op: Int32](gpr: Int64):
        inlined_assembly[
            ".word (0x201000 + ($0 << 5) + 0$1 - ((0$1 >> 4) * 6))",
            NoneType,
            constraints="i,r,~{memory}",
            has_side_effect=True,
        ](op, gpr)

    # The `set` and `clr` take no non-constant operands, and so we pass them as
    # immediate values via meta parameters.
    @staticmethod
    fn _set():
        Self._no_op_imms[17, 0]()

    @staticmethod
    fn _clr():
        Self._no_op_imms[17, 1]()

    @staticmethod
    fn ldx(gpr: Int):
        Self._op_gpr[0](gpr)

    @staticmethod
    fn ldy(gpr: Int):
        Self._op_gpr[1](gpr)

    @staticmethod
    fn stx(gpr: Int):
        Self._op_gpr[2](gpr)

    @staticmethod
    fn sty(gpr: Int):
        Self._op_gpr[3](gpr)

    @staticmethod
    fn ldz(gpr: Int):
        Self._op_gpr[4](gpr)

    @staticmethod
    fn stz(gpr: Int):
        Self._op_gpr[5](gpr)

    @staticmethod
    fn ldzi(gpr: Int):
        Self._op_gpr[6](gpr)

    @staticmethod
    fn stzi(gpr: Int):
        Self._op_gpr[7](gpr)

    @staticmethod
    fn extrx(gpr: Int):
        """
        Extracts a row or moves it to x, result in amx0.
        """
        Self._op_gpr[8](gpr)

    @staticmethod
    fn extry(gpr: Int):
        """
        Extracts a row or moves it to y, result in amx0.
        """
        Self._op_gpr[9](gpr)

    @staticmethod
    fn fma64(gpr: Int):
        """
        Float64 matrix multiply and add.
        """
        Self._op_gpr[10](gpr)

    @staticmethod
    fn fsm64(gpr: Int):
        """
        Float64 matrix multiply and subtract.
        """
        Self._op_gpr[11](gpr)

    @staticmethod
    fn fma32(gpr: Int):
        """
        Float32 matrix multiply and add.
        """
        Self._op_gpr[12](gpr)

    @staticmethod
    fn fsm32(gpr: Int):
        """
        Float32 matrix multiply and subtract.
        """
        Self._op_gpr[13](gpr)

    @staticmethod
    fn mac16(gpr: Int):
        """
        SI16 matrix multiply and add.
        """
        Self._op_gpr[14](gpr)

    @staticmethod
    fn fma16(gpr: Int):
        """
        Float16 matrix multiply and subtract.
        """
        Self._op_gpr[15](gpr)

    @staticmethod
    fn fms16(gpr: Int):
        """
        Float16 matrix multiply and add.
        """
        Self._op_gpr[16](gpr)

    @staticmethod
    fn vec_int__(gpr: Int):
        """
        Horizontal ui16 multiply `z0[i] += x0[i] + y0[i]`.
        """
        Self._op_gpr[18](gpr)

    @staticmethod
    fn vecfp(gpr: Int):
        """
        Horizontal float16 multiply `z0[i] += x0[i] + y0[i]`.
        """
        Self._op_gpr[19](gpr)

    @staticmethod
    fn max_int__(gpr: Int):
        """
        UI16 matrix multiply.
        """
        Self._op_gpr[20](gpr)

    @staticmethod
    fn matfp(gpr: Int):
        """
        Float16 matrix multiply.
        """
        Self._op_gpr[21](gpr)

    @staticmethod
    fn genlut(gpr: Int):
        Self._op_gpr[22](gpr)

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

    @staticmethod
    fn _encode_load_store[
        row_count: Int, type: DType
    ](src: DTypePointer[type], start_index: Int) -> Int:
        """
        Utility to do the bit encoding for load and store ops.
        """
        var src_idx = int(src) | (start_index << 56)
        if row_count == 2:
            src_idx |= 1 << 62
        return src_idx

    @staticmethod
    fn store_x[
        row_count: Int, type: DType
    ](src: DTypePointer[type], start_index: Int):
        Self.ldx(Self._encode_load_store[row_count, type](src, start_index))

    @staticmethod
    fn store_y[
        row_count: Int, type: DType
    ](src: DTypePointer[type], start_index: Int):
        Self.ldy(Self._encode_load_store[row_count, type](src, start_index))

    @staticmethod
    fn store_z[
        row_count: Int, type: DType
    ](src: DTypePointer[type], start_index: Int):
        Self.ldz(Self._encode_load_store[row_count, type](src, start_index))

    @staticmethod
    fn read_x[
        row_count: Int, type: DType
    ](src: DTypePointer[type], start_index: Int):
        Self.stx(Self._encode_load_store[row_count, type](src, start_index))

    @staticmethod
    fn read_y[
        row_count: Int, type: DType
    ](src: DTypePointer[type], start_index: Int):
        Self.sty(Self._encode_load_store[row_count, type](src, start_index))

    @staticmethod
    fn load_z[
        row_count: Int, type: DType
    ](src: DTypePointer[type], start_index: Int):
        Self.stz(Self._encode_load_store[row_count, type](src, start_index))

    @staticmethod
    fn transpose_z_to_x_or_y[
        destination: StringLiteral, type: DType
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
        constrained[type == DType.float32]()

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

        Self.extry(operand)

    @staticmethod
    fn fma[
        mode: StringLiteral, type: DType
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
        constrained[type == DType.float32]()

        alias is_row_mode = mode == "ROW"

        var operand = (
            y_row_index << 6
            | x_row_index << 16
            | z_row_index << 20
            | ((1 << 27) if clear_z else 0)
            | ((1 << 63) if is_row_mode else 0)
        )

        Self.fma32(operand)

    @staticmethod
    fn dot_at_b(
        c: NDBuffer[
            DType.float32,
            2,
            DimList(16, 16),
        ],
        a: NDBuffer[
            DType.float32,
            2,
            DimList(16, 16),
        ],
        b: NDBuffer[
            DType.float32,
            2,
            DimList(16, 16),
        ],
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

        # TODO: We can elide the copy if the data is already is already aligned.
        var a_buffer = stack_allocation[256, Float32, alignment=128]()
        var b_buffer = stack_allocation[256, Float32, alignment=128]()
        var c_buffer = stack_allocation[256, Float32, alignment=128]()

        var num_elements = c.num_elements()
        memcpy(a_buffer, a_pointer, num_elements)
        memcpy(b_buffer, b_pointer, num_elements)
        memset_zero(c_buffer, num_elements)

        Self._set()

        @unroll
        for i in range(8):
            Self.ldx((i << 56) | int(b_buffer.offset(i * b.dim[0]())))
            Self.ldy((i << 56) | int(a_buffer.offset(i * a.dim[0]())))

        Self.fma32(1 << 27)

        @unroll
        for i in range(1, 8):
            Self.fma32((i << 6 << 10) | (i << 6))

        @unroll
        for i in range(8):
            Self.ldx((i << 56) | int(b_buffer.offset((i + 8) * b.dim[0]())))
            Self.ldy((i << 56) | int(a_buffer.offset((i + 8) * a.dim[0]())))

        @unroll
        for i in range(8):
            Self.fma32((i << 6 << 10) | (i << 6))

        @unroll
        for i in range(0, 64, 4):
            Self.stz((i << 56) | int(c_buffer.offset((i >> 2) * c.dim[0]())))

        Self._clr()

        memcpy(c_pointer, c_buffer, num_elements)
