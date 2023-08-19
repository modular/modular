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

from memory.buffer import NDBuffer
from List import DimList
from memory import memset_zero, memcpy
from memory.unsafe import DTypePointer
from sys.info import sizeof


struct amx_detail:
    # All AMX instructions are of the form
    # `0x00201000 | ((op & 0x1F) << 5) | (operand & 0x1F)`
    # where `op` is the operation and `operand` is the register to operate on.

    @staticmethod
    fn _no_op_imms[op: __mlir_type.si32, imm: __mlir_type.si32]():
        # In Apple's Accelerate, instruction 17 is apparently always prefixed by
        # three nops.
        __mlir_op.`pop.inline_asm`[
            _type:None,
            assembly : (
                "nop\nnop\nnop\n.word (0x201000 + ($0 << 5) + $1)"
            ).value,
            constraints : ("i,i,~{memory}").value,
            hasSideEffects : __mlir_attr.unit,
        ](op, imm)

    @staticmethod
    fn _op_gpr[op: __mlir_type.si32](gpr0: Int):
        let gpr = __mlir_op.`index.castu`[_type : __mlir_type.ui64](gpr0.value)
        __mlir_op.`pop.inline_asm`[
            _type:None,
            assembly : (
                ".word (0x201000 + ($0 << 5) + 0$1 - ((0$1 >> 4) * 6))"
            ).value,
            constraints : ("i,r,~{memory}").value,
            hasSideEffects : __mlir_attr.unit,
        ](op, gpr)

    # The `set` and `clr` take no non-constant operands, and so we pass them as
    # immediate values via meta parameters.
    @staticmethod
    fn _set():
        Self._no_op_imms[__mlir_attr.`17:si32`, __mlir_attr.`0:si32`]()

    @staticmethod
    fn _clr():
        Self._no_op_imms[__mlir_attr.`17:si32`, __mlir_attr.`1:si32`]()

    @staticmethod
    fn ldx(gpr: Int):
        Self._op_gpr[__mlir_attr.`0:si32`](gpr)

    @staticmethod
    fn ldy(gpr: Int):
        Self._op_gpr[__mlir_attr.`1:si32`](gpr)

    @staticmethod
    fn stx(gpr: Int):
        Self._op_gpr[__mlir_attr.`2:si32`](gpr)

    @staticmethod
    fn sty(gpr: Int):
        Self._op_gpr[__mlir_attr.`3:si32`](gpr)

    @staticmethod
    fn ldz(gpr: Int):
        Self._op_gpr[__mlir_attr.`4:si32`](gpr)

    @staticmethod
    fn stz(gpr: Int):
        Self._op_gpr[__mlir_attr.`5:si32`](gpr)

    @staticmethod
    fn ldzi(gpr: Int):
        Self._op_gpr[__mlir_attr.`6:si32`](gpr)

    @staticmethod
    fn stzi(gpr: Int):
        Self._op_gpr[__mlir_attr.`7:si32`](gpr)

    @staticmethod
    fn extrx(gpr: Int):
        """
        Extracts a row or moves it to x, result in amx0.
        """
        Self._op_gpr[__mlir_attr.`8:si32`](gpr)

    @staticmethod
    fn extry(gpr: Int):
        """
        Extracts a row or moves it to y, result in amx0.
        """
        Self._op_gpr[__mlir_attr.`9:si32`](gpr)

    @staticmethod
    fn fma64(gpr: Int):
        """
        Float64 matrix multiply and add.
        """
        Self._op_gpr[__mlir_attr.`10:si32`](gpr)

    @staticmethod
    fn fsm64(gpr: Int):
        """
        Float64 matrix multiply and subtract.
        """
        Self._op_gpr[__mlir_attr.`11:si32`](gpr)

    @staticmethod
    fn fma32(gpr: Int):
        """
        Float32 matrix multiply and add.
        """
        Self._op_gpr[__mlir_attr.`12:si32`](gpr)

    @staticmethod
    fn fsm32(gpr: Int):
        """
        Float32 matrix multiply and subtract.
        """
        Self._op_gpr[__mlir_attr.`13:si32`](gpr)

    @staticmethod
    fn mac16(gpr: Int):
        """
        SI16 matrix multiply and add.
        """
        Self._op_gpr[__mlir_attr.`14:si32`](gpr)

    @staticmethod
    fn fma16(gpr: Int):
        """
        Float16 matrix multiply and subtract.
        """
        Self._op_gpr[__mlir_attr.`15:si32`](gpr)

    @staticmethod
    fn fms16(gpr: Int):
        """
        Float16 matrix multiply and add.
        """
        Self._op_gpr[__mlir_attr.`16:si32`](gpr)

    @staticmethod
    fn vecint(gpr: Int):
        """
        Horizontal ui16 multiply `z0[i] += x0[i] + y0[i]`.
        """
        Self._op_gpr[__mlir_attr.`18:si32`](gpr)

    @staticmethod
    fn vecfp(gpr: Int):
        """
        Horizontal float16 multiply `z0[i] += x0[i] + y0[i]`.
        """
        Self._op_gpr[__mlir_attr.`19:si32`](gpr)

    @staticmethod
    fn matint(gpr: Int):
        """
        UI16 matrix multiply.
        """
        Self._op_gpr[__mlir_attr.`20:si32`](gpr)

    @staticmethod
    fn matfp(gpr: Int):
        """
        Float16 matrix multiply.
        """
        Self._op_gpr[__mlir_attr.`21:si32`](gpr)

    @staticmethod
    fn genlut(gpr: Int):
        Self._op_gpr[__mlir_attr.`22:si32`](gpr)

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
        var src_idx = src.__as_index() | (start_index << 56)
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
        destination: __mlir_type.`!kgen.string`, type: DType
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
        constrained[
            __mlir_attr[
                `#kgen.param.expr<in,`,
                destination,
                `,`,
                __mlir_attr.`"X" : !kgen.string`,
                `,`,
                __mlir_attr.`"Y" : !kgen.string`,
                `> : i1`,
            ]
        ]()
        # The type must be Float32.
        constrained[type == DType.float32]()

        # make the y offset field
        #  shift left by 6 to make this an offset in rows,
        #    in fp32 mode, there are 16 elements / 64 byte per row.
        #  The offset field has to be given in bytes.
        let offset = ((z_col_index << 2) | z_row_suboffset) << 20 | (
            xy_row_index << 6
        )

        let is_x_destination = __mlir_attr[
            `#kgen.param.expr<eq,`,
            destination,
            `,`,
            __mlir_attr.`"X" : !kgen.string`,
            `> : i1`,
        ]

        let operand: Int = offset | (
            0x8000000004004000 if is_x_destination else 0x8000000010004000
        )

        Self.extry(operand)

    @staticmethod
    fn fma[
        mode: __mlir_type.`!kgen.string`, type: DType
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
        constrained[
            __mlir_attr[
                `#kgen.param.expr<in,`,
                mode,
                `,`,
                __mlir_attr.`"TILE" : !kgen.string`,
                `,`,
                __mlir_attr.`"ROW" : !kgen.string`,
                `> : i1`,
            ]
        ]()
        # The type must be Float32.
        constrained[type == DType.float32]()

        let is_row_mode = __mlir_attr[
            `#kgen.param.expr<eq,`,
            mode,
            `,`,
            __mlir_attr.`"ROW" : !kgen.string`,
            `> : i1`,
        ]

        let operand = (
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
            2,
            DimList(16, 16),
            DType.float32,
        ],
        a: NDBuffer[
            2,
            DimList(16, 16),
            DType.float32,
        ],
        b: NDBuffer[
            2,
            DimList(16, 16),
            DType.float32,
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

        let a_pointer = a.data
        let b_pointer = b.data
        let c_pointer = c.data

        # TODO: We can elide the copy if the data is already is already aligned.

        alias c256 = (256).value
        alias c128 = (128).value
        let a_buffer: DTypePointer[
            DType.float32
        ] = __mlir_op.`pop.stack_allocation`[
            count:c256,
            alignment:c128,
            _type : __mlir_type.`!pop.pointer<scalar<f32>>`,
        ]()
        let b_buffer: DTypePointer[
            DType.float32
        ] = __mlir_op.`pop.stack_allocation`[
            count:c256,
            alignment:c128,
            _type : __mlir_type.`!pop.pointer<scalar<f32>>`,
        ]()
        let c_buffer: DTypePointer[
            DType.float32
        ] = __mlir_op.`pop.stack_allocation`[
            count:c256,
            alignment:c128,
            _type : __mlir_type.`!pop.pointer<scalar<f32>>`,
        ]()

        let num_elements = c.num_elements()
        memcpy[DType.float32](a_buffer, a_pointer, num_elements)
        memcpy[DType.float32](b_buffer, b_pointer, num_elements)
        memset_zero[DType.float32](c_buffer, num_elements)

        Self._set()

        for i in range(8):
            Self.ldx((i << 56) | b_buffer.offset(i * b.dim[0]()).__as_index())
            Self.ldy((i << 56) | a_buffer.offset(i * a.dim[0]()).__as_index())

        Self.fma32(1 << 27)

        for i in range(1, 8):
            Self.fma32((i << 6 << 10) | (i << 6))

        for i in range(8):
            Self.ldx(
                (i << 56) | b_buffer.offset((i + 8) * b.dim[0]()).__as_index()
            )
            Self.ldy(
                (i << 56) | a_buffer.offset((i + 8) * a.dim[0]()).__as_index()
            )

        for i in range(8):
            Self.fma32((i << 6 << 10) | (i << 6))

        for i in range(0, 64, 4):
            Self.stz(
                (i << 56) | c_buffer.offset((i >> 2) * c.dim[0]()).__as_index()
            )

        Self._clr()

        memcpy[DType.float32](c_pointer, c_buffer, num_elements)
