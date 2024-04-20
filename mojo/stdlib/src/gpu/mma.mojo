# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for working with the
warp-matrix-matrix-multiplication (wmma) instructions."""

from sys import llvm_intrinsic, _RegisterPackType

from memory.unsafe import Pointer, bitcast
from gpu.memory import AddressSpace


fn _split(
    x: SIMD,
) -> StaticTuple[SIMD[x.type, x.size // 2], 2]:
    return StaticTuple[SIMD[x.type, x.size // 2], 2](
        x.slice[x.size // 2](), x.slice[x.size // 2, offset = x.size // 2]()
    )


@always_inline
fn mma(inout d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Performs warp sync Tensor Core based Matrix-multiply and accumulate(MMA) operation.
    """

    # ===------------------------------------------------------------------===#
    # F16 = F16 * F16 + F16
    # ===------------------------------------------------------------------===#
    @parameter
    if (
        d.type == DType.float16
        and d.size == 4
        and a.type == DType.float16
        and a.size == 4
        and b.type == DType.float16
        and b.size == 2
        and c.type == DType.float16
        and c.size == 4
    ):
        var sa = _split(a)
        var sc = _split(c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.f16.f16",
            _RegisterPackType[SIMD[DType.float16, 2], SIMD[DType.float16, 2]],
        ](sa[0], sa[1], b, sc[0], sc[1])

        var d0 = r[0]
        var d1 = r[1]

        d[0] = rebind[Scalar[d.type]](d0[0])
        d[1] = rebind[Scalar[d.type]](d0[1])
        d[2] = rebind[Scalar[d.type]](d1[0])
        d[3] = rebind[Scalar[d.type]](d1[1])
    elif (
        d.type == DType.float16
        and d.size == 2
        and a.type == DType.float16
        and a.size == 1
        and b.type == DType.float16
        and b.size == 1
        and c.type == DType.float16
        and c.size == 2
    ):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m8n8k4.row.col.f16.f16",
            _RegisterPackType[Float16, Float16],
        ](a, b, c)

        d[0] = rebind[Scalar[d.type]](r[0][0])
        d[1] = rebind[Scalar[d.type]](r[1][0])

    # ===------------------------------------------------------------------===#
    # F32 = F16 * F16 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type == DType.float32
        and d.size == 4
        and a.type == DType.float16
        and a.size == 4
        and b.type == DType.float16
        and b.size == 2
        and c.type == DType.float32
        and c.size == 4
    ):
        var sa = _split(a)
        var c0 = c

        var c_ptr = Pointer.address_of(c0).bitcast[Float32]()

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.f32.f32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            sa[0],
            sa[1],
            b,
            c_ptr[0],
            c_ptr[1],
            c_ptr[2],
            c_ptr[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])
    elif (
        d.type == DType.float32
        and d.size == 2
        and a.type == DType.float16
        and a.size == 1
        and b.type == DType.float16
        and b.size == 1
        and c.type == DType.float32
        and c.size == 2
    ):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m8n8k4.row.col.f32.f32",
            _RegisterPackType[Float32, Float32],
        ](a, b, c)

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])

    # ===------------------------------------------------------------------===#
    # F32 = BF16 * BF16 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type == DType.float32
        and d.size == 4
        and a.type == DType.bfloat16
        and a.size == 4
        and b.type == DType.bfloat16
        and b.size == 2
        and c.type == DType.float32
        and c.size == 4
    ):
        var sa = _split(a)
        var c0 = c

        var c_ptr = Pointer.address_of(c0).bitcast[Float32]()

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.bf16",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            bitcast[DType.int32, 1](sa[0]),
            bitcast[DType.int32, 1](sa[1]),
            bitcast[DType.int32, 1](b),
            c_ptr[0],
            c_ptr[1],
            c_ptr[2],
            c_ptr[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])

    elif (
        d.type == DType.float32
        and d.size == 4
        and a.type == DType.bfloat16
        and a.size == 8
        and b.type == DType.bfloat16
        and b.size == 4
        and c.type == DType.float32
        and c.size == 4
    ):
        var sa = _split(a)
        var sa1 = _split(sa[0])
        var sa2 = _split(sa[1])
        var sb = _split(b)
        var c0 = c

        var c_ptr = Pointer.address_of(c0).bitcast[Float32]()

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k16.row.col.bf16",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            bitcast[DType.int32, 1](sa1[0]),
            bitcast[DType.int32, 1](sa1[1]),
            bitcast[DType.int32, 1](sa2[0]),
            bitcast[DType.int32, 1](sa2[1]),
            bitcast[DType.int32, 1](sb[0]),
            bitcast[DType.int32, 1](sb[1]),
            c_ptr[0],
            c_ptr[1],
            c_ptr[2],
            c_ptr[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])

    # ===------------------------------------------------------------------===#
    # F32 = tf32 * tf32 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type == DType.float32
        and d.size == 4
        and a.type == DType.float32
        and a.size == 2
        and b.type == DType.float32
        and b.size == 1
        and c.type == DType.float32
        and c.size == 4
    ):
        var a0 = a
        var b0 = b
        var c0 = c

        var a_ptr = Pointer.address_of(a0).bitcast[UInt32]()
        var b_ptr = Pointer.address_of(b0).bitcast[UInt32]()
        var c_ptr = Pointer.address_of(c0).bitcast[Float32]()

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k4.row.col.tf32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            a_ptr[0],
            a_ptr[1],
            b_ptr[0],
            c_ptr[0],
            c_ptr[1],
            c_ptr[2],
            c_ptr[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])

    elif (
        d.type == DType.float32
        and d.size == 4
        and a.type == DType.float32
        and a.size == 4
        and b.type == DType.float32
        and b.size == 2
        and c.type == DType.float32
        and c.size == 4
    ):
        var a0 = a
        var b0 = b
        var c0 = c

        var a_ptr = Pointer.address_of(a0).bitcast[UInt32]()
        var b_ptr = Pointer.address_of(b0).bitcast[UInt32]()
        var c_ptr = Pointer.address_of(c0).bitcast[Float32]()

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.tf32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            a_ptr[0],
            a_ptr[1],
            a_ptr[2],
            a_ptr[3],
            b_ptr[0],
            b_ptr[1],
            c_ptr[0],
            c_ptr[1],
            c_ptr[2],
            c_ptr[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])
    else:
        constrained[False, "no valid implementation of mma"]()


# ===------------------------------------------------------------------===#
# LDMATRIX Instruction
# ===------------------------------------------------------------------===#


@always_inline
fn ld_matrix[
    type: DType, simd_width: Int, transpose: Bool = False
](ptr: DTypePointer[type, AddressSpace.SHARED]) -> SIMD[type, simd_width]:
    """Performs warp sync copy from shared memory to registers.
    Loads in a fashion that can be used directly by tensor core MMA instructions.
    """

    # The register width is fixed at 4 Bytes (32 bits)
    alias register_width = 4
    alias num_registers = (sizeof[type]() * simd_width) // register_width
    alias base = "llvm.nvvm.ldmatrix.sync.aligned.m8n8"

    var d = SIMD[type, simd_width]()

    @parameter
    fn get_suffix() -> StringLiteral:
        return (".trans" if transpose else "") + ".b16.p3"

    # Here .x1 means every thread would use a single register, x2 is 2 while x4 is 4 registers
    # An mma of shape m16n8k8 of type TF32 means for Matrix A every thread would have 4 registers hence .x4
    # and input simd_width being equal to 4
    @parameter
    if num_registers == 1:
        alias ins = base + ".x1" + get_suffix()
        var r = llvm_intrinsic[ins, UInt32](ptr)
        var r_ptr = Pointer.address_of(r).bitcast[__type_of(d)]()
        d = rebind[SIMD[d.type, d.size]](r_ptr[0])
    elif num_registers == 2:
        alias ins = base + ".x2" + get_suffix()
        var r = llvm_intrinsic[ins, _RegisterPackType[UInt32, UInt32]](ptr)
        var r_ptr = Pointer.address_of(r).bitcast[__type_of(d)]()
        d = rebind[SIMD[d.type, d.size]](r_ptr[0])
    else:
        constrained[
            num_registers == 4,
            "no valid implementation of ldmatrix instruction",
        ]()
        alias ins = base + ".x4" + get_suffix()
        var r = llvm_intrinsic[
            ins, _RegisterPackType[UInt32, UInt32, UInt32, UInt32]
        ](ptr)
        var r_ptr = Pointer.address_of(r).bitcast[__type_of(d)]()
        d = rebind[SIMD[d.type, d.size]](r_ptr[0])
    return d
