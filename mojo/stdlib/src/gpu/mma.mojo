# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for working with the
warp-matrix-matrix-multiplication (wmma) instructions."""

from sys import _RegisterPackType, llvm_intrinsic

from gpu.memory import AddressSpace
from memory import UnsafePointer, bitcast

from utils import StaticTuple


@always_inline
fn mma(inout d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Performs warp sync Tensor Core based Matrix-multiply and accumulate(MMA) operation.
    """

    # ===------------------------------------------------------------------===#
    # F16 = F16 * F16 + F16
    # ===------------------------------------------------------------------===#
    @parameter
    if (
        d.type is DType.float16
        and d.size == 4
        and a.type is DType.float16
        and a.size == 4
        and b.type is DType.float16
        and b.size == 2
        and c.type is DType.float16
        and c.size == 4
    ):
        var sa = a.split()
        var sc = c.split()

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
        d.type is DType.float16
        and d.size == 2
        and a.type is DType.float16
        and a.size == 1
        and b.type is DType.float16
        and b.size == 1
        and c.type is DType.float16
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
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float16
        and a.size == 4
        and b.type is DType.float16
        and b.size == 2
        and c.type is DType.float32
        and c.size == 4
    ):
        var sa = a.split()
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.f32.f32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            sa[0],
            sa[1],
            b,
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])
    elif (
        d.type is DType.float32
        and d.size == 2
        and a.type is DType.float16
        and a.size == 1
        and b.type is DType.float16
        and b.size == 1
        and c.type is DType.float32
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
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.bfloat16
        and a.size == 4
        and b.type is DType.bfloat16
        and b.size == 2
        and c.type is DType.float32
        and c.size == 4
    ):
        var sa = a.split()
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.bf16",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            bitcast[DType.int32, 1](sa[0]),
            bitcast[DType.int32, 1](sa[1]),
            bitcast[DType.int32, 1](b),
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])

    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.bfloat16
        and a.size == 8
        and b.type is DType.bfloat16
        and b.size == 4
        and c.type is DType.float32
        and c.size == 4
    ):
        var sa = a.split()
        var sa1 = sa[0].split()
        var sa2 = sa[1].split()
        var sb = b.split()
        var c0 = bitcast[DType.float32, 4](c)

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
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])

    # ===------------------------------------------------------------------===#
    # F32 = tf32 * tf32 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float32
        and a.size == 2
        and b.type is DType.float32
        and b.size == 1
        and c.type is DType.float32
        and c.size == 4
    ):
        var a0 = bitcast[DType.uint32, 2](a)
        var b0 = bitcast[DType.uint32, 1](b)
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k4.row.col.tf32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            a0[0],
            a0[1],
            b0,
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )

        d[0] = rebind[Scalar[d.type]](r[0])
        d[1] = rebind[Scalar[d.type]](r[1])
        d[2] = rebind[Scalar[d.type]](r[2])
        d[3] = rebind[Scalar[d.type]](r[3])

    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float32
        and a.size == 4
        and b.type is DType.float32
        and b.size == 2
        and c.type is DType.float32
        and c.size == 4
    ):
        var a0 = bitcast[DType.uint32, 4](a)
        var b0 = bitcast[DType.uint32, 2](b)
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.tf32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            a0[0],
            a0[1],
            a0[2],
            a0[3],
            b0[0],
            b0[1],
            c0[0],
            c0[1],
            c0[2],
            c0[3],
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
](ptr: UnsafePointer[Scalar[type], AddressSpace.SHARED]) -> SIMD[
    type, simd_width
]:
    """Performs warp sync copy from shared memory to registers.
    Loads in a fashion that can be used directly by tensor core MMA instructions.
    """

    # TODO: Investigate if fp8 can work with transposed ld_matrix.
    constrained[
        (transpose and type.is_half_float()) or (not transpose),
        "Transposed ld_matrix is only for half precision.",
    ]()

    # The register width is fixed at 4 Bytes (32 bits)
    alias register_btypes = 4
    alias register_width = register_btypes // sizeof[type]()
    alias num_registers = simd_width // register_width

    # Full intrinsic is base + suffix
    alias base = "llvm.nvvm.ldmatrix.sync.aligned.m8n8"

    @parameter
    fn get_suffix() -> StringLiteral:
        alias sfx = ".b16.p3"
        if transpose:
            return ".trans" + sfx
        return sfx

    var d = SIMD[type, simd_width]()

    # Here .x1 means every thread would use a single register, x2 is 2 while x4 is 4 registers
    # An mma of shape m16n8k8 of type TF32 means for Matrix A every thread would have 4 registers hence .x4
    # and input simd_width being equal to 4
    @parameter
    if num_registers == 1:
        alias ins = base + ".x1" + get_suffix()
        var r = llvm_intrinsic[ins, UInt32](ptr)
        var r0 = bitcast[type, register_width](r[0])

        d = rebind[SIMD[type, simd_width]](r0)

    elif num_registers == 2:
        alias ins = base + ".x2" + get_suffix()
        var r = llvm_intrinsic[ins, _RegisterPackType[UInt32, UInt32]](ptr)
        var r0 = bitcast[type, register_width](r[0])
        var r1 = bitcast[type, register_width](r[1])

        d = rebind[SIMD[type, simd_width]](r0.join(r1))

    else:
        constrained[
            num_registers == 4,
            "no valid implementation of ldmatrix instruction",
        ]()
        alias ins = base + ".x4" + get_suffix()
        var r = llvm_intrinsic[
            ins, _RegisterPackType[UInt32, UInt32, UInt32, UInt32]
        ](ptr)

        # Unpack result to 4 vectors (one per register), then concat them to return.
        var r0 = bitcast[type, register_width](r[0])
        var r1 = bitcast[type, register_width](r[1])
        var r2 = bitcast[type, register_width](r[2])
        var r3 = bitcast[type, register_width](r[3])
        d = rebind[SIMD[type, simd_width]](r0.join(r1).join(r2.join(r3)))

        # The following creates additional copies uint32 <-> 2xbf16 in matmul.
        # @parameter
        # for i in range(num_registers):
        #     var vec_per_register = bitcast[type, register_width](
        #         rebind[UInt32](r[i])
        #     )

        #     @parameter
        #     for j in range(register_width):
        #         d[i * register_width + j] = vec_per_register[j]

    return d
