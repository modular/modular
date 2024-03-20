# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for working with the
warp-matrix-matrix-multiplication (wmma) instructions."""

from sys import llvm_intrinsic

from memory.unsafe import Pointer, bitcast


fn _split(
    x: SIMD,
) -> StaticTuple[SIMD[x.type, x.size // 2], 2]:
    return StaticTuple[SIMD[x.type, x.size // 2], 2](
        x.slice[x.size // 2](), x.slice[x.size // 2](x.size // 2)
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
            (SIMD[DType.float16, 2], SIMD[DType.float16, 2]),
        ](sa[0], sa[1], b, sc[0], sc[1])

        var d0 = r.get[0, SIMD[DType.float16, 2]]()
        var d1 = r.get[1, SIMD[DType.float16, 2]]()

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
            "llvm.nvvm.mma.m8n8k4.row.col.f16.f16", (Float16, Float16)
        ](a, b, c)

        var d0 = r.get[0, Float16]()
        var d1 = r.get[1, Float16]()

        d[0] = rebind[Scalar[d.type]](d0[0])
        d[1] = rebind[Scalar[d.type]](d1[0])

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
            (Float32, Float32, Float32, Float32),
        ](
            sa[0],
            sa[1],
            b,
            c_ptr[0],
            c_ptr[1],
            c_ptr[2],
            c_ptr[3],
        )

        d[0] = rebind[Scalar[d.type]](r.get[0, Float32]())
        d[1] = rebind[Scalar[d.type]](r.get[1, Float32]())
        d[2] = rebind[Scalar[d.type]](r.get[2, Float32]())
        d[3] = rebind[Scalar[d.type]](r.get[3, Float32]())
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
            (Float32, Float32),
        ](a, b, c)

        var d0 = r.get[0, Float32]()
        var d1 = r.get[1, Float32]()

        d[0] = rebind[Scalar[d.type]](d0[0])
        d[1] = rebind[Scalar[d.type]](d1[0])

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
            (Float32, Float32, Float32, Float32),
        ](
            bitcast[DType.int32, 1](sa[0]),
            bitcast[DType.int32, 1](sa[1]),
            bitcast[DType.int32, 1](b),
            c_ptr[0],
            c_ptr[1],
            c_ptr[2],
            c_ptr[3],
        )

        d[0] = rebind[Scalar[d.type]](r.get[0, Float32]())
        d[1] = rebind[Scalar[d.type]](r.get[1, Float32]())
        d[2] = rebind[Scalar[d.type]](r.get[2, Float32]())
        d[3] = rebind[Scalar[d.type]](r.get[3, Float32]())

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
            (Float32, Float32, Float32, Float32),
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

        d[0] = rebind[Scalar[d.type]](r.get[0, Float32]())
        d[1] = rebind[Scalar[d.type]](r.get[1, Float32]())
        d[2] = rebind[Scalar[d.type]](r.get[2, Float32]())
        d[3] = rebind[Scalar[d.type]](r.get[3, Float32]())

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
            (Float32, Float32, Float32, Float32),
        ](
            a_ptr[0],
            a_ptr[1],
            b_ptr[0],
            c_ptr[0],
            c_ptr[1],
            c_ptr[2],
            c_ptr[3],
        )

        d[0] = rebind[Scalar[d.type]](r.get[0, Float32]())
        d[1] = rebind[Scalar[d.type]](r.get[1, Float32]())
        d[2] = rebind[Scalar[d.type]](r.get[2, Float32]())
        d[3] = rebind[Scalar[d.type]](r.get[3, Float32]())

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
            (Float32, Float32, Float32, Float32),
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

        d[0] = rebind[Scalar[d.type]](r.get[0, Float32]())
        d[1] = rebind[Scalar[d.type]](r.get[1, Float32]())
        d[2] = rebind[Scalar[d.type]](r.get[2, Float32]())
        d[3] = rebind[Scalar[d.type]](r.get[3, Float32]())
    else:
        constrained[False, "no valid implementation of mma"]()
