# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for working with the
warp-matrix-matrix-multiplication (wmma) instructions."""

from memory.unsafe import bitcast, Pointer
from sys import llvm_intrinsic


# ===----------------------------------------------------------------------===#
# F16 = F16 * F16 + F16
# ===----------------------------------------------------------------------===#


@always_inline
@adaptive
fn mma(inout d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    constrained[d.type == DType.float16 and d.size == 4]()
    constrained[a.type == DType.float16 and a.size == 4]()
    constrained[b.type == DType.float16 and b.size == 2]()
    constrained[c.type == DType.float16 and c.size == 4]()

    let a0 = a.slice[2]()
    let a1 = a.slice[2](2)
    let c0 = c.slice[2]()
    let c1 = c.slice[2](2)

    let r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k8.row.col.f16.f16",
        (SIMD[DType.float16, 2], SIMD[DType.float16, 2]),
    ](
        a0,
        a1,
        b,
        c0,
        c1,
    )

    let d0 = r.get[0, SIMD[DType.float16, 2]]()
    let d1 = r.get[1, SIMD[DType.float16, 2]]()

    d[0] = rebind[Scalar[d.type]](d0[0])
    d[1] = rebind[Scalar[d.type]](d0[1])
    d[2] = rebind[Scalar[d.type]](d1[0])
    d[3] = rebind[Scalar[d.type]](d1[1])


# ===----------------------------------------------------------------------===#
# F32 = tf32 * tf32 + F32
# ===----------------------------------------------------------------------===#


@always_inline
@adaptive
fn mma(inout d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    constrained[d.type == DType.float32 and d.size == 4]()
    constrained[a.type == DType.float32 and a.size == 2]()
    constrained[b.type == DType.float32 and b.size == 1]()
    constrained[c.type == DType.float32 and c.size == 4]()

    var a0 = a
    var b0 = b
    var c0 = c

    let a_ptr = Pointer.address_of(a0).bitcast[UInt32]()
    let b_ptr = Pointer.address_of(b0).bitcast[UInt32]()
    let c_ptr = Pointer.address_of(c0).bitcast[Float32]()

    let r = llvm_intrinsic[
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


@always_inline
@adaptive
fn mma(inout d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    constrained[d.type == DType.float32 and d.size == 4]()
    constrained[a.type == DType.float32 and a.size == 4]()
    constrained[b.type == DType.float32 and b.size == 2]()
    constrained[c.type == DType.float32 and c.size == 4]()

    var a0 = a
    var b0 = b
    var c0 = c

    let a_ptr = Pointer.address_of(a0).bitcast[UInt32]()
    let b_ptr = Pointer.address_of(b0).bitcast[UInt32]()
    let c_ptr = Pointer.address_of(c0).bitcast[Float32]()

    let r = llvm_intrinsic[
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
