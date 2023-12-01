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
fn mma(
    inout d: SIMD[DType.float16, 4],
    a: SIMD[DType.float16, 4],
    b: SIMD[DType.float16, 2],
    c: SIMD[DType.float16, 4],
):
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

    d = d0.join(d1)


# ===----------------------------------------------------------------------===#
# F32 = tf32 * tf32 + F32
# ===----------------------------------------------------------------------===#


@always_inline
fn mma(
    inout d: SIMD[DType.float32, 4],
    a: SIMD[DType.float32, 2],
    b: SIMD[DType.float32, 1],
    c: SIMD[DType.float32, 4],
):
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

    d = SIMD[DType.float32, 4](
        r.get[0, Float32](),
        r.get[1, Float32](),
        r.get[2, Float32](),
        r.get[3, Float32](),
    )


@always_inline
fn mma(
    inout d: SIMD[DType.float32, 4],
    a: SIMD[DType.float32, 4],
    b: SIMD[DType.float32, 2],
    c: SIMD[DType.float32, 4],
):
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

    d = SIMD[DType.float32, 4](
        r.get[0, Float32](),
        r.get[1, Float32](),
        r.get[2, Float32](),
        r.get[3, Float32](),
    )
