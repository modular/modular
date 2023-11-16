# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for working with the
warm-matrix-matrix-multiplication (wmma) instructions."""

from memory.unsafe import bitcast
from .ptx_assembly import ptx_assembly


@always_inline
fn _SM80_16x8x8_F16F16F16F16_TN(
    inout d00: Scalar[DType.float16],
    inout d01: Scalar[DType.float16],
    inout d10: Scalar[DType.float16],
    inout d11: Scalar[DType.float16],
    a00: Scalar[DType.float16],
    a01: Scalar[DType.float16],
    a10: Scalar[DType.float16],
    a11: Scalar[DType.float16],
    b00: Scalar[DType.float16],
    b01: Scalar[DType.float16],
    c00: Scalar[DType.float16],
    c01: Scalar[DType.float16],
    c10: Scalar[DType.float16],
    c11: Scalar[DType.float16],
):
    let a0 = SIMD[DType.float16, 2](a00, a01)
    let a1 = SIMD[DType.float16, 2](a10, a11)
    let b0 = SIMD[DType.float16, 2](b00, b01)
    let c0 = SIMD[DType.float16, 2](c00, c01)
    let c1 = SIMD[DType.float16, 2](c10, c11)

    let r = __mlir_op.`pop.inline_asm`[
        assembly = (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {$0, $1},{$2,"
            " $3},{$4},{$5, $6};"
        ).value,
        _type= (UInt32, UInt32),
        constraints = "=r,=r,r,r,r,r,r".value,
        hasSideEffects = __mlir_attr.unit,
    ](
        bitcast[DType.uint32, 1](a0),
        bitcast[DType.uint32, 1](a1),
        bitcast[DType.uint32, 1](b0),
        bitcast[DType.uint32, 1](c0),
        bitcast[DType.uint32, 1](c1),
    )
    let d0 = bitcast[DType.float16, 2](r.get[0, UInt32]())
    let d1 = bitcast[DType.float16, 2](r.get[0, UInt32]())

    d00 = d0[0]
    d01 = d0[1]

    d10 = d1[0]
    d11 = d1[1]
