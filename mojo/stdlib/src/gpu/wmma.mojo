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

    let r = ptx_assembly[
        (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {$0, $1},{$2,"
            " $3},{$4},{$5, $6};"
        ),
        (UInt32, UInt32),
        constraints="=f,=f,r,r,r,r,r",
    ](
        bitcast[DType.uint32, 1](a0),
        bitcast[DType.uint32, 1](a1),
        bitcast[DType.uint32, 1](b),
        bitcast[DType.uint32, 1](c0),
        bitcast[DType.uint32, 1](c1),
    )
    let d0 = bitcast[DType.float16, 2](r.get[0, UInt32]())
    let d1 = bitcast[DType.float16, 2](r.get[0, UInt32]())

    d = d0.join(d1)
