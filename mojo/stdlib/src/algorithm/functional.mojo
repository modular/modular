# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Int import Int
from Buffer import Buffer
from SIMD import SIMD

# ===----------------------------------------------------------------------===#
# Map
# ===----------------------------------------------------------------------===#


@always_inline
fn map[
    func: __mlir_type[
        `!kgen.signature<[], [],`, `(`, Int, `) force_inline -> !lit.none>`
    ],
](size: Int):
    """
    Map a function over a range from 0 to size.
    """
    var i: Int = 0
    while i < size:
        func(i)
        i += 1


# ===----------------------------------------------------------------------===#
# Vectorize
# ===----------------------------------------------------------------------===#


@always_inline
fn vectorize[
    simd_width: __mlir_type.index,
    func: __mlir_type[
        `!kgen.signature<[simd_width : index], [],`,
        `(`,
        Int,
        `) force_inline -> !lit.none>`,
    ],
](size: Int):
    """Map a function which is parametrized over a simd_Width over a range
    from 0 to size in simd fashion.
    """
    var i: Int = 0
    let vector_end = (size // simd_width) * simd_width
    while i < vector_end:
        func[simd_width](i)
        i += simd_width
    i = vector_end
    while i < size:
        func[1](i)
        i += 1


# ===----------------------------------------------------------------------===#
# reduce
# ===----------------------------------------------------------------------===#
# Implements a simd reduction.
# Consists of 3 steps:
#   1. Iterate over simd_width size chunks of src and use map_fn to update a simd accumulator.
#   2. Reduce the simd accumulator into a single scalar using reduce_fn.
#   3. Iterate over the remainder of src and apply the map_fn on simd elements
#      with simd_width=1.
fn reduce[
    simd_width: __mlir_type.index,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
    acc_type: __mlir_type.`!kgen.dtype`,
    map_fn: __mlir_type[
        `!kgen.signature<[simd_width: index, acc_type : !kgen.dtype, type: !kgen.dtype], [],`,
        `(`,
        SIMD[simd_width, `acc_type`],
        `,`,
        SIMD[simd_width, `type`],
        `) -> `,
        SIMD[simd_width, `acc_type`],
        `>`,
    ],
    reduce_fn: __mlir_type[
        `!kgen.signature<[simd_width: index, type : !kgen.dtype], [],`,
        `(`,
        SIMD[simd_width, `type`],
        `) -> `,
        SIMD[1, `type`],
        `>`,
    ],
](src: Buffer[size, type], init: SIMD[1, acc_type]) -> __mlir_type[
    `!pop.scalar<`, acc_type, `>`
]:
    var i: Int = 0
    var acc_simd = SIMD[simd_width, acc_type].splat(init)
    let len = src.__len__()
    let vector_end = (len // simd_width) * simd_width
    while i < vector_end:
        acc_simd = map_fn[simd_width, acc_type, type](
            acc_simd, src.simd_load[simd_width](i)
        )
        i += simd_width

    i = vector_end
    var acc = reduce_fn[simd_width, acc_type](acc_simd)
    while i < len:
        acc = map_fn[1, acc_type, type](acc, src.__getitem__(i))
        i += 1
    return acc.__getitem__(0)
