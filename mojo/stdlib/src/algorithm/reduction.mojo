# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import Buffer
from SIMD import SIMD
from Numerics import inf, neginf
from Int import Int

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
        `!kgen.signature<<simd_width, acc_type: dtype, type: dtype>(`,
        SIMD[simd_width, `acc_type`],
        `,`,
        SIMD[simd_width, `type`],
        `) -> `,
        SIMD[simd_width, `acc_type`],
        `>`,
    ],
    reduce_fn: __mlir_type[
        `!kgen.signature<<simd_width, type: dtype>(`,
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


# ===----------------------------------------------------------------------===#
# max
# ===----------------------------------------------------------------------===#


fn _simd_max[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    """Helper function that computes the max element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_max()


fn _simd_max_elementwise[
    simd_width: __mlir_type.index,
    acc_type: __mlir_type.`!kgen.dtype`,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, acc_type], y0: SIMD[simd_width, type]) -> SIMD[
    simd_width, acc_type
]:
    """Helper function that computes the elementwise max of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    let y: SIMD[simd_width, acc_type] = __mlir_op.`pop.cast`[
        _type : __mlir_type[`!pop.simd<`, simd_width, `,`, acc_type, `>`]
    ](y0.value)
    return x.max(y)


fn max[
    simd_width: __mlir_type.index,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](src: Buffer[size, type]) -> __mlir_type[`!pop.scalar<`, type, `>`]:
    """Computes the max element in a buffer."""
    return reduce[
        simd_width, size, type, type, _simd_max_elementwise, _simd_max
    ](src, neginf[type]())


# ===----------------------------------------------------------------------===#
# min
# ===----------------------------------------------------------------------===#


fn _simd_min[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    """Helper function that computes the min element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_min()


fn _simd_min_elementwise[
    simd_width: __mlir_type.index,
    acc_type: __mlir_type.`!kgen.dtype`,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, acc_type], y0: SIMD[simd_width, type]) -> SIMD[
    simd_width, acc_type
]:
    """Helper function that computes the elementwise min of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    let y: SIMD[simd_width, acc_type] = __mlir_op.`pop.cast`[
        _type : __mlir_type[`!pop.simd<`, simd_width, `,`, acc_type, `>`]
    ](y0.value)
    return x.min(y)


fn min[
    simd_width: __mlir_type.index,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](src: Buffer[size, type]) -> __mlir_type[`!pop.scalar<`, type, `>`]:
    """Computes the min element in a buffer."""
    return reduce[
        simd_width, size, type, type, _simd_min_elementwise, _simd_min
    ](src, inf[type]())


# ===----------------------------------------------------------------------===#
# sum
# ===----------------------------------------------------------------------===#


fn _simd_sum[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    """Helper function that computes the sum of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_add()


fn _simd_sum_elementwise[
    simd_width: __mlir_type.index,
    acc_type: __mlir_type.`!kgen.dtype`,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, acc_type], y0: SIMD[simd_width, type]) -> SIMD[
    simd_width, acc_type
]:
    """Helper function that computes the elementwise sum of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    let y: SIMD[simd_width, acc_type] = __mlir_op.`pop.cast`[
        _type : __mlir_type[`!pop.simd<`, simd_width, `,`, acc_type, `>`]
    ](y0.value)
    return x + y


fn sum[
    simd_width: __mlir_type.index,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](src: Buffer[size, type]) -> __mlir_type[`!pop.scalar<`, type, `>`]:
    """Computes the sum element in a buffer."""
    return reduce[
        simd_width, size, type, type, _simd_sum_elementwise, _simd_sum
    ](src, 0)


# ===----------------------------------------------------------------------===#
# product
# ===----------------------------------------------------------------------===#


fn _simd_product[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    """Helper function that computes the product of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_mul()


fn _simd_product_elementwise[
    simd_width: __mlir_type.index,
    acc_type: __mlir_type.`!kgen.dtype`,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, acc_type], y0: SIMD[simd_width, type]) -> SIMD[
    simd_width, acc_type
]:
    """Helper function that computes the elementwise product of each element in a
    simd vector and is compatible with the function signature expected by map_fn
    in reduce."""
    let y: SIMD[simd_width, acc_type] = __mlir_op.`pop.cast`[
        _type : __mlir_type[`!pop.simd<`, simd_width, `,`, acc_type, `>`]
    ](y0.value)
    return x * y


fn product[
    simd_width: __mlir_type.index,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](src: Buffer[size, type]) -> __mlir_type[`!pop.scalar<`, type, `>`]:
    """Computes the product element in a buffer."""
    return reduce[
        simd_width, size, type, type, _simd_product_elementwise, _simd_product
    ](src, 1)
