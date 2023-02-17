# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import debug_assert
from Buffer import Buffer
from SIMD import SIMD
from Numerics import inf, neginf
from Int import Int
from Range import range

# ===----------------------------------------------------------------------===#
# reduce
# ===----------------------------------------------------------------------===#
# Implements a simd reduction.
# Consists of 3 steps:
#   1. Iterate over simd_width size chunks of src and use map_fn to update a simd accumulator.
#   2. Reduce the simd accumulator into a single scalar using reduce_fn.
#   3. Iterate over the remainder of src and apply the map_fn on simd elements
#      with simd_width=1.
@always_inline
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
    var acc_simd = SIMD[simd_width, acc_type].splat(init)
    let len = src.__len__()
    let vector_end = (len // simd_width) * simd_width
    for i in range(0, vector_end, simd_width):
        acc_simd = map_fn[simd_width, acc_type, type](
            acc_simd, src.simd_load[simd_width](i)
        )

    var acc = reduce_fn[simd_width, acc_type](acc_simd)
    for ii in range(vector_end, len):  # TODO(#8365) use `i`
        acc = map_fn[1, acc_type, type](acc, src.__getitem__(ii))
    return acc.__getitem__(0)


# ===----------------------------------------------------------------------===#
# max
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_max[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    """Helper function that computes the max element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_max()


@always_inline
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
    ](src, src.__getitem__(0))


# ===----------------------------------------------------------------------===#
# min
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_min[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    """Helper function that computes the min element in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_min()


@always_inline
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
    ](src, src.__getitem__(0))


# ===----------------------------------------------------------------------===#
# sum
# ===----------------------------------------------------------------------===#


@always_inline
fn _simd_sum[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    """Helper function that computes the sum of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_add()


@always_inline
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


@always_inline
fn _simd_product[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    """Helper function that computes the product of elements in a simd vector and is
    compatible with the function signature expected by reduce_fn in reduce."""
    return x.reduce_mul()


@always_inline
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


# ===----------------------------------------------------------------------===#
# mean
# ===----------------------------------------------------------------------===#


fn mean[
    simd_width: __mlir_type.index,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](src: Buffer[size, type]) -> __mlir_type[`!pop.scalar<`, type, `>`]:
    """Computes the mean value of the elements in a buffer."""

    debug_assert(src.__len__() != 0, "input must not be empty")

    return (
        SIMD[1, type](sum[simd_width, size, type](src)) / src.__len__()
    ).__getitem__(0)


# ===----------------------------------------------------------------------===#
# variance
# ===----------------------------------------------------------------------===#


fn variance[
    simd_width: __mlir_type.index,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](src: Buffer[size, type]) -> __mlir_type[`!pop.scalar<`, type, `>`]:
    """Computes the variance value of the elements in a buffer."""

    debug_assert(src.__len__() > 1, "input length must be greater than 1")

    let mean_value = mean[simd_width, size, type](src)

    @always_inline
    fn _simd_variance_elementwise[
        simd_width: __mlir_type.index,
        acc_type: __mlir_type.`!kgen.dtype`,
        type: __mlir_type.`!kgen.dtype`,
    ](x: SIMD[simd_width, acc_type], y0: SIMD[simd_width, type]) -> SIMD[
        simd_width, acc_type
    ]:
        """Helper function that computes the equation $sum (x_i - u)^2 + y$"""
        let y: SIMD[simd_width, acc_type] = __mlir_op.`pop.cast`[
            _type : __mlir_type[`!pop.simd<`, simd_width, `,`, acc_type, `>`]
        ](y0.value)
        let mean_simd = SIMD[simd_width, acc_type].splat(
            __mlir_op.`pop.cast`[
                _type : __mlir_type[`!pop.scalar<`, acc_type, `>`]
            ](mean_value)
        )
        let diff = y - mean_simd
        return x + diff * diff

    let numerator: SIMD[1, type] = reduce[
        simd_width, size, type, type, _simd_variance_elementwise, _simd_sum
    ](src, 0)
    return (numerator / (src.__len__() - 1)).__getitem__(0)
