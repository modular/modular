from Buffer import Buffer
from SIMD import SIMD
from Numerics import neginf
from Int import Int

# ===----------------------------------------------------------------------===#
# simd_max
# ===----------------------------------------------------------------------===#
# Helper function that computes the max element in a simd vector and is compatible
# with the function signature expected by reduce_fn in reduce().
fn simd_max[
    simd_width: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, type]) -> SIMD[1, type]:
    return x.reduce_max()


# ===----------------------------------------------------------------------===#
# simd_max_elementwise
# ===----------------------------------------------------------------------===#
# Helper function that computes the elementwise max of each element in a simd
# vector and is compatible with the function signature expected by map_fn in
# reduce().
fn simd_max_elementwise[
    simd_width: __mlir_type.index,
    acc_type: __mlir_type.`!kgen.dtype`,
    type: __mlir_type.`!kgen.dtype`,
](x: SIMD[simd_width, acc_type], y0: SIMD[simd_width, type]) -> SIMD[
    simd_width, acc_type
]:
    let y = __mlir_op.`kgen.rebind`[_type : SIMD[simd_width, acc_type]](y0)
    return x.max(y)


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
        SIMD[simd_width, acc_type],
        `,`,
        SIMD[simd_width, type],
        `) -> `,
        SIMD[simd_width, acc_type],
        `>`,
    ],
    reduce_fn: __mlir_type[
        `!kgen.signature<[simd_width: index, type : !kgen.dtype], [],`,
        `(`,
        SIMD[simd_width, type],
        `) -> `,
        SIMD[1, type],
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
# max_element
# ===----------------------------------------------------------------------===#
# Computes the max element in src by calling reduce() with a max map_fn and reduce_fn.
fn max[
    simd_width: __mlir_type.index,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](src: Buffer[size, type]) -> __mlir_type[`!pop.scalar<`, type, `>`]:
    let init = SIMD[1, type].splat(neginf[type]())
    return reduce[simd_width, size, type, type, simd_max_elementwise, simd_max](
        src, init
    )
