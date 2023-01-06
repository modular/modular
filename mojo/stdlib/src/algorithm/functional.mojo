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


fn map[
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
    func: __mlir_type[
        `!kgen.signature<[type : !kgen.dtype], [],`,
        `(`,
        SIMD[1, type],
        `) -> `,
        SIMD[1, type],
        `>`,
    ],
](dest: Buffer[size, type], src: Buffer[size, type]):
    """
    Map a unary function over a buffer.
    """
    var i: Int = 0
    while i < dest.__len__():
        let val = func[type](src.__getitem__(i))
        dest.__setitem__(i, val)
        i += 1


fn map[
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
    func: __mlir_type[
        `!kgen.signature<[type : !kgen.dtype], [],`,
        `(`,
        SIMD[1, type],
        `,`,
        SIMD[1, type],
        `) -> `,
        SIMD[1, type],
        `>`,
    ],
](dest: Buffer[size, type], lhs: Buffer[size, type], rhs: Buffer[size, type]):
    """
    Map a binary function over a buffer.
    """
    var i: Int = 0
    while i < dest.__len__():
        let val = func[type](lhs.__getitem__(i), rhs.__getitem__(i))
        dest.__setitem__(i, val)
        i += 1


# ===----------------------------------------------------------------------===#
# Vectorize
# ===----------------------------------------------------------------------===#


fn vectorize[
    simd_width: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
    func: __mlir_type[
        `!kgen.signature<[simd_width : index, type : !kgen.dtype], [],`,
        `(`,
        SIMD[simd_width, type],
        `) -> `,
        SIMD[simd_width, type],
        `>`,
    ],
](dest: Buffer[buffer_size, type], src: Buffer[buffer_size, type]):
    """
    Vectorize a unary function over a buffer.
    """
    var i: Int = 0
    let len = dest.__len__()
    while i < len:
        dest.simd_store[simd_width](
            i, func[simd_width, type](src.simd_load[simd_width](i))
        )
        i += simd_width
    i = (len // simd_width) * simd_width
    while i < len:
        dest.__setitem__(i, func[1, type](src.__getitem__(i)))
        i += 1


fn vectorize[
    simd_width: __mlir_type.index,
    buffer_size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
    func: __mlir_type[
        `!kgen.signature<[simd_width : index, type : !kgen.dtype], [],`,
        `(`,
        SIMD[simd_width, type],
        `,`,
        SIMD[simd_width, type],
        `) -> `,
        SIMD[simd_width, type],
        `>`,
    ],
](
    dest: Buffer[buffer_size, type],
    lhs: Buffer[buffer_size, type],
    rhs: Buffer[buffer_size, type],
):
    """
    Vectorize a binary function over a buffer.
    """
    var i: Int = 0
    let len = dest.__len__()
    while i < len:
        dest.simd_store[simd_width](
            i,
            func[simd_width, type](
                lhs.simd_load[simd_width](i), rhs.simd_load[simd_width](i)
            ),
        )
        i += simd_width
    i = (len // simd_width) * simd_width
    while i < len:
        dest.__setitem__(
            i, func[1, type](lhs.__getitem__(i), rhs.__getitem__(i))
        )
        i += 1
