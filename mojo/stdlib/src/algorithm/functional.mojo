# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Int import Int
from Buffer import Buffer
from SIMD import SIMD


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
