# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Int import Int
from Buffer import Buffer
from SIMD import SIMD

# TODO: Because of #6542 these functions are not parametric on the dtype. As a
# placeholder, we just implement them for f32. In the future, however, we should
# be able to implement them for any dtype as:
#
# fn map[
#     size: __mlir_type.index,
#     type: __mlir_type.`!kgen.dtype`,
#     func: __mlir_type[
#         `!kgen.signature<[type : !kgen.dtype], [],`,
#         `(!pop.scalar<type>) -> !pop.scalar<type>>`,
#     ],
# ](dest: Buffer[size, type], src: Buffer[size, type]):
#
# Also note that because we do not have a way to pass in functions that capture
# variables, we have to write these functions as overloads based on the arity of
# the function.


fn map[
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
    func: __mlir_type.`!kgen.signature<[], [], (!pop.scalar<f32>) -> !pop.scalar<f32>>`,
](dest0: Buffer[size, type], src0: Buffer[size, type]):
    """
    Map a unary function over a buffer.
    """
    var dest = __mlir_op.`kgen.rebind`[
        _type : Buffer[
            size,
            __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
        ]
    ](dest0)
    var src = __mlir_op.`kgen.rebind`[
        _type : Buffer[
            size,
            __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
        ]
    ](src0)
    var i: Int = 0
    while i < dest.__len__():
        let val = func(src.__getitem__(i).__getitem__(0))
        dest.__setitem__(i, val)
        i += 1


fn map[
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
    func: __mlir_type[
        `!kgen.signature<[], [],`,
        `(!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>>`,
    ],
](
    dest0: Buffer[size, type],
    lhs0: Buffer[size, type],
    rhs0: Buffer[size, type],
):
    """
    Map a binary function over a buffer.
    """
    var dest = __mlir_op.`kgen.rebind`[
        _type : Buffer[
            size,
            __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
        ]
    ](dest0)
    var lhs = __mlir_op.`kgen.rebind`[
        _type : Buffer[
            size,
            __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
        ]
    ](lhs0)
    var rhs = __mlir_op.`kgen.rebind`[
        _type : Buffer[
            size,
            __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
        ]
    ](rhs0)
    var i: Int = 0
    while i < dest.__len__():
        let val = func(
            lhs.__getitem__(i).__getitem__(0), rhs.__getitem__(i).__getitem__(0)
        )
        dest.__setitem__(i, val)
        i += 1
