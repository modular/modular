# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import NDBuffer
from Tuple import StaticTuple
from Assert import assert_param
from Int import Int
from SIMD import SIMD


@interface
fn transpose_inplace[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    dtype: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[
        rank,
        __mlir_attr[
            `#kgen.list<`,
            rows,
            `, `,
            cols,
            `> : `,
            `!kgen.list<`,
            __mlir_type.index,
            `[`,
            rank,
            `]>`,
        ],
        dtype,
    ]
):
    ...


@implements(transpose_inplace)
fn transpose_inplace_generic[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    dtype: __mlir_type.`!kgen.dtype`,
](
    buf0: NDBuffer[
        rank,
        __mlir_attr[
            `#kgen.list<`,
            rows,
            `, `,
            cols,
            `> : `,
            `!kgen.list<`,
            __mlir_type.index,
            `[`,
            rank,
            `]>`,
        ],
        dtype,
    ]
):
    assert_param[rank == 2]()
    var buf = __mlir_op.`kgen.rebind`[
        _type : NDBuffer[
            2,
            __mlir_attr[
                `#kgen.list<`,
                rows,
                `, `,
                cols,
                `> : `,
                `!kgen.list<`,
                __mlir_type.index,
                `[`,
                +2,
                `]>`,
            ],
            dtype,
        ]
    ](buf0)
    var i: Int = 0
    while i < rows:
        var j: Int = 0
        while j < cols:
            let pos = StaticTuple[rank, __mlir_type.index].pair(
                i.__as_mlir_index(), j.__as_mlir_index()
            )
            let pos_tr = StaticTuple[rank, __mlir_type.index].pair(
                j.__as_mlir_index(), i.__as_mlir_index()
            )
            let val: SIMD[1, dtype] = buf.__getitem__(pos)
            buf.__setitem__(pos_tr, val)
            j += 1
        i += 1
