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


fn _pair(i: Int, j: Int) -> StaticTuple[2, __mlir_type.index]:
    return StaticTuple[2, __mlir_type.index].pair(
        i.__as_mlir_index(), j.__as_mlir_index()
    )


@interface
fn transpose_inplace[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
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
        type,
    ]
):
    ...


@implements(transpose_inplace)
fn transpose_inplace_4x4[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
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
        type,
    ]
):
    assert_param[rank == 2]()
    assert_param[rows == 4]()
    assert_param[cols == 4]()
    var buf = __mlir_op.`kgen.rebind`[
        _type : NDBuffer[
            2,
            __mlir_attr[
                `#kgen.list<`,
                +4,
                `, `,
                +4,
                `> : `,
                `!kgen.list<`,
                __mlir_type.index,
                `[`,
                +2,
                `]>`,
            ],
            type,
        ]
    ](buf0)

    let row0 = buf.simd_load[4](_pair(0, 0))
    let row1 = buf.simd_load[4](_pair(1, 0))
    let row2 = buf.simd_load[4](_pair(2, 0))
    let row3 = buf.simd_load[4](_pair(3, 0))

    let tmp0 = row0.shuffle[
        4,
        __mlir_attr[`#kgen.list<0, 1, 4, 5> : !kgen.list<index[4]>`],
    ](row1)
    let tmp1 = row0.shuffle[
        4,
        __mlir_attr[`#kgen.list<2, 3, 6, 7> : !kgen.list<index[4]>`],
    ](row1)
    let tmp2 = row2.shuffle[
        4,
        __mlir_attr[`#kgen.list<0, 1, 4, 5> : !kgen.list<index[4]>`],
    ](row3)
    let tmp3 = row2.shuffle[
        4,
        __mlir_attr[`#kgen.list<2, 3, 6, 7> : !kgen.list<index[4]>`],
    ](row3)

    let r0 = tmp0.shuffle[
        4,
        __mlir_attr[`#kgen.list<0, 2, 4, 6> : !kgen.list<index[4]>`],
    ](tmp1)
    let r1 = tmp0.shuffle[
        4,
        __mlir_attr[`#kgen.list<1, 3, 5, 7> : !kgen.list<index[4]>`],
    ](tmp1)
    let r2 = tmp2.shuffle[
        4,
        __mlir_attr[`#kgen.list<0, 2, 4, 6> : !kgen.list<index[4]>`],
    ](tmp3)
    let r3 = tmp2.shuffle[
        4,
        __mlir_attr[`#kgen.list<1, 3, 5, 7> : !kgen.list<index[4]>`],
    ](tmp3)

    buf.simd_store[4](_pair(0, 0), r0)
    buf.simd_store[4](_pair(1, 0), r1)
    buf.simd_store[4](_pair(2, 0), r2)
    buf.simd_store[4](_pair(3, 0), r3)


@implements(transpose_inplace)
fn transpose_inplace_generic[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
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
        type,
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
            type,
        ]
    ](buf0)
    var i: Int = 0
    while i < rows:
        var j: Int = 0
        while j < cols:
            let pos = _pair(i, j)
            let pos_tr = _pair(j, i)
            let val: SIMD[1, type] = buf.__getitem__(pos)
            buf.__setitem__(pos_tr, val)
            j += 1
        i += 1
