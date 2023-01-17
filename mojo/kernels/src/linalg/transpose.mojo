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
from List import create_kgen_list


fn _index2D(rows: Int, cols: Int) -> StaticTuple[2, __mlir_type.index]:
    return StaticTuple[2, __mlir_type.index].pair(
        rows.__as_mlir_index(), cols.__as_mlir_index()
    )


@interface
fn transpose_inplace[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[
        2,
        __mlir_attr[
            `#kgen<list[`,
            rows,
            `, `,
            cols,
            `]> : `,
            `!kgen.list<`,
            __mlir_type.index,
            `[2]>`,
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
        2,
        __mlir_attr[
            `#kgen<list[`,
            rows,
            `, `,
            cols,
            `]> : `,
            `!kgen.list<`,
            __mlir_type.index,
            `[2]>`,
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
            create_kgen_list[__mlir_type.index](4, 4),
            type,
        ]
    ](buf0)

    let row0 = buf.simd_load[4](_index2D(0, 0))
    let row1 = buf.simd_load[4](_index2D(1, 0))
    let row2 = buf.simd_load[4](_index2D(2, 0))
    let row3 = buf.simd_load[4](_index2D(3, 0))

    let tmp0 = row0.shuffle[
        4, create_kgen_list[__mlir_type.index](0, 1, 4, 5)
    ](row1)
    let tmp1 = row2.shuffle[
        4, create_kgen_list[__mlir_type.index](0, 1, 4, 5)
    ](row3)
    let tmp2 = row0.shuffle[
        4, create_kgen_list[__mlir_type.index](2, 3, 6, 7)
    ](row1)
    let tmp3 = row2.shuffle[
        4, create_kgen_list[__mlir_type.index](2, 3, 6, 7)
    ](row3)

    let r0 = tmp0.shuffle[4, create_kgen_list[__mlir_type.index](0, 2, 4, 6)](
        tmp1
    )
    let r1 = tmp0.shuffle[4, create_kgen_list[__mlir_type.index](1, 3, 5, 7)](
        tmp1
    )
    let r2 = tmp2.shuffle[4, create_kgen_list[__mlir_type.index](0, 2, 4, 6)](
        tmp3
    )
    let r3 = tmp2.shuffle[4, create_kgen_list[__mlir_type.index](1, 3, 5, 7)](
        tmp3
    )

    buf.simd_store[4](_index2D(0, 0), r0)
    buf.simd_store[4](_index2D(1, 0), r1)
    buf.simd_store[4](_index2D(2, 0), r2)
    buf.simd_store[4](_index2D(3, 0), r3)


@implements(transpose_inplace)
fn transpose_inplace_generic[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](
    buf0: NDBuffer[
        2,
        __mlir_attr[
            `#kgen<list[`,
            rows,
            `, `,
            cols,
            `]> : `,
            `!kgen.list<`,
            __mlir_type.index,
            `[2]>`,
        ],
        type,
    ]
):
    assert_param[rank == 2]()
    var buf = __mlir_op.`kgen.rebind`[
        _type : NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](rows, cols),
            type,
        ]
    ](buf0)
    var i: Int = 0
    while i < rows:
        var j: Int = i + 1
        while j < cols:
            let pos = _index2D(i, j)
            let pos_tr = _index2D(j, i)
            let tmp = buf.__getitem__(pos)
            buf.__setitem__(pos, buf.__getitem__(pos_tr))
            buf.__setitem__(pos_tr, tmp)
            j += 1
        i += 1
