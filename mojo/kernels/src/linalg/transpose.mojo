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
from TypeUtilities import rebind


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
](buf: NDBuffer[2, create_kgen_list[__mlir_type.index](rows, cols), type]):
    ...


@implements(transpose_inplace)
fn transpose_inplace_4x4[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](buf0: NDBuffer[2, create_kgen_list[__mlir_type.index](rows, cols), type]):
    assert_param[rank == 2]()
    assert_param[rows == 4]()
    assert_param[cols == 4]()
    var buf = rebind[
        NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](rows, cols),
            type,
        ],
        NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](4, 4),
            type,
        ],
    ](buf0)

    let row0 = buf.simd_load[4](_index2D(0, 0))
    let row1 = buf.simd_load[4](_index2D(1, 0))
    let row2 = buf.simd_load[4](_index2D(2, 0))
    let row3 = buf.simd_load[4](_index2D(3, 0))

    let tmp0 = row0.shuffle[4, create_kgen_list[__mlir_type.index](0, 1, 4, 5)](
        row1
    )
    let tmp1 = row2.shuffle[4, create_kgen_list[__mlir_type.index](0, 1, 4, 5)](
        row3
    )
    let tmp2 = row0.shuffle[4, create_kgen_list[__mlir_type.index](2, 3, 6, 7)](
        row1
    )
    let tmp3 = row2.shuffle[4, create_kgen_list[__mlir_type.index](2, 3, 6, 7)](
        row3
    )

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
fn transpose_inplace_8x8[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](buf0: NDBuffer[2, create_kgen_list[__mlir_type.index](rows, cols), type]):
    assert_param[rank == 2]()
    assert_param[rows == 8]()
    assert_param[cols == 8]()
    var buf = rebind[
        NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](rows, cols),
            type,
        ],
        NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](8, 8),
            type,
        ],
    ](buf0)

    let row0 = buf.simd_load[8](_index2D(0, 0))
    let row1 = buf.simd_load[8](_index2D(1, 0))
    let row2 = buf.simd_load[8](_index2D(2, 0))
    let row3 = buf.simd_load[8](_index2D(3, 0))
    let row4 = buf.simd_load[8](_index2D(4, 0))
    let row5 = buf.simd_load[8](_index2D(5, 0))
    let row6 = buf.simd_load[8](_index2D(6, 0))
    let row7 = buf.simd_load[8](_index2D(7, 0))

    alias premute_0 = create_kgen_list[__mlir_type.index](
        0, 8, 1, 9, 4, 12, 5, 13
    )
    alias permute_1 = create_kgen_list[__mlir_type.index](
        2, 10, 3, 11, 6, 14, 7, 15
    )

    let k0 = row0.shuffle[8, premute_0](row1)
    let k1 = row0.shuffle[8, permute_1](row1)
    let k2 = row2.shuffle[8, premute_0](row3)
    let k3 = row2.shuffle[8, permute_1](row3)
    let k4 = row4.shuffle[8, premute_0](row5)
    let k5 = row4.shuffle[8, permute_1](row5)
    let k6 = row6.shuffle[8, premute_0](row7)
    let k7 = row6.shuffle[8, permute_1](row7)

    alias permute_2 = create_kgen_list[__mlir_type.index](
        0, 1, 8, 9, 4, 5, 12, 13
    )
    alias permute_3 = create_kgen_list[__mlir_type.index](
        2, 3, 10, 11, 6, 7, 14, 15
    )

    let k020 = k0.shuffle[8, permute_2](k2)
    let k021 = k0.shuffle[8, permute_3](k2)
    let k130 = k1.shuffle[8, permute_2](k3)
    let k131 = k1.shuffle[8, permute_3](k3)
    let k460 = k4.shuffle[8, permute_2](k6)
    let k461 = k4.shuffle[8, permute_3](k6)
    let k570 = k5.shuffle[8, permute_2](k7)
    let k571 = k5.shuffle[8, permute_3](k7)

    alias permute_4 = create_kgen_list[__mlir_type.index](
        0, 1, 2, 3, 8, 9, 10, 11
    )
    alias permute_5 = create_kgen_list[__mlir_type.index](
        4, 5, 6, 7, 12, 13, 14, 15
    )

    let r0 = k020.shuffle[8, permute_4](k460)
    let r1 = k021.shuffle[8, permute_4](k461)
    let r2 = k130.shuffle[8, permute_4](k570)
    let r3 = k131.shuffle[8, permute_4](k571)
    let r4 = k020.shuffle[8, permute_5](k460)
    let r5 = k021.shuffle[8, permute_5](k461)
    let r6 = k130.shuffle[8, permute_5](k570)
    let r7 = k131.shuffle[8, permute_5](k571)

    buf.simd_store[8](_index2D(0, 0), r0)
    buf.simd_store[8](_index2D(1, 0), r1)
    buf.simd_store[8](_index2D(2, 0), r2)
    buf.simd_store[8](_index2D(3, 0), r3)
    buf.simd_store[8](_index2D(4, 0), r4)
    buf.simd_store[8](_index2D(5, 0), r5)
    buf.simd_store[8](_index2D(6, 0), r6)
    buf.simd_store[8](_index2D(7, 0), r7)


@implements(transpose_inplace)
fn transpose_inplace_generic[
    rank: __mlir_type.index,
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](buf: NDBuffer[2, create_kgen_list[__mlir_type.index](rows, cols), type]):
    assert_param[rank == 2]()
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
