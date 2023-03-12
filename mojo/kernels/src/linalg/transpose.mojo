# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import Buffer, NDBuffer
from Tuple import StaticTuple
from Assert import assert_param, debug_assert
from Int import Int
from Index import StaticIntTuple
from SIMD import SIMD
from List import create_kgen_list
from Pointer import DTypePointer
from DType import DType
from Range import range
from TypeUtilities import rebind
from Functional import unroll


@adaptive
fn transpose_inplace[
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: DType,
](buf0: NDBuffer[2, create_kgen_list[__mlir_type.index](rows, cols), type]):
    assert_param[rows == 4]()
    assert_param[cols == 4]()
    var buf = rebind[
        NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](4, 4),
            type,
        ],
    ](buf0)

    let row0 = buf.simd_load[4](StaticIntTuple[2](0, 0))
    let row1 = buf.simd_load[4](StaticIntTuple[2](1, 0))
    let row2 = buf.simd_load[4](StaticIntTuple[2](2, 0))
    let row3 = buf.simd_load[4](StaticIntTuple[2](3, 0))

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

    buf.simd_store[4](StaticIntTuple[2](0, 0), r0)
    buf.simd_store[4](StaticIntTuple[2](1, 0), r1)
    buf.simd_store[4](StaticIntTuple[2](2, 0), r2)
    buf.simd_store[4](StaticIntTuple[2](3, 0), r3)


@adaptive
fn transpose_inplace[
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: DType,
](buf0: NDBuffer[2, create_kgen_list[__mlir_type.index](rows, cols), type]):
    assert_param[rows == 8]()
    assert_param[cols == 8]()
    var buf = rebind[
        NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](8, 8),
            type,
        ],
    ](buf0)

    let row0 = buf.simd_load[8](StaticIntTuple[2](0, 0))
    let row1 = buf.simd_load[8](StaticIntTuple[2](1, 0))
    let row2 = buf.simd_load[8](StaticIntTuple[2](2, 0))
    let row3 = buf.simd_load[8](StaticIntTuple[2](3, 0))
    let row4 = buf.simd_load[8](StaticIntTuple[2](4, 0))
    let row5 = buf.simd_load[8](StaticIntTuple[2](5, 0))
    let row6 = buf.simd_load[8](StaticIntTuple[2](6, 0))
    let row7 = buf.simd_load[8](StaticIntTuple[2](7, 0))

    alias permute_0 = create_kgen_list[__mlir_type.index](
        0, 8, 1, 9, 4, 12, 5, 13
    )
    alias permute_1 = create_kgen_list[__mlir_type.index](
        2, 10, 3, 11, 6, 14, 7, 15
    )

    let k0 = row0.shuffle[8, permute_0](row1)
    let k1 = row0.shuffle[8, permute_1](row1)
    let k2 = row2.shuffle[8, permute_0](row3)
    let k3 = row2.shuffle[8, permute_1](row3)
    let k4 = row4.shuffle[8, permute_0](row5)
    let k5 = row4.shuffle[8, permute_1](row5)
    let k6 = row6.shuffle[8, permute_0](row7)
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

    buf.simd_store[8](StaticIntTuple[2](0, 0), r0)
    buf.simd_store[8](StaticIntTuple[2](1, 0), r1)
    buf.simd_store[8](StaticIntTuple[2](2, 0), r2)
    buf.simd_store[8](StaticIntTuple[2](3, 0), r3)
    buf.simd_store[8](StaticIntTuple[2](4, 0), r4)
    buf.simd_store[8](StaticIntTuple[2](5, 0), r5)
    buf.simd_store[8](StaticIntTuple[2](6, 0), r6)
    buf.simd_store[8](StaticIntTuple[2](7, 0), r7)


@adaptive
fn transpose_inplace[
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: DType,
](buf0: NDBuffer[2, create_kgen_list[__mlir_type.index](rows, cols), type]):
    assert_param[rows == 16]()
    assert_param[cols == 16]()
    var buf = rebind[
        NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](16, 16),
            type,
        ],
    ](buf0)

    alias permute_0 = create_kgen_list[__mlir_type.index](
        0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29
    )
    alias permute_1 = create_kgen_list[__mlir_type.index](
        2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31
    )
    alias permute_2 = create_kgen_list[__mlir_type.index](
        0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
    )
    alias permute_3 = create_kgen_list[__mlir_type.index](
        2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
    )
    alias permute_4 = create_kgen_list[__mlir_type.index](
        0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
    )
    alias permute_5 = create_kgen_list[__mlir_type.index](
        4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
    )
    alias permute_6 = create_kgen_list[__mlir_type.index](
        0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
    )
    alias permute_7 = create_kgen_list[__mlir_type.index](
        4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
    )

    let row00 = buf.simd_load[16](StaticIntTuple[2](0, 0))
    let row01 = buf.simd_load[16](StaticIntTuple[2](1, 0))
    let row02 = buf.simd_load[16](StaticIntTuple[2](2, 0))
    let row03 = buf.simd_load[16](StaticIntTuple[2](3, 0))
    let row04 = buf.simd_load[16](StaticIntTuple[2](4, 0))
    let row05 = buf.simd_load[16](StaticIntTuple[2](5, 0))
    let row06 = buf.simd_load[16](StaticIntTuple[2](6, 0))
    let row07 = buf.simd_load[16](StaticIntTuple[2](7, 0))
    let row08 = buf.simd_load[16](StaticIntTuple[2](8, 0))
    let row09 = buf.simd_load[16](StaticIntTuple[2](9, 0))
    let row10 = buf.simd_load[16](StaticIntTuple[2](10, 0))
    let row11 = buf.simd_load[16](StaticIntTuple[2](11, 0))
    let row12 = buf.simd_load[16](StaticIntTuple[2](12, 0))
    let row13 = buf.simd_load[16](StaticIntTuple[2](13, 0))
    let row14 = buf.simd_load[16](StaticIntTuple[2](14, 0))
    let row15 = buf.simd_load[16](StaticIntTuple[2](15, 0))

    let k00 = row00.shuffle[16, permute_0](row01)
    let k01 = row00.shuffle[16, permute_1](row01)
    let k02 = row02.shuffle[16, permute_0](row03)
    let k03 = row02.shuffle[16, permute_1](row03)
    let k04 = row04.shuffle[16, permute_0](row05)
    let k05 = row04.shuffle[16, permute_1](row05)
    let k06 = row06.shuffle[16, permute_0](row07)
    let k07 = row06.shuffle[16, permute_1](row07)
    let k08 = row08.shuffle[16, permute_0](row09)
    let k09 = row08.shuffle[16, permute_1](row09)
    let k10 = row10.shuffle[16, permute_0](row11)
    let k11 = row10.shuffle[16, permute_1](row11)
    let k12 = row12.shuffle[16, permute_0](row13)
    let k13 = row12.shuffle[16, permute_1](row13)
    let k14 = row14.shuffle[16, permute_0](row15)
    let k15 = row14.shuffle[16, permute_1](row15)

    let j00 = k00.shuffle[16, permute_2](k02)
    let j01 = k00.shuffle[16, permute_3](k02)
    let j02 = k01.shuffle[16, permute_2](k03)
    let j03 = k01.shuffle[16, permute_3](k03)
    let j04 = k04.shuffle[16, permute_2](k06)
    let j05 = k04.shuffle[16, permute_3](k06)
    let j06 = k05.shuffle[16, permute_2](k07)
    let j07 = k05.shuffle[16, permute_3](k07)
    let j08 = k08.shuffle[16, permute_2](k10)
    let j09 = k08.shuffle[16, permute_3](k10)
    let j10 = k09.shuffle[16, permute_2](k11)
    let j11 = k09.shuffle[16, permute_3](k11)
    let j12 = k12.shuffle[16, permute_2](k14)
    let j13 = k12.shuffle[16, permute_3](k14)
    let j14 = k13.shuffle[16, permute_2](k15)
    let j15 = k13.shuffle[16, permute_3](k15)

    let t00 = j00.shuffle[16, permute_4](j04)
    let t01 = j01.shuffle[16, permute_4](j05)
    let t02 = j02.shuffle[16, permute_4](j06)
    let t03 = j03.shuffle[16, permute_4](j07)
    let t04 = j00.shuffle[16, permute_5](j04)
    let t05 = j01.shuffle[16, permute_5](j05)
    let t06 = j02.shuffle[16, permute_5](j06)
    let t07 = j03.shuffle[16, permute_5](j07)
    let t08 = j08.shuffle[16, permute_4](j12)
    let t09 = j09.shuffle[16, permute_4](j13)
    let t10 = j10.shuffle[16, permute_4](j14)
    let t11 = j11.shuffle[16, permute_4](j15)
    let t12 = j08.shuffle[16, permute_5](j12)
    let t13 = j09.shuffle[16, permute_5](j13)
    let t14 = j10.shuffle[16, permute_5](j14)
    let t15 = j11.shuffle[16, permute_5](j15)

    let r00 = t00.shuffle[16, permute_6](t08)
    let r01 = t01.shuffle[16, permute_6](t09)
    let r02 = t02.shuffle[16, permute_6](t10)
    let r03 = t03.shuffle[16, permute_6](t11)
    let r04 = t04.shuffle[16, permute_6](t12)
    let r05 = t05.shuffle[16, permute_6](t13)
    let r06 = t06.shuffle[16, permute_6](t14)
    let r07 = t07.shuffle[16, permute_6](t15)
    let r08 = t00.shuffle[16, permute_7](t08)
    let r09 = t01.shuffle[16, permute_7](t09)
    let r10 = t02.shuffle[16, permute_7](t10)
    let r11 = t03.shuffle[16, permute_7](t11)
    let r12 = t04.shuffle[16, permute_7](t12)
    let r13 = t05.shuffle[16, permute_7](t13)
    let r14 = t06.shuffle[16, permute_7](t14)
    let r15 = t07.shuffle[16, permute_7](t15)

    buf.simd_store[16](StaticIntTuple[2](0, 0), r00)
    buf.simd_store[16](StaticIntTuple[2](1, 0), r01)
    buf.simd_store[16](StaticIntTuple[2](2, 0), r02)
    buf.simd_store[16](StaticIntTuple[2](3, 0), r03)
    buf.simd_store[16](StaticIntTuple[2](4, 0), r04)
    buf.simd_store[16](StaticIntTuple[2](5, 0), r05)
    buf.simd_store[16](StaticIntTuple[2](6, 0), r06)
    buf.simd_store[16](StaticIntTuple[2](7, 0), r07)
    buf.simd_store[16](StaticIntTuple[2](8, 0), r08)
    buf.simd_store[16](StaticIntTuple[2](9, 0), r09)
    buf.simd_store[16](StaticIntTuple[2](10, 0), r10)
    buf.simd_store[16](StaticIntTuple[2](11, 0), r11)
    buf.simd_store[16](StaticIntTuple[2](12, 0), r12)
    buf.simd_store[16](StaticIntTuple[2](13, 0), r13)
    buf.simd_store[16](StaticIntTuple[2](14, 0), r14)
    buf.simd_store[16](StaticIntTuple[2](15, 0), r15)


@adaptive
fn transpose_inplace[
    rows: __mlir_type.index,
    cols: __mlir_type.index,
    type: DType,
](buf: NDBuffer[2, create_kgen_list[__mlir_type.index](rows, cols), type]):
    for i in range(rows):
        for j in range(i + 1, cols):
            let tmp = buf[i, j]
            buf[StaticIntTuple[2](i, j)] = buf[j, i]
            buf[StaticIntTuple[2](j, i)] = tmp


fn _permute_data[
    size: __mlir_type.index,
    type: DType,
](
    input: DTypePointer[type],
    output: DTypePointer[type],
    perms: DTypePointer[DType.index],
):
    """
    Ensures that output[i] = input[perms[i]] for i ∈ [0, size)
    """
    for axis in range(size):
        let perm_axis = perms.load(axis)[0].value
        let perm_data = input.load(perm_axis)
        output.store(axis, perm_data)


fn _fill_strides[
    rank: __mlir_type.index,
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: DType,
](buf: NDBuffer[rank, input_shape, type], strides: Buffer[rank, DType.index],):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    assert_param[rank > 0]()
    strides[rank - 1] = 1

    @always_inline
    fn _fill_stride_at_idx[idx: Int]():
        alias axis = rank - idx.__as_mlir_index() - 2
        let next_axis_stride = strides[axis + 1]
        let next_axis_dim = buf.dim[axis + 1]()
        let curr_axis_stride = next_axis_stride * next_axis_dim
        strides[axis] = curr_axis_stride

    unroll[rank - 1, _fill_stride_at_idx]()


fn transpose[
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index],
):
    """
    Permute the axis of `input` based on `perms`, and place the result in
    `output`.

    Args:
        output (NDBuffer): the output buffer
        input (NDBuffer): the input buffer
        perms (DTypePointer): permutation of the input axes

    Example:
        transpose(output, input, [2, 0, 1])
        # guarantees output[x, y, z] = input[z, x, y]
    """
    # Compute `permuted_input_strides_buf`
    let input_strides_buf = Buffer[rank, DType.index].stack_allocation()
    let permuted_input_strides_buf = Buffer[
        rank, DType.index
    ].stack_allocation()
    _fill_strides(input, input_strides_buf)
    _permute_data[rank, DType.index](
        input_strides_buf.data, permuted_input_strides_buf.data, perms
    )
    # Compute `output_strides_buf `
    let output_strides_buf = Buffer[rank, DType.index].stack_allocation()
    _fill_strides(output, output_strides_buf)
    # Kickoff; for intuition on permuted input strides, note that
    #   transpose(output, input, [2, 0, 1])
    # guarantees
    #   (let isx denote input_stride_x, etc.)
    #   output[x, y, z] = input[z, x, y]
    # ~ output.at(offset(x*isx + y*isy + z*isz)) = input.at(offset(z*osx + x*osy + y*osz))
    # ~ output.at(offset(x*isx + y*isy + z*isz)) = input.at(offset(x*osy + y*osz + z*osx))
    # ~ output.at(offset([x, y, z], output_strides)) = input.at(offset([x, y, z], permuted_input_strides))
    # ~ output.at(offset(index, output_strides)) = input.at(offset(index, permuted_input_strides))
    alias init_axis = 0
    _copy_with_strides[rank, output_shape, type](
        init_axis,
        output,
        input.data,
        permuted_input_strides_buf.data,
        output_strides_buf.data,
        0,  # input_offset
        0,  # output_offset
    )


fn _copy_with_strides[
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: DType,
](
    axis: Int,
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    input_strides: DTypePointer[DType.index],
    output_strides: DTypePointer[DType.index],
    input_offset: Int,
    output_offset: Int,
):
    """
    Copy data from `input` to `output`, starting at corresponding offsets,
    based on given strides.

    Args:
        output (NDBuffer): the output buffer
        input (NDBuffer): the input buffer
        input_strides (DTypePointer): the stride at each input axis
        output_strides (DTypePointer): the stride at each output axis
        input_offset (Int): The offset at which input data starts
        output_offset (Int): The offset at which output data starts
    """
    debug_assert(axis + 1 <= rank, "out of range")

    let axis_dim = output.dim(axis)
    let input_axis_stride: Int = input_strides.load(axis)[0].value
    let output_axis_stride: Int = output_strides.load(axis)[0].value

    if axis + 1 == rank:
        # TODO speed this up if output_axis_stride is 1, i.e., contiguous?
        var src_ptr = input.offset(input_offset)
        var dst_ptr = output.data.offset(output_offset)
        for i in range(axis_dim):
            dst_ptr.store(0, src_ptr.load(0))
            src_ptr = src_ptr.offset(input_axis_stride)
            dst_ptr = dst_ptr.offset(output_axis_stride)
        return

    let next_axis = axis + 1
    var next_input_offset = input_offset
    var next_output_offset = output_offset
    for _ in range(axis_dim):
        _copy_with_strides[rank, output_shape, type](
            next_axis,
            output,
            input,
            input_strides,
            output_strides,
            next_input_offset,
            next_output_offset,
        )
        next_input_offset += input_axis_stride
        next_output_offset += output_axis_stride
