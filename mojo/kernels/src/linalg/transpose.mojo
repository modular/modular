# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import Buffer, NDBuffer
from Tuple import StaticTuple
from Assert import assert_param
from Int import Int
from SIMD import SIMD
from List import create_kgen_list
from Pointer import DTypePointer
from DType import DType
from Range import range
from TypeUtilities import rebind
from Functional import unroll


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
    for i in range(rows):
        for j in range(i + 1, cols):
            let tmp = buf[i, j]
            buf.__setitem__(_index2D(i, j), buf[j, i])
            buf.__setitem__(_index2D(j, i), tmp)


fn _permute_data[
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.dtype`,
](
    input: DTypePointer[type],
    output: DTypePointer[type],
    perms: DTypePointer[DType.index.value],
):
    """
    Ensures that output[i] = input[perms[i]] for i ∈ [0, size)
    """
    for axis in range(size):
        let perm_axis = perms.load(axis)[0]
        let perm_data = input.load(perm_axis)
        output.store(axis, perm_data)


fn _fill_strides[
    rank: __mlir_type.index,
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    buf: NDBuffer[rank, input_shape, type],
    strides: Buffer[rank, DType.index.value],
):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    assert_param[rank > 0]()
    strides.__setitem__(rank - 1, 1)

    @always_inline
    fn _fill_stride_at_idx[idx: __mlir_type.index]():
        alias axis = rank - idx - 2
        let next_axis_stride = strides[axis + 1]
        let next_axis_dim = buf.dim[axis + 1]()
        let curr_axis_stride = next_axis_stride * next_axis_dim
        strides.__setitem__(axis, curr_axis_stride)

    unroll[rank - 1, _fill_stride_at_idx]()


fn transpose[
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    perms: DTypePointer[DType.index.value],
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
    let input_strides_buf = Buffer[rank, DType.index.value].stack_allocation()
    let permuted_input_strides_buf = Buffer[
        rank, DType.index.value
    ].stack_allocation()
    _fill_strides[rank, input_shape, type](input, input_strides_buf)
    _permute_data[rank, DType.index.value](
        input_strides_buf.data, permuted_input_strides_buf.data, perms
    )
    # Compute `output_strides_buf `
    let output_strides_buf = Buffer[rank, DType.index.value].stack_allocation()
    _fill_strides[rank, output_shape, type](output, output_strides_buf)
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
    _copy_with_strides[init_axis, rank, output_shape, type](
        output,
        input.data,
        permuted_input_strides_buf.data,
        output_strides_buf.data,
        0,  # input_offset
        0,  # output_offset
    )


@interface
fn _copy_with_strides[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    input_strides: DTypePointer[DType.index.value],
    output_strides: DTypePointer[DType.index.value],
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
    ...


@implements(_copy_with_strides)
fn _copy_with_strides_base[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    input_strides: DTypePointer[DType.index.value],
    output_strides: DTypePointer[DType.index.value],
    input_offset: Int,
    output_offset: Int,
):
    """Base case for `_copy_with_strides`"""
    assert_param[axis + 1 == rank]()

    # TODO speed this up if output_axis_stride is 1, i.e., contiguous?
    let axis_dim = output.dim[axis]()
    let output_axis_stride: Int = output_strides.load(axis)[0]
    let input_axis_stride: Int = input_strides.load(axis)[0]
    var src_ptr = input.offset(input_offset)
    var dst_ptr = output.data.offset(output_offset)
    for _ in range(axis_dim):
        dst_ptr.store(0, src_ptr.load(0))
        src_ptr = src_ptr.offset(input_axis_stride)
        dst_ptr = dst_ptr.offset(output_axis_stride)
    return


@implements(_copy_with_strides)
fn _copy_with_strides_iter[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    input_strides: DTypePointer[DType.index.value],
    output_strides: DTypePointer[DType.index.value],
    input_offset: Int,
    output_offset: Int,
):
    """Recursive case for `_copy_with_strides`"""
    assert_param[axis + 1 < rank]()

    let axis_dim = output.dim[axis]()
    let input_axis_stride = input_strides.load(axis)[0]
    let output_axis_stride = output_strides.load(axis)[0]
    alias next_axis = axis + 1
    var next_input_offset = input_offset
    var next_output_offset = output_offset
    for _ in range(axis_dim):
        _copy_with_strides[next_axis, rank, output_shape, type](
            output,
            input,
            input_strides,
            output_strides,
            next_input_offset,
            next_output_offset,
        )
        next_input_offset += input_axis_stride
        next_output_offset += output_axis_stride
