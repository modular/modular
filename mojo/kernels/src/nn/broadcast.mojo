# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import NDBuffer
from Int import Int
from Memory import memcpy
from Pointer import DTypePointer
from Range import range
from TargetInfo import sizeof


# ===----------------------------------------------------------------------===#
# _get_rightmost_broadcast_axis
# ===----------------------------------------------------------------------===#


fn _get_rightmost_broadcast_axis[
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
) -> Int:
    """
    Return the rightmost position (largest axis) at which the dimensions of
    `input_shape` and `output_shape` mismatch, otherwise return -1 (i.e., the
    shapes are equal).

    Args:
        output (NDBuffer): the output buffer
        input (NDBuffer): the input buffer
    """
    return _get_rightmost_broadcast_axis_impl[
        rank - 1, rank, output_shape, input_shape, type
    ](output, input)


@interface
fn _get_rightmost_broadcast_axis_impl[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
) -> Int:
    """
    Return the largest axis ∈ [0, axis] at which the dimensions of
    `input_shape` and `output_shape` mismatch, otherwise return -1 (i.e., the
    shapes are equal).

    Args:
        output (NDBuffer): the output buffer
        input (NDBuffer): the input buffer
    """
    ...


@implements(_get_rightmost_broadcast_axis_impl)
fn _get_rightmost_broadcast_axis_impl_base[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
) -> Int:
    """Base case for `_get_rightmost_broadcast_axis_impl`"""
    assert_param[axis < 0]()
    return -1


@implements(_get_rightmost_broadcast_axis_impl)
fn _get_rightmost_broadcast_axis_impl_iter[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
) -> Int:
    """Recursive case for `_get_rightmost_broadcast_axis_impl`"""
    assert_param[axis >= 0]()
    let in_dim = input.dim[axis]()
    let out_dim = output.dim[axis]()
    if in_dim != out_dim:
        return axis
    return _get_rightmost_broadcast_axis_impl[
        axis - 1, rank, output_shape, input_shape, type
    ](output, input)


# ===----------------------------------------------------------------------===#
# broadcast
# ===----------------------------------------------------------------------===#


fn broadcast[
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
):
    """
    For each axis of `input`, if the dimension is 1, duplicate the data at
    each index of the corresponding axis in `output`, otherwise copy over the
    entire axis to the corresponding axis in `output`.

    Args:
        output (NDBuffer): the output buffer
        input (NDBuffer): the input buffer
    """
    var rightmost_broadcast_axis: Int = _get_rightmost_broadcast_axis[
        rank, output_shape, input_shape, type
    ](output, input)

    let input_output_have_same_shape = rightmost_broadcast_axis == -1
    if input_output_have_same_shape:
        let src_ptr = input.data
        let dst_ptr = output.data
        let elem_size = sizeof[__mlir_type[`!pop.scalar<`, type, `>`]]()
        memcpy(dst_ptr, src_ptr, input.size() * elem_size)
    else:
        alias init_axis = 0
        # imaginary axis before 0
        let init_input_prev_axis_stride = input.size()
        let init_output_prev_axis_stride = output.size()
        broadcast_impl[init_axis, rank, output_shape, input_shape, type](
            output,
            input,
            init_input_prev_axis_stride,
            init_output_prev_axis_stride,
            0,  # input_offset
            0,  # output_offset
            rightmost_broadcast_axis,
        )


@interface
fn broadcast_impl[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    # using `prev` because otherwise computing `next_input_axis_stride` requires
    # dim[axis+1](), which requires more `assert_param` to keep in bound
    input_prev_axis_stride: Int,
    output_prev_axis_stride: Int,
    input_offset: Int,
    output_offset: Int,
    rightmost_broadcast_axis: Int,
):
    """
    For each axis of `input` ∈ [axis, rank), if the dimension is 1, duplicate the data at
    each index of the corresponding axis in `output`, otherwise copy over the
    entire axis to the corresponding axis in `output`.

    Args:
        output (NDBuffer): the output buffer
        input (NDBuffer): the input buffer
        input_prev_axis_stride(Int): the stride at axis `axis - 1` for input
        output_prev_axis_stride(Int): the stride at axis `axis - 1` for output
        input_offset(Int): the offset at which we start copying data from
        output_offset(Int): the offset at which we start copying data to
        rightmost_broadcast_axis(Int): the largest axis at which we need to duplicate `input` data.
    """
    ...


@implements(broadcast_impl)
fn broadcast_impl_base[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    input_axis_stride: Int,
    output_axis_stride: Int,
    input_offset: Int,
    output_offset: Int,
    rightmost_broadcast_axis: Int,
):
    """Base case for `broadcast_impl`"""
    assert_param[axis >= rank]()
    return


fn _tile_1d[
    type: __mlir_type.`!kgen.dtype`
](
    init_dst_ptr: DTypePointer[type],
    src_ptr: DTypePointer[type],
    tile_num_elems: Int,
    n: Int,
):
    """
    Repeat data from `src_ptr[:tile_num_elems]` in `init_dst_ptr` for `n` times
    """
    let elem_bytes = sizeof[__mlir_type[`!pop.scalar<`, type, `>`]]()
    let bytes_to_copy = tile_num_elems * elem_bytes
    var dst_ptr = init_dst_ptr
    for i in range(n):
        memcpy(dst_ptr, src_ptr, bytes_to_copy)
        dst_ptr = dst_ptr.offset(tile_num_elems)


@implements(broadcast_impl)
fn broadcast_impl_iter[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    input_prev_axis_stride: Int,
    output_prev_axis_stride: Int,
    input_offset: Int,
    output_offset: Int,
    rightmost_broadcast_axis: Int,
):
    """Recursive case for `broadcast_impl`"""
    assert_param[axis < rank]()
    let input_axis_stride = input_prev_axis_stride // input.dim[axis]()
    let output_axis_stride = output_prev_axis_stride // output.dim[axis]()
    if Int(axis) == rightmost_broadcast_axis:
        let elems_to_copy = input_axis_stride
        _tile_1d(
            output.data.offset(output_offset),
            input.data.offset(input_offset),
            input_axis_stride,  # elems_to_copy
            output.dim[axis](),
        )
    else:
        alias next_axis = axis + 1
        var next_input_offset = input_offset
        var next_output_offset = output_offset
        for i in range(input.dim[axis]()):
            broadcast_impl[next_axis, rank, output_shape, input_shape, type](
                output,
                input,
                input_axis_stride,
                output_axis_stride,
                next_input_offset,
                next_output_offset,
                rightmost_broadcast_axis,
            )
            next_input_offset += input_axis_stride
            next_output_offset += output_axis_stride
        # dupicate data in output, e.g.,
        #  broadcast([[1]]), shape (1, 1) to shape (2, 3):
        #     [[0, 0, 0], [0, 0, 0]]
        # --> [[1, 1, 1], [0, 0, 0]]   after recursive call to next axis
        # --> [[1, 1, 1], [1, 1, 1]]   after duplicating data in output
        if input.dim[axis]() != output.dim[axis]():
            let output_tile_start = output.data.offset(output_offset)
            _tile_1d(
                output_tile_start.offset(
                    output_axis_stride
                ),  # 1st tile is already there
                output_tile_start,
                output_axis_stride,  # elems_to_copy
                output.dim[axis]() - 1,  # 1st tile is already there
            )
