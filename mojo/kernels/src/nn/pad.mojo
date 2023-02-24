# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------===#
# pad
# ===----------------------------------------------------------------------===#

from Assert import assert_param

from Buffer import Buffer, NDBuffer
from DType import DType
from Functional import vectorize
from Index import Index
from Int import Int
from List import create_kgen_list_unknown
from Memory import memcpy
from Pointer import DTypePointer
from SIMD import SIMD

# TODO Refactor -- we should decide on and put them into a more common file
from Transpose import _fill_strides
from TargetInfo import dtype_simd_width, sizeof
from Range import range
from IO import print


# TODO this probably belongs to `Memory.lit`
fn _fill[
    type: __mlir_type.`!kgen.dtype`
](dst: DTypePointer[type], value: SIMD[1, type], count: Int):
    @always_inline
    fn _set[simd_width: __mlir_type.index](idx: Int):
        let splat_val = SIMD.splat[simd_width, type](value)
        dst.simd_store(idx, splat_val)

    alias vector_width = dtype_simd_width[type]().__as_mlir_index()
    vectorize[vector_width, _set](count)


fn pad[
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    input_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    paddings: DTypePointer[DType.index.value],
    constant: SIMD[1, type],
):
    """
    Fill `output` with values from `input`, and edges padded with `constant`
    based on `paddings`.

    Args:
        output (NDBuffer): the output buffer
        input (NDBuffer): the input buffer
        paddings: (DTypePointer): ordered (before, after) padding sizes for each axis
        constant: (SIMD): the constant to pad output with

    Example:
        let input_shape = (X, Y, Z)
        let paddings = [x0, x1, y0, y1, z0, z1]

        out[x, y, z] =
          input[x - x0, y - y0, z - z0] if x ∈ [x0, x0 + X] && y ∈ [y0, y0 + Y] && z ∈ [z0, z0 + Z]
          else constant
    """

    let input_strides_buf = Buffer[rank, DType.index.value].stack_allocation()
    let output_strides_buf = Buffer[rank, DType.index.value].stack_allocation()
    _fill_strides(input, input_strides_buf)
    _fill_strides(output, output_strides_buf)

    alias init_axis = 0
    _pad_impl[init_axis, rank, output_shape, type](
        output,
        input.data,
        paddings,
        constant,
        output_strides_buf.data,
        input_strides_buf.data,
        0,  # output_offset
        0,  # input_offset
        False,  # pad_with_constant
    )


@interface
fn _pad_impl[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    paddings: DTypePointer[DType.index.value],
    constant: SIMD[1, type],
    output_strides: DTypePointer[DType.index.value],
    input_strides: DTypePointer[DType.index.value],
    output_offset: Int,
    input_offset: Int,
    pad_with_constant: Bool,
):
    """
    Fill axis ∈ [axis, rank) in `output` with values from `input`, and edges
    padded with `constant` based on `paddings`.

    Args:
        output (NDBuffer): the output buffer
        input (DTypePointer): the input buffer
        paddings: (DTypePointer): the (before, after) padding sizes for each axis
        constant: (SIMD): the constant to pad output with
        output_strides (DTypePointer): the stride at each output axis
        input_strides (DTypePointer): the stride at each input axis
        output_offset (Int): The offset at which output data starts
        input_offset (Int): The offset at which input data starts
        pad_with_constant (Bool): whether to always pad remaining region with constant
    """
    ...


@implements(_pad_impl)
fn _pad_impl_base[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    paddings: DTypePointer[DType.index.value],
    constant: SIMD[1, type],
    output_strides: DTypePointer[DType.index.value],
    input_strides: DTypePointer[DType.index.value],
    output_offset: Int,
    input_offset: Int,
    pad_with_constant: Bool,
):
    """Base case for `_pad_impl`"""
    assert_param[axis + 1 == rank]()

    # meta
    let axis_dim = output.dim[axis]()
    let pre_pad = paddings.load(2 * axis).value
    let post_pad = paddings.load(2 * axis + 1).value
    let non_pad = axis_dim - pre_pad - post_pad

    # pointers
    let pre_pad_start_ptr = output.data.offset(output_offset)
    let non_pad_start_ptr = pre_pad_start_ptr.offset(pre_pad)
    let post_pad_start_ptr = non_pad_start_ptr.offset(non_pad)
    let input_start_ptr = input.offset(input_offset)

    # setting values
    if pad_with_constant:
        _fill(pre_pad_start_ptr, constant.value, axis_dim)
    else:
        _fill(pre_pad_start_ptr, constant.value, pre_pad)
        let elem_bytes = sizeof[__mlir_type[`!pop.scalar<`, type, `>`]]()
        let non_pad_bytes = non_pad * elem_bytes
        memcpy(non_pad_start_ptr, input_start_ptr, non_pad_bytes)
        _fill(post_pad_start_ptr, constant.value, post_pad)


@implements(_pad_impl)
fn _pad_impl_iter[
    axis: __mlir_type.index,
    rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
](
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    paddings: DTypePointer[DType.index.value],
    constant: SIMD[1, type],
    output_strides: DTypePointer[DType.index.value],
    input_strides: DTypePointer[DType.index.value],
    output_offset: Int,
    input_offset: Int,
    pad_with_constant: Bool,
):
    """Recursive case for `_pad_impl`"""
    assert_param[axis + 1 < rank]()

    # meta
    let axis_dim = output.dim[axis]()
    let pre_pad = paddings.load(2 * axis).value
    let post_pad = paddings.load(2 * axis + 1).value
    let non_pad = axis_dim - pre_pad - post_pad
    let input_axis_stride = input_strides.load(axis)[0]
    let output_axis_stride = output_strides.load(axis)[0]

    alias next_axis = axis + 1
    var next_input_offset = input_offset
    var next_output_offset = output_offset
    for i in range(axis_dim):
        let is_within_padding = (i < pre_pad) or (pre_pad + non_pad <= i)
        let next_pad_with_constant = pad_with_constant or is_within_padding
        _pad_impl[next_axis, rank, output_shape, type](
            output,
            input,
            paddings,
            constant,
            output_strides,
            input_strides,
            next_output_offset,
            next_input_offset,
            next_pad_with_constant,
        )
        if not is_within_padding:
            next_input_offset += input_axis_stride
        next_output_offset += output_axis_stride
