# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from os import abort

import benchmark
from benchmark import Unit, keep
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from internal_utils import assert_equal
from memory import UnsafePointer, memcpy
from nn.pad import _AxisParams, _do_pad, _fill, pad_constant, pad_reflect
from python import Python
from testing import assert_true

from utils import IndexList, StaticTuple, unroll


@always_inline
fn pad_constant_dispatch[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
    paddings_type: DType,
    constant_type: DType,
    recursive: Int = 0,
](
    output: NDBuffer[mut=True, type, rank, _, output_shape],
    input: NDBuffer[type, rank, _, input_shape],
    paddings: UnsafePointer[Scalar[paddings_type]],
    constant: SIMD[constant_type, 1],
):
    """
    Fill `output` with values from `input`, and edges padded with `constant`
    based on `paddings`.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.
        constant: The constant to pad output with.

    Example:
        var input_shape = (X, Y, Z)
        var paddings = [x0, x1, y0, y1, z0, z1]

        out[x, y, z] =
        input[x - x0, y - y0, z - z0] if x ∈ [x0, x0 + X] &&
                                        y ∈ [y0, y0 + Y] &&
                                        z ∈ [z0, z0 + Z]
        else constant
    """

    @parameter
    if recursive:
        alias init_axis = 0
        var constant_cast = rebind[Scalar[type]](constant[0])

        @__copy_capture(constant_cast)
        @parameter
        fn pad_constant_wrapper(
            output: UnsafePointer[Scalar[type]],
            input: UnsafePointer[Scalar[type]],
            paddings: UnsafePointer[Scalar[paddings_type]],
            output_shape: IndexList[rank],
            output_strides: UnsafePointer[Scalar[DType.index]],
            input_strides: UnsafePointer[Scalar[DType.index]],
        ):
            return _pad_constant_impl_rec[rank, type, paddings_type](
                init_axis,
                output,
                input,
                paddings,
                constant_cast,
                output_shape,
                output_strides,
                input_strides,
                0,  # output_offset
                0,  # input_offset
                False,  # is_fully_padded
            )

        return _do_pad[
            rank,
            output_shape,
            input_shape,
            type,
            paddings_type,
            pad_constant_wrapper,
        ](output, input, paddings)
    else:
        return pad_constant(output, input, paddings, constant)


fn _pad_constant_impl_rec[
    rank: Int,
    type: DType,
    paddings_type: DType,
](
    axis: Int,
    output: UnsafePointer[Scalar[type]],
    input: UnsafePointer[Scalar[type]],
    paddings: UnsafePointer[Scalar[paddings_type]],
    constant: Scalar[type],
    output_shape: IndexList[rank],
    output_strides: UnsafePointer[Scalar[DType.index]],
    input_strides: UnsafePointer[Scalar[DType.index]],
    output_offset: Int,
    input_offset: Int,
    pad_with_constant: Bool,
):
    """
    Fill axis ∈ [axis, rank) in `output` with values from `input`, and edges
    padded with `constant` based on `paddings`.

    Args:
        axis: The axis to operate on.
        output: The output buffer.
        input: The input buffer.
        paddings: The (before, after) padding sizes for each axis.
        constant: the constant to pad output with.
        output_shape: the dynamic shape of the tensor pointed to by output buffer
        output_strides: the stride at each output axis.
        input_strides: the stride at each input axis.
        output_offset: The offset at which output data starts.
        input_offset: The offset at which input data starts.
        pad_with_constant: whether to always pad remaining region with constant.
    """

    # TODO this recursion can add up for larger tensors, optimize in #24565

    var axis_dim = output_shape[axis]
    var pre_pad = Int(paddings[2 * axis])
    var post_pad = Int(paddings[2 * axis + 1])
    var non_pad = axis_dim - pre_pad - post_pad

    if axis + 1 == rank:
        # pointers
        var pre_pad_start_ptr = output.offset(output_offset)
        var non_pad_start_ptr = pre_pad_start_ptr.offset(pre_pad)
        var post_pad_start_ptr = non_pad_start_ptr.offset(non_pad)
        var input_start_ptr = input.offset(input_offset)

        # setting values
        if pad_with_constant:
            _fill(pre_pad_start_ptr, constant, axis_dim)
            return

        _fill(pre_pad_start_ptr, constant, pre_pad)
        memcpy(non_pad_start_ptr, input_start_ptr, non_pad)
        _fill(post_pad_start_ptr, constant, post_pad)
        return

    debug_assert(axis + 1 < rank, "axis is not within range")

    var input_axis_stride = Int(input_strides[axis])
    var output_axis_stride = Int(output_strides[axis])

    var next_input_offset: Int = input_offset
    var next_output_offset: Int = output_offset
    for i in range(axis_dim):
        var is_within_padding = (i < pre_pad) or (pre_pad + non_pad <= i)
        var next_pad_with_constant = pad_with_constant or is_within_padding
        _pad_constant_impl_rec(
            axis + 1,
            output,
            input,
            paddings,
            constant,
            output_shape,
            output_strides,
            input_strides,
            next_output_offset,
            next_input_offset,
            next_pad_with_constant,
        )
        if not is_within_padding:
            next_input_offset += input_axis_stride
        next_output_offset += output_axis_stride


@always_inline
fn pad_reflect_dispatch[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
    paddings_type: DType,
    recursive: Int = 0,
](
    output: NDBuffer[mut=True, type, rank, _, output_shape],
    input: NDBuffer[type, rank, _, input_shape],
    paddings: UnsafePointer[Scalar[paddings_type]],
):
    """
    Fill `output` with values from `input`, and edges padded with reflected
    values from the unpadded region.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.

    Example:
        var input = [[1, 2],
                     [3, 4]]
        var paddings = [2, 2, 1, 0]

        Yields:
        output = [[2, 1, 2],
                  [4, 3, 4],
                  [2, 1, 2],
                  [4, 3, 4],
                  [2, 1, 2],
                  [4, 3, 4]]
    """

    @parameter
    if recursive:
        alias init_axis = 0

        @parameter
        fn pad_reflect_wrapper(
            output: UnsafePointer[Scalar[type]],
            input: UnsafePointer[Scalar[type]],
            paddings: UnsafePointer[Scalar[paddings_type]],
            output_shape: IndexList[rank],
            output_strides: UnsafePointer[Scalar[DType.index]],
            input_strides: UnsafePointer[Scalar[DType.index]],
        ):
            return _pad_reflect_impl_rec[rank, type, paddings_type](
                init_axis,
                output,
                input,
                paddings,
                output_shape,
                output_strides,
                input_strides,
                0,  # output_offset
                0,  # input_offset
            )

        return _do_pad[
            rank,
            output_shape,
            input_shape,
            type,
            paddings_type,
            pad_reflect_wrapper,
        ](output, input, paddings)
    else:
        return pad_reflect(output, input, paddings)


@always_inline
fn _memcpy_regions[
    type: DType
](
    pre_pad: Int,
    post_pad: Int,
    non_pad: Int,
    output_axis_stride: Int,
    pre_pad_start_ptr: UnsafePointer[Scalar[type]],
):
    # now memcpy the fully padded axes from the regions we just filled
    var curr_pre_pad = 0
    var curr_post_pad = 0
    while curr_pre_pad < pre_pad or curr_post_pad < post_pad:
        if curr_pre_pad < pre_pad:
            var copy_to = pre_pad - curr_pre_pad - 1
            var copy_to_ptr = pre_pad_start_ptr.offset(
                (copy_to * output_axis_stride)
            )

            var copy_from: Int
            if non_pad == 1:
                # handle singleton case
                copy_from = pre_pad
            else:
                copy_from = copy_to + ((curr_pre_pad % (non_pad - 1)) + 1) * 2

            var copy_from_ptr = pre_pad_start_ptr.offset(
                (copy_from * output_axis_stride)
            )
            memcpy(copy_to_ptr, copy_from_ptr, output_axis_stride)
            curr_pre_pad += 1

        if curr_post_pad < post_pad:
            var copy_to = pre_pad + non_pad + curr_post_pad
            var copy_to_ptr = pre_pad_start_ptr.offset(
                copy_to * output_axis_stride
            )

            var copy_from: Int
            if non_pad == 1:
                # handle singleton case
                copy_from = pre_pad
            else:
                copy_from = copy_to - ((curr_post_pad % (non_pad - 1)) + 1) * 2

            var copy_from_ptr = pre_pad_start_ptr.offset(
                copy_from * output_axis_stride
            )

            memcpy(copy_to_ptr, copy_from_ptr, output_axis_stride)
            curr_post_pad += 1


fn _pad_reflect_impl_rec[
    rank: Int,
    type: DType,
    paddings_type: DType,
](
    axis: Int,
    output: UnsafePointer[Scalar[type]],
    input: UnsafePointer[Scalar[type]],
    paddings: UnsafePointer[Scalar[paddings_type]],
    output_shape: IndexList[rank],
    output_strides: UnsafePointer[Scalar[DType.index]],
    input_strides: UnsafePointer[Scalar[DType.index]],
    output_offset: Int,
    input_offset: Int,
):
    """
    Fill axis ∈ [axis, rank) in `output` with values from `input`, and edges
    padded with reflected values from the unpadded region

    Args:
        axis: The axis to operate on.
        output: The output buffer.
        input: The input buffer.
        paddings: The (before, after) padding sizes for each axis.
        output_shape: the shape of the tensor passed to `output`
        output_strides: the stride at each output axis.
        input_strides: the stride at each input axis.
        output_offset: The offset at which output data starts.
        input_offset: The offset at which input data starts.
    """

    var axis_dim = output_shape[axis]
    var pre_pad = Int(paddings[2 * axis])
    var post_pad = Int(paddings[2 * axis + 1])
    var non_pad = axis_dim - pre_pad - post_pad
    var pre_pad_start_ptr = output.offset(output_offset)
    var input_start_ptr = input.offset(input_offset)

    var input_axis_stride = Int(input_strides[axis])
    var output_axis_stride = Int(output_strides[axis])

    # first fill the unpadded regions
    if axis + 1 != rank:
        # recurse down to lower dimensions
        var next_input_offset: Int = input_offset
        var next_output_offset: Int = output_offset + (
            output_axis_stride * pre_pad
        )
        # DANGER this uses a lot of recursion. For a rank N tensor the number of calls
        # will be dim(0) * dim(1) * ... dim(N-1)
        for sub_axis in range(pre_pad, pre_pad + non_pad):
            _pad_reflect_impl_rec(
                axis + 1,
                output,
                input,
                paddings,
                output_shape,
                output_strides,
                input_strides,
                next_output_offset,
                next_input_offset,
            )
            next_input_offset += input_axis_stride
            next_output_offset += output_axis_stride
    else:
        # no more dimensions to recurse, copy from input to unpadded region
        var non_pad_start_ptr = pre_pad_start_ptr.offset(pre_pad)
        memcpy(non_pad_start_ptr, input_start_ptr, non_pad)

    _memcpy_regions[type](
        pre_pad, post_pad, non_pad, output_axis_stride, pre_pad_start_ptr
    )


fn pretty_print(
    name: String,
    size: Int,
    rounds: Int,
    time_rec: Float64,
    time_iter: Float64,
    ratio: Float64,
    msg: String,
) raises:
    var py = Python.import_module("builtins")
    _ = py.print(
        py.str(
            "[{:<20}|rounds={}|size={:<4}] recursive: {:>8.3f} (ms), iterative:"
            " {:>8.3f} (ms), ratio: {:>4.2f}x {}"
        ).format(
            name,
            String(rounds),
            String(size),
            time_rec,
            time_iter,
            ratio,
            msg,
        )
    )


fn bench[
    func: fn[
        rank: Int, recursive: Int, size: Int, verify: Bool = False
    ] () raises -> None,
    rank: Int,
    size: Int,
    name: String,
]() raises:
    alias N = 100

    @parameter
    fn runner_iter():
        try:
            for i in range(N):
                var result = func[rank, 0, size]()
                keep(result)
        except e:
            abort(e)

    @parameter
    fn runner_recursive():
        try:
            for i in range(N):
                var result = func[rank, 1, size]()
                keep(result)
        except e:
            abort(e)

    var ms_iter = benchmark.run[runner_iter](1, 10)
    var ms_recursive = benchmark.run[runner_recursive](1, 10)

    var ratio = Int(
        100.0 * ms_recursive.mean(Unit.ns) / ms_iter.mean(Unit.ns)
    ) / 100.0
    var msg: String
    if ratio < 1.0:
        msg = "slowdown"
    else:
        msg = "speedup"
    pretty_print(
        name,
        size,
        N,
        ms_recursive.mean(Unit.ms),
        ms_iter.mean(Unit.ms),
        ratio,
        msg,
    )


fn test_pad_constant_nd[
    rank: Int, recursive: Int, n: Int, verify: Bool = False
]() raises:
    alias d_pre = 3
    alias d_post = 7
    alias d = d_pre + d_post

    @always_inline
    fn get_in_out_shapes[rank: Int = 1]() -> StaticTuple[DimList, 2]:
        var in_shape: DimList = DimList()
        var out_shape: DimList = DimList()

        @parameter
        if rank == 1:
            in_shape = DimList(n)
            out_shape = DimList(n + d)
        elif rank == 2:
            in_shape = DimList(n, n)
            out_shape = DimList(n + d, n + d)
        elif rank == 3:
            in_shape = DimList(n, n, n)
            out_shape = DimList(n + d, n + d, n + d)
        elif rank == 4:
            in_shape = DimList(n, n, n, n)
            out_shape = DimList(n + d, n + d, n + d, n + d)
        return StaticTuple[DimList, 2](in_shape, out_shape)

    alias in_out_shape = get_in_out_shapes[rank]()
    alias in_shape = in_out_shape[0]
    alias out_shape = in_out_shape[1]

    alias in_size = Int(in_shape.product[rank]())
    alias out_size = Int(out_shape.product[rank]())

    # create a big input matrix and fill it with 1
    var input_ptr = UnsafePointer[Scalar[DType.index]].alloc(in_size)
    var input = NDBuffer[DType.index, rank](input_ptr, in_shape)
    input.fill(1)

    # Create a padding array
    var paddings_stack = InlineArray[Scalar[DType.index], 2 * rank](
        uninitialized=True
    )
    var paddings = NDBuffer[DType.index, 1, _, 2 * rank](paddings_stack)

    @parameter
    for i in range(rank):
        paddings[2 * i] = d_pre
        paddings[2 * i + 1] = d_post

    # Create an output matrix and fill with 0
    var output_ptr = UnsafePointer[Scalar[DType.index]].alloc(out_size)
    var output = NDBuffer[DType.index, rank, _, out_shape](
        output_ptr, out_shape
    )
    output.fill(0)

    # constant padding value = 7
    var constant = Scalar[DType.index](7)

    # pad
    pad_constant_dispatch[recursive=recursive](
        output, input, paddings.data, constant
    )

    if verify:
        var output_rec_ptr = UnsafePointer[Scalar[DType.index]].alloc(out_size)
        var output_rec = NDBuffer[DType.index, rank, _, out_shape](
            output_rec_ptr, out_shape
        )
        output_rec.fill(0)
        pad_constant_dispatch[recursive=1](
            output_rec, input, paddings.data, constant
        )
        assert_equal(output, output_rec)

        output_rec_ptr.free()

    input_ptr.free()
    output_ptr.free()


fn test_pad_reflect_nd[
    rank: Int, recursive: Int, n: Int, verify: Bool = False
]() raises:
    alias d_pre = 3
    alias d_post = 7
    alias d = d_pre + d_post

    @always_inline
    fn get_in_out_shapes[rank: Int = 1]() -> StaticTuple[DimList, 2]:
        var in_shape: DimList = DimList()
        var out_shape: DimList = DimList()

        @parameter
        if rank == 1:
            in_shape = DimList(n)
            out_shape = DimList(n + d)
        elif rank == 2:
            in_shape = DimList(n, n)
            out_shape = DimList(n + d, n + d)
        elif rank == 3:
            in_shape = DimList(n, n, n)
            out_shape = DimList(n + d, n + d, n + d)
        elif rank == 4:
            in_shape = DimList(n, n, n, n)
            out_shape = DimList(n + d, n + d, n + d, n + d)
        return StaticTuple[DimList, 2](in_shape, out_shape)

    alias in_out_shape = get_in_out_shapes[rank]()
    alias in_shape = in_out_shape[0]
    alias out_shape = in_out_shape[1]

    alias in_size = Int(in_shape.product[rank]())
    alias out_size = Int(out_shape.product[rank]())

    # create a big input matrix and fill it with 1
    var input_ptr = UnsafePointer[Scalar[DType.index]].alloc(in_size)
    var input = NDBuffer[DType.index, rank](input_ptr, in_shape)
    input.fill(1)

    # Create a padding array
    var paddings_stack = InlineArray[Scalar[DType.index], 2 * rank](
        uninitialized=True
    )
    var paddings = NDBuffer[DType.index, 1, _, 2 * rank](paddings_stack)

    @parameter
    for i in range(rank):
        paddings[2 * i] = d_pre
        paddings[2 * i + 1] = d_post

    # Create an output matrix and fill with 0
    var output_ptr = UnsafePointer[Scalar[DType.index]].alloc(out_size)
    var output = NDBuffer[DType.index, rank, _, out_shape](
        output_ptr, out_shape
    )
    output.fill(0)

    # pad
    pad_reflect_dispatch[recursive=recursive](output, input, paddings.data)

    if verify:
        var output_rec_ptr = UnsafePointer[Scalar[DType.index]].alloc(out_size)
        var output_rec = NDBuffer[DType.index, rank, _, out_shape](
            output_rec_ptr, out_shape
        )
        output_rec.fill(0)
        pad_reflect_dispatch[recursive=1](output_rec, input, paddings.data)

        assert_equal(output, output_rec)

        output_rec_ptr.free()

    input_ptr.free()
    output_ptr.free()


# CHECK-LABEL: test_pad_iterative
def main():
    print("== test_pad_iterative")

    def all[N: Int]():
        bench[test_pad_constant_nd, 1, N, "test_pad_constant_1d"]()
        bench[test_pad_constant_nd, 2, N, "test_pad_constant_2d"]()
        bench[test_pad_constant_nd, 3, N, "test_pad_constant_3d"]()
        # bench[test_pad_constant_nd, 4, N, "test_pad_constant_4d"]()

        bench[test_pad_reflect_nd, 1, N, "test_pad_reflect_1d"]()
        bench[test_pad_reflect_nd, 2, N, "test_pad_reflect_2d"]()
        bench[test_pad_reflect_nd, 3, N, "test_pad_reflect_3d"]()
        # bench[test_pad_reflect_nd, 4, N, "test_pad_reflect_4d"]()

    # all[64]()
    # all[128]()
    # all[256]()
    # all[512]()
    # all[1024]()

    test_pad_constant_nd[1, 0, 64, True]()
    test_pad_constant_nd[2, 0, 64, True]()
    test_pad_constant_nd[3, 0, 64, True]()

    test_pad_reflect_nd[1, 0, 64, True]()
    test_pad_reflect_nd[2, 0, 64, True]()
    test_pad_reflect_nd[3, 0, 64, True]()
