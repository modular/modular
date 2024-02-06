# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------===#
# pad
# ===----------------------------------------------------------------------===#

from sys.info import sizeof

from memory import memcpy
from memory.buffer import Buffer, NDBuffer
from memory.unsafe import DTypePointer

# TODO Refactor -- we should decide on and put them into a more common file
from Transpose import _fill_strides

from utils.index import StaticIntTuple
from utils.list import Dim, DimList

from collections.vector import InlinedFixedVector

from math import min


fn _fill[
    type: DType
](dst: DTypePointer[type], value: SIMD[type, 1], count: Int):
    _ = Buffer[type](dst, count).fill(value)


struct _NestedLoopIter[n_loops: Int]:
    """
    Helper iterable for padding functions meant to represent an n-level loop nest of
    the form:

    for i1 in range(lower_i1, upper_i1):
       for i2 in range(lower_i2, upper_i2):
           for i3 in range(lower_i3, upper_i3):
             .....
    """

    var cur: StaticIntTuple[n_loops]

    alias LoopBoundSpec = InlinedFixedVector[Tuple[Int, Int], n_loops]
    var loop_bounds: _NestedLoopIter[n_loops].LoopBoundSpec
    var early_stop: Bool

    fn __init__(inout self: Self, loop_bounds: self.LoopBoundSpec):
        debug_assert(
            len(loop_bounds) == n_loops,
            (
                "Number of entries in loop_bounds doesn't match the number of"
                " loops specified"
            ),
        )

        self.loop_bounds = loop_bounds

        self.cur = StaticIntTuple[n_loops]()
        self.early_stop = False

        for i in range(n_loops):
            let lb = self._lb_loop(i)
            let ub = self._ub_loop(i)

            self.cur[i] = lb

            let invalid_bound = lb >= ub
            self.early_stop = self.early_stop or invalid_bound

    fn _lb_loop(borrowed self, axis: Int) -> Int:
        return self.loop_bounds[axis].get[0, Int]()

    fn _ub_loop(borrowed self, axis: Int) -> Int:
        return self.loop_bounds[axis].get[1, Int]()

    fn __copyinit__(inout self: Self, other: Self):
        self.cur = other.cur
        self.loop_bounds = other.loop_bounds
        self.early_stop = other.early_stop

    fn __iter__(inout self: Self) -> Self:
        return self

    fn __next__(inout self) -> StaticIntTuple[n_loops]:
        let cur = self.cur

        self.cur[len(self.cur) - 1] += 1

        for i in range(n_loops - 1, 0, -1):
            if self.cur[i] == self._ub_loop(i):
                self.cur[i] = self._lb_loop(i)
                self.cur[i - 1] += 1

        return cur

    fn __len__(inout self) -> Int:
        if self.cur[0] >= self._ub_loop(0) or self.early_stop:
            return 0
        else:
            return 1


fn pad_constant[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
    paddings_type: DType,
    constant_type: DType,
](
    output: NDBuffer[type, rank, output_shape],
    input: NDBuffer[type, rank, input_shape],
    paddings: DTypePointer[paddings_type],
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
        let input_shape = (X, Y, Z)
        let paddings = [x0, x1, y0, y1, z0, z1]

        out[x, y, z] =
          input[x - x0, y - y0, z - z0] if x ∈ [x0, x0 + X] &&
                                           y ∈ [y0, y0 + Y] &&
                                           z ∈ [z0, z0 + Z]
          else constant
    """

    alias init_axis = 0
    let constant_cast = rebind[SIMD[type, 1]](constant[0])

    @__copy_capture(constant_cast)
    @parameter
    fn pad_constant_wrapper(
        output: DTypePointer[type],
        input: DTypePointer[type],
        paddings: DTypePointer[paddings_type],
        output_shape: StaticIntTuple[rank],
        output_strides: DTypePointer[DType.index],
        input_strides: DTypePointer[DType.index],
    ):
        return _pad_constant_impl[rank, type, paddings_type](
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


fn pad_reflect[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
    paddings_type: DType,
](
    output: NDBuffer[type, rank, output_shape],
    input: NDBuffer[type, rank, input_shape],
    paddings: DTypePointer[paddings_type],
):
    """
    Fill `output` with values from `input`, and edges padded with reflected
    values from the unpadded region.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.

    Example:
        let input = [[1, 2],
                     [3, 4]]
        let paddings = [2, 2, 1, 0]

        Yields:
        output = [[2, 1, 2],
                  [4, 3, 4],
                  [2, 1, 2],
                  [4, 3, 4],
                  [2, 1, 2],
                  [4, 3, 4]]
    """
    alias init_axis = 0

    @parameter
    fn pad_reflect_wrapper(
        output: DTypePointer[type],
        input: DTypePointer[type],
        paddings: DTypePointer[paddings_type],
        output_shape: StaticIntTuple[rank],
        output_strides: DTypePointer[DType.index],
        input_strides: DTypePointer[DType.index],
    ):
        return _pad_reflect_impl[rank, type, paddings_type](
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


@always_inline
fn pad_shape[
    input_rank: Int,
    input_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    paddings_buf: NDBuffer[paddings_type, 2],
) raises -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        input_type: Type of the input tensor.
        paddings_type: Type of the padding tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The tensor to pad.
        paddings_buf: The paddings tensor, of shape (input_rank, 2).

    Returns:
        The output shape.
    """

    # TODO add runtime test once we support dynamic rank execution, currently
    # MLIR verifier of `MO::PadLike` prevents testing this with static rank.
    if paddings_buf.dim(0) != input_rank or paddings_buf.dim(1) != 2:
        raise Error("[pad] paddings shape must be (input_rank, 2)")

    # compute and return the output shape
    var output_shape = StaticIntTuple[input_rank]()

    @unroll
    for axis in range(input_rank):
        let pre_pad = int(paddings_buf[axis, 0])
        let post_pad = int(paddings_buf[axis, 1])
        output_shape[axis] = pre_pad + input_buf.dim(axis) + post_pad

    return output_shape


fn _do_pad[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
    paddings_type: DType,
    pad_impl_fn: fn (
        DTypePointer[type],
        DTypePointer[type],
        DTypePointer[paddings_type],
        StaticIntTuple[rank],
        DTypePointer[DType.index],
        DTypePointer[DType.index],
    ) capturing -> None,
](
    output: NDBuffer[type, rank, output_shape],
    input: NDBuffer[type, rank, input_shape],
    paddings: DTypePointer[paddings_type],
):
    let input_strides_buf = Buffer[DType.index, rank].stack_allocation()
    let output_strides_buf = Buffer[DType.index, rank].stack_allocation()
    _fill_strides(input, input_strides_buf)
    _fill_strides(output, output_strides_buf)

    return pad_impl_fn(
        output.data,
        input.data,
        paddings,
        output.dynamic_shape,
        output_strides_buf.data,
        input_strides_buf.data,
    )


fn _pad_constant_impl[
    rank: Int,
    type: DType,
    paddings_type: DType,
](
    axis: Int,
    output: DTypePointer[type],
    input: DTypePointer[type],
    paddings: DTypePointer[paddings_type],
    constant: SIMD[type, 1],
    output_shape: StaticIntTuple[rank],
    output_strides: DTypePointer[DType.index],
    input_strides: DTypePointer[DType.index],
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

    let axis_dim = output_shape[axis]
    let pre_pad = int(paddings[2 * axis])
    let post_pad = int(paddings[2 * axis + 1])
    let non_pad = axis_dim - pre_pad - post_pad

    if axis + 1 == rank:
        # pointers
        let pre_pad_start_ptr = output.offset(output_offset)
        let non_pad_start_ptr = pre_pad_start_ptr.offset(pre_pad)
        let post_pad_start_ptr = non_pad_start_ptr.offset(non_pad)
        let input_start_ptr = input.offset(input_offset)

        # setting values
        if pad_with_constant:
            _fill(pre_pad_start_ptr, constant.value, axis_dim)
            return

        _fill(pre_pad_start_ptr, constant.value, pre_pad)
        memcpy(non_pad_start_ptr, input_start_ptr, non_pad)
        _fill(post_pad_start_ptr, constant.value, post_pad)
        return

    debug_assert(axis + 1 < rank, "axis is not within range")

    let input_axis_stride = input_strides.load(axis)[0].value
    let output_axis_stride = output_strides.load(axis)[0].value

    var next_input_offset: Int = input_offset.value
    var next_output_offset: Int = output_offset.value
    for i in range(axis_dim):
        let is_within_padding = (i < pre_pad) or (pre_pad + non_pad <= i)
        let next_pad_with_constant = pad_with_constant or is_within_padding
        _pad_constant_impl(
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


fn _pad_reflect_impl[
    rank: Int,
    type: DType,
    paddings_type: DType,
](
    axis: Int,
    output: DTypePointer[type],
    input: DTypePointer[type],
    paddings: DTypePointer[paddings_type],
    output_shape: StaticIntTuple[rank],
    output_strides: DTypePointer[DType.index],
    input_strides: DTypePointer[DType.index],
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

    # TODO this recursion can add up for larger tensors, optimize in #24565

    let axis_dim = output_shape[axis]
    let pre_pad = int(paddings[2 * axis])
    let post_pad = int(paddings[2 * axis + 1])
    let non_pad = axis_dim - pre_pad - post_pad
    let pre_pad_start_ptr = output.offset(output_offset)
    let input_start_ptr = input.offset(input_offset)

    let input_axis_stride = input_strides.load(axis)[0].value
    let output_axis_stride = output_strides.load(axis)[0].value

    let input_offsets = StaticIntTuple[2](input_offset, input_offset)
    let output_offsets = StaticIntTuple[2](output_offset, output_offset)

    # first fill the unpadded regions
    if axis + 1 != rank:
        # recurse down to lower dimensions
        var next_input_offset: Int = input_offset.value
        var next_output_offset: Int = output_offset.value + (
            output_axis_stride * pre_pad
        )
        # DANGER this uses a lot of recursion. For a rank N tensor the number of calls
        # will be dim(0) * dim(1) * ... dim(N-1)
        for sub_axis in range(pre_pad, pre_pad + non_pad):
            _pad_reflect_impl(
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
        let non_pad_start_ptr = pre_pad_start_ptr.offset(pre_pad)
        memcpy(non_pad_start_ptr, input_start_ptr, non_pad)

    # now memcpy the fully padded axes from the regions we just filled
    var curr_pre_pad = 0
    var curr_post_pad = 0
    while curr_pre_pad < pre_pad or curr_post_pad < post_pad:
        if curr_pre_pad < pre_pad:
            let copy_to = pre_pad - curr_pre_pad - 1
            let copy_to_ptr = pre_pad_start_ptr.offset(
                (copy_to * output_axis_stride)
            )

            let copy_from: Int
            if non_pad == 1:
                # handle singleton case
                copy_from = pre_pad
            else:
                copy_from = copy_to + ((curr_pre_pad % (non_pad - 1)) + 1) * 2

            let copy_from_ptr = pre_pad_start_ptr.offset(
                (copy_from * output_axis_stride)
            )
            memcpy(copy_to_ptr, copy_from_ptr, output_axis_stride)
            curr_pre_pad += 1

        if curr_post_pad < post_pad:
            let copy_to = pre_pad + non_pad + curr_post_pad
            let copy_to_ptr = pre_pad_start_ptr.offset(
                copy_to * output_axis_stride
            )

            let copy_from: Int
            if non_pad == 1:
                # handle singleton case
                copy_from = pre_pad
            else:
                copy_from = copy_to - ((curr_post_pad % (non_pad - 1)) + 1) * 2

            let copy_from_ptr = pre_pad_start_ptr.offset(
                copy_from * output_axis_stride
            )

            memcpy(copy_to_ptr, copy_from_ptr, output_axis_stride)
            curr_post_pad += 1


@always_inline
fn pad_repeat[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
    paddings_type: DType,
](
    output: NDBuffer[type, rank, output_shape],
    input: NDBuffer[type, rank, input_shape],
    paddings: DTypePointer[paddings_type],
):
    """
    Fill `output` with values from `input`, and edges padded boundary
    values from the unpadded region.

    Parameters:
        rank: Rank of the input/output buffers.
        output_shape: Dimensions of the output buffer.
        input_shape: Dimensions of the input buffer.
        type: DType of the input/output buffer.
        paddings_type: DType of the input, output, and padding buffers.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.

    Example:
        let input = [[1, 2],
                     [3, 4]]
        let paddings = [2, 2, 1, 0]

        Yields:
        output = [[1, 1, 2],
                  [1, 1, 2],
                  [1, 1, 2],
                  [3, 3, 4],
                  [3, 3, 4],
                  [3, 3, 4]]
    """
    let padding_ndbuf = NDBuffer[paddings_type, 2, DimList(rank, 2)](paddings)

    var pre_pads = StaticIntTuple[rank]()
    var post_pads = StaticIntTuple[rank]()

    for axis in range(rank):
        pre_pads[axis] = int(paddings[2 * axis])
        post_pads[axis] = int(paddings[2 * axis + 1])

    var loop_bounds = _NestedLoopIter[rank].LoopBoundSpec(rank)

    for i in range(rank):
        loop_bounds.append((0, input.dim(i)))

    var non_pad_iter = _NestedLoopIter[rank](loop_bounds)

    for input_idx in non_pad_iter:
        let output_idx = input_idx + pre_pads
        output[output_idx] = input[input_idx]

    for axis in range(rank - 1, -1, -1):
        alias sit = StaticIntTuple[2]

        let pre_pad = pre_pads[axis]
        let post_pad = post_pads[axis]

        for i in range(axis):
            loop_bounds[i] = (pre_pads[i], pre_pads[i] + input.dim(i))

        for i in range(axis + 1, rank):
            loop_bounds[i] = (0, output.dim(i))

        # handle pre-padding of the axis
        let pre_lower = 0
        let pre_upper = pre_pads[axis]

        loop_bounds[axis] = (pre_lower, pre_upper)

        var pre_pad_iter = _NestedLoopIter[rank](loop_bounds)

        for write_idx in pre_pad_iter:
            var read_idx = write_idx

            read_idx[axis] = pre_pads[axis]

            output[write_idx] = output[read_idx]

        # and now post-padding
        let post_lower = pre_pads[axis] + input.dim(axis)
        let post_upper = output.dim(axis)

        loop_bounds[axis] = (post_lower, post_upper)

        var post_pad_iter = _NestedLoopIter[rank](loop_bounds)

        for write_idx in post_pad_iter:
            var read_idx = write_idx
            read_idx[axis] = post_lower - 1

            output[write_idx] = output[read_idx]
