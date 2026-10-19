# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements tensor padding with constant or edge values for CPU and GPU."""


# ===-----------------------------------------------------------------------===#
# pad
# ===-----------------------------------------------------------------------===#


from layout import (
    Coord,
    TensorLayout,
    TileTensor,
    row_major,
)

# TODO Refactor -- we should decide on and put them into a more common file
from linalg.transpose import _fill_strides
from std.math import clamp
from std.memory import unsafe_memcpy


from std.utils import IndexList, StaticTuple


@inline(.always)
def _fill[
    dtype: DType
](dst: MutPointer[Scalar[dtype], _], value: Scalar[dtype], count: Int,):
    _ = TileTensor(dst, row_major(count)).fill(value)


# TODO: could this be deleted? maybe replaced with faster collapsed loop.
struct _NestedLoopIter[n_loops: Int](ImplicitlyCopyable, Iterable, Iterator):
    """
    Helper iterable for padding functions meant to represent an n-level loop nest of
    the form:

    for i1 in range(lower_i1, upper_i1):
       for i2 in range(lower_i2, upper_i2):
           for i3 in range(lower_i3, upper_i3):
             .....
    """

    comptime Element = IndexList[Self.n_loops]
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var cur: Self.Element

    comptime LoopBoundSpec = Array[IndexList[2], Self.n_loops]
    var loop_bounds: Self.LoopBoundSpec
    var early_stop: Bool

    def __init__(out self, loop_bounds: Self.LoopBoundSpec):
        assert len(loop_bounds) == Self.n_loops, (
            "Number of entries in loop_bounds doesn't match the number of"
            " loops specified"
        )

        # TODO: Should this function take an `owned loop_bounds` to avoid
        #   a copy in places where the caller already has an owned value?
        self.loop_bounds = loop_bounds.copy()

        self.cur = IndexList[Self.n_loops]()
        self.early_stop = False

        for i in range(Self.n_loops):
            var lb = self._lb_loop(i)
            var ub = self._ub_loop(i)

            self.cur[i] = lb

            var invalid_bound = lb >= ub
            self.early_stop = self.early_stop or invalid_bound

    def _lb_loop(self, axis: Int) -> Int:
        return self.loop_bounds[axis][0]

    def _ub_loop(self, axis: Int) -> Int:
        return self.loop_bounds[axis][1]

    def __init__(out self, *, copy: Self):
        self.cur = copy.cur
        self.loop_bounds = copy.loop_bounds.copy()
        self.early_stop = copy.early_stop

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    def __next__(mut self) raises StopIteration -> Self.Element:
        if self.__len__() <= 0:
            raise StopIteration()

        var cur = self.cur
        self.cur[len(self.cur) - 1] += 1

        for i in range(Self.n_loops - 1, 0, -1):
            if self.cur[i] == self._ub_loop(i):
                self.cur[i] = self._lb_loop(i)
                self.cur[i - 1] += 1

        return cur

    @inline(.always)
    def __len__(self) -> Int:
        if self.cur[0] >= self._ub_loop(0) or self.early_stop:
            return 0
        else:
            return 1


def pad_constant[
    dtype: DType,
    paddings_type: DType,
    constant_type: DType,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    paddings: Pointer[Scalar[paddings_type], _],
    constant: Scalar[constant_type],
):
    """
    Fill `output` with values from `input`, and edges padded with `constant`
    based on `paddings`.

    Parameters:
        dtype: DType of the `input` and `output` buffers.
        paddings_type: DType of the `paddings` buffer.
        constant_type: DType of the `constant` value before it is cast to
            `dtype`.

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
    var constant_cast = rebind[Scalar[dtype]](constant[0])
    comptime output_rank = output.rank

    def pad_constant_wrapper(
        output: MutPointer[Scalar[dtype], address_space=.GENERIC, ...],
        input: Pointer[Scalar[dtype], address_space=.GENERIC, ...],
        paddings: Pointer[Scalar[paddings_type], _],
        output_shape: IndexList[output_rank],
        output_strides: MutPointer[Int, _],
        input_strides: Pointer[Int, _],
    ) {var constant_cast}:
        return _pad_constant_impl[output_rank, dtype, paddings_type](
            output,
            input,
            paddings,
            constant_cast,
            output_shape,
            output_strides,
            input_strides,
        )

    return _do_pad[
        dtype,
        paddings_type,
    ](output, input, paddings, pad_constant_wrapper)


def pad_reflect[
    dtype: DType,
    paddings_type: DType,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    paddings: Pointer[Scalar[paddings_type], _],
):
    """
    Fill `output` with values from `input`, and edges padded with reflected
    values from the unpadded region.

    Parameters:
        dtype: DType of the `input` and `output` buffers.
        paddings_type: DType of the `paddings` buffer.

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

    comptime output_rank = output.rank

    def pad_reflect_wrapper(
        output: MutPointer[Scalar[dtype], address_space=.GENERIC, ...],
        input: Pointer[Scalar[dtype], address_space=.GENERIC, ...],
        paddings: Pointer[Scalar[paddings_type], _],
        output_shape: IndexList[output_rank],
        output_strides: MutPointer[Int, _],
        input_strides: Pointer[Int, _],
    ) {}:
        return _pad_reflect_impl[output_rank, dtype, paddings_type](
            output, input, paddings, output_shape, output_strides, input_strides
        )

    return _do_pad[
        dtype,
        paddings_type,
    ](output, input, paddings, pad_reflect_wrapper)


@inline(.always)
def pad_shape[
    input_type: DType,
    paddings_type: DType,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    paddings_buf: TileTensor[mut=False, paddings_type, ...],
) raises -> IndexList[input_buf.rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_type: Type of the input tensor.
        paddings_type: Type of the padding tensor.

    Args:
        input_buf: The tensor to pad.
        paddings_buf: The paddings tensor, of shape (input_rank, 2).

    Returns:
        The output shape.
    """
    comptime assert (
        paddings_buf.flat_rank == 1
    ), "paddings_buf must be of rank 1"

    # TODO add runtime test once we support dynamic rank execution, currently
    # MLIR verifier of `MO::PadLike` prevents testing this with static rank.
    if paddings_buf.dim[0]() != Scalar[paddings_buf.linear_idx_type](
        2 * input_buf.rank
    ):
        raise Error("[pad] paddings shape must be (2 * input_rank)")

    # compute and return the output shape
    var output_shape = IndexList[input_buf.rank]()

    comptime for axis in range(input_buf.rank):
        var pre_pad = Int(paddings_buf[2 * axis])
        var post_pad = Int(paddings_buf[2 * axis + 1])
        output_shape[axis] = pre_pad + Int(input_buf.dim[axis]()) + post_pad

    return output_shape


def _do_pad[
    OutputLayoutType: TensorLayout,
    //,
    dtype: DType,
    paddings_type: DType,
    PadImplFn: ImplicitlyCopyable
    & def(
        MutPointer[Scalar[dtype], address_space=.GENERIC, ...],
        Pointer[Scalar[dtype], address_space=.GENERIC, ...],
        Pointer[Scalar[paddings_type], _],
        IndexList[OutputLayoutType.rank],
        MutPointer[Int, _],
        Pointer[Int, _],
    ) -> None,
](
    output: TileTensor[
        mut=True, dtype, OutputLayoutType, address_space=.GENERIC, ...
    ],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    paddings: Pointer[Scalar[paddings_type], _],
    pad_impl_fn: PadImplFn,
):
    var input_strides_stack = Array[Int, output.rank](uninitialized=True)
    var input_strides_buf = TileTensor(
        input_strides_stack, row_major[input.rank]()
    )
    var output_strides_stack = Array[Int, output.rank](uninitialized=True)
    var output_strides_buf = TileTensor(
        output_strides_stack, row_major[output.rank]()
    )
    _fill_strides(input, input_strides_buf)
    _fill_strides(output, output_strides_buf)

    var output_shape = IndexList[output.rank]()

    comptime for axis in range(output.rank):
        output_shape[axis] = Int(output.dim[axis]())

    return pad_impl_fn(
        output.ptr,
        input.ptr,
        paddings,
        output_shape,
        output_strides_buf.ptr,
        input_strides_buf.ptr,
    )


@inline(.always)
def _pad_reflect_repeat_src_index[
    mode: StaticString
](p: Int, in_len: Int, pre: Int) -> Int:
    """Map output position `p` along one axis to the input index to read,
    shared by the reflect and repeat (edge) pad modes on both the CPU and
    GPU paths.

    "reflect" folds `p - pre` into a period of `2 * (in_len - 1)` and
    mirrors it back, so positions in `[0, in_len)` map to themselves and
    positions beyond the input are reflected at the boundary. When
    `in_len == 1` the period is zero, so the source is always 0.

    "repeat" clamps `p - pre` to `[0, in_len)`, repeating the boundary
    value outside the input.

    Args:
        p: The output position along the axis.
        in_len: The input length along the axis.
        pre: The pre-padding amount along the axis.

    Returns:
        The input index to read for output position `p`.
    """
    var src = p - pre
    comptime if mode == "reflect":
        if in_len == 1:
            return 0
        var period = 2 * (in_len - 1)
        var r = src % period
        if r < 0:
            r += period
        if r < in_len:
            return r
        return period - r
    else:
        comptime assert mode == "repeat", t"unknown pad mode {mode}"
        return clamp(src, 0, in_len - 1)


struct _PadAxis[rank: Int, paddings_type: DType](TrivialRegisterPassable):
    """Per-axis padding geometry and the shared output-to-input coordinate
    mapping used by all pad modes.

    The geometry is derived once per axis from the `(before, after)` padding
    pair and the output extent:

    - `pre_pad`: output positions `[0, pre_pad)` precede the input.
    - `non_pad`: output positions `[pre_pad, pre_pad + non_pad)` hold the
      input values, so the input extent is `non_pad`.
    - `post_pad`: output positions `[pre_pad + non_pad, ...)` follow the
      input.

    `map_index` is the single bounds/coordinate mapping shared by the modes:
    the input index to read for a given output position. `constant` mode
    only needs `is_within_padding` (out-of-bounds positions are filled with
    the constant instead of being mapped); `reflect` and `repeat` map every
    output position to an in-bounds input index.
    """

    var pre_pad: Int
    var post_pad: Int
    var non_pad: Int

    @inline(.always)
    def __init__(
        out self,
        axis: Int,
        paddings: Pointer[Scalar[Self.paddings_type], _],
        output_shape: IndexList[Self.rank],
    ):
        var axis_dim = output_shape[axis]
        var pre_pad = Int(paddings[2 * axis])
        var post_pad = Int(paddings[2 * axis + 1])
        var non_pad = axis_dim - pre_pad - post_pad

        self.pre_pad = pre_pad
        self.post_pad = post_pad
        self.non_pad = non_pad

    @inline(.always)
    def is_within_padding(self, i: Int) -> Bool:
        """Whether output position `i` on this axis is a padded position,
        i.e. outside `[pre_pad, pre_pad + non_pad)`."""
        return (i < self.pre_pad) or (self.pre_pad + self.non_pad <= i)

    @inline(.always)
    def in_bounds(self, i: Int) -> Bool:
        """Whether input index `i` on this axis lies in `[0, non_pad)`."""
        return 0 <= i < self.non_pad

    @inline(.always)
    def map_index[mode: StaticString](self, p: Int) -> Int:
        """Map output position `p` on this axis to the input index to
        read, via the shared `_pad_reflect_repeat_src_index` mapping.

        - `mode == "reflect"`: folds `p - pre_pad` into a period of
          `2 * (non_pad - 1)` and mirrors it back at the boundaries; when
          `non_pad == 1` the source is always 0.
        - `mode == "repeat"`: clamps `p - pre_pad` to `[0, non_pad)`,
          repeating the boundary value outside the input.
        """
        return _pad_reflect_repeat_src_index[mode](
            p, self.non_pad, self.pre_pad
        )


struct _AxisParams[rank: Int, dtype: DType, paddings_type: DType](
    TrivialRegisterPassable
):
    """Constant-mode per-axis state, layered on the shared `_PadAxis`
    bounds/coordinate mapping.
    """

    var pad_axis: _PadAxis[Self.rank, Self.paddings_type]

    var output_offset: Int
    var input_offset: Int
    var pad_with_constant: Bool
    var is_within_padding: Bool
    var next_pad_with_constant: Bool

    """
    output_offset: The offset at which output data starts.
    input_offset: The offset at which input data starts.
    pad_with_constant: whether to always pad remaining region with constant.
    """

    @inline(.always)
    def pre_pad(self) -> Int:
        return self.pad_axis.pre_pad

    @inline(.always)
    def post_pad(self) -> Int:
        return self.pad_axis.post_pad

    @inline(.always)
    def non_pad(self) -> Int:
        return self.pad_axis.non_pad

    @inline(.always)
    def __init__(
        out self,
        axis: Int,
        paddings: Pointer[Scalar[Self.paddings_type], _],
        output_shape: IndexList[Self.rank],
    ):
        self.pad_axis = _PadAxis[Self.rank, Self.paddings_type](
            axis, paddings, output_shape
        )
        self.output_offset = 0
        self.input_offset = 0
        self.pad_with_constant = False
        self.is_within_padding = False
        self.next_pad_with_constant = False

    @inline(.always)
    def init_offsets(
        mut self,
        output_offset: Int,
        input_offset: Int,
        pad_with_constant: Bool,
    ):
        self.output_offset = output_offset
        self.input_offset = input_offset
        self.pad_with_constant = pad_with_constant

    @inline(.always)
    def pre_check(mut self, i: Int):
        self.is_within_padding = self.pad_axis.is_within_padding(i)
        self.next_pad_with_constant = (
            self.pad_with_constant or self.is_within_padding
        )

    @inline(.always)
    def post_check(mut self, output_axis_stride: Int, input_axis_stride: Int):
        if not self.is_within_padding:
            self.input_offset += input_axis_stride
        self.output_offset += output_axis_stride

    @inline(.always)
    def base(
        mut self,
        output: MutPointer[Scalar[Self.dtype], _],
        input: Pointer[Scalar[Self.dtype], _],
        constant: Scalar[Self.dtype],
        axis_dim: Int,
    ):
        var pre_pad_start_ptr = output.unsafe_offset(self.output_offset)

        # setting values
        if self.pad_with_constant:
            _fill(pre_pad_start_ptr, constant, axis_dim)
        else:
            var non_pad_start_ptr = pre_pad_start_ptr.unsafe_offset(
                self.pre_pad()
            )
            var post_pad_start_ptr = non_pad_start_ptr.unsafe_offset(
                self.non_pad()
            )
            var input_start_ptr = input.unsafe_offset(self.input_offset)
            _fill(pre_pad_start_ptr, constant, self.pre_pad())
            unsafe_memcpy(
                dest=non_pad_start_ptr,
                src=input_start_ptr,
                count=self.non_pad(),
            )
            _fill(post_pad_start_ptr, constant, self.post_pad())


@inline(.always)
def _pad_constant_axis[
    rank: Int, dtype: DType, paddings_type: DType, axis: Int
](
    output: MutPointer[Scalar[dtype], _],
    input: Pointer[Scalar[dtype], _],
    constant: Scalar[dtype],
    output_shape: IndexList[rank],
    output_strides: Pointer[Int, _],
    input_strides: Pointer[Int, _],
    var axis_params: StaticTuple[_AxisParams[rank, dtype, paddings_type], rank],
):
    comptime if axis == (rank - 1):
        axis_params[axis].base(output, input, constant, output_shape[axis])
    else:
        var output_axis_stride = Int(output_strides[axis])
        var input_axis_stride = Int(input_strides[axis])
        for i in range(output_shape[axis]):
            axis_params[axis].pre_check(i)

            axis_params[axis + 1].init_offsets(
                axis_params[axis].output_offset,
                axis_params[axis].input_offset,
                axis_params[axis].next_pad_with_constant,
            )
            _pad_constant_axis[rank, dtype, paddings_type, axis + 1](
                output,
                input,
                constant,
                output_shape,
                output_strides,
                input_strides,
                axis_params,
            )

            axis_params[axis].post_check(output_axis_stride, input_axis_stride)


def _pad_constant_impl[
    rank: Int, dtype: DType, paddings_type: DType
](
    output: MutPointer[Scalar[dtype], _],
    input: Pointer[Scalar[dtype], _],
    paddings: Pointer[Scalar[paddings_type], _],
    constant: Scalar[dtype],
    output_shape: IndexList[rank],
    output_strides: Pointer[Int, _],
    input_strides: Pointer[Int, _],
):
    """
    Fill axis ∈ [axis, rank) in `output` with values from `input`, and edges
    padded with `constant` based on `paddings`.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: The (before, after) padding sizes for each axis.
        constant: the constant to pad output with.
        output_shape: the dynamic shape of the tensor pointed to by output buffer
        output_strides: the stride at each output axis.
        input_strides: the stride at each input axis.
    """

    # allocate 'rank' axis-data vector, only use the ones in range[axis,rank)
    var axis_params = StaticTuple[
        _AxisParams[rank, dtype, paddings_type], rank
    ]()

    comptime for r in range(rank):
        axis_params[r] = _AxisParams[rank, dtype, paddings_type](
            r, paddings, output_shape
        )

    # CRITICAL: should be setting output_offset=0, input_offset=0, and
    # pad_with_constant=False for axis=0 in padding. However, this is
    # already addressed in the constructor of _AxisParams.
    # axis_params[0].init_offsets(output_offset, input_offset, pad_with_constant)

    _pad_constant_axis[rank, dtype, paddings_type, 0](
        output,
        input,
        constant,
        output_shape,
        output_strides,
        input_strides,
        axis_params,
    )


@inline(.always)
def _memcpy_regions_fast[
    dtype: DType
](
    pre_pad: Int,
    post_pad: Int,
    non_pad: Int,
    output_axis_stride: Int,
    pre_pad_start_ptr: MutPointer[Scalar[dtype], _],
):
    @inline(.always)
    def _copy_reflected_rows[pre_copy: Bool]() {var}:
        # Padded output positions map to in-bounds input positions via the
        # shared reflect fold (`_pad_reflect_repeat_src_index`); every copy
        # reads the input region and writes a padded position, so the
        # iteration order is irrelevant.
        var first = 0 if pre_copy else pre_pad + non_pad
        var count = pre_pad if pre_copy else post_pad
        for copy_to in range(first, first + count):
            var copy_from = pre_pad + _pad_reflect_repeat_src_index["reflect"](
                copy_to, non_pad, pre_pad
            )

            var copy_to_ptr = pre_pad_start_ptr.unsafe_offset(
                copy_to * output_axis_stride
            )
            var copy_from_ptr = pre_pad_start_ptr.unsafe_offset(
                copy_from * output_axis_stride
            )

            # dest and src are non-overlapping slices of the same buffer
            # (shared origin). Opt out of exclusivity with an unsafe any-origin:
            # unsafe_memcpy's non-overlap requirement is a caller contract the
            # exclusivity checker can't prove.
            unsafe_memcpy(
                dest=copy_to_ptr,
                src=copy_from_ptr.as_unsafe_any_origin(),
                count=output_axis_stride,
            )

    _copy_reflected_rows[pre_copy=True]()
    _copy_reflected_rows[pre_copy=False]()


struct _AxisParamsReflect[rank: Int, dtype: DType, paddings_type: DType](
    TrivialRegisterPassable
):
    """Reflect-mode per-axis state, layered on the shared `_PadAxis`
    bounds/coordinate mapping.
    """

    var pad_axis: _PadAxis[Self.rank, Self.paddings_type]

    var next_input_offset: Int
    var next_output_offset: Int

    @inline(.always)
    def pre_pad(self) -> Int:
        return self.pad_axis.pre_pad

    @inline(.always)
    def post_pad(self) -> Int:
        return self.pad_axis.post_pad

    @inline(.always)
    def non_pad(self) -> Int:
        return self.pad_axis.non_pad

    @inline(.always)
    def __init__(
        out self,
        axis: Int,
        paddings: Pointer[Scalar[Self.paddings_type], _],
        output_shape: IndexList[Self.rank],
    ):
        self.pad_axis = _PadAxis[Self.rank, Self.paddings_type](
            axis, paddings, output_shape
        )
        self.next_input_offset = 0
        self.next_output_offset = 0

    @inline(.always)
    def init_offsets(
        mut self,
        output_offset: Int,
        input_offset: Int,
        output_axis_stride: Int,
    ):
        # setting offsets for the lower dimensions
        self.next_input_offset = input_offset
        self.next_output_offset = output_offset + (
            output_axis_stride * self.pre_pad()
        )

    @inline(.always)
    def update_next_offsets(
        mut self, output_axis_stride: Int, input_axis_stride: Int
    ):
        self.next_output_offset += output_axis_stride
        self.next_input_offset += input_axis_stride

    @inline(.always)
    def base(
        mut self,
        output_offset: Int,
        input_offset: Int,
        output: MutPointer[Scalar[Self.dtype], address_space=.GENERIC, ...],
        input: Pointer[Scalar[Self.dtype], address_space=.GENERIC, ...],
    ):
        # no more dimensions to recurse, copy from input to unpadded region
        var non_pad_start_ptr = output.unsafe_offset(
            output_offset + self.pre_pad()
        )
        var input_start_ptr = input.unsafe_offset(input_offset)
        unsafe_memcpy(
            dest=non_pad_start_ptr, src=input_start_ptr, count=self.non_pad()
        )

    @inline(.always)
    def memcpy_regions(
        mut self,
        output_axis_stride: Int,
        output_offset: Int,
        output: MutPointer[Scalar[Self.dtype], _],
    ):
        var pre_pad_start_ptr = output.unsafe_offset(output_offset)

        _memcpy_regions_fast(
            self.pre_pad(),
            self.post_pad(),
            self.non_pad(),
            output_axis_stride,
            pre_pad_start_ptr,
        )


@inline(.always)
def _pad_reflect_axis[
    rank: Int,
    dtype: DType,
    paddings_type: DType,
    axis: Int,
](
    output: MutPointer[Scalar[dtype], address_space=.GENERIC, ...],
    input: Pointer[Scalar[dtype], address_space=.GENERIC, ...],
    output_strides: Pointer[Int, _],
    input_strides: Pointer[Int, _],
    var axis_params: StaticTuple[
        _AxisParamsReflect[rank, dtype, paddings_type], rank
    ],
):
    var output_axis_stride = Int(output_strides[axis])
    var input_offset: Int
    var output_offset: Int

    comptime if axis == 0:
        input_offset = 0
        output_offset = 0
        # CRITICAL: setting output_offset, input_offset, output and input pointers for the first axis in padding.
        # output_offset: The offset at which output data starts.
        # input_offset: The offset at which input data starts.
        axis_params[0].init_offsets(
            output_offset, input_offset, output_axis_stride
        )
    else:
        output_offset = axis_params[axis - 1].next_output_offset
        input_offset = axis_params[axis - 1].next_input_offset

    comptime if axis == rank - 1:
        axis_params[axis].base(output_offset, input_offset, output, input)
    else:
        var output_axis_stride_next = Int(output_strides[axis + 1])
        var input_axis_stride = Int(input_strides[axis])
        for _ in range(
            axis_params[axis].pre_pad(),
            axis_params[axis].pre_pad() + axis_params[axis].non_pad(),
        ):
            axis_params[axis + 1].init_offsets(
                axis_params[axis].next_output_offset,
                axis_params[axis].next_input_offset,
                output_axis_stride_next,
            )
            _pad_reflect_axis[rank, dtype, paddings_type, axis + 1](
                output, input, output_strides, input_strides, axis_params
            )

            axis_params[axis].update_next_offsets(
                output_axis_stride, input_axis_stride
            )
    axis_params[axis].memcpy_regions(output_axis_stride, output_offset, output)


def _pad_reflect_impl[
    rank: Int,
    dtype: DType,
    paddings_type: DType,
](
    output: MutPointer[Scalar[dtype], address_space=.GENERIC, ...],
    input: Pointer[Scalar[dtype], address_space=.GENERIC, ...],
    paddings: Pointer[Scalar[paddings_type], _],
    output_shape: IndexList[rank],
    output_strides: MutPointer[Int, _],
    input_strides: Pointer[Int, _],
):
    """
    Fill axis ∈ [axis, rank) in `output` with values from `input`, and edges
    padded with reflected values from the unpadded region

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: The (before, after) padding sizes for each axis.
        output_shape: the shape of the tensor passed to `output`
        output_strides: the stride at each output axis.
        input_strides: the stride at each input axis.
    """

    var axis_params = StaticTuple[
        _AxisParamsReflect[rank, dtype, paddings_type], rank
    ]()

    for r in range(rank):
        axis_params[r] = _AxisParamsReflect[rank, dtype, paddings_type](
            r, paddings, output_shape
        )

    _pad_reflect_axis[rank, dtype, paddings_type, 0](
        output, input, output_strides, input_strides, axis_params
    )


@inline(.always)
def pad_repeat[
    dtype: DType,
    paddings_type: DType,
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[mut=False, dtype, ...],
    paddings: Pointer[Scalar[paddings_type], _],
):
    """
    Fill `output` with values from `input`, and edges padded boundary
    values from the unpadded region.

    Parameters:
        dtype: DType of the input/output buffer.
        paddings_type: DType of the input, output, and padding buffers.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.

    Example:
        var input = [[1, 2],
                     [3, 4]]
        var paddings = [2, 2, 1, 0]

        Yields:
        output = [[1, 1, 2],
                  [1, 1, 2],
                  [1, 1, 2],
                  [3, 3, 4],
                  [3, 3, 4],
                  [3, 3, 4]]
    """

    # Per-axis padding geometry via the shared `_PadAxis` mapping: repeat
    # clamps the source index to the input extent (`map_index[mode="repeat"]`
    # in `_PadAxis`), so each padded position copies the nearest border
    # value. `pre_pads`/`post_pads` below are just unpacked views for the
    # loop bounds.
    comptime rank = output.rank
    var output_shape = IndexList[rank]()
    comptime for axis in range(rank):
        output_shape[axis] = Int(output.dim[axis]())

    var pad_axes = StaticTuple[_PadAxis[rank, paddings_type], rank]()
    var pre_pads = IndexList[rank]()
    var post_pads = IndexList[rank]()

    for axis in range(comptime (rank)):
        var pad_axis = _PadAxis[rank, paddings_type](
            axis, paddings, output_shape
        )
        pad_axes[axis] = pad_axis
        pre_pads[axis] = pad_axis.pre_pad
        post_pads[axis] = pad_axis.post_pad

    var loop_bounds = _NestedLoopIter[output.rank].LoopBoundSpec(
        fill=IndexList[2](0)
    )

    comptime for i in range(output.rank):
        loop_bounds[i] = IndexList[2](0, Int(input.dim[i]()))

    var non_pad_iter = _NestedLoopIter[output.rank](loop_bounds)

    for input_idx in non_pad_iter:
        var output_idx = input_idx + pre_pads
        output.store(
            Coord(output_idx),
            input.load[width=1](Coord(input_idx)),
        )

    for axis in reversed(range(comptime (output.rank))):
        for i in range(axis):
            loop_bounds[i] = IndexList[2](
                pre_pads[i], pre_pads[i] + Int(input.dim(i))
            )

        for i in range(axis + 1, comptime (output.rank)):
            loop_bounds[i] = IndexList[2](0, Int(output.dim(i)))

        # handle pre-padding of the axis
        var pre_lower = 0
        var pre_upper = pre_pads[axis]

        loop_bounds[axis] = IndexList[2](pre_lower, pre_upper)

        var pre_pad_iter = _NestedLoopIter[output.rank](loop_bounds)

        for write_idx in pre_pad_iter:
            # Shared repeat-mode coordinate map (see `_PadAxis.map_index`):
            # clamps to the first input position; the border row is read in
            # output space, i.e. at `pre_pad + mapped input index`.
            var read_idx = write_idx
            read_idx[axis] = pre_pads[axis] + pad_axes[axis].map_index[
                mode="repeat"
            ](write_idx[axis])

            output[Coord(write_idx)] = output[Coord(read_idx)]

        # and now post-padding
        var post_lower = pre_pads[axis] + Int(input.dim(axis))
        var post_upper = Int(output.dim(axis))

        loop_bounds[axis] = IndexList[2](post_lower, post_upper)

        var post_pad_iter = _NestedLoopIter[output.rank](loop_bounds)

        for write_idx in post_pad_iter:
            # Shared repeat-mode coordinate map (see `_PadAxis.map_index`):
            # clamps to the last input position, read in output space.
            var read_idx = write_idx
            read_idx[axis] = pre_pads[axis] + pad_axes[axis].map_index[
                mode="repeat"
            ](write_idx[axis])

            output[Coord(write_idx)] = output[Coord(read_idx)]
