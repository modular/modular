# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from math import clamp

from algorithm import elementwise
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from runtime.asyncrt import DeviceContextPtr

from utils._select import _select_register_value as select
from utils.index import IndexList


@always_inline("nodebug")
fn _normalize_and_clamp_dim(start: Int, step: Int, dim_i: Int) -> Int:
    # Normalize the start/stop indices
    var normalized_idx = select(start < 0, start + dim_i, start)

    # Compute the min/max for clamping start/end
    var idx_min = select(step > 0, 0, -1)
    var idx_max = select(step > 0, dim_i, dim_i - 1)

    # Allow start and stop to truncate like numpy and torch allow.
    return clamp(normalized_idx, idx_min, idx_max)


# ===-----------------------------------------------------------------------===#
# slice_dim_as_view
# ===-----------------------------------------------------------------------===#


@always_inline
fn slice_dim_as_view[
    dtype: DType, dim: Int
](
    tensor: LayoutTensor[dtype, **_], start: Int, end: Int, step: Int
) -> LayoutTensor[
    dtype,
    Layout.row_major[tensor.rank](),
    tensor.origin,
    address_space = tensor.address_space,
]:
    var new_shape = tensor.runtime_layout.shape.value.canonicalize()
    var new_stride = tensor.runtime_layout.stride.value.canonicalize()

    var dim_i = tensor.dim(dim)
    var old_stride = tensor.stride(dim)

    # Normalize the start/stop indices
    var clamped_start = _normalize_and_clamp_dim(start, step, dim_i)
    var clamped_stop = _normalize_and_clamp_dim(end, step, dim_i)

    var new_offset = clamped_start * old_stride

    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = tensor.ptr + new_offset

    # Stride == number of elements to the next index in this dimension.
    # So to step we can just increase the stride.
    new_stride[dim] = old_stride * step

    # If the steps are positive we traverse from start, if negative from
    # stop.
    new_shape[dim] = len(range(clamped_start, clamped_stop, step))

    # Create the new view
    return LayoutTensor[
        dtype,
        Layout.row_major[tensor.rank](),
        tensor.origin,
        address_space = tensor.address_space,
    ](
        new_data,
        RuntimeLayout[Layout.row_major[tensor.rank]()](new_shape, new_stride),
    )


# ===-----------------------------------------------------------------------===#
# slice_as_view
# ===-----------------------------------------------------------------------===#


@always_inline
fn slice_as_view[
    dtype: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
](
    tensor: LayoutTensor[dtype, **_],
    starts: LayoutTensor[start_type, **_],
    ends: LayoutTensor[end_type, **_],
    steps: LayoutTensor[step_type, **_],
) -> LayoutTensor[
    dtype,
    Layout.row_major[tensor.rank](),
    tensor.origin,
    address_space = tensor.address_space,
]:
    var new_shape = IndexList[tensor.rank]()
    var new_stride = IndexList[tensor.rank]()

    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = tensor.ptr

    @parameter
    for i in range(tensor.rank):
        var start = Int(starts[i])
        var stop = Int(ends[i])
        var step = Int(steps[i])
        var dim_i = tensor.dim(i)
        var stride_i = tensor.stride(i)

        # Normalize the start/stop indices
        start = _normalize_and_clamp_dim(start, step, dim_i)
        stop = _normalize_and_clamp_dim(stop, step, dim_i)

        var new_offset = start * stride_i
        new_data = new_data.offset(new_offset)

        # Stride == number of elements to the next index in this dimension.
        # So to step we can just increase the stride.
        new_stride[i] = stride_i * step

        # If the steps are positive we traverse from start, if negative from
        # stop.
        new_shape[i] = len(range(start, stop, step))

    # Create the new view
    return LayoutTensor[
        dtype,
        Layout.row_major[tensor.rank](),
        tensor.origin,
        address_space = tensor.address_space,
    ](
        new_data,
        RuntimeLayout[Layout.row_major[tensor.rank]()](
            new_shape,
            new_stride,
        ),
    )


# ===-----------------------------------------------------------------------===#
# copy_to_slice
# ===-----------------------------------------------------------------------===#


@always_inline
fn copy_to_slice[
    dtype: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
    target: StaticString = "cpu",
](
    buffer: LayoutTensor[mut=True, dtype, **_],
    in_slice: LayoutTensor[dtype, **_],
    start: LayoutTensor[start_type, **_],
    end: LayoutTensor[end_type, **_],
    step: LayoutTensor[step_type, **_],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    var expected_shape = slice_shape[single_thread_blocking_override=True](
        buffer, start, end, step
    )

    if expected_shape != rebind[IndexList[buffer.rank]](
        in_slice.runtime_layout.shape.value.canonicalize()
    ):
        raise Error(
            "Shape mismatch for mo.mutable.store.slice: expected 'slice'",
            " operand to have shape: ",
            expected_shape,
            " but got: ",
            in_slice.runtime_layout.shape.value.canonicalize(),
        )

    var buffer_slice_view = slice_as_view(buffer, start, end, step)

    @always_inline
    @__copy_capture(in_slice, buffer_slice_view)
    @parameter
    fn copy[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        var coords = rebind[IndexList[in_slice.rank]](idx)
        var buf_index = buffer_slice_view.runtime_layout(
            RuntimeTuple[
                fill_like(buffer_slice_view.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        var slice_index = in_slice.runtime_layout(
            RuntimeTuple[fill_like(in_slice.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        buffer_slice_view.ptr.store[width=simd_width](
            buf_index, in_slice.ptr.load[width=simd_width](slice_index)
        )

    elementwise[copy, 1, target=target](
        buffer_slice_view.runtime_layout.shape.value.canonicalize(),
        context,
    )


# ===-----------------------------------------------------------------------===#
# slice_as_copy
# ===-----------------------------------------------------------------------===#


@always_inline
fn slice_as_copy[
    dtype: DType,
    index_type: DType,
](
    output: LayoutTensor[mut=True, dtype, **_],
    tensor: LayoutTensor[dtype, **_],
    start: LayoutTensor[index_type, **_],
    end: LayoutTensor[index_type, **_],
    step: LayoutTensor[index_type, **_],
) raises:
    constrained[output.rank == tensor.rank]()
    # Apply slice to the tensor
    var sliced = slice_as_view(tensor, start, end, step)

    # Copy lambda sliced view into output buffer.
    @always_inline
    @__copy_capture(sliced)
    @parameter
    fn copy[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        var index = rebind[IndexList[tensor.rank]](idx)
        var output_index = output.runtime_layout(
            RuntimeTuple[fill_like(output.layout.shape, UNKNOWN_VALUE)](index)
        )
        var slice_index = sliced.runtime_layout(
            RuntimeTuple[fill_like(sliced.layout.shape, UNKNOWN_VALUE)](index)
        )
        output.ptr.store[width=simd_width](
            output_index, sliced.ptr.load[width=simd_width](slice_index)
        )

    # Invoke copy.
    elementwise[copy, 1](output.runtime_layout.shape.value.canonicalize())


# ===-----------------------------------------------------------------------===#
# slice_shape
# ===-----------------------------------------------------------------------===#


@always_inline
fn slice_shape[
    input_type: DType,
    start_type: DType,
    stop_type: DType,
    step_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: LayoutTensor[input_type, **_],
    start_buf: LayoutTensor[start_type, **_],
    stop_buf: LayoutTensor[stop_type, **_],
    step_buf: LayoutTensor[step_type, **_],
) raises -> IndexList[input_buf.rank]:
    constrained[start_buf.rank == 1, "start_buf.rank must be 1"]()
    constrained[stop_buf.rank == 1, "stop_buf.rank must be 1"]()
    constrained[step_buf.rank == 1, "step_buf.rank must be 1"]()

    if input_buf.rank != start_buf.dim[0]():
        raise Error("[slice] start indices size must equal input rank")
    if input_buf.rank != stop_buf.dim[0]():
        raise Error("[slice] stop indices size must equal input rank")
    if input_buf.rank != step_buf.dim[0]():
        raise Error("[slice] step indices size must equal input rank")

    for axis in range(input_buf.rank):
        if step_buf[axis] == 0:
            raise Error("[slice] step must be non-zero")

    var output_shape = IndexList[input_buf.rank]()

    for i in range(input_buf.rank):
        var start = Int(start_buf[i])
        var stop = Int(stop_buf[i])
        var step = Int(step_buf[i])
        var dim_i = input_buf.dim(i)

        start = _normalize_and_clamp_dim(start, step, dim_i)
        stop = _normalize_and_clamp_dim(stop, step, dim_i)

        if step > 0 and stop < start:
            raise Error(
                "[slice] normalized stop cannot be smaller than start for"
                " positive step"
            )

        if step < 0 and start < stop:
            raise Error(
                "[slice] normalized start cannot be smaller than stop for"
                " negative step"
            )

        output_shape[i] = len(range(start, stop, step))

    return output_shape
