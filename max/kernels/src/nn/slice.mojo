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

from math import clamp

from algorithm import elementwise
from gpu.host import DeviceContext, get_gpu_target
from layout._coord import Coord, DynamicCoord, Idx, coord_to_index_list
from layout._layout import Layout, row_major
from layout._tile_tensor import TileTensor
from runtime.asyncrt import DeviceContextPtr
from sys.info import simd_width_of, _current_target

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
    tensor: TileTensor[dtype, ...], start: Int, end: Int, step: Int
) -> TileTensor[
    dtype,
    Layout[
        shape_types = DynamicCoord[DType.int64, tensor.rank].element_types,
        stride_types = DynamicCoord[DType.int64, tensor.rank].element_types,
    ],
    tensor.origin,
    address_space = tensor.address_space,
]:
    var new_shape = coord_to_index_list(tensor.layout.shape_coord())
    var new_stride = coord_to_index_list(tensor.layout.stride_coord())

    var dim_i = Int(tensor.dim(dim))
    var old_stride = Int(tensor.dynamic_stride(dim))

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
    return {
        new_data,
        Layout(
            Coord(rebind[IndexList[tensor.rank]](new_shape)),
            Coord(rebind[IndexList[tensor.rank]](new_stride)),
        ),
    }


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
    tensor: TileTensor[dtype, ...],
    starts: TileTensor[start_type, ...],
    ends: TileTensor[end_type, ...],
    steps: TileTensor[step_type, ...],
) -> TileTensor[
    dtype,
    Layout[
        shape_types = DynamicCoord[DType.int64, tensor.rank].element_types,
        stride_types = DynamicCoord[DType.int64, tensor.rank].element_types,
    ],
    tensor.origin,
    address_space = tensor.address_space,
]:
    comptime assert starts.flat_rank == 1
    comptime assert ends.flat_rank == 1
    comptime assert steps.flat_rank == 1

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
        var dim_i = Int(tensor.dim(i))
        var stride_i = Int(tensor.dynamic_stride(i))

        # Normalize the start/stop indices
        start = _normalize_and_clamp_dim(start, step, dim_i)
        stop = _normalize_and_clamp_dim(stop, step, dim_i)

        var new_offset = start * stride_i
        new_data = new_data + new_offset

        # Stride == number of elements to the next index in this dimension.
        # So to step we can just increase the stride.
        new_stride[i] = stride_i * step

        # If the steps are positive we traverse from start, if negative from
        # stop.
        new_shape[i] = len(range(start, stop, step))

    # Create the new view
    return {
        new_data,
        Layout(
            Coord(new_shape),
            Coord(rebind[type_of(new_shape)](new_stride)),
        ),
    }


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
    buffer: TileTensor[mut=True, dtype, ...],
    in_slice: TileTensor[dtype, ...],
    start: TileTensor[start_type, ...],
    end: TileTensor[end_type, ...],
    step: TileTensor[step_type, ...],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    var expected_shape = slice_shape[single_thread_blocking_override=True](
        buffer, start, end, step
    )

    if expected_shape != rebind[IndexList[buffer.rank]](
        coord_to_index_list(in_slice.layout.shape_coord())
    ):
        raise Error(
            "Shape mismatch for mo.mutable.store.slice: expected 'slice'",
            " operand to have shape: ",
            expected_shape,
            " but got: ",
            coord_to_index_list(in_slice.layout.shape_coord()),
        )

    var buffer_slice_view = slice_as_view(buffer, start, end, step)

    @always_inline
    @__copy_capture(in_slice, buffer_slice_view)
    @parameter
    fn copy[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        var coords = rebind[IndexList[in_slice.rank]](idx)
        var buf_index = buffer_slice_view.layout(Coord(coords))
        var slice_index = in_slice.layout(Coord(coords))
        buffer_slice_view.ptr.store[width=simd_width](
            buf_index, in_slice.ptr.load[width=simd_width](slice_index)
        )

    elementwise[copy, 1, target=target](
        coord_to_index_list(buffer_slice_view.layout.shape_coord()),
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
    output: TileTensor[mut=True, dtype, ...],
    tensor: TileTensor[dtype, ...],
    start: TileTensor[index_type, ...],
    end: TileTensor[index_type, ...],
    step: TileTensor[index_type, ...],
) raises:
    comptime assert output.flat_rank == tensor.flat_rank
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
        var output_index = output.layout(Coord(index))
        var slice_index = sliced.layout(Coord(index))
        output.ptr.store[width=simd_width](
            output_index, sliced.ptr.load[width=simd_width](slice_index)
        )

    # Invoke copy.
    elementwise[copy, 1](coord_to_index_list(output.layout.shape_coord()))


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
    input_buf: TileTensor[input_type, ...],
    start_buf: TileTensor[start_type, ...],
    stop_buf: TileTensor[stop_type, ...],
    step_buf: TileTensor[step_type, ...],
) raises -> IndexList[input_buf.rank]:
    comptime assert start_buf.flat_rank == 1, "start_buf.rank must be 1"
    comptime assert stop_buf.flat_rank == 1, "stop_buf.rank must be 1"
    comptime assert step_buf.flat_rank == 1, "step_buf.rank must be 1"

    if input_buf.rank != Int(start_buf.dim[0]()):
        raise Error("[slice] start indices size must equal input rank")
    if input_buf.rank != Int(stop_buf.dim[0]()):
        raise Error("[slice] stop indices size must equal input rank")
    if input_buf.rank != Int(step_buf.dim[0]()):
        raise Error("[slice] step indices size must equal input rank")

    for axis in range(input_buf.rank):
        if step_buf[axis] == 0:
            raise Error("[slice] step must be non-zero")

    var output_shape = IndexList[input_buf.rank]()

    for i in range(input_buf.rank):
        var start = Int(start_buf[i])
        var stop = Int(stop_buf[i])
        var step = Int(step_buf[i])
        var dim_i = Int(input_buf.dim(i))

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


# ===-----------------------------------------------------------------------===#
# sliced_add
# ===-----------------------------------------------------------------------===#


fn sliced_add[
    dtype: DType,
    //,
    target: StaticString,
](
    c: TileTensor[mut=True, dtype, ...],
    a: TileTensor[dtype, ...],
    b: TileTensor[dtype, ...],
    lora_end_idx: TileTensor[DType.int64, ...],
    ctx: Optional[DeviceContext],
) raises:
    """Adds tensors a and b element-wise for rows < lora_end_idx, otherwise copies a.

    This is used for LoRA where only some sequences have LoRA applied.
    For rows in [0, lora_end_idx): c = a + b
    For rows in [lora_end_idx, batch_seq_len): c = a

    Args:
        c: Output tensor.
        a: First input tensor.
        b: Second input tensor.
        lora_end_idx: Scalar tensor with end index of LoRA token portion (rows to apply add).
        ctx: Device context for GPU operations.
    """
    comptime assert lora_end_idx.flat_rank == 1

    var batch_end_idx = Int(lora_end_idx[0])

    @parameter
    @__copy_capture(batch_end_idx, c, a, b)
    fn _sliced_add[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        var out_val: SIMD[dtype, width]
        var coords = Coord(idx)

        comptime assert a.flat_rank == coords.flat_rank
        comptime assert b.flat_rank == coords.flat_rank
        comptime assert c.flat_rank == coords.flat_rank

        if idx[0] >= batch_end_idx:
            out_val = a.load[width](coords)
        else:
            var a_val = a.load[width](coords)
            var b_val = b.load[width](coords)
            out_val = a_val + b_val

        c.store[width](coords, out_val)

    @parameter
    if target == "gpu":
        debug_assert(ctx is not None, "DeviceContext required for GPU target")
        comptime compile_target = get_gpu_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[_sliced_add, simd_width, target=target](
            coord_to_index_list(c.layout.shape_coord()),
            ctx.value(),
        )
    else:
        comptime compile_target = _current_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[_sliced_add, simd_width, target=target](
            coord_to_index_list(c.layout.shape_coord())
        )
