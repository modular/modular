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

from sys.info import sizeof

from layout import LayoutTensor, Layout
from memory import UnsafePointer, memcpy
from memory.unsafe_pointer import _default_alignment

# ===-----------------------------------------------------------------------===#
# _get_rightmost_broadcast_axis
# ===-----------------------------------------------------------------------===#


fn _get_rightmost_broadcast_axis[
    type: DType,
](
    input: LayoutTensor[type, **_],
    output: LayoutTensor[mut=True, type, **_],
) -> Int:
    """
    Return the rightmost position (largest axis) at which the dimensions of
    `input_shape` and `output_shape` mismatch, otherwise return -1 (i.e., the
    shapes are equal).

    Args:
        input: the input buffer
        output: the output buffer
    """
    # TODO: consider manually unrolling this loop
    for axis in reversed(range(input.rank)):
        var in_dim = input.runtime_layout.dim(axis)
        var out_dim = output.runtime_layout.dim(axis)
        if in_dim != out_dim:
            return axis
    return -1


# ===-----------------------------------------------------------------------===#
# broadcast
# ===-----------------------------------------------------------------------===#


fn broadcast[
    type: DType,
](
    output: LayoutTensor[
        mut=True, type, address_space = AddressSpace.GENERIC, **_
    ],
    input: LayoutTensor[type, address_space = AddressSpace.GENERIC, **_],
):
    """
    For each axis of `input`, if the dimension is 1, duplicate the data at
    each index of the corresponding axis in `output`, otherwise copy over the
    entire axis to the corresponding axis in `output`.

    Args:
        output: The output buffer.
        input: The input buffer.
    """
    # short-circuit if any dimension of the output is 0, this way we don't need
    # to worry about such cases in the kernel implementation.
    if output.size() == 0:
        return

    var rightmost_broadcast_axis: Int = _get_rightmost_broadcast_axis[type](
        input, output
    )

    var input_output_have_same_shape = rightmost_broadcast_axis == -1
    if input_output_have_same_shape:
        var src_ptr = input.ptr
        var dst_ptr = output.ptr
        memcpy(dst_ptr, src_ptr, input.size())
        return

    alias init_axis = 0
    # imaginary axis before 0
    var init_input_prev_axis_stride = input.size()
    var init_output_prev_axis_stride = output.size()
    broadcast_impl[type](
        init_axis,
        output,
        input,
        init_input_prev_axis_stride,
        init_output_prev_axis_stride,
        0,  # input_offset
        0,  # output_offset
        rightmost_broadcast_axis,
    )


fn broadcast_impl[
    type: DType,
](
    axis: Int,
    output: LayoutTensor[
        mut=True, type, address_space = AddressSpace.GENERIC, **_
    ],
    input: LayoutTensor[type, address_space = AddressSpace.GENERIC, **_],
    # using `prev` because otherwise computing `next_input_axis_stride` requires
    # dim[axis+1](), which requires more `constrained` to keep in bound
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
        axis: The axis value.
        output: The output buffer.
        input: The input buffer.
        input_prev_axis_stride: The stride at axis `axis - 1` for input.
        output_prev_axis_stride: The stride at axis `axis - 1` for output.
        input_offset: The offset at which we start copying data from.
        output_offset: The offset at which we start copying data to.
        rightmost_broadcast_axis: The largest axis at which we need to duplicate `input` data.
    """
    if axis >= input.rank:
        return
    var input_axis_stride = input_prev_axis_stride // input.runtime_layout.dim(
        axis
    )

    if axis == rightmost_broadcast_axis:
        var elems_to_copy = input_axis_stride
        _tile_1d(
            output.ptr.offset(output_offset),
            input.ptr.offset(input_offset),
            input_axis_stride,  # elems_to_copy
            output.runtime_layout.dim(axis),
        )
        return

    var output_axis_stride = output_prev_axis_stride // output.runtime_layout.dim(
        axis
    )

    var next_input_offset = input_offset
    var next_output_offset = output_offset
    for _ in range(input.runtime_layout.dim(axis)):
        broadcast_impl[type](
            axis + 1,
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
    # duplicate data in output, e.g.,
    #  broadcast([[1]]), shape (1, 1) to shape (2, 3):
    #     [[0, 0, 0], [0, 0, 0]]
    # --> [[1, 1, 1], [0, 0, 0]]   after recursive call to next axis
    # --> [[1, 1, 1], [1, 1, 1]]   after duplicating data in output
    if input.runtime_layout.dim(axis) != output.runtime_layout.dim(axis):
        var output_tile_start = output.ptr.offset(output_offset)
        _tile_1d(
            output_tile_start.offset(
                output_axis_stride
            ),  # 1st tile is already there
            output_tile_start,
            output_axis_stride,  # elems_to_copy
            output.runtime_layout.dim(axis) - 1,  # 1st tile is already there
        )


fn _tile_1d[
    type: DType,
](
    init_dst_ptr: UnsafePointer[
        Scalar[type],
        address_space = AddressSpace.GENERIC,
        mut=True, **_,
    ],
    src_ptr: UnsafePointer[
        Scalar[type],
        address_space = AddressSpace.GENERIC, **_,
    ],
    tile_num_elems: Int,
    n: Int,
):
    """
    Repeat data from `src_ptr[:tile_num_elems]` in `init_dst_ptr` for `n` times
    """
    var dst_ptr = init_dst_ptr
    for i in range(n):
        memcpy(dst_ptr, src_ptr, tile_num_elems)
        dst_ptr = dst_ptr.offset(tile_num_elems)
