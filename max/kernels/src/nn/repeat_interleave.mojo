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

from sys import simd_width_of

from algorithm.functional import elementwise
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.layout import is_row_major
from layout.int_tuple import fill_like
from buffer import NDBuffer
from tensor_internal._indexing import _row_major_strides

from utils import IndexList


fn _collapse_dims_around_axis(
    shape: IndexList, axis: Int, out result: IndexList[3]
) raises:
    if axis >= shape.size:
        raise Error("axis larger than provided shape")

    @parameter
    if shape.size == 0:
        return IndexList[3](1, 1, 1)

    var split = shape[axis]

    var before = 1
    for i in range(axis):
        before *= shape[i]

    var after = 1
    for i in range(axis + 1, shape.size):
        after *= shape[i]

    return IndexList[3](before, split, after)


@always_inline
fn repeat_interleave[
    dtype: DType,
    type_repeats: DType,
](
    input: LayoutTensor[dtype, *_, **_],
    repeats: LayoutTensor[type_repeats, *_, **_],
    axis: Int,
    output: LayoutTensor[mut=True, dtype, *_, **_],
) raises:
    """
    Fill `output` by repeating values from `input` along `axis` based on the
    values in `repeats` buffer.

    This is intended to implement the same functionality as torch.repeat:
    https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html

    Args:
        input: The input buffer.
        repeats: The number of repetitions each element in input.
        axis: The axis along which to repeat values.
        output: The output buffer.
    """
    debug_assert(
        is_row_major[input.rank](input.layout)
        and is_row_major[output.rank](output.layout)
    )
    constrained[input.rank == output.rank]()
    constrained[repeats.rank == 1]()

    # Compute the shape of the input and result buffers.
    # These are the shapes of the buffers we will be working on.
    var collapsed_input_shape = _collapse_dims_around_axis(
        input.runtime_layout.shape.value, axis
    )
    var collapsed_output_shape = _collapse_dims_around_axis(
        output.runtime_layout.shape.value, axis
    )

    debug_assert(collapsed_output_shape[0] == collapsed_input_shape[0])
    debug_assert(collapsed_output_shape[2] == collapsed_input_shape[2])

    var collapsed_input = LayoutTensor[dtype, Layout.row_major[3](), _, **_](
        input.ptr,
        RuntimeLayout[Layout.row_major[3]()].row_major(collapsed_input_shape),
    )

    var collapsed_output = LayoutTensor[dtype, Layout.row_major[3](), _, **_](
        output.ptr,
        RuntimeLayout[Layout.row_major[3]()].row_major(collapsed_output_shape),
    )

    var input_repeat_dim = collapsed_input_shape[1]
    var output_repeat_dim = collapsed_output_shape[1]
    var repeat_stride = Int(repeats.dim[0]() > 1)

    # Mapping from offsets in the input tensor to offsets in the output tensor
    # along the repeat axis.
    var offset_mapping = List[Int](unsafe_uninit_length=output_repeat_dim)

    var repeat_offset = 0
    var output_offset = 0
    for input_offset in range(input_repeat_dim):
        var repeat_val = Int(repeats[repeat_offset])
        if repeat_val < 0:
            raise Error("all repeat values must be non-negative")

        for _ in range(repeat_val):
            offset_mapping[output_offset] = input_offset
            output_offset += 1

        repeat_offset += repeat_stride

    @always_inline
    @parameter
    fn func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var output_index = rebind[IndexList[3]](idx)
        var input_index = output_index
        input_index[1] = offset_mapping[output_index[1]]

        var input_idx = collapsed_input.runtime_layout(
            RuntimeTuple[
                fill_like(collapsed_input.layout.shape, UNKNOWN_VALUE)
            ](input_index)
        )
        var input_value = collapsed_input.ptr.load[width=width](input_idx)

        var output_idx = collapsed_output.runtime_layout(
            RuntimeTuple[
                fill_like(collapsed_output.layout.shape, UNKNOWN_VALUE)
            ](output_index)
        )
        collapsed_output.ptr.store(output_idx, input_value)

    elementwise[func, simd_width_of[output.dtype]()](
        collapsed_output.runtime_layout.shape.value
    )


@always_inline
fn repeat_interleave_shape[
    type_repeats: DType,
](
    input: LayoutTensor[*_, **_],
    repeats: LayoutTensor[type_repeats, _, _, **_],
    axis: Int,
) raises -> IndexList[input.rank]:
    constrained[type_repeats.is_integral()]()
    constrained[repeats.rank == 1]()

    var repeats_size = repeats.dim[0]()
    if repeats_size != 1 and repeats_size != input.dim(axis):
        raise Error(
            "repeat_interleave: repeats must be size 1 or equal to "
            "the size of input[axis]"
        )

    var total_repeats = 0
    for i in range(repeats_size):
        total_repeats += Int(repeats[i])

    var result = rebind[IndexList[input.rank]](
        input.runtime_layout.shape.value.canonicalize()
    )

    # If the repeats is size 1, the repeat is treated as a broadcast
    if repeats_size == 1:
        result[axis] *= total_repeats
    else:
        result[axis] = total_repeats

    return result
