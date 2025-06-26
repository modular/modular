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

from sys import simdwidthof

from algorithm.functional import elementwise
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
    rank: Int,
    type_repeats: DType,
](
    input: NDBuffer[dtype, rank],
    repeats: NDBuffer[type_repeats, 1],
    axis: Int,
    output: NDBuffer[mut=True, dtype, rank],
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
        _row_major_strides(input.get_shape()) == input.get_strides()
        and _row_major_strides(output.get_shape()) == output.get_strides()
    )

    # Compute the shape of the input and result buffers.
    # These are the shapes of the buffers we will be working on.
    var collapsed_input_shape = _collapse_dims_around_axis(
        input.get_shape(), axis
    )
    var collapsed_output_shape = _collapse_dims_around_axis(
        output.get_shape(), axis
    )

    debug_assert(collapsed_output_shape[0] == collapsed_input_shape[0])
    debug_assert(collapsed_output_shape[2] == collapsed_input_shape[2])

    var collapsed_input = NDBuffer[dtype, 3](
        input.data,
        dynamic_shape=collapsed_input_shape,
    )
    var collapsed_output = NDBuffer[dtype, 3](
        output.data,
        dynamic_shape=collapsed_output_shape,
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
    fn func[width: Int, rank: Int](idx: IndexList[rank]):
        var output_index = rebind[IndexList[3]](idx)
        var input_index = output_index
        input_index[1] = offset_mapping[output_index[1]]

        var input_value = collapsed_input.load[width=width](input_index)
        collapsed_output.store(output_index, input_value)

    elementwise[func, simdwidthof[output.type]()](collapsed_output.get_shape())


@always_inline
fn repeat_interleave_shape[
    type_repeats: DType,
](
    input: NDBuffer, repeats: NDBuffer[type_repeats, 1], axis: Int
) raises -> IndexList[input.rank]:
    constrained[type_repeats.is_integral()]()

    var repeats_size = repeats.dim[0]()
    if repeats_size != 1 and repeats_size != input.dim(axis):
        raise Error(
            "repeat_interleave: repeats must be size 1 or equal to "
            "the size of input[axis]"
        )

    var total_repeats = 0
    for i in range(repeats_size):
        total_repeats += Int(repeats[i])

    var result = input.get_shape()

    # If the repeats is size 1, the repeat is treated as a broadcast
    if repeats_size == 1:
        result[axis] *= total_repeats
    else:
        result[axis] = total_repeats

    return result
