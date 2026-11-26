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
"""Implements the reshape operation that reinterprets a contiguous tensor's data under a new shape."""

from layout import Coord, TileTensor
from layout.coord import CoordLike, DynamicCoord
from layout.layout import Layout


# Reshape assumes inputs are contiguous. It should always be fused last and
# a non-contiguous tensor cannot be fused *into* this as input.
@inline(.always)
def reshape[
    dtype: DType,
    shape_types: TypeList[Trait=CoordLike, ...],
    //,
](
    input: TileTensor[dtype, ...],
    new_shape: Coord[*shape_types],
) -> TileTensor[
    dtype,
    Layout[
        shape_types=DynamicCoord[.int64, shape_types.length].element_types,
        stride_types=DynamicCoord[.int64, shape_types.length].element_types,
    ],
    input.origin,
    address_space=input.address_space,
    Engine=input.Engine,
]:
    """Returns a view of the contiguous `input` tensor reinterpreted under `new_shape` with matching element count.

    Computes row-major (contiguous) strides for the requested shape and
    constructs a new `TileTensor` that shares the input's underlying buffer,
    origin, address space, and engine. The caller is responsible for ensuring
    the input is contiguous.

    Parameters:
        dtype: Element type of the input tensor.
        shape_types: Element types of the requested output shape.

    Args:
        input: Contiguous source tensor whose data is reinterpreted.
        new_shape: Desired output shape, whose element product must match
            the number of elements in `input`.

    Returns:
        A `TileTensor` view over the same buffer with the new shape and
        contiguous strides.
    """
    comptime output_rank = shape_types.length
    var shape = DynamicCoord[.int64, output_rank]()
    var stride = DynamicCoord[.int64, output_rank]()
    var acc = 1
    comptime for i in reversed(range(output_rank)):
        var dim = Int(new_shape[i])
        shape.replace_dynamic[i](Scalar[.int64](dim))
        stride.replace_dynamic[i](Scalar[.int64](acc))
        acc *= dim
    return {input._storage, Layout(shape, stride)}


@inline(.always)
def reshape_shape[
    output_rank: Int,
    input_type: DType,
    target_shape_type: DType,
](
    input_buf: TileTensor[input_type, ...],
    target_shape_buf: TileTensor[target_shape_type, ...],
) raises -> DynamicCoord[.int, output_rank]:
    """Reads a target shape from a buffer and returns it as a `Coord`, inferring any single `-1` dimension.

    Copies the rank-1 `target_shape_buf` into a `Coord` of rank `output_rank`,
    allowing at most one dimension to be specified as `-1`, which is then inferred
    from the input's element count. Raises an error if the constraints are
    violated or the inferred shape does not match the input's number of elements.

    Parameters:
        output_rank: Expected rank of the target shape.
        input_type: Element type of the input buffer.
        target_shape_type: Element type of the target shape buffer.

    Args:
        input_buf: Source tensor whose element count constrains the inferred shape.
        target_shape_buf: Rank-1 tensor holding the desired output dimensions,
            where at most one entry may be `-1`.

    Returns:
        The target shape with any `-1` dimension resolved.

    Raises:
        Error: If `target_shape_buf` is not rank 1, the rank does not match
            `output_rank`, more than one `-1` is present, a negative value
            other than `-1` appears, or the resulting element count does not
            match the input.
    """
    comptime assert (
        target_shape_buf.flat_rank == 1
    ), "target_shape_buf must be rank 1"
    if output_rank != Int(target_shape_buf.dim(0)):
        raise Error("[reshape] requires (len(target_shape) == output_rank)")

    # move the target shape from buffer into an array; also check and record
    # if there's any to-be-inferred dimension (-1).
    var target_shape = Array[Int, output_rank](fill=0)
    var to_be_inferred_axis = -1
    var non_negative_dim_product = 1
    for axis in range(output_rank):
        var target_dim = Int(target_shape_buf[axis])
        target_shape[axis] = target_dim
        if target_dim < 0:
            if target_dim != -1:
                raise Error(
                    "[reshape] only -1 is allowed as a negative value in target"
                    " shape"
                )
            if to_be_inferred_axis != -1:
                raise Error("[reshape] only one -1 is allowed in target shape")
            to_be_inferred_axis = axis
        else:
            non_negative_dim_product *= target_dim

    var input_num_elems = input_buf.num_elements()
    var output_num_elems = non_negative_dim_product
    # Infer a dimension as the remaining elements, if needed.
    if to_be_inferred_axis != -1:
        target_shape[to_be_inferred_axis] = (
            input_num_elems // non_negative_dim_product
        )
        output_num_elems *= target_shape[to_be_inferred_axis]

    if output_num_elems != input_num_elems:
        raise Error("[reshape] input and output number of elements must match")

    return Coord(target_shape)
