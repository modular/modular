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

from layout import Layout, LayoutTensor, RuntimeLayout
from register import register_internal

from utils.index import IndexList


# Reshape assumes inputs are contiguous. It should always be fused last and
# a non-contiguous tensor cannot be fused *into* this as input.
@always_inline
fn reshape[
    dtype: DType, //,
    output_rank: Int,
    single_thread_blocking_override: Bool = True,
](
    input: LayoutTensor[dtype, **_],
    new_shape: IndexList[output_rank],
) -> LayoutTensor[
    dtype,
    Layout.row_major[output_rank](),
    input.origin,
    address_space = input.address_space,
]:
    var stride_tuple = __type_of(new_shape)()
    var stride: Int = 1

    # Create contiguous strides.
    @parameter
    for i in reversed(range(output_rank)):
        # Start from the back so we can accumulate the strides.
        stride_tuple[i] = stride
        stride *= new_shape[i]

    # Return the a view with the new shape.
    return LayoutTensor[
        dtype,
        Layout.row_major[output_rank](),
        address_space = input.address_space,
    ](
        input.ptr,
        RuntimeLayout[Layout.row_major[output_rank]()](new_shape, stride_tuple),
    )


@register_internal("layout_tensor_reshape")
@always_inline
fn layout_tensor_reshape[
    output_rank: Int,
    dtype: DType,
    single_thread_blocking_override: Bool,
](
    input: LayoutTensor[dtype, **_],
    new_shape: IndexList[output_rank],
) -> LayoutTensor[
    dtype,
    Layout.row_major[output_rank](),
    input.origin,
    address_space = input.address_space,
]:
    return reshape[
        output_rank,
        single_thread_blocking_override=single_thread_blocking_override,
    ](input, new_shape)


@always_inline
fn reshape_shape[
    output_rank: Int,
    input_type: DType,
    target_shape_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: LayoutTensor[input_type, **_],
    target_shape_buf: LayoutTensor[target_shape_type, **_],
) raises -> IndexList[output_rank]:
    constrained[target_shape_buf.rank == 1, "target_shape_buf must be rank 1"]()
    if output_rank != target_shape_buf.dim(0):
        raise Error("[reshape] requires (len(target_shape) == output_rank)")

    # move the target shape from buffer into a static int tuple; also check and
    # record if there's any to-be-inferred dimension (-1).
    var target_shape = IndexList[output_rank]()
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

    var input_num_elems = input_buf.size()
    var output_num_elems = non_negative_dim_product
    # Infer a dimension as the remaining elements, if needed.
    if to_be_inferred_axis != -1:
        target_shape[to_be_inferred_axis] = (
            input_num_elems // non_negative_dim_product
        )
        output_num_elems *= target_shape[to_be_inferred_axis]

    if output_num_elems != input_num_elems:
        raise Error("[reshape] input and output number of elements must match")

    return target_shape
