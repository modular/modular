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

from math import ceildiv

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import block_dim, global_idx
from gpu.host import DeviceContext
from memory import memcpy
from testing import assert_false

from utils.index import Index

# This is DeviceAttribute.MAX_THREADS_PER_BLOCK (in ONNXRT it is a global
# with value of 256).
alias MAX_THREADS_PER_BLOCK = 256


# TODO: Follow-up: Eliminate offsets calculations and use NDBuffers directly.
fn scatter_nd_gpu[
    type: DType,
    indices_type: DType,
](
    output_data_ptr: UnsafePointer[Scalar[type]],
    indices_data_ptr: UnsafePointer[Scalar[indices_type]],
    element_counts_and_input_dims_ptr: UnsafePointer[Int64],
    updates_data_ptr: UnsafePointer[Scalar[type]],
    num_indices: Int,
    last_index_dimension: Int,
    num_updates_elements: Int,
):
    var id: UInt = global_idx.x
    if id >= num_indices:
        return

    var element_counts_and_input_dims = NDBuffer[DType.int64, 1](
        element_counts_and_input_dims_ptr, Index(last_index_dimension * 2)
    )

    var data_offset = 0

    var indices_start = last_index_dimension * id
    var indices_end = indices_start + last_index_dimension

    for i in range(indices_start, indices_end):
        var index = Int(indices_data_ptr.load(i))
        var element_count_dim = Int(
            element_counts_and_input_dims[i - indices_start]
        )
        var dim_value = Int(
            element_counts_and_input_dims[
                i - indices_start + last_index_dimension
            ]
        )

        # Clamp the index if out of range.
        # This would have been an error in the CPU kernel, but throwing in the CUDA EP
        # is hard. This is the approach taken by other frameworks for out of bound indices
        # in their corresponding GPU backends as well.
        # index >= -dim_value && index < dim_value
        if index >= 0:
            if index >= dim_value:
                index = dim_value - 1
        else:
            if index < -dim_value:
                index = 0
            else:
                index += dim_value

        data_offset += index * element_count_dim

    # Set updates_data_base to appropriate offset (from where to copy).
    var updates_data_base = updates_data_ptr.offset(num_updates_elements * id)
    # Set output_data_base to appropriate offset (where to copy).
    var output_data_base = output_data_ptr.offset(data_offset)

    # Start copying appropriate amount of elements.
    for i in range(num_updates_elements):
        output_data_base[i] = updates_data_base[i]


# TODO: Extend for using reduce function if needed.
fn scatter_nd[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    updates_rank: Int,
](
    data: NDBuffer[type, data_rank, *_],
    indices: NDBuffer[indices_type, indices_rank, *_],
    updates: NDBuffer[type, updates_rank, *_],
    output: NDBuffer[type, data_rank, *_],
    ctx: DeviceContext,
) raises:
    """
    Implements ONNX ScatterND operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND.

    Parameters:
        type: Type of data, updates, and output tensors.
        indices_type: Type of the indices tensor.
        data_rank: Rank of input (data) tensor (data_rank >= 1).
        indices_rank: Rank of input (data) tensor (indices_rank >= 1).
        updates_rank: Rank of updates tensor (updates_rank = data_rank +
                      indices_rank - indices_shape[-1] - 1).

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank containing indices for the scatter
                 operation.
        updates: Tensor containing values to update output tensor based on
                 indices tensor.
        output: Tensor of rank data_rank, shaped the same as data tensor.
        ctx: DeviceContext.
    """
    if data.get_shape() != output.get_shape():
        print("Input and output shapes in scatter_nd must be the same.")

    if (
        len(updates.get_shape())
        != data_rank + indices_rank - indices.get_shape()[indices_rank - 1] - 1
    ):
        print(
            "updates rank must be: data_rank + indices_rank -"
            " indices_shape[-1] - 1"
        )

    # Copy input data to output (appropriate elements will be updated as needed
    # by the end of scatternd kernel).
    var output_flat = output.flatten()
    var data_flat = data.flatten()
    memcpy(output_flat.data, data_flat.data, len(output_flat))

    # Get shapes of buffers to be used in subsequent calculations.
    var data_shape = data.get_shape()
    var indices_shape = indices.get_shape()
    var last_shape_of_indices = indices_shape[indices_rank - 1]
    var updates_shape = updates.get_shape()

    # Depending on r_minus_m = data_rank - last_shape_of_indices,
    # we will be copying:
    #   element (r_minus_m = 0),
    #   row (r_minus_m = 1),
    #   sheet (r_minus_m = 2),
    #   cuboid (r_minus_m = 3), etc.
    var r_minus_m = data_rank - last_shape_of_indices
    # Calculate how many elements to copy/scatter (this is from the innermost
    # dimensions, and is contiguous memory locations).
    var count_copy = 1
    for i in range(r_minus_m):
        count_copy = count_copy * data_shape[data_rank - 1 - i]

    # Calculate number of (input) data elements to copy to GPU.
    var data_count_copy = 1
    for i in range(data_rank):
        data_count_copy = data_count_copy * data_shape[data_rank - 1 - i]

    # Calculate number of indices NDBuffer elements to copy to GPU.
    var indices_count_copy = 1
    for i in range(indices_rank):
        indices_count_copy = (
            indices_count_copy * indices_shape[indices_rank - 1 - i]
        )

    # Calculate number of updates NDBuffer elements to copy to GPU.
    var updates_count_copy = 1
    for i in range(updates_rank):
        updates_count_copy = (
            updates_count_copy * updates_shape[updates_rank - 1 - i]
        )

    # NDBuffer below will store both input_strides and data NDBuffer dimensions.
    # (combine both in one to reduce number of memcpy from H->D).
    var ptr = UnsafePointer[Int64].alloc(last_shape_of_indices * 2)
    var element_counts_and_input_dims = NDBuffer[DType.int64, 1](
        ptr, DimList(last_shape_of_indices * 2)
    )

    # input_strides
    # e.g., for a shape of 2, 3, 4, 5
    #       input_strides --> [3*4*5, 4*5, 5, 1]
    var input_strides = NDBuffer[
        DType.int64, 1, MutableAnyOrigin, DimList(data_rank)
    ]().stack_allocation()
    for i in range(data_rank):
        var total_stride = 1
        for j in range(i + 1, data_rank):
            total_stride *= data_shape[j]
        input_strides[i] = total_stride

    for i in range(last_shape_of_indices):
        element_counts_and_input_dims[i] = input_strides[i]
        element_counts_and_input_dims[i + last_shape_of_indices] = data_shape[i]

    # Allocate and copy output data, elements_counts_and_input_dims, updates,
    # indices to GPU.
    var output_device = ctx.enqueue_create_buffer[type](data_count_copy)
    var element_counts_and_input_dims_device = ctx.enqueue_create_buffer[
        DType.int64
    ](last_shape_of_indices * 2)
    var updates_device = ctx.enqueue_create_buffer[type](updates_count_copy)
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_count_copy
    )
    ctx.enqueue_copy(output_device, output_flat.data)
    ctx.enqueue_copy(
        element_counts_and_input_dims_device,
        element_counts_and_input_dims.data,
    )
    ctx.enqueue_copy(updates_device, updates.data)
    ctx.enqueue_copy(indices_device, indices.data)

    # Number of indices (that is without last dimension).
    # Each thread will handle one index.
    # e.g., 3,2,3 ==> 6
    var num_indices = 1
    for i in range(len(indices.get_shape()) - 1):
        num_indices *= indices.get_shape()[i]

    var num_updates_elements = count_copy

    ctx.enqueue_function[scatter_nd_gpu[type=type, indices_type=indices_type]](
        output_device,
        indices_device,
        element_counts_and_input_dims_device,
        updates_device,
        num_indices,
        last_shape_of_indices,
        num_updates_elements,
        grid_dim=(ceildiv(num_indices, MAX_THREADS_PER_BLOCK)),
        block_dim=(MAX_THREADS_PER_BLOCK),
    )

    # Copy back output data from GPU to CPU.
    ctx.enqueue_copy(output.data, output_device)
    ctx.synchronize()

    _ = output_device
    _ = element_counts_and_input_dims_device
    _ = updates_device
    _ = indices_device

    ptr.free()


fn linear_fill[
    type: DType
](buf: NDBuffer[type, *_], elems: VariadicList[Scalar[type]]):
    debug_assert(
        buf.num_elements() == len(elems), "must fill all elements of tensor"
    )

    for i in range(buf.num_elements()):
        buf[i] = elems[i]


fn test_case[
    type: DType,
    input_shape: DimList,
    indices_shape: DimList,
    updates_shape: DimList,
](
    data_vals: VariadicList[Scalar[type]],
    indices_vals: VariadicList[Int64],
    updates_vals: VariadicList[Scalar[type]],
    output_ref_vals: VariadicList[Scalar[type]],
) raises:
    var data = NDBuffer[
        type, 3, MutableAnyOrigin, input_shape
    ].stack_allocation()
    linear_fill(data, data_vals)
    var indices = NDBuffer[
        DType.int64, 2, MutableAnyOrigin, indices_shape
    ].stack_allocation()
    linear_fill(indices, indices_vals)
    var updates = NDBuffer[
        type, 3, MutableAnyOrigin, updates_shape
    ].stack_allocation()
    linear_fill(updates, updates_vals)
    var output = NDBuffer[
        type, 3, MutableAnyOrigin, input_shape
    ].stack_allocation()

    # Note: This is for the specific set of examples
    #      (due to _to_ndbuffer[] parameters).
    with DeviceContext() as ctx:
        scatter_nd(data, indices, updates, output, ctx)

    _ = data
    _ = indices
    _ = updates

    var output_ref = NDBuffer[
        type, 3, MutableAnyOrigin, input_shape
    ].stack_allocation()
    linear_fill(output_ref, output_ref_vals)

    for i in range(output.size()):
        if output_ref[i] != output[i]:
            print("FAILURE: Mismatch at idx: ", end="")
            print(i)
            assert_false(True)


fn main():
    fn test_scatternd_gpu():
        print("== test_scatternd_gpu")
        var data = VariadicList[Float32](
            # fmt: off
            1, 2, 3, 4,
            5, 6, 7, 8,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 2, 3, 4,
            5, 6, 7, 8,
            8, 7, 6, 5,
            4, 3, 2, 1,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 2, 3, 4,
            5, 6, 7, 8,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 2, 3, 4,
            5, 6, 7, 8,
            # fmt: on
        )

        var indices = VariadicList[Int64](0, 2)

        var updates = VariadicList[Float32](
            # fmt: off
            5, 5, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7,
            8, 8, 8, 8,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4,
            # fmt: on
        )

        var output_ref = VariadicList[Float32](
            # fmt: off
            5, 5, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7,
            8, 8, 8, 8,
            1, 2, 3, 4,
            5, 6, 7, 8,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 2, 3, 4,
            5, 6, 7, 8,
            # fmt: on
        )

        _ = test_case[
            DType.float32,
            input_shape = DimList(4, 4, 4),
            indices_shape = DimList(2, 1),
            updates_shape = DimList(2, 4, 4),
        ]
        (
            data,
            indices,
            updates,
            output_ref,
        )

    test_scatternd_gpu()
