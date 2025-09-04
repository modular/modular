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
from sys import has_apple_gpu_accelerator

from gpu import global_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from testing import (
    assert_equal,
)

alias float_dtype = DType.float32
alias VECTOR_WIDTH = 10
alias BLOCK_SIZE = 5
alias layout = Layout.row_major(VECTOR_WIDTH)


struct Vector:
    var ptr: UnsafePointer[Float32]
    var size: Int

    #    var nested_vector: NestedVector

    fn __init__(out self, ptr: UnsafePointer[Float32], size: Int):
        self.ptr = ptr
        self.size = size


#        self.nested_vector = NestedVector(ptr, size)


def main():
    constrained[
        has_apple_gpu_accelerator(),
        "This example requires a supported GPU",
    ]()

    # Get context for the attached GPU
    var ctx = DeviceContext()

    # Allocate data on the GPU address space
    var lhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var rhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var out_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)

    # Calculate the number of blocks needed to cover the vector
    var grid_dim = ceildiv(VECTOR_WIDTH, BLOCK_SIZE)

    # Fill in values across the entire width
    _ = lhs_buffer.enqueue_fill(1.25)
    _ = rhs_buffer.enqueue_fill(2.5)

    # Launch the vector_addition function as a GPU kernel
    ctx.enqueue_function[vector_addition_all_unsafe](
        lhs_buffer.unsafe_ptr(),
        rhs_buffer.unsafe_ptr(),
        out_buffer.unsafe_ptr(),
        VECTOR_WIDTH,
        grid_dim=grid_dim,
        block_dim=BLOCK_SIZE,
    )

    # Map to host so that values can be printed from the CPU
    with out_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[float_dtype, layout](host_buffer)
        print("Resulting vector with unsafe pointers:", host_tensor)
        for i in range(VECTOR_WIDTH):
            assert_equal(host_buffer[i], 3.75)

    _ = lhs_buffer.enqueue_fill(2.25)
    _ = rhs_buffer.enqueue_fill(3.5)

    var lhs_vector = Vector(ptr=lhs_buffer.unsafe_ptr(), size=VECTOR_WIDTH)
    var rhs_vector = Vector(ptr=rhs_buffer.unsafe_ptr(), size=VECTOR_WIDTH)
    var out_vector = Vector(ptr=out_buffer.unsafe_ptr(), size=VECTOR_WIDTH)

    # Launch the vector_addition function as a GPU kernel
    ctx.enqueue_function[vector_addition_vector](
        lhs_vector,
        rhs_vector,
        out_vector,
        grid_dim=grid_dim,
        block_dim=BLOCK_SIZE,
    )

    # Map to host so that values can be printed from the CPU
    with out_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[float_dtype, layout](host_buffer)
        print(
            "Resulting vector with custom vector struct pointers:", host_tensor
        )
        for i in range(VECTOR_WIDTH):
            assert_equal(host_buffer[i], 5.75)

    _ = lhs_buffer.enqueue_fill(4.25)
    _ = rhs_buffer.enqueue_fill(5.5)
    _ = out_buffer.enqueue_fill(0.0)
    var lhs_tensor = LayoutTensor[float_dtype, layout](lhs_buffer)
    var rhs_tensor = LayoutTensor[float_dtype, layout](rhs_buffer)
    var out_tensor = LayoutTensor[float_dtype, layout](out_buffer)

    ctx.enqueue_function[vector_addition_tensor](
        lhs_tensor,
        rhs_tensor,
        out_tensor,
        VECTOR_WIDTH,
        grid_dim=grid_dim,
        block_dim=BLOCK_SIZE,
    )

    # Map to host so that values can be printed from the CPU
    with out_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[float_dtype, layout](host_buffer)
        print("Resulting vector with layout tensor:", host_tensor)
        for i in range(VECTOR_WIDTH):
            assert_equal(host_buffer[i], 9.75)


fn vector_addition_vector(
    lhs_tensor: Vector,
    rhs_tensor: Vector,
    out_tensor: Vector,
):
    """The calculation to perform across the vector on the GPU."""
    var global_tid = global_idx.x
    if global_tid < UInt(lhs_tensor.size):
        out_tensor.ptr[global_tid] = (
            lhs_tensor.ptr[global_tid] + rhs_tensor.ptr[global_tid]
        )


fn vector_addition_all_unsafe(
    lhs_tensor: UnsafePointer[Float32],
    rhs_tensor: UnsafePointer[Float32],
    out_tensor: UnsafePointer[Float32],
    size: Int,
):
    """The calculation to perform across the vector on the GPU."""
    var global_tid = global_idx.x
    if global_tid < UInt(size):
        out_tensor[global_tid] = lhs_tensor[global_tid] + rhs_tensor[global_tid]


fn vector_addition_tensor(
    lhs_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    rhs_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    out_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    size: Int,
):
    """The calculation to perform across the vector on the GPU."""
    var global_tid = global_idx.x
    if global_tid < UInt(size):
        out_tensor[global_tid] = (
            lhs_tensor.ptr[global_tid] + rhs_tensor.ptr[global_tid]
        )
