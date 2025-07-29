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

from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from math import ceildiv
from sys import has_accelerator

# Vector data type and size
alias float_dtype = DType.float32
alias vector_size = 1000
alias layout = Layout.row_major(vector_size)

# Calculate the number of thread blocks needed by dividing the vector size
# by the block size and rounding up.
alias block_size = 256
alias num_blocks = ceildiv(vector_size, block_size)


fn vector_addition(
    lhs_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    rhs_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    out_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
):
    """Calculate the element-wise sum of two vectors on the GPU."""

    # Calculate the index of the vector element for the thread to process
    var tid = block_idx.x * block_dim.x + thread_idx.x

    # Don't process out of bounds elements
    if tid < vector_size:
        out_tensor[tid] = lhs_tensor[tid] + rhs_tensor[tid]


def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        # Get the context for the attached GPU
        ctx = DeviceContext()

        # Create HostBuffers for input vectors
        lhs_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )
        rhs_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )
        ctx.synchronize()

        # Initialize the input vectors
        for i in range(vector_size):
            lhs_host_buffer[i] = Float32(i)
            rhs_host_buffer[i] = Float32(i * 0.5)

        print("LHS buffer: ", lhs_host_buffer)
        print("RHS buffer: ", rhs_host_buffer)

        # Create DeviceBuffers for the input vectors
        lhs_device_buffer = ctx.enqueue_create_buffer[float_dtype](vector_size)
        rhs_device_buffer = ctx.enqueue_create_buffer[float_dtype](vector_size)

        # Copy the input vectors from the HostBuffers to the DeviceBuffers
        ctx.enqueue_copy(dst_buf=lhs_device_buffer, src_buf=lhs_host_buffer)
        ctx.enqueue_copy(dst_buf=rhs_device_buffer, src_buf=rhs_host_buffer)

        # Create a DeviceBuffer for the result vector
        result_device_buffer = ctx.enqueue_create_buffer[float_dtype](
            vector_size
        )

        # Wrap the DeviceBuffers in LayoutTensors
        lhs_tensor = LayoutTensor[float_dtype, layout](lhs_device_buffer)
        rhs_tensor = LayoutTensor[float_dtype, layout](rhs_device_buffer)
        result_tensor = LayoutTensor[float_dtype, layout](result_device_buffer)

        # Compile and enqueue the kernel
        ctx.enqueue_function_checked[vector_addition, vector_addition](
            lhs_tensor,
            rhs_tensor,
            result_tensor,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        # Create a HostBuffer for the result vector
        result_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )

        # Copy the result vector from the DeviceBuffer to the HostBuffer
        ctx.enqueue_copy(
            dst_buf=result_host_buffer, src_buf=result_device_buffer
        )

        # Finally, synchronize the DeviceContext to run all enqueued operations
        ctx.synchronize()

        print("Result vector:", result_host_buffer)
