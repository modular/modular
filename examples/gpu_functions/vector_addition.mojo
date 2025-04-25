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

from gpu import thread_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import ceildiv
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

alias float_dtype = DType.float32
alias VECTOR_WIDTH = 10
alias layout = Layout.row_major(VECTOR_WIDTH)


def main():
    constrained[
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator(),
        "This example requires a supported GPU",
    ]()

    # Get context for the attached GPU
    var ctx = DeviceContext()

    # Allocate data on the GPU address space
    var lhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var rhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var out_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)

    # Fill in values across the entire width
    _ = lhs_buffer.enqueue_fill(1.25)
    _ = rhs_buffer.enqueue_fill(2.5)

    # Wrap the device buffers in tensors
    var lhs_tensor = LayoutTensor[float_dtype, layout](lhs_buffer)
    var rhs_tensor = LayoutTensor[float_dtype, layout](rhs_buffer)
    var out_tensor = LayoutTensor[float_dtype, layout](out_buffer)

    # Launch the vector_addition function as a GPU kernel
    ctx.enqueue_function[vector_addition](
        lhs_tensor,
        rhs_tensor,
        out_tensor,
        grid_dim=1,
        block_dim=VECTOR_WIDTH,
    )

    # Map to host so that values can be printed from the CPU
    with out_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[float_dtype, layout](host_buffer)
        print("Resulting vector:", host_tensor)


fn vector_addition(
    lhs_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    rhs_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    out_tensor: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
):
    """The calculation to perform across the vector on the GPU."""
    var tid = thread_idx.x
    out_tensor[tid] = lhs_tensor[tid] + rhs_tensor[tid]
