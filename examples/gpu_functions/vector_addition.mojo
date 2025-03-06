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

from gpu.host import Dim
from gpu.id import block_dim, block_idx, thread_idx
from layout import LayoutTensor, Layout
from math import ceildiv
from max.driver import (
    Accelerator,
    Device,
    Tensor,
    accelerator,
    cpu,
)
from sys import has_nvidia_gpu_accelerator

alias float_dtype = DType.float32
alias tensor_rank = 1


fn vector_addition[
    lhs_layout: Layout,
    rhs_layout: Layout,
    out_layout: Layout,
](
    length: Int,
    lhs: LayoutTensor[float_dtype, lhs_layout, MutableAnyOrigin],
    rhs: LayoutTensor[float_dtype, rhs_layout, MutableAnyOrigin],
    out: LayoutTensor[float_dtype, out_layout, MutableAnyOrigin],
):
    """The calculation to perform across the vector on the GPU."""
    tid = block_dim.x * block_idx.x + thread_idx.x
    if tid < length:
        var result = lhs[tid] + rhs[tid]
        out[tid] = result


def main():
    @parameter
    if has_nvidia_gpu_accelerator():
        # Attempt to connect to a compatible GPU. If one is not found, this will
        # error out and exit.
        gpu_device = accelerator()
        host_device = cpu()

        alias VECTOR_WIDTH = 10

        # Allocate the two input tensors on the host.
        lhs_tensor = Tensor[float_dtype, 1]((VECTOR_WIDTH), host_device)
        rhs_tensor = Tensor[float_dtype, 1]((VECTOR_WIDTH), host_device)

        # Fill them with initial values.
        for i in range(VECTOR_WIDTH):
            lhs_tensor[i] = 1.25
            rhs_tensor[i] = 2.5

        # Move the input tensors to the accelerator.
        lhs_tensor = lhs_tensor.move_to(gpu_device)
        rhs_tensor = rhs_tensor.move_to(gpu_device)

        # Allocate a tensor on the accelerator to host the calculation results.
        out_tensor = Tensor[float_dtype, tensor_rank](
            (VECTOR_WIDTH), gpu_device
        )
        lhs_layout_tensor = lhs_tensor.to_layout_tensor()
        rhs_layout_tensor = rhs_tensor.to_layout_tensor()
        out_layout_tensor = out_tensor.to_layout_tensor()

        # Compile the function to run across a grid on the GPU.
        gpu_function = Accelerator.compile[
            vector_addition[
                lhs_layout_tensor.layout,
                rhs_layout_tensor.layout,
                out_layout_tensor.layout,
            ]
        ](gpu_device)

        # The grid is divided up into blocks, making sure there's an extra
        # full block for any remainder. This hasn't been tuned for any specific
        # GPU.
        alias BLOCK_SIZE = 16
        var num_blocks = ceildiv(VECTOR_WIDTH, BLOCK_SIZE)

        # Launch the compiled function on the GPU. The target device is specified
        # first, followed by all function arguments. The last two named parameters
        # are the dimensions of the grid in blocks, and the block dimensions.
        gpu_function(
            gpu_device,
            lhs_layout_tensor,
            rhs_layout_tensor,
            out_layout_tensor,
            VECTOR_WIDTH,
            grid_dim=Dim(num_blocks),
            block_dim=Dim(BLOCK_SIZE),
        )

        # Move the output tensor back onto the CPU so that we can read the results.
        out_tensor = out_tensor.move_to(host_device)

        print("Resulting vector:", out_tensor)
    else:
        print(
            "These examples require a MAX-compatible NVIDIA GPU and none was"
            " detected."
        )
