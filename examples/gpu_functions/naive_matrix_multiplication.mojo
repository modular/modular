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
from layout import LayoutTensor
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
alias tensor_rank = 2


fn naive_matrix_multiplication(
    i: Int,
    j: Int,
    k: Int,
    m: LayoutTensor[float_dtype],
    n: LayoutTensor[float_dtype],
    p: LayoutTensor[float_dtype],
):
    """Naive matrix multiplication of M_ij x N_jk = P_ik."""
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x

    if row < i and col < k:
        for j_index in range(j):
            p[row, col] = p[row, col] + m[row, j_index] * n[j_index, col]


def main():
    @parameter
    if has_nvidia_gpu_accelerator():
        # Attempt to connect to a compatible GPU. If one is not found, this will
        # error out and exit.
        gpu_device = accelerator()
        host_device = cpu()

        alias I = 5
        alias J = 4
        alias K = 6

        # Allocate the two input matrices on the host.
        m_tensor = Tensor[float_dtype, tensor_rank]((I, J), host_device)
        n_tensor = Tensor[float_dtype, tensor_rank]((J, K), host_device)

        # Fill them with initial values.
        for m_row in range(I):
            for m_col in range(J):
                m_tensor[m_row, m_col] = m_row - m_col

        for n_row in range(J):
            for n_col in range(K):
                n_tensor[n_row, n_col] = n_row + n_col

        print("M matrix:", m_tensor)
        print("N matrix:", n_tensor)

        # Move the input matrices to the accelerator.
        m_tensor = m_tensor.move_to(gpu_device)
        n_tensor = n_tensor.move_to(gpu_device)

        # Allocate a tensor on the accelerator to host the calculation results.
        p_tensor = Tensor[float_dtype, tensor_rank]((I, K), gpu_device)

        # Compile the function to run across a grid on the GPU.
        gpu_function = Accelerator.compile[naive_matrix_multiplication](
            gpu_device
        )

        # The grid is divided up into blocks, making sure there's an extra
        # full block for any remainder. This hasn't been tuned for any specific
        # GPU.
        alias BLOCK_SIZE = 16
        num_col_blocks = ceildiv(I, BLOCK_SIZE)
        num_row_blocks = ceildiv(J, BLOCK_SIZE)

        # Launch the compiled function on the GPU. The target device is specified
        # first, followed by all function arguments. The last two named parameters
        # are the dimensions of the grid in blocks, and the block dimensions.
        gpu_function(
            gpu_device,
            I,
            J,
            K,
            m_tensor.to_layout_tensor(),
            n_tensor.to_layout_tensor(),
            p_tensor.to_layout_tensor(),
            grid_dim=Dim(num_col_blocks, num_row_blocks),
            block_dim=Dim(BLOCK_SIZE, BLOCK_SIZE),
        )

        # Move the output tensor back onto the CPU so that we can read the results.
        p_tensor = p_tensor.move_to(host_device)

        print("Resulting matrix:", p_tensor)
    else:
        print(
            "These examples require a MAX-compatible NVIDIA GPU and none was"
            " detected."
        )
