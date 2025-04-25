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


from gpu import global_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import ceildiv
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

alias float_dtype = DType.float32

alias I = 5
alias J = 4
alias K = 6

alias m_layout = Layout.row_major(I, J)
alias n_layout = Layout.row_major(J, K)
alias p_layout = Layout.row_major(I, K)


def main():
    constrained[
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator(),
        "This example requires a supported GPU",
    ]()

    var ctx = DeviceContext()
    var m_buffer = ctx.enqueue_create_buffer[float_dtype](m_layout.size())
    var n_buffer = ctx.enqueue_create_buffer[float_dtype](n_layout.size())
    var p_buffer = ctx.enqueue_create_buffer[float_dtype](p_layout.size())

    # Map input buffers to host to fill with values from CPU
    with m_buffer.map_to_host() as host_buffer:
        var m_tensor = LayoutTensor[float_dtype, m_layout](host_buffer)
        for m_row in range(I):
            for m_col in range(J):
                m_tensor[m_row, m_col] = m_row - m_col
        print("M matrix:", m_tensor)

    with n_buffer.map_to_host() as host_buffer:
        var n_tensor = LayoutTensor[float_dtype, n_layout](host_buffer)
        for n_row in range(J):
            for n_col in range(K):
                n_tensor[n_row, n_col] = n_row + n_col
        print("N matrix:", n_tensor)

    # Wrap device buffers in `LayoutTensor`
    var m_tensor = LayoutTensor[float_dtype, m_layout](m_buffer)
    var n_tensor = LayoutTensor[float_dtype, n_layout](n_buffer)
    var p_tensor = LayoutTensor[float_dtype, p_layout](p_buffer)

    # The grid is divided up into blocks, making sure there's an extra
    # full block for any remainder. This hasn't been tuned for any specific
    # GPU.
    alias BLOCK_SIZE = 16
    alias num_col_blocks = ceildiv(I, BLOCK_SIZE)
    alias num_row_blocks = ceildiv(J, BLOCK_SIZE)

    # Launch the compiled function on the GPU. The target device is specified
    # first, followed by all function arguments. The last two named parameters
    # are the dimensions of the grid in blocks, and the block dimensions.
    ctx.enqueue_function[naive_matrix_multiplication](
        m_tensor,
        n_tensor,
        p_tensor,
        grid_dim=(num_col_blocks, num_row_blocks),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    # Move the output tensor back onto the CPU so that we can read the results.
    with p_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[float_dtype, p_layout](host_buffer)
        print("Resulting matrix:", host_tensor)


fn naive_matrix_multiplication(
    m: LayoutTensor[float_dtype, m_layout, MutableAnyOrigin],
    n: LayoutTensor[float_dtype, n_layout, MutableAnyOrigin],
    p: LayoutTensor[float_dtype, p_layout, MutableAnyOrigin],
):
    """Naive matrix multiplication of M_ij x N_jk = P_ik."""
    var row = global_idx.y
    var col = global_idx.x

    var m_dim = p.dim(0)
    var n_dim = p.dim(1)
    var k_dim = m.dim(1)

    if row < m_dim and col < n_dim:
        for j_index in range(k_dim):
            p[row, col] = p[row, col] + m[row, j_index] * n[j_index, col]
