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

from gpu import block_idx, global_idx, thread_idx
from gpu.host import DeviceContext
from testing import assert_equal

alias buffer_size = 1024
alias block_dim = 32


def test_fill_thread_idx(ctx: DeviceContext):
    var output_host = UnsafePointer[Scalar[DType.int]].alloc(buffer_size)
    var output_buffer = ctx.enqueue_create_buffer[DType.int](
        buffer_size
    ).enqueue_fill(9)

    fn kernel(output: UnsafePointer[Scalar[DType.int]]):
        output[global_idx.x] = thread_idx.x

    ctx.enqueue_function_checked[kernel, kernel](
        output_buffer,
        grid_dim=buffer_size // block_dim,
        block_dim=block_dim,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    for i in range(0, buffer_size, block_dim):
        for j in range(block_dim):
            assert_equal(output_host[i + j], j)

    output_host.free()


def test_fill_block_idx(ctx: DeviceContext):
    var output_host = UnsafePointer[Scalar[DType.int]].alloc(buffer_size)
    var output_buffer = ctx.enqueue_create_buffer[DType.int](
        buffer_size
    ).enqueue_fill(9)

    fn kernel(output: UnsafePointer[Scalar[DType.int]]):
        output[global_idx.x] = block_idx.x

    ctx.enqueue_function_checked[kernel, kernel](
        output_buffer,
        grid_dim=buffer_size // block_dim,
        block_dim=block_dim,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    for i in range(0, buffer_size, block_dim):
        for j in range(block_dim):
            assert_equal(output_host[i + j], i // block_dim)

    output_host.free()


def main():
    with DeviceContext() as ctx:
        test_fill_thread_idx(ctx)
        test_fill_block_idx(ctx)
