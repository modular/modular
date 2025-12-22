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

from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from itertools import product
from linalg.matmul.gpu import matmul_kernel
from testing import assert_false

from utils.index import Index

# Tile size for tiling in shared memory.
# Thread block would have shape (tile_size, tile_size, 1)
comptime tile_size = 32

fn run_matmul[
    M: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext) raises:
    print("== run_matmul_kernel")
    var a_host_ptr = alloc[Float32](M * K)
    var a_host = NDBuffer[DType.float32, 2, _, DimList(M, K)](a_host_ptr)
    var b_host_ptr = alloc[Float32](K * N)
    var b_host = NDBuffer[DType.float32, 2, _, DimList(K, N)](b_host_ptr)
    var c_host_ptr = alloc[Float32](M * N)
    var c_host = NDBuffer[DType.float32, 2, _, DimList(M, N)](c_host_ptr)

    for i in range(M):
        for j in range(K):
            a_host[Index(i, j)] = 1

    for i in range(K):
        for j in range(N):
            b_host[Index(i, j)] = 1

    for i in range(M):
        for j in range(N):
            c_host[Index(i, j)] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime kernel = matmul_kernel[
        DType.float32,
        DType.float32,
        DType.float32,
        tile_size,
    ]

    ctx.enqueue_function_checked[kernel, kernel](
        c_device,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(N, tile_size), ceildiv(M, tile_size)),
        block_dim=(tile_size, tile_size),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    var failed = False
    for i in range(M - 10, M):
        for j in range(N - 10, N):
            if c_host[i, j] != Float32(K):
                print(
                    "Fail at index = [",
                    i,
                    ",",
                    j,
                    "] the value is",
                    c_host[i, j],
                    "the golden value is",
                    K,
                )
                failed = True

    assert_false(failed)
    if not failed:
        print("succeed")

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_host
    _ = b_host
    _ = c_host


def main():
    # Should be able to handle non-divisible values.
    comptime size_list = [513, 502, 511]
    with DeviceContext() as ctx:
        @parameter
        for N, M, K in product(size_list, size_list, size_list):
            run_matmul[N, M, K](ctx)
