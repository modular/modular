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

from random import rand

from gpu import block_dim, block_idx, grid_dim, thread_idx
from gpu.host import DeviceContext
from gpu.semaphore import Semaphore
from memory import memset_zero
from testing import assert_equal


fn semaphore_vector_reduce[
    dtype: DType,
    N: Int,
    num_parts: Int,
](
    c_ptr: UnsafePointer[Scalar[dtype]],
    a_ptr: UnsafePointer[Scalar[dtype]],
    locks: UnsafePointer[Int32],
):
    var tid = thread_idx.x
    var block_idx = block_idx.x
    var sema = Semaphore(locks.offset(0), tid)

    sema.fetch()
    # for each block the partition id is the same as block_idx

    sema.wait(block_idx)

    c_ptr[tid] += a_ptr[block_idx * N + tid]
    var lx: Int
    if num_parts == (block_idx + 1):
        lx = 0
    else:
        lx = block_idx + 1
    sema.release(lx)


fn run_vector_reduction[
    dtype: DType,
    N: Int,
    num_parts: Int,
](ctx: DeviceContext,) raises:
    print(
        "== run_semaphore vector reduction kernel => ",
        String(dtype),
        N,
        num_parts,
    )

    alias PN = N * num_parts
    var a_host = UnsafePointer[Scalar[dtype]].alloc(PN)
    var c_host = UnsafePointer[Scalar[dtype]].alloc(N)
    var c_host_ref = UnsafePointer[Scalar[dtype]].alloc(N)

    rand[dtype](a_host, PN)
    memset_zero(c_host, N)
    memset_zero(c_host_ref, N)

    var a_device = ctx.enqueue_create_buffer[dtype](PN)
    var c_device = ctx.enqueue_create_buffer[dtype](N)
    var lock_dev = ctx.enqueue_create_buffer[DType.int32](1)

    ctx.enqueue_memset(lock_dev, 0)
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(c_device, c_host)

    alias kernel = semaphore_vector_reduce[dtype, N, num_parts]
    ctx.enqueue_function_checked[kernel, kernel](
        c_device,
        a_device,
        lock_dev,
        grid_dim=num_parts,
        block_dim=N,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    for i in range(N):
        for j in range(num_parts):
            c_host_ref[i] += a_host[j * N + i]

    for i in range(N):
        assert_equal(c_host[i], c_host_ref[i])

    _ = a_device
    _ = c_device
    _ = lock_dev

    a_host.free()
    c_host.free()
    c_host_ref.free()


fn semaphore_matrix_reduce[
    dtype: DType, M: Int, N: Int, num_parts: Int
](
    c_ptr: UnsafePointer[Scalar[dtype]],
    a_ptr: UnsafePointer[Scalar[dtype]],
    locks: UnsafePointer[Int32],
):
    var tid = thread_idx.x
    var block_idx = block_idx.x
    var sema = Semaphore(locks.offset(0), tid)

    sema.fetch()

    sema.wait(block_idx)
    for x in range(tid, M * N, block_dim.x):
        var row = x // N
        var col = x % N
        c_ptr[row * N + col] += a_ptr[
            row * (N * num_parts) + (block_idx * num_parts + col)
        ]

    var lx: Int
    if num_parts == (block_idx + 1):
        lx = 0
    else:
        lx = block_idx + 1
    sema.release(lx)


fn run_matrix_reduction[
    dtype: DType,
    M: Int,
    N: Int,
    num_parts: Int,
](ctx: DeviceContext,) raises:
    print(
        "== run_semaphore matrix reduction kernel => ",
        String(dtype),
        M,
        N,
        num_parts,
    )

    alias PX = M * N * num_parts
    var a_host = UnsafePointer[Scalar[dtype]].alloc(PX)
    var c_host = UnsafePointer[Scalar[dtype]].alloc(M * N)
    var c_host_ref = UnsafePointer[Scalar[dtype]].alloc(M * N)

    rand[dtype](a_host, PX)
    memset_zero(c_host, M * N)
    memset_zero(c_host_ref, M * N)

    var a_device = ctx.enqueue_create_buffer[dtype](PX)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var lock_dev = ctx.enqueue_create_buffer[DType.int32](1)

    ctx.enqueue_memset(lock_dev, 0)
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(c_device, c_host)

    var block_size = 1024

    alias kernel = semaphore_matrix_reduce[dtype, M, N, num_parts]
    ctx.enqueue_function_checked[kernel, kernel](
        c_device,
        a_device,
        lock_dev,
        grid_dim=num_parts,
        block_dim=block_size,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    for r in range(M):
        for c in range(N):
            for i in range(num_parts):
                c_host_ref[r * N + c] += a_host[
                    r * (N * num_parts) + (i * num_parts + c)
                ]

    for i in range(M * N):
        assert_equal(c_host[i], c_host_ref[i])

    _ = a_device
    _ = c_device
    _ = lock_dev

    a_host.free()
    c_host.free()
    c_host_ref.free()


def main():
    with DeviceContext() as ctx:
        run_vector_reduction[DType.float32, 128, 4](ctx)
        run_matrix_reduction[DType.float32, 128, 128, 4](ctx)
