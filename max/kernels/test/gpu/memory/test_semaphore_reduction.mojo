# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.random import rand

from std.gpu import (
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    grid_dim,
    thread_idx_uint as thread_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.sync.semaphore import Semaphore
from std.memory import memset_zero
from std.testing import assert_equal


def semaphore_vector_reduce[
    dtype: DType,
    N: Int,
    num_parts: Int,
](
    c_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    locks: UnsafePointer[Int32, MutAnyOrigin],
):
    var tid = thread_idx.x
    var block_idx = block_idx.x
    var sema = Semaphore(locks, Int(tid))

    sema.fetch()
    # for each block the partition id is the same as block_idx

    sema.wait(Int(block_idx))

    c_ptr[tid] += a_ptr[block_idx * UInt(N) + tid]
    var lx: Int
    if num_parts == Int(block_idx + 1):
        lx = 0
    else:
        lx = Int(block_idx + 1)
    sema.release(Int32(lx))


def run_vector_reduction[
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

    comptime PN = N * num_parts
    var a_host = alloc[Scalar[dtype]](PN)
    var c_host = alloc[Scalar[dtype]](N)
    var c_host_ref = alloc[Scalar[dtype]](N)

    rand[dtype](a_host, PN)
    memset_zero(c_host, N)
    memset_zero(c_host_ref, N)

    var a_device = ctx.enqueue_create_buffer[dtype](PN)
    var c_device = ctx.enqueue_create_buffer[dtype](N)
    var lock_dev = ctx.enqueue_create_buffer[DType.int32](1)

    ctx.enqueue_memset(lock_dev, 0)
    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(c_device, c_host)

    comptime kernel = semaphore_vector_reduce[dtype, N, num_parts]
    ctx.enqueue_function_experimental[kernel](
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


def semaphore_matrix_reduce[
    dtype: DType, M: Int, N: Int, num_parts: Int
](
    c_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    locks: UnsafePointer[Int32, MutAnyOrigin],
):
    var tid = thread_idx.x
    var block_idx = block_idx.x
    var sema = Semaphore(locks, Int(tid))

    sema.fetch()

    sema.wait(Int(block_idx))
    for x in range(Int(tid), M * N, Int(block_dim.x)):
        var row = x // N
        var col = x % N
        c_ptr[row * N + col] += a_ptr[
            row * (N * num_parts) + Int(block_idx * UInt(num_parts) + UInt(col))
        ]

    var lx: Int
    if num_parts == Int(block_idx + 1):
        lx = 0
    else:
        lx = Int(block_idx + 1)
    sema.release(Int32(lx))


def run_matrix_reduction[
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

    comptime PX = M * N * num_parts
    var a_host = alloc[Scalar[dtype]](PX)
    var c_host = alloc[Scalar[dtype]](M * N)
    var c_host_ref = alloc[Scalar[dtype]](M * N)

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

    comptime kernel = semaphore_matrix_reduce[dtype, M, N, num_parts]
    ctx.enqueue_function_experimental[kernel](
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


def main() raises:
    with DeviceContext() as ctx:
        run_vector_reduction[DType.float32, 128, 4](ctx)
        run_matrix_reduction[DType.float32, 128, 128, 4](ctx)
