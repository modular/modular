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
from random import random_si64

from gpu import WARP_SIZE, barrier, lane_id, thread_idx
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from gpu.mma import ld_matrix, mma
from gpu.mma_util import store_matrix_d
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.matmul_gpu import matmul_kernel_naive
from memory import stack_allocation
from testing import assert_almost_equal

from utils.numerics import get_accum_type


fn test_ldmatrix_fp32(
    c_ptr: UnsafePointer[Float32],
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m: UInt = 16
    alias mma_n: UInt = 8
    alias mma_k: UInt = 8

    var d_reg = SIMD[DType.float32, 4](0)
    var tid = thread_idx.x
    var a_shared = stack_allocation[
        mma_m * mma_k,
        DType.float32,
        alignment=32,
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        mma_n * mma_k,
        DType.float32,
        alignment=32,
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(tid, mma_m * mma_k, WARP_SIZE):
        a_shared[i] = a_ptr[i]

    # Transpose B to fit ld_matrix layout.
    for i in range(tid, mma_k * mma_n, WARP_SIZE):
        var x = i % mma_n
        var y = i // mma_n
        b_shared[x * mma_k + y] = b_ptr[i]

    barrier()

    var a_reg = ld_matrix[4](
        a_shared + Int((lane_id() % 16) * 8 + (lane_id() // 16) * 4)
    )
    var b_reg = ld_matrix[2](
        b_shared + Int((lane_id() % 8) * 8 + (lane_id() // 8) * 4)
    )

    mma(d_reg, a_reg, b_reg, d_reg)
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, 0, 0, n)


fn test_ldmatrix_transposed[
    input_type: DType, output_type: DType
](
    c_ptr: UnsafePointer[Scalar[output_type]],
    a_ptr: UnsafePointer[Scalar[input_type]],
    b_ptr: UnsafePointer[Scalar[input_type]],
):
    alias accum_type = get_accum_type[input_type]()
    alias mma_shape = get_mma_shape[input_type, accum_type]()
    alias M = mma_shape[0]
    alias N = mma_shape[1]
    alias K = mma_shape[2]
    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    var lane = lane_id()
    var d = SIMD[accum_type, c_frag_size](0)

    var a_shared = stack_allocation[
        M * K, input_type, alignment=32, address_space = AddressSpace.SHARED
    ]()
    var b_shared = stack_allocation[
        N * K, input_type, alignment=32, address_space = AddressSpace.SHARED
    ]()

    for i in range(lane, M * K, WARP_SIZE):
        a_shared[i] = a_ptr[i]

    # Transpose B to fit ld_matrix layout.
    for i in range(lane, N * K, WARP_SIZE):
        b_shared[i] = b_ptr[i]

    barrier()

    var a_reg = ld_matrix[a_frag_size](
        a_shared + Int((lane % M) * K + (lane // M) * K // 2)
    )
    var b_reg = ld_matrix[b_frag_size, transpose=True](
        b_shared + Int((lane % K) * N + (lane // K) * N // 2)
    )

    mma(d, a_reg, b_reg, d)
    store_matrix_d[M, N, K](
        c_ptr,
        # Store matrix is hardcoded to store 4 elements.
        rebind[SIMD[output_type, 4]](d.cast[output_type]()),
        0,
        0,
        mma_shape[1],
    )


fn check_ldmatrix_transposed_bf16[
    input_type: DType,
    output_type: DType,
](ctx: DeviceContext) raises:
    print("== test ldmatrix transposed bf16")

    # Shape for a single mma.
    alias accum_type = get_accum_type[input_type]()
    alias mma_shape = get_mma_shape[input_type, accum_type]()
    alias M = mma_shape[0]
    alias N = mma_shape[1]
    alias K = mma_shape[2]

    var a_host = UnsafePointer[Scalar[input_type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[input_type]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[output_type]].alloc(M * N)
    var c_host_ref = UnsafePointer[Scalar[output_type]].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = m * K + k

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = k * N + n

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[input_type](M * K)
    var b_device = ctx.enqueue_create_buffer[input_type](K * N)
    var c_device = ctx.enqueue_create_buffer[output_type](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[output_type](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    ctx.enqueue_function[test_ldmatrix_transposed[input_type, output_type]](
        c_device,
        a_device,
        b_device,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    alias BLOCK_DIM = 16
    ctx.enqueue_function[
        matmul_kernel_naive[output_type, input_type, input_type, BLOCK_DIM]
    ](
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host.load(i)
        var out_ref = c_host_ref.load(i)
        assert_almost_equal(out_val, out_ref)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref


fn check_ldmatrix(
    M: Int, N: Int, K: Int, rand_min: Int64, rand_max: Int64, ctx: DeviceContext
) raises:
    print("== test ldmatrix instruction")

    var a_host = UnsafePointer[Float32].alloc(M * K)
    var b_host = UnsafePointer[Float32].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    ctx.enqueue_function[test_ldmatrix_fp32](
        c_device,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    alias BLOCK_DIM = 16

    ctx.enqueue_function[
        matmul_kernel_naive[
            DType.float32, DType.float32, DType.float32, BLOCK_DIM
        ]
    ](
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.synchronize()
    ctx.enqueue_copy(c_host_ref, c_device_ref)

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_ref[i]
        assert_almost_equal(out_val, out_ref)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref


def main():
    with DeviceContext() as ctx:
        check_ldmatrix(16, 8, 8, -100, 100, ctx)
        check_ldmatrix_transposed_bf16[DType.bfloat16, DType.bfloat16](ctx)
