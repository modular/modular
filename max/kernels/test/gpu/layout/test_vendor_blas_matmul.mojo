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

from math import ceildiv
from os import abort
from sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator

from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from random import rand
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu import matmul_kernel_naive
from linalg.matmul.vendor.blas import matmul
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]


fn test_matmul[
    input_type: DType, M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    print("== test_vendor_blas", input_type, "x", M, "x", N, "x", K)

    comptime static_a_shape = DimList(M, K)
    comptime static_b_shape = DimList(N, K)
    comptime static_c_shape = DimList(M, N)

    var a_host_ptr = UnsafePointer[Scalar[input_type]].alloc(M * K)
    var a_host = NDBuffer[input_type, 2, _, static_a_shape](a_host_ptr)
    var b_size = N * K
    var b_host_ptr = UnsafePointer[Scalar[input_type]].alloc(b_size)
    var b_host = NDBuffer[input_type, 2, _, static_b_shape](b_host_ptr)
    var c_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(M * N)
    var c_host = NDBuffer[DType.float32, 2, _, static_c_shape](c_host_ptr)
    var c_host_ref_ptr = UnsafePointer[Scalar[DType.float32]].alloc(M * N)
    var c_host_ref = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_host_ref_ptr
    )

    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())

    c_host.zero()
    c_host_ref.zero()

    var a_device = ctx.enqueue_create_buffer[input_type](M * K)
    var a_device_nd = NDBuffer[input_type, 2, _, static_a_shape](
        a_device.unsafe_ptr()
    )
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var b_device_nd = NDBuffer[input_type, 2, _, static_b_shape](
        b_device.unsafe_ptr()
    )
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var c_device_nd = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_device.unsafe_ptr()
    )
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)
    var c_device_ref_nd = NDBuffer[DType.float32, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr()
    )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    matmul(
        ctx,
        c_device_nd,
        a_device_nd,
        b_device_nd,
        transpose_b=True,
        c_row_major=True,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    var c_tensor_ref = from_ndbuffer_row_major(c_device_ref_nd)
    var a_tensor = from_ndbuffer_row_major(a_device_nd)
    var b_tensor = from_ndbuffer_row_major(b_device_nd)

    # Run naive matmul.
    comptime BLOCK_DIM = 16
    comptime kernel = matmul_kernel_naive[
        DType.float32,
        input_type,
        input_type,
        c_tensor_ref.layout,
        a_tensor.layout,
        b_tensor.layout,
        BLOCK_DIM,
        transpose_b=True,
    ]
    ctx.enqueue_function_experimental[kernel](
        c_tensor_ref,
        a_tensor,
        b_tensor,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=0.01,
        rtol=0.01,
    )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


fn test_matmul[input_types: List[DType]]() raises:
    with DeviceContext() as ctx:
        comptime for input_type in input_types:
            test_matmul[input_type, 64, 16, 32](ctx)
            test_matmul[input_type, 512, 2560, 512](ctx)


def main():
    comptime if has_amd_gpu_accelerator():
        test_matmul[[DType.float8_e4m3fnuz, DType.bfloat16]]()
    elif has_nvidia_gpu_accelerator():
        test_matmul[[DType.float8_e4m3fn, DType.bfloat16]]()
    else:
        abort("Unknown GPU Accelerator.")
