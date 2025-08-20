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
from os import abort
from sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator

from buffer import DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from linalg.matmul_gpu import matmul_kernel_naive
from linalg.vendor_blas import matmul
from layout._ndbuffer_stub import from_ndbuffer_row_major


fn test_matmul[
    input_type: DType, M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    print("== test_vendor_blas", input_type, "x", M, "x", N, "x", K)

    alias transpose_b = True
    alias static_a_shape = DimList(M, K)
    alias static_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    alias static_c_shape = DimList(M, N)

    var a_host = HostNDBuffer[input_type, 2, static_a_shape]()
    var b_host = HostNDBuffer[input_type, 2, static_b_shape]()
    var c_host = HostNDBuffer[DType.float32, 2, static_c_shape]()
    var c_host_ref = HostNDBuffer[DType.float32, 2, static_c_shape]()

    random(a_host.tensor)
    random(b_host.tensor)

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    var a_device = DeviceNDBuffer[input_type, 2, static_a_shape](ctx=ctx)
    var b_device = DeviceNDBuffer[input_type, 2, static_b_shape](ctx=ctx)
    var c_device = DeviceNDBuffer[DType.float32, 2, static_c_shape](ctx=ctx)
    var c_device_ref = DeviceNDBuffer[DType.float32, 2, static_c_shape](ctx=ctx)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    matmul(
        ctx,
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        transpose_b=True,
        c_row_major=True,
    )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)

    var c_tensor_ref = from_ndbuffer_row_major(c_device_ref.tensor)
    var a_tensor = from_ndbuffer_row_major(a_device.tensor)
    var b_tensor = from_ndbuffer_row_major(b_device.tensor)

    # Run naive matmul.
    alias BLOCK_DIM = 16
    ctx.enqueue_function[
        matmul_kernel_naive[
            DType.float32,
            input_type,
            input_type,
            c_tensor_ref.layout,
            a_tensor.layout,
            b_tensor.layout,
            BLOCK_DIM,
            transpose_b=True,
        ]
    ](
        c_tensor_ref,
        a_tensor,
        b_tensor,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.01,
        rtol=0.01,
    )

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref


fn test_matmul[input_types: List[DType]]() raises:
    with DeviceContext() as ctx:

        @parameter
        for input_type in input_types:
            test_matmul[input_type, 64, 16, 32](ctx)
            test_matmul[input_type, 512, 2560, 512](ctx)


fn main() raises:
    @parameter
    if has_amd_gpu_accelerator():
        test_matmul[[DType.float8_e4m3fnuz, DType.bfloat16]]()
    elif has_nvidia_gpu_accelerator():
        test_matmul[[DType.float8_e4m3fn, DType.bfloat16]]()
    else:
        abort("Unknown GPU Accelerator.")
