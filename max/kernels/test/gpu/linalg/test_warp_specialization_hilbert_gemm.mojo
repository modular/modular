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
# UNSUPPORTED: AMD-GPU
# REQUIRES: H100-GPU
from gpu.host.info import H100

import linalg.vendor_blas
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul_sm90 import warp_specialize_gemm_with_multicasting
from linalg.matmul_tile_scheduler import MatmulSchedule
from linalg.utils_gpu import MatmulConfig

from utils.index import Index

alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4


def test_warp_specialize_gemm[
    wgmma_n: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    num_consumer: Int = 1,
    transpose_b: Bool = True,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        "wgmma_n",
        wgmma_n,
        " : ",
        M,
        "x",
        N,
        "x",
        K,
        " with num_consumer -> ",
        num_consumer,
    )

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Initialize matmul operands
    print("Randomizing host tensor A")
    random(a_host.tensor)
    print("Randomizing host tensor B")
    random(b_host.tensor)
    print("Zeroing host tensor C")
    zero(c_host.tensor)
    print("Zeroing host reference tensor C")
    zero(c_host_ref.tensor)

    # Move operands to the Device
    print("Copying tensor A to device")
    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    print("Copying tensor B to device")
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)
    print("Copying tensor C to device")
    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    print("Copying reference tensor C to device")
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    print("Converting device tensors to row-major format")
    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)

    alias block_tile_shape = Index(128, wgmma_n, 64)

    alias matmul_config = MatmulConfig[
        a_type, b_type, c_type, transpose_b, mma_shape = Index(64, wgmma_n, 16)
    ](
        block_tile_shape=block_tile_shape,
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=num_consumer,
        partitioned_multicast=False,
    )

    print("Running warp specialize gemm with hilbert swizzle")
    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        config=matmul_config,
        schedule=schedule,
        hilbert_swizzle=True,
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        M,
        N,
        K,
        ctx,
    )

    ctx.synchronize()

    print("Running reference matmul using vendor BLAS")
    vendor_blas.matmul(
        ctx,
        c_device_ref.tensor,
        a_device.tensor,
        b_device.tensor,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    print("Copying results back to host")
    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()
    alias rtol = 1e-2
    print("Verifying results match reference implementation")
    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=rtol,
    )
    print("Test passed successfully")
    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device

    _ = a
    _ = b
    _ = c


def main():
    with DeviceContext() as ctx:

        @parameter
        if ctx.device_info is H100:
            alias M = 8192
            alias N = 6144
            alias K = 4096
            test_warp_specialize_gemm[
                64, DType.bfloat16, DType.bfloat16, DType.bfloat16
            ](ctx, dynamic(M), static[N](), static[K]())
        else:
            print("Skipping test - requires NVIDIA H100 GPU")
