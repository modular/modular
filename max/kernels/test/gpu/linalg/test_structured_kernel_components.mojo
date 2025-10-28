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

from linalg.matmul.gpu.amd.warp_spec_matmul import warp_specialized_matmul
from gpu import WARP_SIZE
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from layout._fillers import random
import linalg.matmul.vendor.blas as vendor_blas
from testing import assert_equal
from random import random_si64


def test_warp_specialization_amd[
    M: Int,
    N: Int,
    K: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    producer_warps_a: Int,
    producer_warps_b: Int,
    consumer_warps: Int,
    pipeline_stages: Int = 1,
](ctx: DeviceContext):
    var device_a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var device_b = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var device_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b, device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        host_c = host_c.enqueue_fill(0)
        host_c_ref = host_c_ref.enqueue_fill(0)

        ctx.synchronize()

        for i in range(M * K):
            var val = random_si64(0, 20)
            host_a[i] = val.cast[DType.bfloat16]()

        for i in range(K * N):
            var val = random_si64(0, 20)
            host_b[i] = val.cast[DType.bfloat16]()

        ctx.enqueue_copy(device_a, host_a)
        ctx.enqueue_copy(device_b, host_b)
        ctx.enqueue_copy(device_c, host_c)
        ctx.enqueue_copy(device_c_ref, host_c_ref)
        ctx.synchronize()

        var a_device_tensor = LayoutTensor[
            DType.bfloat16,
            Layout.row_major(M, K),
        ](device_a)

        var b_device_tensor = LayoutTensor[
            DType.bfloat16, Layout.row_major(N, K)
        ](device_b)

        var c_device_tensor = LayoutTensor[
            DType.float32, Layout.row_major(M, N)
        ](device_c)

        var c_device_ref_tensor = LayoutTensor[
            DType.float32, Layout.row_major(M, N)
        ](device_c_ref)

        var global_c_device_tensor = c_device_tensor.address_space_cast[
            AddressSpace.GLOBAL
        ]()
        var global_a_device_tensor = a_device_tensor.address_space_cast[
            AddressSpace.GLOBAL
        ]()
        var global_b_device_tensor = b_device_tensor.address_space_cast[
            AddressSpace.GLOBAL
        ]()

        alias kernel = warp_specialized_matmul[
            a_device_tensor.dtype,
            c_device_tensor.dtype,
            a_device_tensor.layout,
            b_device_tensor.layout,
            c_device_tensor.layout,
            BM,
            BN,
            BK,
            WM,
            WN,
            WK,
            producer_warps_a=producer_warps_a,
            producer_warps_b=producer_warps_b,
            consumer_warps=consumer_warps,
            pipeline_stages=pipeline_stages,
        ]

        ctx.enqueue_function_checked[kernel, kernel](
            global_a_device_tensor,
            global_b_device_tensor,
            global_c_device_tensor,
            grid_dim=(M // BM, N // BN),
            block_dim=(
                WARP_SIZE
                * (producer_warps_a + producer_warps_b + consumer_warps)
            ),
        )

        vendor_blas.matmul(
            ctx,
            c_device_ref_tensor,
            a_device_tensor,
            b_device_tensor,
            c_row_major=True,
            transpose_b=True,
        )

        ctx.synchronize()

        ctx.enqueue_copy(host_c, device_c)
        ctx.enqueue_copy(host_c_ref, device_c_ref)
        ctx.synchronize()

        var errors = 0
        for i in range(M * N):
            # print(i // N, i % N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
            if host_c[i] != host_c_ref[i]:  # and errors < 100:
                # print(i // N, i % N, host_c[i], host_c_ref[i])
                errors += 1

            # if errors < 100:
            #     print(i//N, i%N, host_c[i], host_c_ref[i])

        assert_equal(errors, 0)


def main():
    with DeviceContext() as ctx:
        print("Running AMD Warp Specialization Tests")
        test_warp_specialization_amd[
            4096, 4096, 4096, 64, 64, 64, 32, 32, 64, 2, 2, 4, pipeline_stages=2
        ](ctx)

        test_warp_specialization_amd[
            1024, 1024, 256, 64, 64, 64, 32, 32, 64, 2, 2, 4, pipeline_stages=2
        ](ctx)

        test_warp_specialization_amd[
            32, 32, 32, 32, 32, 32, 16, 16, 32, 2, 2, 2
        ](ctx)
        print("==== AMD Warp Specialization Tests passed ====")
