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
from linalg.matmul.gpu.amd.warp_spec_matmul import warp_specialized_matmul


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
    a_producer_warps: Int,
    b_producer_warps: Int,
    consumer_warps: Int,
    pipeline_stages: Int = 1,
](ctx: DeviceContext):
    var device_a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var device_b = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var device_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            var val = random_si64(0, 20)
            host_a[i] = val.cast[DType.bfloat16]()

        for i in range(K * N):
            var val = random_si64(0, 20)
            host_b[i] = val.cast[DType.bfloat16]()

    var a_device_tensor = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(M, K),
    ](device_a)

    var b_device_tensor = LayoutTensor[DType.bfloat16, Layout.row_major(N, K)](
        device_b
    )

    var c_device_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N)](
        device_c
    )

    var c_device_ref_tensor = LayoutTensor[
        DType.float32, Layout.row_major(M, N)
    ](device_c_ref)

    warp_specialized_matmul[
        M,
        N,
        K,
        BM,
        BN,
        BK,
        WM,
        WN,
        WK,
        a_producer_warps,
        b_producer_warps,
        consumer_warps,
        pipeline_stages,
    ](
        a_device_tensor,
        b_device_tensor,
        c_device_tensor,
        ctx,
    )

    vendor_blas.matmul(
        ctx,
        c_device_ref_tensor,
        a_device_tensor,
        b_device_tensor,
        c_row_major=True,
        transpose_b=True,
    )

    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
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
