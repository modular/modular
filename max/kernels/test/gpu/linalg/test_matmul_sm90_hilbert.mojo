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

from gpu.host.info import H100
import linalg.vendor_blas
from gpu.host import DeviceContext
from internal_utils._utils import dynamic, static
from linalg.matmul_sm90_testbed import test_matmul_sm90
from linalg.matmul_tile_scheduler import MatmulSchedule
from utils.index import Index

# NOTE: This test originally tested hilbert_swizzle=True functionality,
# but the testbed doesn't currently support the hilbert_swizzle parameter.
# To properly test hilbert swizzle, the testbed would need to be updated
# to include this parameter and pass it to warp_specialize_gemm_with_multicasting.

# Helper to calculate block_tile_shape - fixed for bfloat16
alias block_tile_shape[wgmma_n: Int] = Index(128, wgmma_n, 64)

# Helper to calculate wgmma_shape - fixed for bfloat16
alias wgmma_shape[wgmma_n: Int] = Index(64, wgmma_n, 16)


def main():
    with DeviceContext() as ctx:
        alias M = 8192
        alias N = 6144
        alias K = 4096

        print(
            "Running warp specialize gemm test (Note: hilbert swizzle"
            " not supported in testbed)"
        )
        print("Test configuration: M=", M, ", N=", N, ", K=", K)

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),
            block_tile_shape[64],
            wgmma_shape[64],
        ](ctx, dynamic(M), static[N](), static[K]())

        print("Test completed successfully")
