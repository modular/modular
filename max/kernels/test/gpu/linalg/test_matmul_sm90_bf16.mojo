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

import linalg.vendor_blas
from gpu.host import DeviceContext
from internal_utils._utils import dynamic, static
from linalg.matmul_sm90_testbed import test_matmul_sm90
from linalg.matmul_tile_scheduler import MatmulSchedule
from utils.index import Index

# Helper to calculate block_tile_shape based on num_consumer and wgmma_n
alias block_tile_shape[num_consumer: Int, wgmma_n: Int] = Index(
    64 * num_consumer, wgmma_n, 64
)

# Helper to calculate wgmma_shape - fixed for bfloat16
alias wgmma_shape[wgmma_n: Int] = Index(64, wgmma_n, 16)


def main():
    with DeviceContext() as ctx:
        alias wgmma_n = List[Int](128, 256)

        @parameter
        for i in range(len(wgmma_n)):

            @parameter
            for j in range(1, 3):
                test_matmul_sm90[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 1, 1),  # cluster_shape
                    block_tile_shape[j, wgmma_n[i]],
                    wgmma_shape[wgmma_n[i]],
                    num_consumer=j,
                    num_pipeline_stages=4,
                    schedule = MatmulSchedule.TILE2D,
                ](
                    ctx,
                    static[1024](),
                    static[512](),
                    static[128](),
                )

                test_matmul_sm90[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 1, 1),  # cluster_shape
                    block_tile_shape[j, wgmma_n[i]],
                    wgmma_shape[wgmma_n[i]],
                    num_consumer=j,
                    num_pipeline_stages=4,
                    schedule = MatmulSchedule.TILE2D,
                ](
                    ctx,
                    dynamic(99),
                    static[1024](),
                    static[1024](),
                )

                test_matmul_sm90[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 1, 1),  # cluster_shape
                    block_tile_shape[j, wgmma_n[i]],
                    wgmma_shape[wgmma_n[i]],
                    num_consumer=j,
                    num_pipeline_stages=4,
                    schedule = MatmulSchedule.TILE2D,
                ](
                    ctx,
                    dynamic(100),
                    static[512](),
                    static[256](),
                )

                # Test K not multiple of tile size.
                test_matmul_sm90[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    Index(1, 1, 1),  # cluster_shape
                    block_tile_shape[j, wgmma_n[i]],
                    wgmma_shape[wgmma_n[i]],
                    num_consumer=j,
                    num_pipeline_stages=4,
                    schedule = MatmulSchedule.TILE2D,
                ](
                    ctx,
                    dynamic(201),
                    static[2048](),
                    static[200](),
                )

        # K is aligned by 8B
        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            block_tile_shape[2, 128],
            wgmma_shape[128],
            num_consumer=2,
            num_pipeline_stages=4,
        ](ctx, dynamic(150), static[3200](), static[588]())

        # K is aligned by 4B
        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            block_tile_shape[2, 256],
            wgmma_shape[256],
            num_consumer=2,
            num_pipeline_stages=4,
        ](ctx, dynamic(90), static[256](), static[270]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            block_tile_shape[2, 128],
            wgmma_shape[128],
            num_consumer=2,
            num_pipeline_stages=4,
        ](ctx, dynamic(213), static[1111](), static[128]())
