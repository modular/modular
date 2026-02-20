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

import linalg.matmul.vendor.blas as vendor_blas
from gpu.host import DeviceContext
from internal_utils._utils import dynamic, static
from linalg.matmul.gpu.sm90.testbed import test_matmul_sm90
from linalg.matmul.gpu.tile_scheduler import MatmulSchedule

from utils.index import Index

# Helper to calculate block_tile_shape based on num_consumer and wgmma_n
comptime block_tile_shape[num_consumer: Int, wgmma_n: Int] = Index(
    64 * num_consumer, wgmma_n, 64
)

# Helper to calculate wgmma_shape - fixed for bfloat16
comptime wgmma_shape[wgmma_n: Int] = Index(64, wgmma_n, 16)


def main():
    with DeviceContext() as ctx:
        comptime wgmma_n: List[Int] = [128, 256]

        comptime for i in range(len(wgmma_n)):
            comptime for j in range(1, 3):
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

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(128, 64, 64),
            Index(64, 64, 16),
            num_consumer=2,
            num_pipeline_stages=8,
        ](ctx, dynamic(256), static[4096](), static[1536]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(64, 48, 64),
            Index(64, 48, 16),
            num_consumer=1,
            num_pipeline_stages=8,
            k_group_size=2,
        ](ctx, dynamic(256), static[1536](), static[4096]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(64, 48, 64),
            Index(64, 48, 16),
            num_consumer=1,
            num_pipeline_stages=12,
            k_group_size=4,
        ](ctx, dynamic(256), static[1536](), static[4096]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(64, 32, 64),
            Index(64, 32, 16),
            num_consumer=1,
            num_pipeline_stages=17,
        ](ctx, dynamic(2), static[4096](), static[1536]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(64, 8, 64),
            Index(64, 8, 16),
            num_consumer=1,
            num_pipeline_stages=20,
        ](ctx, dynamic(16), static[64](), static[256]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(64, 256, 64),
            Index(64, 256, 16),
            num_consumer=1,
            num_pipeline_stages=5,
        ](ctx, dynamic(20), static[84](), static[4096]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(128, 256, 64),
            Index(64, 256, 16),
            num_consumer=2,
            num_pipeline_stages=2,
        ](ctx, dynamic(476), static[1024](), static[128]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(128, 48, 64),
            Index(64, 48, 16),
            num_consumer=2,
            num_pipeline_stages=2,
        ](ctx, dynamic(1536), static[48](), static[4096]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 1, 1),  # cluster_shape
            Index(128, 8, 64),
            Index(64, 8, 16),
            num_consumer=2,
            num_pipeline_stages=2,
        ](ctx, dynamic(1536), static[13](), static[4096]())
