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

from gpu.host import DeviceContext
from internal_utils._utils import dynamic, static
from linalg.matmul.gpu.sm90.testbed import test_matmul_sm90
from linalg.matmul.gpu.tile_scheduler import MatmulSchedule

from utils.index import Index

# Helper to calculate block_tile_shape based on dtype and wgmma_n
alias block_tile_shape[wgmma_n: Int, a_dtype: DType] = Index(
    128, wgmma_n, 128
) if a_dtype is DType.float8_e4m3fn else Index(128, wgmma_n, 64)

# Helper to calculate wgmma_shape based on dtype and wgmma_n
alias wgmma_shape[wgmma_n: Int, a_dtype: DType] = Index(
    64, wgmma_n, 32
) if a_dtype is DType.float8_e4m3fn else Index(64, wgmma_n, 16)


def main():
    with DeviceContext() as ctx:
        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[80, DType.bfloat16],
            wgmma_shape[80, DType.bfloat16],
            num_consumer=2,
            num_pipeline_stages=8,
            partitioned_multicast=False,
            grid_shape = Index(32, 4),
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, dynamic(512), static[2560](), static[8192]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape = Index(10, 13),
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, dynamic(8192), static[2560](), static[8192]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, dynamic(4096), static[2560](), static[8192]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape = Index(4, 33),
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, dynamic(8192), static[8192](), static[2048]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, dynamic(4096), static[8192](), static[2048]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            use_tma_store=True,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, dynamic(4096), static[8192](), static[2048]())

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape = Index(8, 16),
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(8192),
            static[14336](),
            static[8192](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape = Index(8, 16),
            use_tma_store=True,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(8192),
            static[14336](),
            static[8192](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(4096),
            static[14336](),
            static[8192](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            use_tma_store=True,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(4096),
            static[14336](),
            static[8192](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape = Index(4, 33),
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[8192](),
            static[8192](),
            static[7168](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape = Index(4, 33),
            use_tma_store=True,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[8192](),
            static[8192](),
            static[7168](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[4096](),
            static[8192](),
            static[7168](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            use_tma_store=True,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[4096](),
            static[8192](),
            static[7168](),
        )

        @parameter
        for multicast_mode in range(2):
            test_matmul_sm90[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 2, 1),
                block_tile_shape[80, DType.bfloat16],
                wgmma_shape[80, DType.bfloat16],
                num_consumer=2,
                partitioned_multicast = Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                static[256](),
                static[80](),
                static[128](),
            )

            test_matmul_sm90[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 2, 1),
                block_tile_shape[256, DType.bfloat16],
                wgmma_shape[256, DType.bfloat16],
                num_consumer=2,
                partitioned_multicast = Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                static[256](),
                static[256](),
                static[128](),
            )

            test_matmul_sm90[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(1, 2, 1),
                block_tile_shape[64, DType.bfloat16],
                wgmma_shape[64, DType.bfloat16],
                partitioned_multicast = Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                static[256](),
                static[64](),
                static[128](),
            )

            test_matmul_sm90[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                block_tile_shape[256, DType.bfloat16],
                wgmma_shape[256, DType.bfloat16],
                num_consumer=2,
                partitioned_multicast = Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                static[128](),
                static[512](),
                static[128](),
            )

            test_matmul_sm90[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 1, 1),
                block_tile_shape[64, DType.bfloat16],
                wgmma_shape[64, DType.bfloat16],
                partitioned_multicast = Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                static[128](),
                static[128](),
                static[128](),
            )

            test_matmul_sm90[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 2, 1),
                block_tile_shape[256, DType.bfloat16],
                wgmma_shape[256, DType.bfloat16],
                partitioned_multicast = Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                static[256](),
                static[512](),
                static[128](),
            )

            test_matmul_sm90[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                Index(2, 2, 1),
                block_tile_shape[64, DType.bfloat16],
                wgmma_shape[64, DType.bfloat16],
                num_consumer=2,
                partitioned_multicast = Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                static[256](),
                static[128](),
                static[128](),
            )

        print("# 2x1 warp specialized gemm with multicasting tests")

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, DType.bfloat16],
            wgmma_shape[64, DType.bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
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
            Index(1, 2, 1),
            block_tile_shape[64, DType.bfloat16],
            wgmma_shape[64, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(1024),
            static[512](),
            static[128](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, DType.bfloat16],
            wgmma_shape[64, DType.bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(199),
            static[1024](),
            static[1024](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, DType.bfloat16],
            wgmma_shape[64, DType.bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(200),
            static[512](),
            static[256](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, DType.bfloat16],
            wgmma_shape[64, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(201),
            static[2048](),
            static[256](),
        )

        print("# 1x2 warp specialized gemm with multicasting tests")

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.bfloat16],
            wgmma_shape[128, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
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
            Index(2, 1, 1),
            block_tile_shape[128, DType.bfloat16],
            wgmma_shape[128, DType.bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(1024),
            static[512](),
            static[128](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.bfloat16],
            wgmma_shape[128, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
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
            Index(2, 1, 1),
            block_tile_shape[128, DType.bfloat16],
            wgmma_shape[128, DType.bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(100),
            static[512](),
            static[256](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.bfloat16],
            wgmma_shape[128, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(201),
            static[2048](),
            static[256](),
        )

        print("# 2x2 warp specialized gemm with multicasting tests")

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
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
            Index(2, 2, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(1024),
            static[512](),
            static[128](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(199),
            static[1024](),
            static[1024](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(200),
            static[512](),
            static[256](),
        )

        test_matmul_sm90[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, DType.bfloat16],
            wgmma_shape[256, DType.bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(201),
            static[2048](),
            static[256](),
        )
