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

from ..tile_scheduler import MatmulSchedule, RasterOrder
from internal_utils import TuningConfig


@register_passable("trivial")
struct TuningConfigSM90(TuningConfig):
    var M: Int
    var N: Int
    var K: Int

    var mma_shape: IndexList[3]
    var block_tile_shape: IndexList[3]
    var num_pipeline_stages: UInt
    var cluster_shape: IndexList[3]
    var num_consumer: UInt
    var partitioned_multicast: Bool
    var grid_shape: OptionalReg[IndexList[2]]  # = None
    var schedule: MatmulSchedule  # =  MatmulSchedule.NONE
    var splits: OptionalReg[Int]
    var raster_order: OptionalReg[RasterOrder]

    fn __init__(
        out self,
        M: Int,
        N: Int,
        K: Int,
        mma_shape: IndexList[3],
        block_tile_shape: IndexList[3],
        num_pipeline_stages: UInt,
        cluster_shape: IndexList[3],
        num_consumer: UInt,
        partitioned_multicast: Bool,
        grid_shape: OptionalReg[IndexList[2]] = None,
        schedule: MatmulSchedule = MatmulSchedule.NONE,
        splits: OptionalReg[Int] = None,
        raster_order: OptionalReg[RasterOrder] = None,
    ):
        self.M = M
        self.N = N
        self.K = K
        self.mma_shape = mma_shape
        self.block_tile_shape = block_tile_shape
        self.num_pipeline_stages = num_pipeline_stages
        self.cluster_shape = cluster_shape
        self.num_consumer = num_consumer
        self.partitioned_multicast = partitioned_multicast
        self.grid_shape = grid_shape
        self.schedule = schedule
        self.splits = splits
        self.raster_order = raster_order

    fn __str__(self) -> String:
        return String("config: ", "m:", self.M, "/n:", self.N, "/k:", self.K)


fn _get_tuning_list_bf16[mma_k: Int, BK: Int]() -> List[TuningConfigSM90]:
    # kprofile -s open-source/max/max/kernels/src/linalg/matmul/gpu/sm90/tuning.mojo.snippet open-source/max/max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml
    # ----------------BEGIN-TUNING-LIST-BF16----------------
    alias config_list = List(
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [0]
        TuningConfigSM90(
            M=1,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [1]
        TuningConfigSM90(
            M=2,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [2]
        TuningConfigSM90(
            M=4,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [3]
        TuningConfigSM90(
            M=8,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [4]
        TuningConfigSM90(
            M=16,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [5]
        TuningConfigSM90(
            M=32,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [6]
        TuningConfigSM90(
            M=48,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 40, mma_k),
            block_tile_shape=Index(64 * 1, 40, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [7]
        TuningConfigSM90(
            M=64,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 40, mma_k),
            block_tile_shape=Index(64 * 1, 40, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [8]
        TuningConfigSM90(
            M=80,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [9]
        TuningConfigSM90(
            M=96,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [10]
        TuningConfigSM90(
            M=112,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 2, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [11]
        TuningConfigSM90(
            M=128,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 2, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [12]
        TuningConfigSM90(
            M=192,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 160, mma_k),
            block_tile_shape=Index(64 * 1, 160, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [13]
        TuningConfigSM90(
            M=256,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 2, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [14]
        TuningConfigSM90(
            M=1,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [15]
        TuningConfigSM90(
            M=2,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [16]
        TuningConfigSM90(
            M=4,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [17]
        TuningConfigSM90(
            M=8,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [18]
        TuningConfigSM90(
            M=16,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [19]
        TuningConfigSM90(
            M=32,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [20]
        TuningConfigSM90(
            M=48,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [21]
        TuningConfigSM90(
            M=64,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 40, mma_k),
            block_tile_shape=Index(64 * 1, 40, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [22]
        TuningConfigSM90(
            M=80,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [23]
        TuningConfigSM90(
            M=96,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [24]
        TuningConfigSM90(
            M=112,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=11,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [25]
        TuningConfigSM90(
            M=128,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=11,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [26]
        TuningConfigSM90(
            M=192,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 160, mma_k),
            block_tile_shape=Index(64 * 1, 160, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [27]
        TuningConfigSM90(
            M=256,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 2, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [28]
        TuningConfigSM90(
            M=1,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [29]
        TuningConfigSM90(
            M=2,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [30]
        TuningConfigSM90(
            M=4,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [31]
        TuningConfigSM90(
            M=8,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [32]
        TuningConfigSM90(
            M=16,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [33]
        TuningConfigSM90(
            M=32,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [34]
        TuningConfigSM90(
            M=48,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [35]
        TuningConfigSM90(
            M=64,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [36]
        TuningConfigSM90(
            M=80,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [37]
        TuningConfigSM90(
            M=96,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 1, 256, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=5,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [38]
        TuningConfigSM90(
            M=112,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 2, 128, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [39]
        TuningConfigSM90(
            M=128,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [40]
        TuningConfigSM90(
            M=192,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 2, 216, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [41]
        TuningConfigSM90(
            M=256,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 2, 216, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [42]
        TuningConfigSM90(
            M=1,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [43]
        TuningConfigSM90(
            M=2,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [44]
        TuningConfigSM90(
            M=4,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 1, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [45]
        TuningConfigSM90(
            M=8,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 1, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [46]
        TuningConfigSM90(
            M=16,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 1, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [47]
        TuningConfigSM90(
            M=32,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 1, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [48]
        TuningConfigSM90(
            M=48,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 1, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [49]
        TuningConfigSM90(
            M=64,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 1, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [50]
        TuningConfigSM90(
            M=80,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 1, 192, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [51]
        TuningConfigSM90(
            M=96,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 2, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [52]
        TuningConfigSM90(
            M=112,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 2, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [53]
        TuningConfigSM90(
            M=128,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 2, 144, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [54]
        TuningConfigSM90(
            M=192,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [55]
        TuningConfigSM90(
            M=256,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
    )
    # ----------------END-TUNING-LIST-BF16-SMALL----------------

    return materialize[config_list]()
