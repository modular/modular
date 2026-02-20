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

from ...tile_scheduler import RasterOrder
from internal_utils import Table, TuningConfig


struct TuningConfigSM100(TrivialRegisterPassable, TuningConfig):
    # The kernel parameters are optimal for shape in [M:M_end]xNxK.
    var M: Int
    var M_end: Int
    var N: Int
    var K: Int

    # Kernel parameters
    var mma_shape: IndexList[3]
    var block_tile_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var block_swizzle_size: UInt
    var rasterize_order: RasterOrder
    var cta_group: Int
    var swapAB: Bool
    var k_group_size: UInt
    var num_accum_pipeline_stages: UInt
    var num_clc_pipeline_stages: UInt
    var num_split_k: Int

    fn __init__(
        out self,
        M: Int,
        N: Int,
        K: Int,
        mma_shape: IndexList[3],
        block_tile_shape: IndexList[3],
        cluster_shape: IndexList[3],
        block_swizzle_size: UInt,
        rasterize_order: RasterOrder,
        cta_group: Int = 2,
        swapAB: Bool = False,
        k_group_size: UInt = 1,
        num_accum_pipeline_stages: UInt = 2,
        num_clc_pipeline_stages: UInt = 2,
        num_split_k: Int = 1,
    ):
        self.M = M
        self.M_end = M + 1
        self.N = N
        self.K = K
        self.mma_shape = mma_shape
        self.block_tile_shape = block_tile_shape
        self.cluster_shape = cluster_shape
        self.block_swizzle_size = block_swizzle_size
        self.rasterize_order = rasterize_order
        self.cta_group = cta_group
        self.swapAB = swapAB
        self.k_group_size = k_group_size
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_split_k = num_split_k

    fn __str__(self) -> String:
        return String("config: ", "m:", self.M, "/n:", self.N, "/k:", self.K)

    fn __init__(
        out self,
        M: Int,
        M_end: Int,
        N: Int,
        K: Int,
        mma_shape: IndexList[3],
        cta_group: Int,
        cluster_shape: IndexList[3],
        block_swizzle_size: UInt,
        rasterize_order: RasterOrder,
        swapAB: Bool = False,
        k_group_size: UInt = 1,
        num_accum_pipeline_stages: UInt = 2,
        num_clc_pipeline_stages: UInt = 2,
        num_split_k: Int = 1,
    ):
        self.M = M
        self.M_end = M_end
        self.N = N
        self.K = K
        self.mma_shape = mma_shape
        self.cta_group = cta_group
        self.block_tile_shape = Index(
            mma_shape[0] // cta_group,
            mma_shape[1] // cta_group,
            mma_shape[2] * 4,
        )
        self.cluster_shape = cluster_shape
        self.block_swizzle_size = block_swizzle_size
        self.rasterize_order = rasterize_order
        self.swapAB = swapAB
        self.k_group_size = k_group_size
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_split_k = num_split_k


# codegen template
# TuningConfigSM100(
#     M=[@M],
#     N=[@N],
#     K=[@K],
#     mma_shape=Index([@TUNE_BM] * 2, [@TUNE_BN] * 2, mma_k),
#     block_tile_shape=Index([@TUNE_BM], [@TUNE_BN], bk),
#     cluster_shape=Index([@TUNE_CLUSTER_DIM_X], [@TUNE_CLUSTER_DIM_Y], [@TUNE_CLUSTER_DIM_Z]),
#     block_swizzle_size=[@TUNE_BLOCK_SWIZZLE_SIZE],
#     rasterize_order=RasterOrder([@TUNE_RASTER_ORDER]),
# )

# ===----------------------------------------------------------------------=== #
# BF16 outliers
# ===----------------------------------------------------------------------=== #


fn _get_tuning_list_sm100_bf16() -> List[TuningConfigSM100]:
    return [
        TuningConfigSM100(
            M=3456,
            M_end=3456 + 64,
            N=43008,
            K=5376,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        TuningConfigSM100(
            M=48000,
            M_end=48000 + 64,
            N=5376,
            K=21504,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # ------------ llama3-8b-tp2 ------------#
        TuningConfigSM100(
            M=87,
            M_end=129,
            N=3072,
            K=4096,
            mma_shape=Index(128, 48, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            k_group_size=2,
        ),
        TuningConfigSM100(
            M=449,
            M_end=513,
            N=3072,
            K=4096,
            mma_shape=Index(128, 192, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            k_group_size=1,
        ),
        TuningConfigSM100(
            M=9,
            M_end=32,
            N=4096,
            K=7168,
            mma_shape=Index(128, 16, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            k_group_size=4,
        ),
        # ------------ llama3-8b-tp1 ------------#
        TuningConfigSM100(
            M=65,
            M_end=81,
            N=6144,
            K=4096,
            mma_shape=Index(128, 80, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            k_group_size=2,
        ),
        TuningConfigSM100(
            M=81,
            M_end=97,
            N=6144,
            K=4096,
            mma_shape=Index(128, 96, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            k_group_size=2,
        ),
        TuningConfigSM100(
            M=9,
            M_end=32,
            N=6144,
            K=4096,
            mma_shape=Index(128, 32, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            k_group_size=2,
        ),
        TuningConfigSM100(
            M=9,
            M_end=17,
            N=4096,
            K=4096,
            mma_shape=Index(128, 16, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            k_group_size=2,
        ),
        TuningConfigSM100(
            M=17,
            M_end=64,
            N=4096,
            K=4096,
            mma_shape=Index(128, 32, 16),
            cta_group=2,
            cluster_shape=Index(2, 2, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            k_group_size=2,
        ),
        TuningConfigSM100(
            M=9,
            M_end=22,
            N=28672,
            K=4096,
            mma_shape=Index(128, 32, 16),
            cta_group=2,
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            k_group_size=4,
        ),
        TuningConfigSM100(
            M=23,
            M_end=47,
            N=28672,
            K=4096,
            mma_shape=Index(128, 48, 16),
            cta_group=2,
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            k_group_size=4,
        ),
        TuningConfigSM100(
            M=48,
            M_end=65,
            N=28672,
            K=4096,
            mma_shape=Index(256, 64, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            k_group_size=2,
        ),
        TuningConfigSM100(
            M=65,
            M_end=81,
            N=28672,
            K=4096,
            mma_shape=Index(256, 80, 16),
            cta_group=2,
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            k_group_size=2,
        ),
    ]


# ===----------------------------------------------------------------------=== #
# FP8 Shapes
# ===----------------------------------------------------------------------=== #


fn _get_tuning_list_sm100_fp8[mma_k: Int, bk: Int]() -> List[TuningConfigSM100]:
    # ----------------BEGIN-TUNING-LIST-SM100-FP8----------------
    comptime config_list = [
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [0]
        TuningConfigSM100(
            M=150,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [1]
        TuningConfigSM100(
            M=225,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [2]
        TuningConfigSM100(
            M=256,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 2, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [3]
        TuningConfigSM100(
            M=300,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [4]
        TuningConfigSM100(
            M=450,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [5]
        TuningConfigSM100(
            M=512,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [6]
        TuningConfigSM100(
            M=600,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(64, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [7]
        TuningConfigSM100(
            M=750,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 48 * 2, mma_k),
            block_tile_shape=Index(128, 48, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [8]
        TuningConfigSM100(
            M=768,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(64, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [9]
        TuningConfigSM100(
            M=1024,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(64, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [10]
        TuningConfigSM100(
            M=2048,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [11]
        TuningConfigSM100(
            M=4096,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [12]
        TuningConfigSM100(
            M=6144,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [13]
        TuningConfigSM100(
            M=8192,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [14]
        TuningConfigSM100(
            M=150,
            N=4608,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [15]
        TuningConfigSM100(
            M=225,
            N=4608,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [16]
        TuningConfigSM100(
            M=300,
            N=4608,
            K=16384,
            mma_shape=Index(64 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(64, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [17]
        TuningConfigSM100(
            M=450,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(128, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [18]
        TuningConfigSM100(
            M=600,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [19]
        TuningConfigSM100(
            M=750,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [20]
        TuningConfigSM100(
            M=2048,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [21]
        TuningConfigSM100(
            M=4096,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [22]
        TuningConfigSM100(
            M=6144,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [23]
        TuningConfigSM100(
            M=8192,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [24]
        TuningConfigSM100(
            M=150,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [25]
        TuningConfigSM100(
            M=225,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [26]
        TuningConfigSM100(
            M=256,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [27]
        TuningConfigSM100(
            M=300,
            N=13312,
            K=16384,
            mma_shape=Index(64 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(64, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [28]
        TuningConfigSM100(
            M=450,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [29]
        TuningConfigSM100(
            M=512,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [30]
        TuningConfigSM100(
            M=600,
            N=13312,
            K=16384,
            mma_shape=Index(64 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(64, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [31]
        TuningConfigSM100(
            M=750,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [32]
        TuningConfigSM100(
            M=768,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [33]
        TuningConfigSM100(
            M=1024,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [34]
        TuningConfigSM100(
            M=2048,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 104 * 2, mma_k),
            block_tile_shape=Index(128, 104, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [35]
        TuningConfigSM100(
            M=4096,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [36]
        TuningConfigSM100(
            M=6144,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [37]
        TuningConfigSM100(
            M=8192,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 88 * 2, mma_k),
            block_tile_shape=Index(128, 88, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [38]
        TuningConfigSM100(
            M=150,
            N=16384,
            K=2048,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [39]
        TuningConfigSM100(
            M=225,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [40]
        TuningConfigSM100(
            M=256,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [41]
        TuningConfigSM100(
            M=300,
            N=16384,
            K=2048,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [42]
        TuningConfigSM100(
            M=450,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [43]
        TuningConfigSM100(
            M=512,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [44]
        TuningConfigSM100(
            M=600,
            N=16384,
            K=2048,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [45]
        TuningConfigSM100(
            M=750,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [46]
        TuningConfigSM100(
            M=768,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [47]
        TuningConfigSM100(
            M=1024,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [48]
        TuningConfigSM100(
            M=2048,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [49]
        TuningConfigSM100(
            M=4096,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [50]
        TuningConfigSM100(
            M=6144,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [51]
        TuningConfigSM100(
            M=8192,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [52]
        TuningConfigSM100(
            M=150,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [53]
        TuningConfigSM100(
            M=225,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [54]
        TuningConfigSM100(
            M=300,
            N=16384,
            K=4096,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [55]
        TuningConfigSM100(
            M=450,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [56]
        TuningConfigSM100(
            M=600,
            N=16384,
            K=4096,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [57]
        TuningConfigSM100(
            M=750,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [58]
        TuningConfigSM100(
            M=2048,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [59]
        TuningConfigSM100(
            M=4096,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [60]
        TuningConfigSM100(
            M=6144,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [61]
        TuningConfigSM100(
            M=8192,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [62]
        TuningConfigSM100(
            M=150,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [63]
        TuningConfigSM100(
            M=225,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [64]
        TuningConfigSM100(
            M=256,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [65]
        TuningConfigSM100(
            M=300,
            N=16384,
            K=6656,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [66]
        TuningConfigSM100(
            M=450,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [67]
        TuningConfigSM100(
            M=512,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [68]
        TuningConfigSM100(
            M=600,
            N=16384,
            K=6656,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [69]
        TuningConfigSM100(
            M=750,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [70]
        TuningConfigSM100(
            M=768,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [71]
        TuningConfigSM100(
            M=1024,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [72]
        TuningConfigSM100(
            M=2048,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [73]
        TuningConfigSM100(
            M=4096,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [74]
        TuningConfigSM100(
            M=6144,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [75]
        TuningConfigSM100(
            M=8192,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [76]
        TuningConfigSM100(
            M=150,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [77]
        TuningConfigSM100(
            M=225,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [78]
        TuningConfigSM100(
            M=300,
            N=16384,
            K=13312,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [79]
        TuningConfigSM100(
            M=450,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [80]
        TuningConfigSM100(
            M=600,
            N=16384,
            K=13312,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [81]
        TuningConfigSM100(
            M=750,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [82]
        TuningConfigSM100(
            M=2048,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [83]
        TuningConfigSM100(
            M=4096,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [84]
        TuningConfigSM100(
            M=6144,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [85]
        TuningConfigSM100(
            M=8192,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [86]
        TuningConfigSM100(
            M=150,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [87]
        TuningConfigSM100(
            M=225,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [88]
        TuningConfigSM100(
            M=300,
            N=26624,
            K=16384,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [89]
        TuningConfigSM100(
            M=450,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 120 * 2, mma_k),
            block_tile_shape=Index(128, 120, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [90]
        TuningConfigSM100(
            M=600,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [91]
        TuningConfigSM100(
            M=750,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [92]
        TuningConfigSM100(
            M=2048,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 120 * 2, mma_k),
            block_tile_shape=Index(128, 120, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [93]
        TuningConfigSM100(
            M=4096,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 104 * 2, mma_k),
            block_tile_shape=Index(128, 104, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [94]
        TuningConfigSM100(
            M=6144,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 120 * 2, mma_k),
            block_tile_shape=Index(128, 120, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [95]
        TuningConfigSM100(
            M=8192,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
    ]
    # ----------------END-TUNING-LIST-SM100-FP8----------------

    return materialize[config_list]()


fn _get_tuning_list_sm100_nvfp4() -> List[TuningConfigSM100]:
    comptime config_list = [
        TuningConfigSM100(
            M=1,
            M_end=2,
            N=7168,
            K=16384,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
        ),
    ]

    return materialize[config_list]()


fn _get_tuning_list_sm100_mxfp8() -> List[TuningConfigSM100]:
    comptime config_list = [
        TuningConfigSM100(
            M=1,
            M_end=2,
            N=7168,
            K=16384,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
        ),
    ]

    return materialize[config_list]()
