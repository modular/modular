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
from utils import IndexList

from .utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    _bk_base,
    select_config,
)


fn create_matmul_configs_ampere[
    key: String, a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    alias dict = get_dispatch_table[a_type, b_type, c_type, transpose_b]()
    try:
        return dict[key]
    except error:
        return MatmulConfig[a_type, b_type, c_type, transpose_b](
            num_pipeline_stages=0,
        )  # 128x128_4


fn get_dispatch_table[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
]() -> Dict[String, MatmulConfig[a_type, b_type, c_type, transpose_b]]:
    var tile_configs = Dict[
        String, MatmulConfig[a_type, b_type, c_type, transpose_b]
    ]()

    @always_inline
    fn insert(
        name: StaticString,
        *,
        block_tile_shape: IndexList[3],
        warp_tile_shape: IndexList[3],
        num_pipeline_stages: Int,
        num_k_partitions: Int,
        num_warp_k_partitions: Int,
    ):
        tile_configs[name] = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=block_tile_shape,
            warp_tile_shape=warp_tile_shape,
            num_pipeline_stages=num_pipeline_stages,
            num_k_partitions=num_k_partitions,
            num_warp_k_partitions=num_warp_k_partitions,
        )

    # ===------------------------------------------------------------------=== #
    # Static_NK = (4096, 4096)
    # ===------------------------------------------------------------------=== #

    insert(
        "16_4096_4096",
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "32_4096_4096",
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "64_4096_4096",
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "128_4096_4096",
        block_tile_shape=(64, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "256_4096_4096",
        block_tile_shape=(64, 256, 32),
        warp_tile_shape=(32, 128, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "512_4096_4096",
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    insert(
        "768_4096_4096",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1280_4096_4096",
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1606_4096_4096",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "2048_4096_4096",
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # ===------------------------------------------------------------------=== #
    # Static_NK = (4096, 14336)
    # ===------------------------------------------------------------------=== #

    insert(
        "16_4096_14336",
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "32_4096_14336",
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "64_4096_14336",
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "128_4096_14336",
        block_tile_shape=(64, 256, 64),
        warp_tile_shape=(32, 128, 32),
        num_pipeline_stages=4,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    insert(
        "512_4096_14336",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    insert(
        "768_4096_14336",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "896_4096_14336",
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    insert(
        "1024_4096_14336",
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    insert(
        "1152_4096_14336",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    insert(
        "1280_4096_14336",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    insert(
        "1606_4096_14336",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "2048_4096_14336",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    # ===------------------------------------------------------------------=== #
    # Static_NK = (128256, 4096)
    # ===------------------------------------------------------------------=== #

    insert(
        "128_128256_4096",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "896_128256_4096",
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "2048_128256_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "768_128256_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # ===------------------------------------------------------------------=== #
    # Static_NK = (28672, 4096)
    # ===------------------------------------------------------------------=== #

    insert(
        "16_28672_4096",
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "32_28672_4096",
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "64_28672_4096",
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "128_28672_4096",
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "256_28672_4096",
        block_tile_shape=(64, 256, 64),
        warp_tile_shape=(32, 128, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "512_28672_4096",
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "768_28672_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "896_28672_4096",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1024_28672_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1152_28672_4096",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1280_28672_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1606_28672_4096",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "2048_28672_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # ===------------------------------------------------------------------=== #
    # Static_NK = (6144, 4096)
    # ===------------------------------------------------------------------=== #

    insert(
        "16_6144_4096",
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "32_6144_4096",
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "64_6144_4096",
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "128_6144_4096",
        block_tile_shape=(64, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "256_6144_4096",
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "512_6144_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "768_6144_4096",
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "896_6144_4096",
        block_tile_shape=(128, 256, 32),
        warp_tile_shape=(64, 128, 16),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1024_6144_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1152_6144_4096",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1280_6144_4096",
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "1606_6144_4096",
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    insert(
        "2048_6144_4096",
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(64, 64, 64),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # ===------------------------------------------------------------------=== #
    # Static_NK = (14336, 4096)
    # ===------------------------------------------------------------------=== #

    insert(
        "16_14336_4096",
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "32_14336_4096",
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    insert(
        "64_14336_4096",
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    return tile_configs
