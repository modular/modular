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

from collections import Dict, InlineArray, OptionalReg
from pathlib import Path

from algorithm.functional import elementwise, tile_and_unswitch
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext

from .matmul_gpu import multistage_gemm
from .utils import elementwise_epilogue_type
from .utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    _bk_base,
    _get_block_warp_tile_shape,
    select_config,
)


fn create_tile_configs[
    key: String, a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    alias input = key

    var tile_configs = Dict[
        String, MatmulConfig[a_type, b_type, c_type, transpose_b]
    ]()
    # --------------------------------K=4096, N=4096--------------------------------

    tile_configs["16_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["128_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["256_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 256, 32),
        warp_tile_shape=(32, 128, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["512_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    tile_configs["768_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1280_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1606_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["2048_4096_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # K=4096, N=14336
    tile_configs["16_14336_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_14336_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_14336_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    # --------------------------------K=14336, N=4096--------------------------------
    tile_configs["16_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["128_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 256, 64),
        warp_tile_shape=(32, 128, 32),
        num_pipeline_stages=4,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    tile_configs["512_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    tile_configs["768_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["896_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    tile_configs["1024_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    tile_configs["1152_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    tile_configs["1280_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    tile_configs["1606_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["2048_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    # --------------------------------K=4096, N=128256--------------------------------
    tile_configs["128_128256_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["896_128256_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["2048_128256_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["768_128256_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # --------------------------------K=4096, N=28672--------------------------------
    tile_configs["16_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["128_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["256_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 256, 64),
        warp_tile_shape=(32, 128, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["512_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["768_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["896_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1024_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1152_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1280_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1606_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["2048_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # --------------------------------K=4096, N=6144--------------------------------
    tile_configs["16_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["128_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["256_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["512_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["768_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["896_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 32),
        warp_tile_shape=(64, 128, 16),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # --------------------------------K=4096, N=14336--------------------------------
    tile_configs["16_14336_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_14336_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_14336_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    # --------------------------------K=14336, N=4096--------------------------------
    tile_configs["16_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["128_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 256, 64),
        warp_tile_shape=(32, 128, 32),
        num_pipeline_stages=4,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    tile_configs["512_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    tile_configs["768_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["896_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    tile_configs["1024_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    tile_configs["1152_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=3,
        num_warp_k_partitions=1,
    )

    tile_configs["1280_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    tile_configs["1606_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["2048_4096_14336"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=2,
        num_warp_k_partitions=1,
    )

    # --------------------------------K=4096, N=128256--------------------------------
    tile_configs["128_128256_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["896_128256_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["2048_128256_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["768_128256_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # --------------------------------K=4096, N=28672--------------------------------
    tile_configs["16_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["128_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["256_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 256, 64),
        warp_tile_shape=(32, 128, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["512_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["768_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["896_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1024_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1152_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1280_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1606_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["2048_28672_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # --------------------------------K=4096, N=6144--------------------------------
    tile_configs["16_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(16, 64, 128),
        warp_tile_shape=(16, 32, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["32_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(32, 64, 64),
        warp_tile_shape=(16, 64, 32),
        num_pipeline_stages=6,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["64_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 64, 64),
        warp_tile_shape=(32, 64, 32),
        num_pipeline_stages=5,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["128_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(64, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=2,
    )

    tile_configs["256_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 64),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["512_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["768_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["896_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 32),
        warp_tile_shape=(64, 128, 16),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1024_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 64, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1152_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1280_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 128, 32),
        warp_tile_shape=(64, 64, 32),
        num_pipeline_stages=4,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["1606_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(128, 256, 64),
        warp_tile_shape=(64, 128, 32),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    tile_configs["2048_6144_4096"] = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        block_tile_shape=(256, 128, 64),
        warp_tile_shape=(128, 128, 64),
        num_pipeline_stages=3,
        num_k_partitions=1,
        num_warp_k_partitions=1,
    )

    # The except is a dead branch, it will not execute because if the input is
    # not in the dictionary, then the check_in_dict function will have already
    # raised an error. The only reason the except here exists is because the
    # compiler does not allow a direct return because a dictionary could, in
    # theory, return an error.
    try:
        return tile_configs[input]
    except not_here:
        return MatmulConfig[a_type, b_type, c_type, transpose_b](
            num_pipeline_stages=0,
        )  # 128x128_4
