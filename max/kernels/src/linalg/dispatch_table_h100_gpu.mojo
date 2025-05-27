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
from gpu.host.info import H100

from utils import IndexList, Index

from .matmul_gpu import multistage_gemm
from .utils import elementwise_epilogue_type
from .utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    _bk_base,
    _get_block_warp_tile_shape,
    select_config,
)


fn get_grid_shape[key: String]() -> IndexList[2]:
    if key == "8192_2560_8192":
        return Index(10, H100.sm_count // 10)
    elif key == "8192_8192_2048":
        return Index(4, H100.sm_count // 4)
    elif key == "8192_14336_8192":
        return Index(8, H100.sm_count // 8)
    elif key == "8192_8192_7168":
        return Index(8, H100.sm_count // 8)
    else:
        return IndexList[2](1, 1)


fn get_config_h100[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    key: String,
    mma_shape: IndexList[3],
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape]:
    if key == "512_2560_8192":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 80, 64),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
        )
    elif key == "8192_2560_8192":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
        )
    elif key == "4096_2560_8192":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
        )

    elif key == "8192_8192_2048":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
        )
    elif key == "4096_8192_2048":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
        )

    elif key == "8192_14336_8192":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
        )
    elif key == "4096_14336_8192":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
        )

    elif key == "8192_8192_7168":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
        )
    elif key == "4096_8192_7168":
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape,](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
        )

    else:
        return MatmulConfig[a_type, b_type, c_type, transpose_b, mma_shape](
            num_pipeline_stages=0
        )
