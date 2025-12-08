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

from math import align_up
from sys import argv, size_of

from bit import next_power_of_two, prev_power_of_two
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.compile import _compile_code, get_gpu_target
from gpu.host.info import B200
from gpu.host.nvidia.tma import TensorMapSwizzle
from internal_utils import assert_almost_equal, random, zero
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.matmul.gpu.sm100.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from layout.layout import Layout
from layout.tma_async import _tma_desc_tile_layout
from linalg.matmul.gpu.sm100.config import MatmulConfig
from linalg.matmul.gpu.sm100.tile_scheduler import RasterOrder
from linalg.matmul.gpu.sm100.matmul import (
    blackwell_tma_umma_warp_specialized_kernel,
)
from testdata.matmul_sm100_ptx import reference_ptx
from testing import assert_equal

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


fn test_ptx[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]() raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    # constraint for bfloat16 matmul
    constrained[
        (a_type != DType.bfloat16) or (MMA_M != 128) or (MMA_N % 32 == 0),
        "if MMA_M is 128, then MMA_N must be a multiple of 32",
    ]()

    # constraint for fp8 matmul
    constrained[
        (a_type != DType.float8_e4m3fn) or (MMA_N % 64 == 0),
        "MMA_N must be a multiple of 64 for fp8 matmul",
    ]()

    comptime a_swizzle = config.a_swizzle
    comptime b_swizzle = config.b_swizzle

    comptime cluster_shape = config.cluster_shape

    comptime a_tma_shape = Index(BM // cluster_shape[1], BK)
    comptime a_tma_layout = Layout.row_major(a_tma_shape[0], a_tma_shape[1])
    comptime a_tma_desc_layout = _tma_desc_tile_layout[
        a_type, 2, a_tma_shape, True, a_swizzle
    ]()

    comptime b_tma_shape = Index(
        BN // (cluster_shape[0] // config.cta_group), BK
    )
    comptime b_tma_layout = Layout.row_major(b_tma_shape[0], b_tma_shape[1])
    comptime b_tma_desc_layout = _tma_desc_tile_layout[
        b_type, 2, b_tma_shape, True, b_swizzle
    ]()

    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if MMA_M == 256 else c_tma_tile_shape_mma128

    comptime c_tma_shape = c_tma_tile_shape if not config.AB_swapped else Index(
        c_tma_tile_shape[0], c_tma_tile_shape[1] // 8
    )
    comptime c_tma_layout = Layout.row_major(c_tma_shape[0], c_tma_shape[1])
    comptime c_tma_desc_layout = _tma_desc_tile_layout[
        c_type, 2, c_tma_shape, True, config.c_swizzle
    ]()

    comptime kernel = blackwell_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_layout,
        b_tma_layout,
        c_tma_layout,
        a_tma_desc_layout,
        b_tma_desc_layout,
        c_tma_desc_layout,
        transpose_b=transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        elementwise_compute_lambda_fn=None,
    ]

    var ptx = _compile_code[kernel, target = get_gpu_target["sm_100a"]()]().asm
    comptime M = c_layout.shape[0].value()
    comptime N = c_layout.shape[1].value()
    comptime K = a_layout.shape[1].value()
    var expected_ptx = reference_ptx[M, N, K]()
    assert_equal(ptx, expected_ptx)


def main():
    comptime BK = 64
    comptime MMA_K = 16
    comptime a_type = DType.bfloat16
    comptime b_type = DType.bfloat16
    comptime c_type = DType.bfloat16
    comptime transpose_b = True

    comptime M = 4096
    comptime N = 4096
    comptime K = 4096
    comptime c_layout = Layout.row_major(M, N)
    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(N, K)

    comptime block_tile_shape = Index(128, 128, BK)
    comptime mma_shape = Index(
        block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
    )
    comptime cluster_shape = Index(2, 1, 1)
    comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        cta_group=2,
        block_swizzle_size=0,
        raster_order=RasterOrder.AlongM,
    )
    test_ptx[
        c_type,
        c_layout,
        a_type,
        a_layout,
        b_type,
        b_layout,
        transpose_b=transpose_b,
        config=config,
    ]()
