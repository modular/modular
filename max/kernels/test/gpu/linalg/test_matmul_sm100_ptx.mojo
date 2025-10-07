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
from gpu.host._nvidia_cuda import TensorMapSwizzle
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.matmul.gpu.sm100.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from layout.layout import Layout
from layout.tma_async import _tma_desc_tile_layout
from linalg.utils_gpu import MatmulConfig
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
    cta_group: Int = 1,
    num_clc_pipeline_stages: UInt = 2,
    block_swizzle_size: Int = 0,
    rasterize_order: RasterOrder = RasterOrder.AlongM,
    num_pipeline_stages: Optional[UInt] = None,
    transpose_c: Bool = False,
]() raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    alias MMA_M = config.mma_shape[0]
    alias MMA_N = config.mma_shape[1]
    alias MMA_K = config.mma_shape[2]

    alias BM = MMA_M // cta_group
    alias BN = MMA_N // cta_group
    alias BK = config.block_tile_shape[2]

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

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B

    alias cluster_shape = config.cluster_shape

    alias a_tma_shape = Index(BM // cluster_shape[1], BK)
    alias a_tma_layout = Layout.row_major(a_tma_shape[0], a_tma_shape[1])
    alias a_tma_desc_layout = _tma_desc_tile_layout[
        a_type, 2, a_tma_shape, True, a_swizzle
    ]()

    alias b_tma_shape = Index(BN // (cluster_shape[0] // cta_group), BK)
    alias b_tma_layout = Layout.row_major(b_tma_shape[0], b_tma_shape[1])
    alias b_tma_desc_layout = _tma_desc_tile_layout[
        b_type, 2, b_tma_shape, True, b_swizzle
    ]()

    alias width = 32 if (MMA_M == 256 and MMA_N % 32 == 0) or (
        MMA_M == 128 and BN % 32 == 0
    ) else 16
    alias output_tile_shape = Index(128, width) if not transpose_c else Index(
        width, 128
    )
    alias split_tile_shape = Index(64, width) if not transpose_c else Index(
        width, 64
    )
    alias c_tma_tile_shape = output_tile_shape if MMA_M == 256 else split_tile_shape
    alias c_swizzle = TensorMapSwizzle.SWIZZLE_32B if transpose_c else (
        TensorMapSwizzle.SWIZZLE_64B if width
        == 32 else TensorMapSwizzle.SWIZZLE_32B
    )
    # transpose_c => MMA_M == 256 is the same as (not transpose_c) or MMA_M == 256
    constrained[
        (not transpose_c) or MMA_M == 256,
        "swapAB is only supported for MMA_M == 256",
    ]()
    alias c_tma_shape = c_tma_tile_shape if not transpose_c else Index(
        c_tma_tile_shape[0], c_tma_tile_shape[1] // 8
    )
    alias c_tma_layout = Layout.row_major(
        c_tma_tile_shape[0], c_tma_tile_shape[1]
    )
    alias c_tma_desc_layout = _tma_desc_tile_layout[
        c_type, 2, c_tma_shape, True, c_swizzle
    ]()

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    alias b200_smem = B200.shared_memory_per_multiprocessor - 1024
    alias a_smem_bytes_per_stage = BM * BK * size_of[a_type]()
    alias b_smem_bytes_per_stage = BN * BK * size_of[b_type]()
    # A and B per pipeline stage
    alias AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage
    # Support double-buffer for output stages.
    alias num_output_stages = 2

    alias c_smem_bytes = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages * size_of[c_type]()

    alias MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
    alias CLC_RESPONSE_BYTES = size_of[Int128]()  # 16 bytes per response
    alias TMEM_ADDR_BYTES = size_of[
        Int32
    ]()  # 4 bytes or 32 bits for tensor memory address
    # the 'N' dimension of tensor memory is 512
    alias TMEM_N = 512
    # the maximum different number of mma's that can be run in parallel is TMEM_N/MMA_N
    alias max_accum_pipeline_stages = TMEM_N // next_power_of_two(MMA_N)
    # Mainloop barrier
    alias accum_full_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages
    alias accum_empty_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages

    alias clc_response_bytes = CLC_RESPONSE_BYTES * num_clc_pipeline_stages
    alias clc_full_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    alias clc_empty_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    alias clc_throttle_full_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    alias clc_throttle_empty_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages

    alias tmem_addr_bytes = TMEM_ADDR_BYTES
    alias tmem_dealloc_mbar_bytes = MBAR_BYTES

    alias tmem_writeout_smem = c_smem_bytes + tmem_addr_bytes + tmem_dealloc_mbar_bytes
    alias accum_smem = accum_full_mbar_bytes + accum_empty_mbar_bytes
    alias clc_smem = (
        clc_response_bytes
        + clc_full_mbar_bytes
        + clc_empty_mbar_bytes
        + clc_throttle_full_mbar_bytes
        + clc_throttle_empty_mbar_bytes
    )
    alias smem_leftover = (b200_smem) - (
        clc_smem + accum_smem + tmem_writeout_smem
    )

    alias tma_mbar_bytes_per_stage = MBAR_BYTES
    alias mma_mbar_bytes_per_stage = MBAR_BYTES

    alias producer_consumer_smem_per_stage = (
        AB_smem_per_stage + tma_mbar_bytes_per_stage + mma_mbar_bytes_per_stage
    )

    alias max_pipeline_stages: UInt = smem_leftover // producer_consumer_smem_per_stage

    constrained[
        max_pipeline_stages >= 1, "Max pipeline stages must be at least 1"
    ]()

    @parameter
    if num_pipeline_stages:
        constrained[
            num_pipeline_stages.value() <= max_pipeline_stages,
            "Pipeline stage must be less than or equal to max pipeline stages",
        ]()

    alias pipeline_stage = num_pipeline_stages.value() if num_pipeline_stages else max_pipeline_stages
    alias producer_consumer_smem = producer_consumer_smem_per_stage * pipeline_stage

    alias smem_size = (
        clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
    )

    alias kernel = blackwell_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_layout,
        b_tma_layout,
        c_tma_layout,
        a_tma_desc_layout,
        b_tma_desc_layout,
        c_tma_desc_layout,
        config.block_tile_shape,
        config.mma_shape,
        transpose_b=transpose_b,
        cluster_shape = StaticTuple[Int32, 3](
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=cta_group,
        num_pipeline_stages = UInt(pipeline_stage),
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        num_accum_pipeline_stages = UInt(max_accum_pipeline_stages),
        num_output_stages = UInt(num_output_stages),
        output_tile_shape=output_tile_shape,
        elementwise_compute_lambda_fn=None,
        block_swizzle_size=block_swizzle_size,
        rasterize_order=rasterize_order,
        transpose_c=transpose_c,
    ]

    var ptx = _compile_code[kernel, target = get_gpu_target["sm_100a"]()]().asm
    alias M = c_layout.shape[0].value()
    alias N = c_layout.shape[1].value()
    alias K = a_layout.shape[1].value()
    var expected_ptx = reference_ptx[M, N, K]()
    assert_equal(ptx, expected_ptx)


def main():
    alias BK = 64
    alias MMA_K = 16
    alias a_type = DType.bfloat16
    alias b_type = DType.bfloat16
    alias c_type = DType.bfloat16
    alias transpose_b = True

    alias M = 4096
    alias N = 4096
    alias K = 4096
    alias c_layout = Layout.row_major(M, N)
    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(N, K)

    alias block_tile_shape = Index(128, 128, BK)
    alias mma_shape = Index(
        block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
    )
    alias cluster_shape = Index(2, 1, 1)
    alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=block_tile_shape,
        mma_shape=mma_shape,
        cluster_shape=cluster_shape,
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
        cta_group=2,
        block_swizzle_size=0,
        rasterize_order = RasterOrder.AlongM,
    ]()
