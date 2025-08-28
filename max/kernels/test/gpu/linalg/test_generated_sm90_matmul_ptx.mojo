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

from collections import OptionalReg
from math import ceildiv
from sys import size_of
from testing import assert_true
from gpu.globals import WARPGROUP_SIZE
from gpu.host.compile import _compile_code
from gpu.host import get_gpu_target
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import H100
from layout import Layout
from linalg.matmul_tile_scheduler import MatmulSchedule
from stdlib.bit import log2_floor

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from linalg.utils_gpu import (
    MatmulConfig,
)
from layout.tma_async import _tma_desc_tile_layout

from linalg.matmul_sm90 import (
    _is_valid_grid_shape,
    _get_grid_shape,
    _get_c_smem_layout,
    tma_wgmma_warp_specialized_gemm_kernel,
    tma_wgmma_warp_specialized_gemm_kernel_persistent,
)


fn compile_sm90_matmul_ptx[
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    *,
    M: Int,
    N: Int,
    K: Int,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    hilbert_swizzle: Bool = False,
]() raises:
    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    constrained[
        (a_type == b_type is DType.float8_e4m3fn)
        or (a_type == b_type and a_type in (DType.bfloat16, DType.float32)),
        "Unsupported input dtype",
    ]()

    constrained[
        a_type != DType.float8_e4m3fn or BK == 128,
        "BK must be 128 for fp8 data type for numerical accuracy correctness",
    ]()

    constrained[
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None,
        "Either the epilogue lambda or the compute lambda can be used",
    ]()

    constrained[
        BM > 64 or (BM == 64 and config.num_consumer == 1),
        "Only support 1 consumer for BM=64",
    ]()

    @parameter
    if schedule == MatmulSchedule.DS_SCHEDULER:
        constrained[
            grid_shape is not None,
            "Grid shape must be provided for DS scheduler",
        ]()
        alias ds_grid_shape = grid_shape.value()
        constrained[
            ds_grid_shape[0] <= H100.sm_count and ds_grid_shape[1] == 1,
            "Deepseek scheduler only accepts grid shape with 1 column",
        ]()

    elif grid_shape:
        constrained[
            _is_valid_grid_shape[grid_shape.value(), config.cluster_shape](
                ceildiv(N, BN)
            ),
            String(
                "grid shape:",
                grid_shape.value(),
                "is not compatible with cluster shape:",
                config.cluster_shape,
                "and static N:",
                N,
                sep=" ",
            ),
        ]()

    alias grid_shape_adjusted = grid_shape.value() if grid_shape else _get_grid_shape[
        config.cluster_shape
    ](
        ceildiv(N, BN)
    )

    alias cluster_shape = StaticTuple[Int32, 3](
        config.cluster_shape[0],
        config.cluster_shape[1],
        config.cluster_shape[2],
    )

    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])

    alias c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        config.num_pipeline_stages,
    ]()
    alias c_smem_tile = Index(
        c_smem_layout.shape[0].value(),
        c_smem_layout.shape[1].value() // config.num_consumer,
    )

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    # make sure TMA_BN = 64 -> 128B swizzle, 32 -> 64B swizzle and etc.
    alias c_swizzle = TensorMapSwizzle(
        min(log2_floor(c_smem_tile[1] // 8), 3)
    ) if use_tma_store else TensorMapSwizzle.SWIZZLE_NONE

    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    alias c_layout = Layout.row_major(M, N)

    alias a_tile_shape = Index(
        BM // CLUSTER_N, BK
    ) if config.partitioned_multicast else Index(BM, BK)
    alias b_tile_shape = Index(
        BN // CLUSTER_M, BK
    ) if config.partitioned_multicast else Index(BN, BK)

    alias a_tile_layout = Layout.row_major(a_tile_shape[0], a_tile_shape[1])
    alias b_tile_layout = Layout.row_major(b_tile_shape[0], b_tile_shape[1])
    alias c_tile_layout = Layout.row_major(c_smem_tile[0], c_smem_tile[1])

    alias a_tma_desc_layout = _tma_desc_tile_layout[
        a_type, 2, a_tile_shape, is_k_major=True, swizzle_mode=a_swizzle
    ]()
    alias b_tma_desc_layout = _tma_desc_tile_layout[
        b_type, 2, b_tile_shape, is_k_major=True, swizzle_mode=b_swizzle
    ]()
    alias c_tma_desc_layout = Layout.row_major(c_smem_tile[0], c_smem_tile[1])

    alias num_threads = WARPGROUP_SIZE * config.num_consumer + WARPGROUP_SIZE
    alias smem_size = Int(config.num_pipeline_stages) * (
        BM * BK * size_of[a_type]()
        + BN * BK * size_of[b_type]()
        + (size_of[Int64]() * 2)
    ) + c_smem_layout.size() * size_of[c_type]()

    constrained[
        smem_size <= H100.shared_memory_per_multiprocessor - 1024,
        "requested SMEM size exceeds 227KB limit.",
    ]()

    @parameter
    if schedule != MatmulSchedule.NONE:
        alias kernel = tma_wgmma_warp_specialized_gemm_kernel_persistent[
            a_type,
            b_type,
            c_type,
            a_layout,
            b_layout,
            a_tile_layout,
            b_tile_layout,
            c_layout,
            config.block_tile_shape,
            config.mma_shape,
            a_tma_desc_layout,
            b_tma_desc_layout,
            c_tma_desc_layout,
            c_tile_layout,
            c_smem_layout,
            c_swizzle=c_swizzle,
            cluster_shape=cluster_shape,
            grid_shape=grid_shape_adjusted,
            schedule=schedule,
            transpose_b=True,
            num_threads=num_threads,
            pipeline_stages = config.num_pipeline_stages,
            partitioned_multicast = config.partitioned_multicast,
            use_tma_store=use_tma_store,
            pdl_level = config.pdl_level(),
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        ]

        var asm = _compile_code[
            kernel,
            target = get_gpu_target["sm_90"](),
        ]().asm
        assert_true("ld.local" not in asm and "st.local" not in asm)
    else:
        alias kernel = tma_wgmma_warp_specialized_gemm_kernel[
            a_type,
            b_type,
            c_type,
            a_layout,
            b_layout,
            a_tile_layout,
            b_tile_layout,
            c_layout,
            config.block_tile_shape,
            config.mma_shape,
            a_tma_desc_layout,
            b_tma_desc_layout,
            c_tma_desc_layout,
            c_tile_layout,
            c_smem_layout,
            c_swizzle=c_swizzle,
            cluster_shape=cluster_shape,
            transpose_b=True,
            num_threads=num_threads,
            pipeline_stages = config.num_pipeline_stages,
            partitioned_multicast = config.partitioned_multicast,
            use_tma_store=use_tma_store,
            pdl_level = config.pdl_level(),
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            hilbert_swizzle=hilbert_swizzle,
        ]
        var asm = _compile_code[
            kernel,
            target = get_gpu_target["sm_90"](),
        ]().asm
        assert_true("ld.local" not in asm and "st.local" not in asm)


fn test_local_memory_access() raises:
    print("== test_local_memory_access")

    alias M8192_N14336_K8192_config = MatmulConfig[
        DType.bfloat16,
        DType.bfloat16,
        DType.bfloat16,
        transpose_b=True,
    ](
        block_tile_shape=Index(128, 256, 64),
        mma_shape=Index(64, 256, 16),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
    )
    compile_sm90_matmul_ptx[
        M=8192,
        N=14336,
        K=8192,
        transpose_b=True,
        config=M8192_N14336_K8192_config,
        grid_shape = Index(8, H100.sm_count // 8),
        schedule = MatmulSchedule.TILE2D,
    ]()

    alias M2048_N14336_K8192_config = MatmulConfig[
        DType.bfloat16,
        DType.bfloat16,
        DType.bfloat16,
        transpose_b=True,
    ](
        block_tile_shape=Index(128, 256, 64),
        mma_shape=Index(64, 256, 16),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
    )
    compile_sm90_matmul_ptx[
        M=2048,
        N=14336,
        K=8192,
        transpose_b=True,
        config=M2048_N14336_K8192_config,
        grid_shape = Index(8, H100.sm_count // 8),
        schedule = MatmulSchedule.TILE2D,
    ]()

    alias M2048_N4096_K256_config = MatmulConfig[
        DType.bfloat16,
        DType.bfloat16,
        DType.bfloat16,
        transpose_b=True,
    ](
        block_tile_shape=Index(128, 256, 64),
        mma_shape=Index(64, 256, 16),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=1,
        partitioned_multicast=True,
    )
    compile_sm90_matmul_ptx[
        M=2048,
        N=4096,
        K=256,
        transpose_b=True,
        config=M2048_N4096_K256_config,
    ]()

    alias M512_N2560_K8192_config_fp8 = MatmulConfig[
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        DType.bfloat16,
        transpose_b=True,
    ](
        block_tile_shape=Index(64, 64, 128),
        mma_shape=Index(64, 64, 32),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=7,
        num_consumer=1,
        partitioned_multicast=False,
    )
    compile_sm90_matmul_ptx[
        M=512,
        N=2560,
        K=8192,
        transpose_b=True,
        config=M512_N2560_K8192_config_fp8,
        schedule = MatmulSchedule.DS_SCHEDULER,
        grid_shape = Index(H100.sm_count, 1),
    ]()

    alias M8192_N14336_K8192_config_fp8 = MatmulConfig[
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        DType.bfloat16,
        transpose_b=True,
    ](
        block_tile_shape=Index(128, 128, 128),
        mma_shape=Index(64, 128, 32),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=6,
        num_consumer=2,
        partitioned_multicast=False,
    )
    compile_sm90_matmul_ptx[
        M=8192,
        N=14336,
        K=8192,
        transpose_b=True,
        config=M8192_N14336_K8192_config_fp8,
        grid_shape = Index(8, H100.sm_count // 8),
        schedule = MatmulSchedule.TILE2D,
    ]()

    alias M2048_N14336_K8192_config_fp8 = MatmulConfig[
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        DType.bfloat16,
        transpose_b=True,
    ](
        block_tile_shape=Index(128, 128, 128),
        mma_shape=Index(64, 256, 32),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
    )
    compile_sm90_matmul_ptx[
        M=2048,
        N=14336,
        K=8192,
        transpose_b=True,
        config=M2048_N14336_K8192_config_fp8,
        grid_shape = Index(8, H100.sm_count // 8),
        schedule = MatmulSchedule.TILE2D,
    ]()

    alias M2048_N4096_K256_config_fp8 = MatmulConfig[
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        DType.bfloat16,
        transpose_b=True,
    ](
        block_tile_shape=Index(128, 128, 128),
        mma_shape=Index(64, 256, 32),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=3,
        num_consumer=2,
        partitioned_multicast=True,
    )
    compile_sm90_matmul_ptx[
        M=2048,
        N=4096,
        K=256,
        transpose_b=True,
        config=M2048_N4096_K256_config_fp8,
    ]()


fn main() raises:
    test_local_memory_access()
