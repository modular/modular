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
from math import ceildiv
from sys import (
    align_of,
    env_get_bool,
    env_get_int,
    simd_width_of,
    size_of,
    has_nvidia_gpu_accelerator,
)

from algorithm import elementwise
from buffer.buffer import NDBuffer
from gpu.primitives.grid_controls import PDLLevel
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
    lt_to_tt,
)
from logger import Logger

from utils.index import Index, IndexList

from .....utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from .....utils_gpu import MatmulKernels, _vendor_blas_fallback_disabled
from ..structured_kernels.config import (
    MatmulConfig,
    build_configs,
    choose_config,
)
from ... import matmul_kernel_naive, gemv_gpu, multistage_gemm
from ....vendor.matmul import matmul as matmul_vendor
from ...tile_scheduler import RasterOrder
from .matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
    matmul_sm100_fallback,
)
from internal_utils import Table
from .tuning_configs import (
    _get_tuning_list_sm100_fp8,
    TuningConfigSM100,
    _get_tuning_list_sm100_bf16,
)

comptime DISPATCH_MISS = 0
comptime DISPATCH_HIT = 1

comptime logger = Logger()


@always_inline
fn matmul_dispatch_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_lambda_wrapper: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    constrained[a_type == b_type, "a_type and b_type must be the same"]()

    var m = c.dim[0]()
    comptime static_N = c.shape.get[1]()
    comptime static_K = a.shape.get[1]()

    @parameter
    if env_get_bool["AUTOTUNING_MODE", False]():
        var c_tensor = lt_to_tt(from_ndbuffer_row_major(c))
        var a_tensor = lt_to_tt(from_ndbuffer_row_major(a))
        var b_tensor = lt_to_tt(from_ndbuffer_row_major(b))

        comptime BM = env_get_int["TUNE_BM", 128]()
        comptime BN = env_get_int["TUNE_BN", 64]()
        comptime BK = (
            TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]()
        )
        comptime MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
        comptime CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 2]()
        comptime CLUSTER_DIM_Y = env_get_int["TUNE_CLUSTER_DIM_Y", 1]()
        comptime CLUSTER_DIM_Z = env_get_int["TUNE_CLUSTER_DIM_Z", 1]()
        comptime CLUSTER_DIM = Index(
            CLUSTER_DIM_X, CLUSTER_DIM_Y, CLUSTER_DIM_Z
        )
        comptime BLOCK_SWIZZLE_SIZE = env_get_int[
            "TUNE_BLOCK_SWIZZLE_SIZE", 0
        ]()
        comptime RASTERIZE_ORDER = env_get_int["TUNE_RASTER_ORDER", 1]()
        comptime CTA_GROUP = env_get_int["TUNE_CTA_GROUP", 2]()
        comptime K_GROUP_SIZE = env_get_int["TUNE_K_GROUP_SIZE", 1]()
        comptime AB_SWAPPED = env_get_bool["TUNE_AB_SWAPPED", False]()

        comptime umma_shape = Index(BM * CTA_GROUP, BN * CTA_GROUP, MMA_K)

        comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            mma_shape=umma_shape,
            cluster_shape=CLUSTER_DIM,
            block_swizzle_size=BLOCK_SWIZZLE_SIZE,
            raster_order=RasterOrder(Int32(RASTERIZE_ORDER)),
            cta_group=CTA_GROUP,
            AB_swapped=AB_SWAPPED,
            k_group_size=K_GROUP_SIZE,
        )

        return blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=config,
            register_based_epilogue=register_based_epilogue,
        ](c_tensor, a_tensor, b_tensor, ctx)

    @parameter
    if _vendor_blas_fallback_disabled():

        @parameter
        if (
            c_type == DType.bfloat16
            and static_N * size_of[c_type]() % 16 == 0
            and static_K * size_of[a_type]() % 16 == 0
            and transpose_b
        ):
            var status = heuristic_and_outliers_dispatch[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            if status:
                return
            else:
                raise Error("Heuristic and outliers dispatch failed.")
        else:
            constrained[
                False,
                "Unsupported shape for benchmarking mode.",
            ]()

    var epilogue_type = String("None")

    @parameter
    if elementwise_compute_lambda_fn:
        epilogue_type = String("Compute Epilogue")
    elif elementwise_lambda_fn:
        epilogue_type = String("Normal Epilogue")

    logger.info("------ Dispatching to SM100 (B200+) ------")
    logger.info(
        "Input Data Types: ",
        a_type,
        ", ",
        b_type,
        " Output Data Type: ",
        c_type,
        " Problem Shape: MNK=[",
        m,
        ", ",
        static_N,
        ", ",
        static_K,
        "]",
        " Epilogue Type: ",
        epilogue_type,
    )

    # default matmul config for sm100
    comptime MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    # 1. for m==1 our gemv matmul is faster than cublas for skinny bfloat16 matmuls
    # 2. Our GEMV matmul dosen't support float8 yet.
    # 3. static_N=1 is not supported on SM100 due to the output buffer TMA requirements. (`N * size_of(c_type) % 16 == 0`).
    @parameter
    if a_type == DType.bfloat16:
        if static_N == 1 or m == 1:
            logger.info("------ Executing GEMV Matmul------")
            gemv_gpu[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return

    # SM100 kernel requirements:
    # 1. `N * size_of(c_type) % 16B == 0` for output buffer (TMA requirement)
    # 2. `c_type == DType.bfloat16` SM100 kernel only supports bfloat16 for output buffer
    @parameter
    if (
        c_type == DType.bfloat16
        and static_N * size_of[c_type]() % 16 == 0
        and static_K * size_of[a_type]() % 16 == 0
        and transpose_b
    ):
        var status = DISPATCH_MISS

        @parameter
        if a_type == b_type == DType.bfloat16:
            status = matmul_dispatch_sm100_bf16[
                c_type=c_type,
                a_type=a_type,
                b_type=b_type,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

        elif a_type == b_type == DType.float8_e4m3fn:
            status = matmul_dispatch_sm100_fp8[
                c_type=c_type,
                a_type=a_type,
                b_type=b_type,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

        if status:
            logger.info("------ Executing MOJO SM100 Matmul------")
            return

    # fallback to vendor matmul for untuned shapes
    # We assume that this will always be a hit as in the worst case it will be a navie matmul.
    return _vendor_blas_matmul_sm100[
        c_type,
        a_type,
        b_type,
        transpose_b,
        elementwise_lambda_wrapper=elementwise_lambda_wrapper,
        pdl_level=pdl_level,
    ](c, a, b, ctx)


# NOTE:
# 1. SM100 matmul supports compute lambdas so we should just use normal and compute lambdas.
fn matmul_dispatch_sm100_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises -> Int:
    comptime static_N = c.shape.get[1]()
    comptime static_K = a.shape.get[1]()

    comptime MMA_K = 32
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())
    var m = c.dim[0]()

    if m <= 128:
        return heuristic_and_outliers_dispatch[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

    @parameter
    @always_inline("nodebug")
    fn _dispatch[entry: TuningConfigSM100]() raises:
        comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            mma_shape=entry.mma_shape,
            cluster_shape=entry.cluster_shape,
            block_swizzle_size=Int(entry.block_swizzle_size),
        )

        return _matmul_dispatch_sm100[
            transpose_b=transpose_b,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

    @parameter
    @always_inline("nodebug")
    fn _search[
        T: Table[TuningConfigSM100],
        domain: List[Int] = List[Int](),
    ]() raises -> Int:
        @parameter
        @always_inline
        fn get_m(x: TuningConfigSM100) -> Int:
            return x.M

        comptime m_values = T.query_values[Int, get_m, domain]()

        @parameter
        for static_m in m_values:

            @parameter
            @always_inline
            fn rule_eq_m(x: TuningConfigSM100) -> Bool:
                return x.M == static_m

            if m <= static_m:
                comptime idx_list = T.query_index[rule_eq_m, domain=domain]()

                @parameter
                if idx_list:
                    comptime entry = T.configs[idx_list[0]]
                    _dispatch[entry]()
                    return DISPATCH_HIT
                else:
                    # dynamic m is in the range but cannot find any corresponding config in the table.
                    break

        return DISPATCH_MISS

    comptime tuning_list = _get_tuning_list_sm100_fp8[mma_k=MMA_K, bk=BK]()
    comptime tuning_table = Table(tuning_list, "tuning_table_sm100_fp8")

    @parameter
    @always_inline
    fn rule_eq_nk(x: TuningConfigSM100) -> Bool:
        return x.K == static_K and x.N == static_N

    comptime nk_idx_list = tuning_table.query_index[rule_eq_nk]()

    # TODO: re-enable the following tuning dispatch.
    # make sure the domain (nk_idx_list) is not empty!
    if m > 128:

        @parameter
        if nk_idx_list:
            if _search[tuning_table, domain=nk_idx_list]() == DISPATCH_HIT:
                return DISPATCH_HIT

    # gemma-3-27b-it-prefill (TP1)
    @parameter
    if static_N == 5376 and static_K == 21504:
        if m == 224 or m == 256:
            comptime block_tile_shape = Index(128, 64, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 288:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 512:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=4,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 1024:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(8, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2000 or m == 2048:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(8, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3000 or m == 3500:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 4096:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=4,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    elif static_N == 43008 and static_K == 5376:
        if m == 224:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 256:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 288 or m == 512:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 1024 or m == 2048:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2000 or m == 3500 or m == 4096 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3000 or m == 7000:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    elif static_N == 8192 and static_K == 5376:
        if m == 224 or m == 256:
            comptime block_tile_shape = Index(128, 64, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 288 or m == 1024:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 512:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2000:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2048 or m == 3000:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3500:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 4096 or m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    elif static_N == 5376 and static_K == 4096:
        if m == 224:
            comptime block_tile_shape = Index(128, 64, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 256:
            comptime block_tile_shape = Index(128, 64, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 288:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 512:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 1024:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2000 or m == 2048:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(8, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3000:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3500 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 4096:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 7000:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    elif static_N == 21504 and static_K == 5376:
        if m == 224 or m == 256:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 288:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=4,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 512:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 1024 or m == 4096:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2000 or m == 3500:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2048 or m == 3000 or m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    # gemma-3-27b-it-prefill (TP2)
    @parameter
    if static_N == 4096 and static_K == 5376:
        if m == 2000:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3000 or m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3500:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    elif static_N == 5376 and static_K == 2048:
        if m == 2000:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3000:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=4,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3500:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=4,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    elif static_N == 5376 and static_K == 10752:
        if m == 2000:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(8, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3000:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=0,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3500:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    elif static_N == 10752 and static_K == 5376:
        if m == 2000 or m == 3000 or m == 3500 or m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    # TODO (KERN-2084): Enable default matmul for large shapes to increase accuracy
    # # fallback to default matmul for large shapes
    # alias block_tile_shape = Index(128, 128, BK)
    # alias umma_shape = Index(
    #     block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
    # )
    # alias cluster_shape = Index(2, 1, 1)
    # alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
    #     block_tile_shape=block_tile_shape,
    #     mma_shape=umma_shape,
    #     cluster_shape=cluster_shape,
    # )
    # _matmul_dispatch_sm100[
    #     transpose_b=transpose_b,
    #     config=config,
    #     elementwise_lambda_fn=elementwise_lambda_fn,
    #     elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
    #     pdl_level=pdl_level,
    #     block_swizzle_size=0,
    # ](c, a, b, ctx)
    # return DISPATCH_HIT
    return DISPATCH_MISS


fn heuristic_and_outliers_dispatch[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises -> Int:
    var m = c.dim[0]()
    comptime static_N = c.shape.get[1]()
    comptime static_K = a.shape.get[1]()

    constrained[
        a_type == b_type and a_type in (DType.bfloat16, DType.float8_e4m3fn),
        "Only support bfloat16 and float8_e4m3fn input types",
    ]()

    comptime MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    comptime outliers = Table(
        _get_tuning_list_sm100_bf16(), "bf16_heuristic_outliers"
    ) if a_type == DType.bfloat16 else Table(
        _get_tuning_list_sm100_fp8[MMA_K, BK](), "fp8_heuristic_outliers"
    )

    @parameter
    @always_inline
    fn rule(x: TuningConfigSM100) -> Bool:
        return x.K == static_K and x.N == static_N

    comptime outlier_configs = outliers.find[rule]()

    @parameter
    for tuning_config in outlier_configs:
        if m >= tuning_config.M and m < tuning_config.M_end:
            comptime matmul_config = MatmulConfig[
                a_type, b_type, c_type, transpose_b
            ](
                mma_shape=tuning_config.mma_shape,
                cta_group=tuning_config.cta_group,
                cluster_shape=tuning_config.cluster_shape,
                block_swizzle_size=Int(tuning_config.block_swizzle_size),
                raster_order=tuning_config.rasterize_order,
                AB_swapped=tuning_config.swapAB,
                num_accum_pipeline_stages=Int(
                    tuning_config.num_accum_pipeline_stages
                ),
                num_clc_pipeline_stages=Int(
                    tuning_config.num_clc_pipeline_stages
                ),
                k_group_size=Int(tuning_config.k_group_size),
                num_split_k=tuning_config.num_split_k,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=matmul_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

            return DISPATCH_HIT

    comptime configs = build_configs[
        a_type, b_type, c_type, static_N, static_K, transpose_b
    ]()
    var config_runtime = choose_config[a_type, b_type, c_type, transpose_b](
        m, static_N, static_K
    )

    @parameter
    for config in configs:
        if config_runtime == config:
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    return DISPATCH_MISS


# NOTE:
# 1. SM100 matmul supports compute lambdas so we should just use normal and compute lambdas.
fn matmul_dispatch_sm100_bf16[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises -> Int:
    var m = c.dim[0]()
    comptime static_N = c.shape.get[1]()
    comptime static_K = a.shape.get[1]()

    comptime MMA_K = 16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    comptime llama3_8b_NK = [
        # TP1
        Index(6144, 4096),
        Index(4096, 4096),
        Index(28672, 4096),
        Index(4096, 14336),
        # TP2
        Index(3072, 4096),
        Index(4096, 2048),
        Index(14336, 4096),
        Index(4096, 7168),
    ]

    comptime miscellaneous_NK = [
        Index(1536, 4096),
        Index(4096, 1536),
    ]

    @parameter
    if Index(static_N, static_K) in miscellaneous_NK:
        return heuristic_and_outliers_dispatch[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

    @parameter
    if Index(static_N, static_K) in llama3_8b_NK:
        if m <= 128:
            return heuristic_and_outliers_dispatch[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

    # gemma-3-27b-it-prefill (TP1)
    @parameter
    if static_N == 8192 and static_K == 5376:
        if m == 48000:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2000 or m == 3000 or m == 3500:
            comptime block_tile_shape = Index(128, 112, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 4096 or m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 512:
            comptime block_tile_shape = Index(128, 112, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    @parameter
    if static_N == 5376 and static_K == 4096:
        if m == 2000:
            comptime block_tile_shape = Index(128, 104, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

        elif m == 3000 or m == 48000:
            comptime block_tile_shape = Index(128, 112, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

        elif m == 3500 or m == 7000 or m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 4096:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 512:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    @parameter
    if static_N == 43008 and static_K == 5376:
        if m == 2000:
            comptime block_tile_shape = Index(128, 112, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

        elif (
            m == 512
            or m == 3000
            or m == 3500
            or m == 4096
            or m == 7000
            or m == 48000
        ):
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    @parameter
    if static_N == 5376 and static_K == 21504:
        if m == 2000:
            comptime block_tile_shape = Index(128, 104, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3000:
            comptime block_tile_shape = Index(128, 112, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=1,
                num_pipeline_stages=7,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 3500 or m == 4096 or m == 7000 or m == 48000:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=4,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            comptime block_tile_shape = Index(128, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 512:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(4, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=8,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    @parameter
    if static_N == 262208 and static_K == 5376:
        if m == 1:
            comptime block_tile_shape = Index(64, 128, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    # gemma-3-27b-it-prefill (TP=2)
    @parameter
    if static_N == 4096 and static_K == 5376:
        if m == 3000:
            comptime block_tile_shape = Index(128, 88, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    # llama-3-1-8b-it-prefill (TP=2)
    @parameter
    if static_N == 3072 and static_K == 4096:
        if m == 4096:
            comptime block_tile_shape = Index(128, 120, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 2048:
            comptime block_tile_shape = Index(128, 96, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 2, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 512:
            comptime block_tile_shape = Index(128, 48, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=2,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    @parameter
    if static_N == 4096 and static_K == 2048:
        if m == 512:
            comptime block_tile_shape = Index(128, 56, BK)
            comptime umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            comptime cluster_shape = Index(2, 1, 1)
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
                block_swizzle_size=4,
            )
            _matmul_dispatch_sm100[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    return DISPATCH_MISS


# NOTE: vendor blas, naive matmul, and multistage gemm dosen't support compute lambdas so we need to wrap them in a lambda function.
# if there is no compute lambda, then this wrapper will be a simple element wise lambda.
@always_inline
fn _vendor_blas_matmul_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool = False,
    elementwise_lambda_wrapper: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    comptime K = a.shape.get[1]()
    comptime a_shape = a.shape
    comptime b_shape = b.shape
    comptime c_shape = c.shape
    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    try:
        logger.info("Executing vendor BLAS (cuBLAS/cublasLt)")
        return matmul_vendor[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
        ](c, a, b, ctx)

    except:
        # fallback to multistage/naive gemms if the cublas failed. This is a workaround for now for KERN-1812
        logger.warning("Vendor BLAS failed")

        @parameter
        if not a_type.is_float8() and K * size_of[a_type]() >= 8 * 16:
            logger.info("Executing Multistage matmul kernel")
            comptime kernels = MatmulKernels[
                a_type, b_type, c_type, transpose_b
            ]()
            comptime config = kernels.ampere_256x64_4
            multistage_gemm[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                config,
                ctx,
            )
        else:
            comptime BLOCK_DIM = 16
            logger.info("Executing Naive matmul kernel")

            var c_layout_tensor = from_ndbuffer_row_major(c)
            var a_layout_tensor = from_ndbuffer_row_major(a)
            var b_layout_tensor = from_ndbuffer_row_major(b)

            comptime kernel = matmul_kernel_naive[
                c_type,
                a_type,
                b_type,
                c_layout_tensor.layout,
                a_layout_tensor.layout,
                b_layout_tensor.layout,
                BLOCK_DIM,
                transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
            ]

            ctx.enqueue_function[kernel, kernel](
                c_layout_tensor,
                a_layout_tensor,
                b_layout_tensor,
                m,
                n,
                k,
                grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )
        return


fn _matmul_dispatch_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    """Our sm100 matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations if there is any.
    """

    var c_tensor = lt_to_tt(from_ndbuffer_row_major(c))
    var a_tensor = lt_to_tt(from_ndbuffer_row_major(a))
    var b_tensor = lt_to_tt(from_ndbuffer_row_major(b))

    constrained[
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None,
        "Either the epilogue lambda or the compute lambda can be used",
    ]()

    @parameter
    if not elementwise_lambda_fn:
        if not c.data:
            raise "c must be allocated!"

        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c_tensor, a_tensor, b_tensor, ctx)
        return

    else:
        comptime epilogue = elementwise_lambda_fn.value()
        # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
        # arch support 32B load/store to global memory, see KERN-2037.
        comptime use_32b_simd = (
            has_nvidia_gpu_accelerator()
            and ctx.default_device_info.compute >= B200.compute
        )
        comptime simd_size = 32 // size_of[c.type]() if use_32b_simd else (
            simd_width_of[c.type, target = get_gpu_target()]()
        )

        @parameter
        @__copy_capture(c)
        fn epilogue_wrapper[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = c.load[
                width=simd_width,
                # Load takes alignment in bytes, lambda takes number of elements
                alignment = alignment * size_of[c.type](),
            ](c_coord)
            epilogue[c.type, simd_width, alignment=alignment](c_coord, c_val)

        # If c is already allocated, we can just use the sm100 matmul and
        # apply the epilogue.
        if c.data:
            var m = c.dim[0]()
            var n = c.dim[1]()

            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c_tensor, a_tensor, b_tensor, ctx)

            elementwise[epilogue_wrapper, simd_size, target="gpu"](
                Index(m, n), ctx
            )
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var tmp_device_buffer = ctx.enqueue_create_buffer[c.type](
            c.num_elements()
        )

        # Construct a new buffer with external origin pointing to the temporary storage.
        var c_tmp = NDBuffer[c.type, 2, MutExternalOrigin](
            rebind[UnsafePointer[Scalar[c.type], MutExternalOrigin]](
                tmp_device_buffer.unsafe_ptr()
            ),
            IndexList[2](c.dim[0](), c.dim[1]()),
        )

        _matmul_dispatch_sm100[
            transpose_b=transpose_b,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c_tmp, a, b, ctx)

        _ = tmp_device_buffer^
