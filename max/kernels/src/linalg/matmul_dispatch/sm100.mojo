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
from sys import size_of, env_get_bool, env_get_int
from buffer.buffer import NDBuffer
from gpu.grid_controls import PDLLevel
from gpu.host import DeviceContext
from layout._ndbuffer_stub import (
    from_ndbuffer_row_major,
)
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import B200
from utils.index import Index
from logger import Logger
from ..utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from ..utils_gpu import MatmulConfig, MatmulKernels
from ..matmul_sm100 import (
    blackwell_matmul_tma_umma_warp_specialized,
    matmul_sm100_fallback,
)
from linalg.matmul_tile_scheduler import RasterOrder
from utils.index import Index, IndexList
from sys import align_of, simd_width_of
from algorithm import elementwise
from gpu.host import get_gpu_target
from linalg.matmul_gpu import matmul_kernel_naive
from ..matmul_vendor import matmul as matmul_vendor

alias DISPATCH_MISS = 0
alias DISPATCH_HIT = 1


@always_inline
fn matmul_dispatch_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises -> Int:
    var c_tensor = from_ndbuffer_row_major(c)
    var a_tensor = from_ndbuffer_row_major(a)
    var b_tensor = from_ndbuffer_row_major(b)

    @parameter
    if env_get_bool["AUTOTUNING_MODE", False]():
        alias BM = env_get_int["BM", 128]()
        alias BN = env_get_int["BN", 64]()
        alias BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())
        alias CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 2]()
        alias CLUSTER_DIM_Y = env_get_int["TUNE_CLUSTER_DIM_Y", 1]()
        alias CLUSTER_DIM_Z = env_get_int["TUNE_CLUSTER_DIM_Z", 1]()
        alias CLUSTER_DIM = Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, CLUSTER_DIM_Z)
        alias BLOCK_SWIZZLE_SIZE = env_get_int["TUNE_BLOCK_SWIZZLE_SIZE", 8]()
        alias RASTERIZE_ORDER = env_get_int["TUNE_RASTER_ORDER", 1]()
        alias block_tile_shape = Index(BM, BN, BK)
        alias MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
        alias UmmaShape = Index(BM * 2, BN * 2, MMA_K)

        alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=block_tile_shape,
            mma_shape=UmmaShape,
            cluster_shape=CLUSTER_DIM,
        )

        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=config,
            cta_group=2,
            block_swizzle_size=BLOCK_SWIZZLE_SIZE,
            rasterize_order = RasterOrder(RASTERIZE_ORDER),
        ](c_tensor, a_tensor, b_tensor, ctx)

        return DISPATCH_HIT

    @parameter
    if elementwise_lambda_fn:
        alias umma_shape = Index(64, 128, 16)
        alias BK = 64
        alias block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

        matmul_sm100_fallback[
            transpose_b=transpose_b,
            umma_shape=umma_shape,
            block_tile_shape=block_tile_shape,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c_tensor, a_tensor, b_tensor, ctx)

        return DISPATCH_HIT

    constrained[
        a_type == b_type == c_type,
        "a_type and b_type and c_type must be the same",
    ]()

    alias BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())
    alias block_tile_shape = Index(128, 64, BK)
    alias MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
    alias umma_shape = Index(
        block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
    )

    alias static_N = c.shape.get[1]()  # mxk
    alias static_K = a.shape.get[1]()  # mxn

    var m = c.dim[0]()

    # 8192x8192x2048: BM=128 / BN=128 / CLUSTER=(2,1,1)
    # 4096x8192x2048: BM=128 / BN=128 / CLUSTER=(2,1,1)
    # 512x8192x2048: BM=64 / BN=112 / CLUSTER=(2,1,1)
    @parameter
    if static_N == 8192 and static_K == 2048:
        if m == 512:
            alias block_tile_shape = Index(64, 112, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=2,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT
        elif m == 4096:
            alias block_tile_shape = Index(128, 128, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=2,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            alias block_tile_shape = Index(128, 128, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=1,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT

    # 4096x8192x7168: BM=128 / BN=128 / CLUSTER=(2,1,1)
    # 8192x8192x7168: BM=128 / BN=128 / CLUSTER=(4,1,1)
    # 512x8192x7168: BM=128 / BN=112 / CLUSTER=(2,1,1)
    @parameter
    if static_N == 8192 and static_K == 7168:
        if m == 512:
            alias block_tile_shape = Index(128, 112, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=8,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT
        elif m == 4096:
            alias block_tile_shape = Index(128, 128, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b, config=config, cta_group=2
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            alias block_tile_shape = Index(128, 128, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=8,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT

    # 4096x14336x8192: BM=128 / BN=112 / CLUSTER=(2,1,1)
    # 8192x14336x8192: BM=128 / BN=112 / CLUSTER=(4,1,1)
    # 512x14336x8192: BM=128 / BN=112 / CLUSTER=(4,1,1)
    @parameter
    if static_N == 14336 and static_K == 8192:
        if m == 512:
            alias block_tile_shape = Index(128, 104, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=1,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT
        elif m == 4096:
            alias block_tile_shape = Index(128, 112, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=8,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            alias block_tile_shape = Index(128, 128, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=8,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT

    # 8192x2560x8192: BM=128 / BN=80 / CLUSTER=(2,1,1)
    # 4096x2560x8192: BM=128 / BN=80 / CLUSTER=(2,1,1)
    # 512x2560x8192: BM=64 / BN=80 / CLUSTER=(4,1,1)
    @parameter
    if static_N == 2560 and static_K == 8192:
        if m == 512:
            alias block_tile_shape = Index(64, 80, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(4, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b, config=config, cta_group=2
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT
        elif m == 4096:
            alias block_tile_shape = Index(128, 72, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=8,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            alias block_tile_shape = Index(128, 80, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b, config=config, cta_group=2
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT

    # 4096x4096x4096: BM=128 / BN=128 / CLUSTER=(2,1,1)
    @parameter
    if static_N == 4096 and static_K == 4096:
        if m == 4096:
            alias block_tile_shape = Index(128, 128, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=0,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT

    @parameter
    if static_N == 8192 and static_K == 8192:
        if m == 8192:
            alias block_tile_shape = Index(128, 128, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias cluster_shape = Index(2, 1, 1)
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=cluster_shape,
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                cta_group=2,
                block_swizzle_size=8,
                rasterize_order = RasterOrder.AlongM,
            ](c_tensor, a_tensor, b_tensor, ctx)
            return DISPATCH_HIT

    # Global fallback for any unmatched cases
    alias fallback_block_tile_shape = Index(128, 64, BK)
    alias fallback_umma_shape = Index(
        fallback_block_tile_shape[0] * 2,
        fallback_block_tile_shape[1] * 2,
        MMA_K,
    )
    alias fallback_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=fallback_block_tile_shape,
        mma_shape=fallback_umma_shape,
        cluster_shape=Index(2, 1, 1),
    )
    # (KERN-2026) block_swizzle_size=8 fails for some special cases, so we use 0 here.
    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=fallback_config,
        cta_group=2,
        block_swizzle_size=0,
    ](c_tensor, a_tensor, b_tensor, ctx)

    return DISPATCH_HIT


fn matmul_dispatch_sm100_seperate_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    block_swizzle_size: Int = 0,  # (KERN-2026) block_swizzle_size=8 fails for some special cases, so we use 0 here.
    cta_group: Int = 2,
    num_pipeline_stages: Optional[UInt] = None,
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

    var c_tensor = from_ndbuffer_row_major(c)
    var a_tensor = from_ndbuffer_row_major(a)
    var b_tensor = from_ndbuffer_row_major(b)

    @parameter
    if not elementwise_lambda_fn:
        if not c.data:
            raise "c must be allocated!"

        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=config,
            cta_group=cta_group,
            block_swizzle_size=block_swizzle_size,
            num_pipeline_stages=num_pipeline_stages,
        ](c_tensor, a_tensor, b_tensor, ctx)
        return

    else:
        alias epilogue = elementwise_lambda_fn.value()
        # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
        # arch support 32B load/store to global memory, see KERN-2037.
        alias simd_size = 32 // size_of[
            c.type
        ]() if ctx.default_device_info >= B200 else simd_width_of[
            c.type, target = get_gpu_target()
        ]()

        @parameter
        @__copy_capture(c)
        fn epilogue_wrapper[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = c.load[
                width=simd_width,
                # Load takes alignment in bytes, lambda takes number of elements
                alignment = alignment * size_of[c_type](),
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
                cta_group=cta_group,
                block_swizzle_size=block_swizzle_size,
            ](c_tensor, a_tensor, b_tensor, ctx)

            elementwise[epilogue_wrapper, simd_size, target="gpu"](
                Index(m, n), ctx
            )
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var tmp_device_buffer = ctx.enqueue_create_buffer[c.type](
            c.num_elements()
        )

        # We do not want to mark c as `mut` in the function signature, so we
        # create a new shallow copy of c as a temporary buffer.
        var c_tmp = c
        c_tmp.data = tmp_device_buffer._unsafe_ptr()

        matmul_dispatch_sm100_seperate_epilogue[
            transpose_b=transpose_b,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
            num_pipeline_stages=num_pipeline_stages,
        ](c_tmp, a, b, ctx)

        _ = tmp_device_buffer^


@always_inline
fn matmul_sm100_entrypoint[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises -> Int:
    constrained[
        a_type == b_type and a_type is DType.bfloat16,
        "a_type and b_type must be bfloat16",
    ]()

    var c_tensor = from_ndbuffer_row_major(c)
    var a_tensor = from_ndbuffer_row_major(a)
    var b_tensor = from_ndbuffer_row_major(b)

    @parameter
    if env_get_bool["AUTOTUNING_MODE", False]():
        alias BM = env_get_int["TUNE_BM", 128]()
        alias BN = env_get_int["TUNE_BN", 64]()
        alias BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())
        alias CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 2]()
        alias CLUSTER_DIM_Y = env_get_int["TUNE_CLUSTER_DIM_Y", 1]()
        alias CLUSTER_DIM_Z = env_get_int["TUNE_CLUSTER_DIM_Z", 1]()
        alias CLUSTER_DIM = Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, CLUSTER_DIM_Z)
        alias BLOCK_SWIZZLE_SIZE = env_get_int["TUNE_BLOCK_SWIZZLE_SIZE", 0]()
        alias RASTERIZE_ORDER = env_get_int["TUNE_RASTER_ORDER", 1]()
        alias PIPELINE_STAGE = env_get_int["TUNE_PIPELINE_STAGE", 4]()
        alias block_tile_shape = Index(BM, BN, BK)
        alias MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
        alias UmmaShape = Index(BM * 2, BN * 2, MMA_K)

        alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=block_tile_shape,
            mma_shape=UmmaShape,
            cluster_shape=CLUSTER_DIM,
        )

        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=config,
            cta_group=2,
            block_swizzle_size=BLOCK_SWIZZLE_SIZE,
            rasterize_order = RasterOrder(RASTERIZE_ORDER),
            num_pipeline_stages = UInt(PIPELINE_STAGE),
        ](c_tensor, a_tensor, b_tensor, ctx)

        return DISPATCH_HIT

    var m = c.dim[0]()
    alias static_N = c.shape.get[1]()
    alias static_K = a.shape.get[1]()

    var logger = Logger()
    logger.info("------ Dispatching to SM100 (B200+) ------")
    logger.info("Problem Shape: MNK=[", m, ", ", static_N, ", ", static_K, "]")

    # default matmul config for sm100
    alias MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
    alias BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    # SM100 kernel requirements:
    # 1. `N * size_of(c_type) % 16B == 0` for output buffer (TMA requirement)
    # 2. TODO: (KERN-2028) `K % BK == 0` SM100 kernel accuracy check fails when `K` is not a multiple of `BK`
    # 3. `c_type == DType.bfloat16` SM100 kernel only supports bfloat16 for output buffer
    @parameter
    if (
        c_type == DType.bfloat16
        and static_N * size_of[c_type]() % 16 == 0
        and static_K % BK == 0
    ):
        logger.info("Executing SM100 GEMM")

        # gemma-3-27b-it-prefill and decode(TP=1)
        @parameter
        if static_N == 8192 and static_K == 5376:
            if m == 48000:
                alias block_tile_shape = Index(128, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=8,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 2000 or m == 3000 or m == 3500:
                alias block_tile_shape = Index(128, 112, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 4096 or m == 7000 or m == 8192:
                alias block_tile_shape = Index(128, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=8,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 512:
                alias block_tile_shape = Index(128, 112, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=0,
                ](c, a, b, ctx)
                return DISPATCH_HIT

        elif static_N == 5376 and static_K == 4096:
            if m == 2000:
                alias block_tile_shape = Index(128, 104, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)
                return DISPATCH_HIT

            elif m == 3000 or m == 48000:
                alias block_tile_shape = Index(128, 112, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)
                return DISPATCH_HIT

            elif m == 3500 or m == 7000 or m == 8192:
                alias block_tile_shape = Index(128, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=8,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 4096:
                alias block_tile_shape = Index(128, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=2,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 512:
                alias block_tile_shape = Index(128, 96, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=1,
                ](c, a, b, ctx)
                return DISPATCH_HIT

        elif static_N == 43008 and static_K == 5376:
            if m == 2000:
                alias block_tile_shape = Index(128, 112, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
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
                alias block_tile_shape = Index(128, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=1,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 8192:
                alias block_tile_shape = Index(128, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=8,
                ](c, a, b, ctx)
                return DISPATCH_HIT

        elif static_N == 5376 and static_K == 21504:
            if m == 2000:
                alias block_tile_shape = Index(128, 104, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 3000:
                alias block_tile_shape = Index(128, 112, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    num_pipeline_stages = UInt(7),
                    block_swizzle_size=1,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 3500 or m == 4096 or m == 7000 or m == 48000:
                alias block_tile_shape = Index(128, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=4,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 8192:
                alias block_tile_shape = Index(128, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=8,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 512:
                alias block_tile_shape = Index(128, 96, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(4, 2, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                    block_swizzle_size=8,
                ](c, a, b, ctx)
                return DISPATCH_HIT

        elif static_N == 262208 and static_K == 5376:
            if m == 1:
                alias block_tile_shape = Index(64, 128, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)
                return DISPATCH_HIT

        # gemma-3-27b-it-prefill (TP=2)
        @parameter
        if static_N == 4096 and static_K == 5376:
            if m == 3000:
                alias block_tile_shape = Index(128, 88, BK)
                alias umma_shape = Index(
                    block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
                )
                alias cluster_shape = Index(2, 1, 1)
                alias config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=block_tile_shape,
                    mma_shape=umma_shape,
                    cluster_shape=cluster_shape,
                )
                matmul_dispatch_sm100_seperate_epilogue[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)
                return DISPATCH_HIT

        # Our GEMV matmul dosen't support float8 yet.
        # m=1 and N=1 is not supported on SM100 due to the output buffer TMA requirements. (`N * size_of(c_type) % 16 == 0`).
        @parameter
        if a_type is DType.bfloat16:
            if m == 1:
                return DISPATCH_MISS

        # fallback to vendor matmul for untuned shapes
        matmul_vendor[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, b, ctx)
        return DISPATCH_HIT

    else:
        # Our GEMV matmul dosen't support float8 yet.
        # m=1 and N=1 is not supported on SM100 due to the output buffer TMA requirements. (`N * size_of(c_type) % 16 == 0`).
        @parameter
        if a_type is DType.bfloat16:
            if m == 1 or static_N == 1:
                return DISPATCH_MISS

        logger.info("Executing vendor BLAS (cuBLAS)")
        matmul_vendor[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, b, ctx)
        return DISPATCH_HIT
