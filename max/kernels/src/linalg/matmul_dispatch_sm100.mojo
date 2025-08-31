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
from utils.index import Index
from logger import Logger
from .utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from .utils_gpu import MatmulConfig, MatmulKernels
from .matmul_sm100 import (
    blackwell_matmul_tma_umma_warp_specialized,
    matmul_sm100_fallback,
)

from utils.index import Index, IndexList

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
    @parameter
    if env_get_bool["AUTOTUNING_MODE", False]():
        alias BM = env_get_int["BM", 128]()
        alias BN = env_get_int["BN", 64]()
        alias BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())
        alias CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 2]()
        alias CLUSTER_DIM_Y = env_get_int["TUNE_CLUSTER_DIM_Y", 1]()
        alias CLUSTER_DIM_Z = env_get_int["TUNE_CLUSTER_DIM_Z", 1]()
        alias CLUSTER_DIM = Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, CLUSTER_DIM_Z)
        alias block_tile_shape = Index(BM, BN, BK)
        alias MMA_K = 32 if a_type == DType.float8_e4m3fn else 16
        alias UmmaShape = Index(BM * 2, BN * 2, MMA_K)

        alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=block_tile_shape,
            mma_shape=UmmaShape,
            cluster_shape=CLUSTER_DIM,
        )

        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b, config=config, cta_group=2
        ](c, a, b, ctx)

        return DISPATCH_HIT

    @parameter
    if elementwise_lambda_fn:
        var a_tensor = from_ndbuffer_row_major(a)
        var b_tensor = from_ndbuffer_row_major(b)
        var c_tensor = from_ndbuffer_row_major(c)

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
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
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
            ](c, a, b, ctx)
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
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
            return DISPATCH_HIT
        else:
            # Fallback
            alias block_tile_shape = Index(128, 64, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=Index(2, 1, 1),
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
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
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
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
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            alias block_tile_shape = Index(128, 128, BK)
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
            ](c, a, b, ctx)
            return DISPATCH_HIT
        else:
            # Fallback
            alias block_tile_shape = Index(128, 64, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=Index(2, 1, 1),
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
            return DISPATCH_HIT

    # 4096x14336x8192: BM=128 / BN=112 / CLUSTER=(2,1,1)
    # 8192x14336x8192: BM=128 / BN=112 / CLUSTER=(4,1,1)
    # 512x14336x8192: BM=128 / BN=112 / CLUSTER=(4,1,1)
    @parameter
    if static_N == 14336 and static_K == 8192:
        if m == 512:
            alias block_tile_shape = Index(128, 112, BK)
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
            ](c, a, b, ctx)
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
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            alias block_tile_shape = Index(128, 112, BK)
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
            ](c, a, b, ctx)
            return DISPATCH_HIT
        else:
            # Fallback
            alias block_tile_shape = Index(128, 64, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=Index(2, 1, 1),
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
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
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 4096:
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
            ](c, a, b, ctx)
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
            ](c, a, b, ctx)
            return DISPATCH_HIT
        else:
            # Fallback
            alias block_tile_shape = Index(128, 64, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=Index(2, 1, 1),
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
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
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
            return DISPATCH_HIT
        else:
            # Fallback
            alias block_tile_shape = Index(128, 64, BK)
            alias umma_shape = Index(
                block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
            )
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=block_tile_shape,
                mma_shape=umma_shape,
                cluster_shape=Index(2, 1, 1),
            )
            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b, config=config, cta_group=2
            ](c, a, b, ctx)
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
    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b, config=fallback_config, cta_group=2
    ](c, a, b, ctx)

    return DISPATCH_HIT
