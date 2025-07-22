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
from sys import sizeof, env_get_bool, env_get_int

from buffer.buffer import NDBuffer
from gpu.grid_controls import PDLLevel
from gpu.host import DeviceContext
from gpu.host.info import H100
from linalg.matmul_tile_scheduler import MatmulSchedule, RasterOrder

from utils.index import Index
from logger import Logger
from .utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from .utils_gpu import MatmulConfig
from .matmul_sm90 import warp_specialize_gemm_with_multicasting
from .matmul_sm90_splitk import warp_specialize_gemm_with_multicasting_splitk

# TODO: Move to a general location and use for all dispatch
alias DISPATCH_MISS = 0
alias DISPATCH_HIT = 1


fn matmul_dispatch_sm90[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises -> Int:
    alias is_AB_fp8 = a_type == b_type == DType.float8_e4m3fn
    alias is_AB_bf16 = a_type == b_type == DType.bfloat16
    alias is_AB_fp32 = a_type == b_type == DType.float32

    alias input_type_supported = is_AB_fp8 or is_AB_bf16 or is_AB_fp32

    # fmt: off
    alias has_static_NK = b.shape.all_known[2]() \
                      and a.shape.has_value[1]() \
                      and c.shape.has_value[1]()
    # fmt: on

    alias N = c.shape.get[1]()
    alias N_multiple_of_8 = N % 8 == 0

    var logger = Logger()
    logger.info("------ Dispatching to sm90 ------")

    # Support K multiple of 16B for FP8 due to using TMA.
    # 4B and 8B alignments are supported for BF16/FP32 by using
    # cp.async.ca.
    alias K = a.shape.get[1]()
    alias K_multiple_of_16B = K * sizeof[a_type]() % 16 == 0
    alias K_multiple_of_4B = K * sizeof[a_type]() % 4 == 0
    alias K_align_supported = (K_multiple_of_16B and is_AB_fp8) or (
        K_multiple_of_4B and (is_AB_bf16 or is_AB_fp32)
    )

    # General constraints for H100 matmul
    # fmt: off
    @parameter
    if not (
        input_type_supported and \
        transpose_b and \
        has_static_NK and \
        K_align_supported
    ):
        return DISPATCH_MISS
    # fmt: on

    @parameter
    if is_AB_fp8:
        logger.info("------ Dispatching to sm90 FP8 ------")
        return matmul_dispatch_sm90_fp8[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

    elif is_AB_bf16 or is_AB_fp32:
        logger.info("------ Dispatching to sm90 BF16/FP32 ------")
        return matmul_dispatch_sm90_bf16_fp32[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

    logger.info("SM90 dispatch miss - no matching path")
    return DISPATCH_MISS


# ===----------------------------------------------------------------------=== #
# FP8 (e4m3fn) Dispatch
# ===----------------------------------------------------------------------=== #


fn matmul_dispatch_sm90_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises -> Int:
    alias static_N = c.shape.get[1]()
    alias static_K = a.shape.get[1]()

    var m = c.dim[0]()

    @parameter
    if env_get_bool["AUTOTUNING_MODE", False]():
        alias NUM_PIPELINE_STAGES = env_get_int["TUNE_NUM_PIPELINE_STAGES", 4]()
        alias NUM_CONSUMER = env_get_int["TUNE_NUM_CONSUMER", 1]()
        alias WGMMA_N = env_get_int["TUNE_WGMMA_N", 128]()
        alias CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 1]()
        alias GRID_DIM_X = env_get_int["TUNE_GRID_DIM_X", 1]()
        alias GRID_DIM_Y = H100.sm_count // GRID_DIM_X
        alias BLOCK_TILE_DIM_M = 64 * NUM_CONSUMER

        alias SCHEDULE_TYPE = MatmulSchedule(
            env_get_int["TUNE_SCHEDULE_TYPE", 1]()
        )

        alias H100_FP8_TUNING_CONFIG = MatmulConfig[
            a_type,
            b_type,
            c_type,
            transpose_b,
            mma_shape = Index(64, WGMMA_N, 32),
        ](
            block_tile_shape=Index(BLOCK_TILE_DIM_M, WGMMA_N, 128),
            cluster_shape=Index(CLUSTER_DIM_X, 1, 1),
            num_pipeline_stages=NUM_PIPELINE_STAGES,
            num_consumer=NUM_CONSUMER,
            partitioned_multicast=False,
            pdl_level=pdl_level,
        )
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            config=H100_FP8_TUNING_CONFIG,
            grid_shape = Index(128, 1),
            schedule = MatmulSchedule.DS_SCHEDULER,
        ](
            rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
            rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
            rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
            ctx,
        )
        return DISPATCH_HIT

    # llama-405B-FP8 gemm shapes
    @parameter
    if static_N == 16384 and static_K == 2048:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(64, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                grid_shape = Index(H100.sm_count, 1),
                schedule = MatmulSchedule.DS_SCHEDULER,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(H100.sm_count, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 512:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(H100.sm_count, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 1024:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                grid_shape = Index(H100.sm_count, 1),
                schedule = MatmulSchedule.DS_SCHEDULER,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        else:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                grid_shape = Index(8, H100.sm_count // 8),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif static_N == 2304 and static_K == 16384:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 48, 32),
            ](
                block_tile_shape=Index(64, 48, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(H100.sm_count, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 48, 32),
            ](
                block_tile_shape=Index(64, 48, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(H100.sm_count, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 96, 32),
            ](
                block_tile_shape=Index(64, 96, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(H100.sm_count, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 512:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 144, 32),
            ](
                block_tile_shape=Index(128, 144, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(H100.sm_count, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 1024:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 144, 32),
            ](
                block_tile_shape=Index(128, 144, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                grid_shape = Index(H100.sm_count, 1),
                schedule = MatmulSchedule.DS_SCHEDULER,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 2048:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 144, 32),
            ](
                block_tile_shape=Index(128, 144, 128),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                grid_shape = Index(16, 8),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        else:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # grid_shape = Index(16, 8),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif static_N == 13312 and static_K == 16384:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(64, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # grid_shape = Index(H100.sm_count, 1),
                # schedule = MatmulSchedule.DS_SCHEDULER,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 208, 32),
            ](
                block_tile_shape=Index(128, 208, 128),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(H100.sm_count, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 512:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(H100.sm_count, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 1024:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # grid_shape = Index(H100.sm_count, 1),
                # schedule = MatmulSchedule.DS_SCHEDULER,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        else:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                grid_shape = Index(8, H100.sm_count // 8),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif static_N == 16384 and static_K == 6656:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(64, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 1024:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        else:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                grid_shape = Index(8, H100.sm_count // 8),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    # llama-8B-FP8 gemm shapes
    elif (
        (static_N == 6144 and static_K == 4096)
        or (static_N == 4096 and static_K == 4096)
        or (static_N == 28672 and static_K == 4096)
        or (static_N == 4096 and static_K == 14336)
    ):
        if m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(64, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 1024:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=6,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        else:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128, 32),
            ](
                block_tile_shape=Index(128, 128, 128),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=6,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                grid_shape = Index(8, H100.sm_count // 8),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
    else:
        # for gemms with small n and k we fall back the naive kernel
        alias BN = _find_largest_bn_for_sm90_matmul[a_type, static_N]()
        alias BK = 128

        @parameter
        if BN != -1 and static_K % BK == 0:
            # If the number of blocks is less than the number of SMs, it's probably better to not use any persistent kernel
            if ceildiv(m, 64) * ceildiv(static_N, BN) <= H100.sm_count:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, BN, 32),
                ](
                    block_tile_shape=Index(64, BN, BK),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=6,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                    ctx,
                )
                return DISPATCH_HIT
            elif m <= 1024:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, BN, 32),
                ](
                    block_tile_shape=Index(64, BN, BK),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=6,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                    ctx,
                )
                return DISPATCH_HIT
            else:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, BN, 32),
                ](
                    block_tile_shape=Index(128, BN, BK),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                    ctx,
                )
                return DISPATCH_HIT
    return DISPATCH_MISS


# ===----------------------------------------------------------------------=== #
# BF16 and FP32 Dispatch
# ===----------------------------------------------------------------------=== #


fn matmul_dispatch_sm90_bf16_fp32[
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises -> Int:
    @parameter
    if env_get_bool["AUTOTUNING_MODE", False]():
        # CLUSTER_DIM_X = 2^m for m in range[0-3]
        alias CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 1]()

        # GRID_DIM_X = 2^n for n in range[0-7]
        alias GRID_DIM_X = env_get_int["TUNE_GRID_DIM_X", 1]()
        alias GRID_DIM_Y = H100.sm_count // GRID_DIM_X

        alias H100_TUNING_CONFIG = MatmulConfig[
            a_type,
            b_type,
            c_type,
            transpose_b,
            mma_shape = Index(64, 256, 16),
        ](
            block_tile_shape=Index(128, 256, 64),
            cluster_shape=Index(CLUSTER_DIM_X, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            pdl_level=pdl_level,
        )
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            config=H100_TUNING_CONFIG,
            grid_shape = Index(GRID_DIM_X, GRID_DIM_Y),
            schedule = MatmulSchedule.TILE2D,
        ](
            rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
            rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
            rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
            ctx,
        )
        return DISPATCH_HIT

    alias static_N = c.shape.get[1]()
    alias static_K = a.shape.get[1]()
    alias a_is_bfloat16_or_float32 = a_type in (
        DType.bfloat16,
        DType.float32,
    )
    alias size_factor = 2 if a_type is DType.float32 else 1
    alias mma_k = 16 // size_factor
    alias BK = 64 // size_factor

    var m = c.dim[0]()

    # We have fast gemv for BF16 and FP32, skip H100 matmul here
    # and continue dispatching outside to reach the fast gemv.
    if m == 1:
        return DISPATCH_MISS

    # Internvl 2xH100 shapes
    @parameter
    if a_is_bfloat16_or_float32 and static_N == 2560 and static_K == 5120:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 32 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 32 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=10,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(132, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 64 // size_factor, BK),
                cluster_shape=Index(2, 2, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(132, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 5120 and static_K == 3584:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 40 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 40 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=10,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 40 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 40 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 80 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 80 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=7,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 5120 and static_K == 27648:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 40 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 40 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 80 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 80 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 13824 and static_K == 5120:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 128 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(2, 2, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 3200 and static_K == 6400:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 32 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 32 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 32 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 32 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=9,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 64 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 6400 and static_K == 3200:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 128 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=6,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 3200 and static_K == 4992:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 32 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 32 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 32 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 32 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=9,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 64 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 3200 and static_K == 4608:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 32 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 32 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=9,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 64 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 1664 and static_K == 3200:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 16 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 16 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 32 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 32 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=10,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 1536 and static_K == 3200:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 16 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 16 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 32 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 32 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=10,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 5120 and static_K == 75837:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 40 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 40 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 80 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 80 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    elif a_is_bfloat16_or_float32 and static_N == 12800 and static_K == 2560:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 128 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 128:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 128 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=5,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 256:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    # GTC matmul configs
    @parameter
    if a_is_bfloat16_or_float32 and static_N == 2560 and static_K == 8192:
        if m <= 16:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=True,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                # schedule = MatmulSchedule.DS_SCHEDULER,
                # grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting_splitk[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                splits=2,
                raster_order = RasterOrder.AlongM,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m == 512:
            alias M512_N2560_K8192_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 80 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 80 // size_factor, BK),
                cluster_shape=Index(1, 2, 1),
                num_pipeline_stages=8,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M512_N2560_K8192_config,
                # grid_shape = Index(32, 4),
                # schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

        elif m == 8192:
            alias M8192_N2560_K8192_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M8192_N2560_K8192_config,
                grid_shape = Index(10, H100.sm_count // 10),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

        elif m == 4096:
            alias M4096_N2560_K8192_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M4096_N2560_K8192_config,
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    @parameter
    if a_is_bfloat16_or_float32 and static_N == 8192 and static_K == 2048:
        if m <= 16:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m == 8192:
            alias M8192_N8192_K2048_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M8192_N8192_K2048_config,
                grid_shape = Index(4, H100.sm_count // 4),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

        elif m == 4096:
            alias M4096_N8192_K2048_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M4096_N8192_K2048_config,
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    @parameter
    if a_is_bfloat16_or_float32 and static_N == 14336 and static_K == 8192:
        if m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 112 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 112 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m == 8192:
            alias M8192_N14336_K8192_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M8192_N14336_K8192_config,
                grid_shape = Index(8, H100.sm_count // 8),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

        elif m == 4096:
            alias M4096_N14336_K8192_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M4096_N14336_K8192_config,
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    @parameter
    if a_is_bfloat16_or_float32 and static_N == 8192 and static_K == 7168:
        if m <= 16:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m <= 64:
            alias config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 64 // size_factor, mma_k),
            ](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = MatmulSchedule.DS_SCHEDULER,
                grid_shape = Index(128, 1),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        elif m == 8192:
            alias M8192_N8192_K7168_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M8192_N8192_K7168_config,
                grid_shape = Index(8, H100.sm_count // 8),
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

        elif m == 4096:
            alias M4096_N8192_K7168_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 256 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M4096_N8192_K7168_config,
                schedule = MatmulSchedule.TILE2D,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    @parameter
    if (
        a_is_bfloat16_or_float32
        and static_N == 3840
        and static_K in (15360, 4096)
    ):
        if m <= 512:
            alias M512_N3840_K15360_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, 128 // size_factor, mma_k),
            ](
                block_tile_shape=Index(128, 128 // size_factor, BK),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M512_N3840_K15360_config,
                schedule = MatmulSchedule.NONE,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )

            return DISPATCH_HIT

    alias BN = _find_largest_bn_for_sm90_matmul[
        a_type, static_N
    ]() // size_factor

    # `audio_decoder/test_residual_fsq.py::test_fsq` test fails if
    # we enable float32 here.
    # Fallback path with vectorized output and cp.async.ca load if K
    # is not multiple of 16B.
    @parameter
    if a_type is DType.bfloat16 and BN != -1:
        if m <= 128:
            alias default_bf16_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, BN, mma_k),
            ](
                block_tile_shape=Index(64, BN, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=default_bf16_config,
                schedule = MatmulSchedule.NONE,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT
        else:
            alias default_bf16_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
                mma_shape = Index(64, BN, mma_k),
            ](
                block_tile_shape=Index(128, BN, BK),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=default_bf16_config,
                schedule = MatmulSchedule.NONE,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

    # Fallback path, will use scalar 2B output and lots of OOB check.
    @parameter
    if a_type is DType.bfloat16:
        alias BN = 256
        alias default_bf16_config = MatmulConfig[
            a_type,
            b_type,
            c_type,
            transpose_b,
            mma_shape = Index(64, BN, mma_k),
        ](
            block_tile_shape=Index(128, BN, 64),
            num_pipeline_stages=4,
            num_consumer=2,
            pdl_level=pdl_level,
        )
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            config=default_bf16_config,
            schedule = MatmulSchedule.NONE,
        ](
            rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
            rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
            rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
            ctx,
        )
        return DISPATCH_HIT

    return DISPATCH_MISS


fn _find_largest_bn_for_sm90_matmul[dtype: DType, N: Int]() -> Int:
    @parameter
    if N % 8 != 0:
        return -1

    fn _get_max_bn() capturing -> Int:
        # For float8_e4m3fn maximum BN that will not result in register spilling is 160
        var BN = 160 if dtype == DType.float8_e4m3fn else 256
        while BN >= 8:
            if N % BN == 0:
                return BN
            BN -= 8
        return 8

    return _get_max_bn()
