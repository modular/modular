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
from sys import env_get_bool, env_get_int, size_of

from buffer.buffer import NDBuffer
from gpu.grid_controls import PDLLevel
from gpu.host import DeviceContext
from gpu.host.info import H100
from internal_utils import Table, TuningConfig
from logger import Logger

from utils.index import Index, IndexList

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import MatmulConfig
from ..tile_scheduler import MatmulSchedule, RasterOrder
from .matmul import warp_specialize_gemm_with_multicasting

alias MAX_M = Int.MAX

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
    alias K_multiple_of_16B = K * size_of[a_type]() % 16 == 0
    alias K_multiple_of_4B = K * size_of[a_type]() % 4 == 0
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


@register_passable("trivial")
struct TuningConfigSM90(TuningConfig):
    var M: Int
    var N: Int
    var K: Int

    var mma_shape: IndexList[3]
    var block_tile_shape: IndexList[3]
    var num_pipeline_stages: UInt
    var cluster_shape: IndexList[3]
    var num_consumer: UInt
    var partitioned_multicast: Bool
    var grid_shape: OptionalReg[IndexList[2]]  # = None
    var schedule: MatmulSchedule  # =  MatmulSchedule.NONE
    var splits: OptionalReg[Int]
    var raster_order: OptionalReg[RasterOrder]

    fn __init__(
        out self,
        M: Int,
        N: Int,
        K: Int,
        mma_shape: IndexList[3],
        block_tile_shape: IndexList[3],
        num_pipeline_stages: UInt,
        cluster_shape: IndexList[3],
        num_consumer: UInt,
        partitioned_multicast: Bool,
        grid_shape: OptionalReg[IndexList[2]] = None,
        schedule: MatmulSchedule = MatmulSchedule.NONE,
        splits: OptionalReg[Int] = None,
        raster_order: OptionalReg[RasterOrder] = None,
    ):
        self.M = M
        self.N = N
        self.K = K
        self.mma_shape = mma_shape
        self.block_tile_shape = block_tile_shape
        self.num_pipeline_stages = num_pipeline_stages
        self.cluster_shape = cluster_shape
        self.num_consumer = num_consumer
        self.partitioned_multicast = partitioned_multicast
        self.grid_shape = grid_shape
        self.schedule = schedule
        self.splits = splits
        self.raster_order = raster_order

    fn __str__(self) -> String:
        return String("config: ", "m:", self.M, "/n:", self.N, "/k:", self.K)


# llama-405B-FP8 gemm shapes
alias llama_405b_fp8_list = List(
    ##############################
    # N=16384 and K=2048
    TuningConfigSM90(
        M=64,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(64, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        grid_shape=Index(128, 1),
        schedule=MatmulSchedule.DS_SCHEDULER,
    ),
    TuningConfigSM90(
        M=128,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(H100.sm_count, 1),
        schedule=MatmulSchedule.DS_SCHEDULER,
    ),
    TuningConfigSM90(
        M=256,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(H100.sm_count, 1),
        schedule=MatmulSchedule.DS_SCHEDULER,
    ),
    TuningConfigSM90(
        M=512,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=1024,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(H100.sm_count, 1),
        schedule=MatmulSchedule.DS_SCHEDULER,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(8, H100.sm_count // 8),
        schedule=MatmulSchedule.TILE2D,
    ),
    ##############################
    # N=2304 and K=16384
    TuningConfigSM90(
        M=64,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 48, 32),
        block_tile_shape=Index(64, 48, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=128,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 48, 32),
        block_tile_shape=Index(64, 48, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=256,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 96, 32),
        block_tile_shape=Index(64, 96, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=512,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 144, 32),
        block_tile_shape=Index(128, 144, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=1024,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 144, 32),
        block_tile_shape=Index(128, 144, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=2048,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 144, 32),
        block_tile_shape=Index(128, 144, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(16, 8),
        schedule=MatmulSchedule.TILE2D,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=None,
        schedule=MatmulSchedule.TILE2D,
    ),
    ##############################
    # N=13312 and K=16384
    TuningConfigSM90(
        M=64,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(64, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(128, 1),
    ),
    TuningConfigSM90(
        M=128,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=256,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 208, 32),
        block_tile_shape=Index(128, 208, 128),
        cluster_shape=Index(1, 2, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=512,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=1024,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(8, H100.sm_count // 8),
        schedule=MatmulSchedule.TILE2D,
    ),
    ##############################
    # N=16384 and K=6656
    TuningConfigSM90(
        M=64,
        N=16384,
        K=6656,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(64, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(128, 1),
    ),
    TuningConfigSM90(
        M=1024,
        N=16384,
        K=6656,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=16384,
        K=6656,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(8, H100.sm_count // 8),
        schedule=MatmulSchedule.TILE2D,
    ),
)

alias llama_405b_fp8_table = Table(llama_405b_fp8_list, "llama_405b_fp8")

# llama-8B-FP8 gemm shapes
alias llama_8b_fp8_list = List(
    ##############################
    # ignore N and K for this table.
    TuningConfigSM90(
        M=128,
        N=-1,
        K=-1,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(64, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=True,
        grid_shape=None,
        schedule=MatmulSchedule.NONE,
    ),
    TuningConfigSM90(
        M=1024,
        N=-1,
        K=-1,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=6,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=None,
        schedule=MatmulSchedule.NONE,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=-1,
        K=-1,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=6,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(8, H100.sm_count // 8),
        schedule=MatmulSchedule.TILE2D,
    ),
)

alias llama_8b_fp8_table = Table(llama_8b_fp8_list, "llama_8b_fp8")


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
        ](
            block_tile_shape=Index(BLOCK_TILE_DIM_M, WGMMA_N, 128),
            mma_shape=Index(64, WGMMA_N, 32),
            cluster_shape=Index(CLUSTER_DIM_X, 1, 1),
            num_pipeline_stages=UInt(NUM_PIPELINE_STAGES),
            num_consumer=UInt(NUM_CONSUMER),
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

    @parameter
    @always_inline("nodebug")
    fn _dispatch[entry: TuningConfigSM90]() raises:
        alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=entry.block_tile_shape,
            mma_shape=entry.mma_shape,
            cluster_shape=entry.cluster_shape,
            num_pipeline_stages=entry.num_pipeline_stages,
            num_consumer=entry.num_consumer,
            partitioned_multicast=entry.partitioned_multicast,
            pdl_level=pdl_level,
        )
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            config=config,
            schedule = entry.schedule,
            grid_shape = entry.grid_shape,
        ](
            rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
            rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
            rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
            ctx,
        )

    @parameter
    @always_inline("nodebug")
    fn _search[
        T: Table[TuningConfigSM90], domain: List[Int] = List[Int]()
    ]() raises -> Int:
        @parameter
        @always_inline
        fn get_m(x: TuningConfigSM90) -> Int:
            return x.M

        alias m_values = T.query_values[Int, get_m, domain]()

        @parameter
        for static_m in m_values:

            @parameter
            @always_inline
            fn rule_eq_m(x: TuningConfigSM90) -> Bool:
                return x.M == static_m

            if m <= static_m:
                alias idx_list = T.query_index[rule_eq_m, domain=domain]()

                @parameter
                if idx_list:
                    alias entry = T.configs[idx_list[0]]
                    _dispatch[entry]()
                    return DISPATCH_HIT
                else:
                    # dynamic m is in the range but cannot find any corresponding config in the table.
                    break

        return DISPATCH_MISS

    # llama-405B-FP8 gemm shapes
    @parameter
    if (
        (static_N == 16384 and static_K == 2048)
        or (static_N == 2304 and static_K == 16384)
        or (static_N == 13312 and static_K == 16384)
        or (static_N == 16384 and static_K == 6656)
    ):

        @parameter
        @always_inline
        fn rule_eq_nk(x: TuningConfigSM90) -> Bool:
            return x.K == static_K and x.N == static_N

        # First, filter by static params N and K
        alias nk_idx_list = llama_405b_fp8_table.query_index[rule_eq_nk]()
        # Search the table for matching values of M within domain
        if _search[llama_405b_fp8_table, domain=nk_idx_list]() == DISPATCH_HIT:
            return DISPATCH_HIT

    # llama-8B-FP8 gemm shapes
    elif (
        (static_N == 6144 and static_K == 4096)
        or (static_N == 4096 and static_K == 4096)
        or (static_N == 28672 and static_K == 4096)
        or (static_N == 4096 and static_K == 14336)
    ):
        # Search the table for matching values of M, no domain specified.
        if _search[llama_8b_fp8_table]() == DISPATCH_HIT:
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
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, BN, BK),
                    mma_shape=Index(64, BN, 32),
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
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, BN, BK),
                    mma_shape=Index(64, BN, 32),
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
                ](
                    block_tile_shape=Index(128, BN, BK),
                    mma_shape=Index(64, BN, 32),
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


fn _get_miscellaneous_list[
    size_factor: Int, mma_k: Int, BK: Int
]() -> List[TuningConfigSM90]:
    return List(
        TuningConfigSM90(
            M=128,
            N=1536,
            K=4096,
            mma_shape=IndexList[3](64, 32, mma_k),
            block_tile_shape=Index(64, 32, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(H100.sm_count, 1),
            schedule=MatmulSchedule.DS_SCHEDULER,
        ),
        TuningConfigSM90(
            M=128,
            N=4096,
            K=1536,
            mma_shape=IndexList[3](64, 32, mma_k),
            block_tile_shape=Index(128, 32, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(H100.sm_count, 1),
            schedule=MatmulSchedule.DS_SCHEDULER,
        ),
        TuningConfigSM90(
            M=128,
            N=1536,
            K=4608,
            mma_shape=IndexList[3](64, 32, mma_k),
            block_tile_shape=Index(64, 32, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            schedule=MatmulSchedule.NONE,
        ),
    )


fn _get_internvl_list[
    size_factor: Int, mma_k: Int, BK: Int
]() -> List[TuningConfigSM90]:
    return List(
        ##############################
        # static_N == 2560 and static_K == 5120:
        TuningConfigSM90(
            M=64,
            N=2560,
            K=5120,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(64, 32 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=2560,
            K=5120,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=10,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=256,
            N=2560,
            K=5120,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(128, 64 // size_factor, BK),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 5120 and static_K == 3584:
        TuningConfigSM90(
            M=64,
            N=5120,
            K=3584,
            mma_shape=IndexList[3](64, 40 // size_factor, mma_k),
            block_tile_shape=Index(64, 40 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=10,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=5120,
            K=3584,
            mma_shape=IndexList[3](64, 40 // size_factor, mma_k),
            block_tile_shape=Index(128, 40 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
        ),
        TuningConfigSM90(
            M=256,
            N=5120,
            K=3584,
            mma_shape=IndexList[3](64, 80 // size_factor, mma_k),
            block_tile_shape=Index(128, 80 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=7,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ################################
        TuningConfigSM90(
            M=64,
            N=5120,
            K=27648,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=5120,
            K=27648,
            mma_shape=IndexList[3](64, 40 // size_factor, mma_k),
            block_tile_shape=Index(128, 40 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=256,
            N=5120,
            K=27648,
            mma_shape=IndexList[3](64, 80 // size_factor, mma_k),
            block_tile_shape=Index(128, 80 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##########################
        TuningConfigSM90(
            M=64,
            N=13824,
            K=5120,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=13824,
            K=5120,
            mma_shape=IndexList[3](64, 128 // size_factor, mma_k),
            block_tile_shape=Index(128, 128 // size_factor, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=256,
            N=13824,
            K=5120,
            mma_shape=IndexList[3](64, 256 // size_factor, mma_k),
            block_tile_shape=Index(128, 256 // size_factor, BK),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 3200 and static_K == 6400:
        TuningConfigSM90(
            M=64,
            N=3200,
            K=6400,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(64, 32 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=3200,
            K=6400,
            mma_shape=Index(64, 32 // size_factor, mma_k),
            block_tile_shape=IndexList[3](128, 32 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=9,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=256,
            N=3200,
            K=6400,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(128, 64 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 6400 and static_K == 3200:
        TuningConfigSM90(
            M=64,
            N=6400,
            K=3200,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=6400,
            K=3200,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(128, 64 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=256,
            N=6400,
            K=3200,
            mma_shape=IndexList[3](64, 128 // size_factor, mma_k),
            block_tile_shape=Index(128, 128 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 3200 and static_K == 4992:
        TuningConfigSM90(
            M=64,
            N=3200,
            K=4992,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(64, 32 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=3200,
            K=4992,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(128, 32 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=9,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=256,
            N=3200,
            K=4992,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(128, 64 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 3200 and static_K == 4608:
        TuningConfigSM90(
            M=64,
            N=3200,
            K=4608,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(64, 32 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=3200,
            K=4608,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
        ),
        TuningConfigSM90(
            M=256,
            N=3200,
            K=4608,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(128, 64 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 1664 and static_K == 3200:
        TuningConfigSM90(
            M=64,
            N=1664,
            K=3200,
            mma_shape=IndexList[3](64, 16 // size_factor, mma_k),
            block_tile_shape=Index(64, 16 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=1664,
            K=3200,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(64, 32 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=10,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=256,
            N=1664,
            K=3200,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 1536 and static_K == 3200:
        TuningConfigSM90(
            M=64,
            N=1536,
            K=3200,
            mma_shape=IndexList[3](64, 16 // size_factor, mma_k),
            block_tile_shape=Index(64, 16 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=1536,
            K=3200,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(64, 32 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=10,
            num_consumer=1,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
        ),
        TuningConfigSM90(
            M=256,
            N=1536,
            K=3200,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 5120 and static_K == 75837:
        TuningConfigSM90(
            M=64,
            N=5120,
            K=75837,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=128,
            N=5120,
            K=75837,
            mma_shape=IndexList[3](64, 40 // size_factor, mma_k),
            block_tile_shape=Index(128, 40 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        TuningConfigSM90(
            M=256,
            N=5120,
            K=75837,
            mma_shape=IndexList[3](64, 80 // size_factor, mma_k),
            block_tile_shape=Index(128, 80 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule.NONE,
        ),
        ##############################
        # static_N == 12800 and static_K == 2560:
        TuningConfigSM90(
            M=64,
            N=12800,
            K=2560,
            mma_shape=IndexList[3](64, 128 // size_factor, mma_k),
            block_tile_shape=Index(64, 128 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
        ),
        TuningConfigSM90(
            M=128,
            N=12800,
            K=2560,
            mma_shape=IndexList[3](64, 128 // size_factor, mma_k),
            block_tile_shape=Index(128, 128 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=True,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
        ),
        TuningConfigSM90(
            M=256,
            N=12800,
            K=2560,
            mma_shape=IndexList[3](64, 256 // size_factor, mma_k),
            block_tile_shape=Index(128, 256 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
        ),
    )


# shapes for llama3.3.70b
fn _get_llama_3_3_70b_list[
    size_factor: Int, mma_k: Int, BK: Int
]() -> List[TuningConfigSM90]:
    return List(
        # static_N == 2560 and static_K == 8192
        TuningConfigSM90(
            M=16,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=True,
            schedule=MatmulSchedule.NONE,
            grid_shape=None,
        ),
        TuningConfigSM90(
            M=64,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(64, 64 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            schedule=MatmulSchedule.NONE,
            grid_shape=None,
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=512,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 80 // size_factor, mma_k),
            block_tile_shape=Index(128, 80 // size_factor, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
        ),
        TuningConfigSM90(
            M=4096,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 256 // size_factor, mma_k),
            block_tile_shape=Index(128, 256 // size_factor, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.TILE2D,
        ),
        TuningConfigSM90(
            M=8192,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 256 // size_factor, mma_k),
            block_tile_shape=Index(128, 256 // size_factor, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(10, H100.sm_count // 10),
            schedule=MatmulSchedule.TILE2D,
        ),
    )


# shapes for gemma.3.27b
fn _get_gemma_3_27b_list[
    size_factor: Int, mma_k: Int, BK: Int
]() -> List[TuningConfigSM90]:
    return List(
        TuningConfigSM90(
            M=16,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=16,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(3),
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=16,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 1, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(3),
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=16,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 1, 96, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=32,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=32,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(3),
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=32,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 112, mma_k),
            block_tile_shape=Index(64 * 1, 112, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(3),
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=32,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 1, 96, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=2,
            raster_order=RasterOrder.AlongM,
        ),
        TuningConfigSM90(
            M=224,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 2, 96, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=224,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 2, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=224,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=224,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 2, 96, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=256,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 2, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=256,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=256,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 2, 96, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=256,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 2, 96, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=288,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=288,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 168, mma_k),
            block_tile_shape=Index(64 * 2, 168, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=288,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=512,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=512,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=512,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 168, mma_k),
            block_tile_shape=Index(64 * 2, 168, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=512,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 168, mma_k),
            block_tile_shape=Index(64 * 2, 168, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=1024,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=1024,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=1024,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 168, mma_k),
            block_tile_shape=Index(64 * 2, 168, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=2000,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=2000,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=2000,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=2000,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=2048,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=2048,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=2048,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=2048,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=3072,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=3072,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=3072,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=3072,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=3500,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=3500,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=3500,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=3500,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=4096,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=4096,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=4096,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=4096,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=7000,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=7000,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=7000,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=7000,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=8192,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=8192,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=8192,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=8192,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=48000,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=48000,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
        ),
        TuningConfigSM90(
            M=48000,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
        ),
        TuningConfigSM90(
            M=48000,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
        ),
    )


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
    alias size_factor = 2 if a_type is DType.float32 else 1
    alias mma_k = 16 // size_factor
    alias BK = 64 // size_factor

    @parameter
    if env_get_bool["AUTOTUNING_MODE", False]():
        alias static_N = c.shape.get[1]()
        alias static_K = a.shape.get[1]()

        alias IS_LARGE_GEMM_SHAPE = env_get_bool[
            "TUNE_LARGE_GEMM_SHAPE", True
        ]()
        alias CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 1]()
        alias CLUSTER_DIM_Y = env_get_int["TUNE_CLUSTER_DIM_Y", 1]()
        alias NUM_PIPELINE_STAGES = env_get_int["TUNE_NUM_PIPELINE_STAGES", 4]()
        alias NUM_CONSUMER = env_get_int["TUNE_NUM_CONSUMER", 1]()
        alias WGMMA_N = env_get_int["TUNE_WGMMA_N", 128]()
        alias BLOCK_TILE_DIM_M = 64 * NUM_CONSUMER
        alias PARTITIONED_MULTICAST = env_get_bool[
            "TUNE_PARTITIONED_MULTICAST", False
        ]()
        alias SCHEDULE_TYPE = MatmulSchedule(
            env_get_int["TUNE_SCHEDULE_TYPE", 0]()
        )

        @parameter
        if IS_LARGE_GEMM_SHAPE:
            # GRID_DIM_X = 2^n for n in range[0-7]
            alias GRID_DIM_X = env_get_int["TUNE_GRID_DIM_X", 1]()
            alias GRID_DIM_Y = H100.sm_count // GRID_DIM_X

            alias H100_TUNING_CONFIG = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
            ](
                block_tile_shape=Index(
                    BLOCK_TILE_DIM_M, WGMMA_N // size_factor, BK
                ),
                mma_shape=Index(64, WGMMA_N // size_factor, mma_k),
                cluster_shape=Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, 1),
                num_pipeline_stages=UInt(NUM_PIPELINE_STAGES),
                num_consumer=UInt(NUM_CONSUMER),
                partitioned_multicast=PARTITIONED_MULTICAST,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=H100_TUNING_CONFIG,
                grid_shape = Index(GRID_DIM_X, GRID_DIM_Y),
                schedule=SCHEDULE_TYPE,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
            return DISPATCH_HIT

        else:
            alias IS_SPLITK = env_get_bool["TUNE_IS_SPLITK", False]()

            @parameter
            if not IS_SPLITK:
                alias NUM_PIPELINE_STAGES = env_get_int[
                    "TUNE_NUM_PIPELINE_STAGES", 4
                ]()
                alias GRID_DIM_X = H100.sm_count
                alias GRID_DIM_Y = 1

                constrained[
                    SCHEDULE_TYPE != MatmulSchedule.DS_SCHEDULER
                    or (
                        CLUSTER_DIM_X == 1
                        and CLUSTER_DIM_Y == 1
                        and (not PARTITIONED_MULTICAST)
                    ),
                    "Deepseek scheduler dose not support multicasting",
                ]()

                alias SMALL_SHAPE_H100_BF16_TUNING_CONFIG_NON_SPLITK = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                ](
                    block_tile_shape=Index(BLOCK_TILE_DIM_M, WGMMA_N, BK),
                    cluster_shape=Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, 1),
                    num_pipeline_stages=UInt(NUM_PIPELINE_STAGES),
                    num_consumer=UInt(NUM_CONSUMER),
                    partitioned_multicast=PARTITIONED_MULTICAST,
                    pdl_level=pdl_level,
                    mma_shape=Index(64, WGMMA_N, 16),
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=SMALL_SHAPE_H100_BF16_TUNING_CONFIG_NON_SPLITK,
                    grid_shape = Index(GRID_DIM_X, GRID_DIM_Y),
                    schedule=SCHEDULE_TYPE,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                    ctx,
                )
                return DISPATCH_HIT

            else:
                alias SPLITS = env_get_int["TUNE_SPLITS", 2]()

                alias SMALL_SHAPE_H100_BF16_TUNING_CONFIG_SPLITK = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                ](
                    block_tile_shape=Index(BLOCK_TILE_DIM_M, WGMMA_N, BK),
                    cluster_shape=Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, 1),
                    num_pipeline_stages=UInt(NUM_PIPELINE_STAGES),
                    num_consumer=UInt(NUM_CONSUMER),
                    partitioned_multicast=PARTITIONED_MULTICAST,
                    pdl_level=pdl_level,
                    mma_shape=Index(64, WGMMA_N, 16),
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=SMALL_SHAPE_H100_BF16_TUNING_CONFIG_SPLITK,
                    splits=SPLITS,
                    raster_order = RasterOrder.AlongM,
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

    var m = c.dim[0]()

    # We have fast gemv for BF16 and FP32, skip H100 matmul here
    # and continue dispatching outside to reach the fast gemv.
    if m == 1:
        return DISPATCH_MISS

    # load custom tables
    # Internvl gemm shapes
    alias internvl_list = _get_internvl_list[size_factor, mma_k, BK]()
    alias internvl_table = Table(internvl_list, "internvl")

    alias llama_3_3_70b_list = _get_llama_3_3_70b_list[size_factor, mma_k, BK]()
    alias llama_3_3_70b_table = Table(llama_3_3_70b_list, "llama_3_3_70b")

    alias gemma_3_27b_list = _get_gemma_3_27b_list[size_factor, mma_k, BK]()
    alias gemma_3_27b_table = Table(gemma_3_27b_list, "gemma_3_27b")

    alias miscellaneous_list = _get_miscellaneous_list[size_factor, mma_k, BK]()
    alias miscellaneous_table = Table(miscellaneous_list, "miscellaneous")

    @parameter
    @always_inline("nodebug")
    fn _dispatch[entry: TuningConfigSM90]() raises:
        alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=entry.block_tile_shape,
            mma_shape=entry.mma_shape,
            cluster_shape=entry.cluster_shape,
            num_pipeline_stages=entry.num_pipeline_stages,
            num_consumer=entry.num_consumer,
            partitioned_multicast=entry.partitioned_multicast,
            pdl_level=pdl_level,
        )

        @parameter
        if not entry.splits:
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = entry.schedule,
                grid_shape = entry.grid_shape,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )
        else:
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule = entry.schedule,
                grid_shape = entry.grid_shape,
                splits = entry.splits.value(),
                raster_order = entry.raster_order.value(),
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                ctx,
            )

    @parameter
    @always_inline("nodebug")
    fn _search[
        T: Table[TuningConfigSM90],
        domain: List[Int] = List[Int](),
    ]() raises -> Int:
        @parameter
        @always_inline
        fn get_m(x: TuningConfigSM90) -> Int:
            return x.M

        alias m_values = T.query_values[Int, get_m, domain]()

        @parameter
        for static_m in m_values:

            @parameter
            @always_inline
            fn rule_eq_m(x: TuningConfigSM90) -> Bool:
                return x.M == static_m

            if m <= static_m:
                alias idx_list = T.query_index[rule_eq_m, domain=domain]()

                @parameter
                if idx_list:
                    alias entry = T.configs[idx_list[0]]
                    _dispatch[entry]()
                    return DISPATCH_HIT
                else:
                    # dynamic m is in the range but cannot find any corresponding config in the table.
                    break

        return DISPATCH_MISS

    @parameter
    @always_inline
    fn rule_eq_nk(x: TuningConfigSM90) -> Bool:
        return x.K == static_K and x.N == static_N

    @parameter
    if a_is_bfloat16_or_float32 and (
        (static_N == 1536 and static_K == 4096)
        or (static_N == 4096 and static_K == 1536)
        or (static_N == 1536 and static_K == 4608)
    ):
        alias nk_idx_list = miscellaneous_table.query_index[rule_eq_nk]()
        if _search[miscellaneous_table, domain=nk_idx_list]() == DISPATCH_HIT:
            return DISPATCH_HIT

    # Internvl 2xH100 shapes
    @parameter
    if a_is_bfloat16_or_float32 and (
        (static_N == 2560 and static_K == 5120)
        or (static_N == 5120 and static_K == 3584)
        or (static_N == 5120 and static_K == 27648)
        or (static_N == 13824 and static_K == 5120)
        or (static_N == 3200 and static_K == 6400)
        or (static_N == 6400 and static_K == 3200)
        or (static_N == 3200 and static_K == 4992)
        or (static_N == 3200 and static_K == 4608)
        or (static_N == 1664 and static_K == 3200)
        or (static_N == 1536 and static_K == 3200)
        or (static_N == 5120 and static_K == 75837)
        or (static_N == 12800 and static_K == 2560)
    ):
        # First, filter by static params N and K
        alias nk_idx_list = internvl_table.query_index[rule_eq_nk]()
        # Search the table for matching values of M within domain
        if _search[internvl_table, domain=nk_idx_list]() == DISPATCH_HIT:
            return DISPATCH_HIT

    # matmul configs for llama_3_3_70b
    @parameter
    if a_is_bfloat16_or_float32 and static_N == 2560 and static_K == 8192:
        alias nk_idx_list = llama_3_3_70b_table.query_index[rule_eq_nk]()

        # In this case for m>64 the ranges are not supported.
        # TODO: add ranges for <=256, 512, 1024, 2048
        if m <= 64 or m in [512, 4096, 8192]:
            if (
                _search[llama_3_3_70b_table, domain=nk_idx_list]()
                == DISPATCH_HIT
            ):
                return DISPATCH_HIT

    # matmul configs for gemma_3_27b
    @parameter
    if a_is_bfloat16_or_float32 and (
        (static_N == 5376 and static_K == 21504)
        or (static_N == 5376 and static_K == 4096)
        # or (static_N == 262208 and static_K == 5376)
        or (static_N == 43008 and static_K == 5376)
        or (static_N == 8192 and static_K == 5376)
    ):
        alias nk_idx_list = gemma_3_27b_table.query_index[rule_eq_nk]()

        @parameter
        if nk_idx_list:
            # TODO: add ranges for <=256, 512, 1024, 2048
            if (
                m >= 16
                and _search[gemma_3_27b_table, domain=nk_idx_list]()
                == DISPATCH_HIT
            ):
                return DISPATCH_HIT

    @parameter
    if a_is_bfloat16_or_float32 and static_N == 8192 and static_K == 2048:
        if m <= 16:
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                mma_shape=Index(64, 64 // size_factor, mma_k),
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
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                mma_shape=Index(64, 64 // size_factor, mma_k),
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
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
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
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
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
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 112 // size_factor, BK),
                mma_shape=Index(64, 112 // size_factor, mma_k),
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
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
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
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
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
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                mma_shape=Index(64, 64 // size_factor, mma_k),
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
            alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                mma_shape=Index(64, 64 // size_factor, mma_k),
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
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
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
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
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
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 128 // size_factor, BK),
                mma_shape=Index(64, 128 // size_factor, mma_k),
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
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(64, BN, BK),
                mma_shape=Index(64, BN, mma_k),
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
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, BN, BK),
                mma_shape=Index(64, BN, mma_k),
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
            a_type, b_type, c_type, transpose_b
        ](
            block_tile_shape=Index(128, BN, 64),
            mma_shape=Index(64, BN, mma_k),
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
