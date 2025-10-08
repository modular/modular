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
from sys import simd_width_of, size_of

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from gpu.cluster import cluster_sync, cluster_sync_relaxed, elect_one_sync
from gpu.globals import WARPGROUP_SIZE
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import H100
from gpu.id import block_dim, block_id_in_cluster, grid_dim, thread_idx
from gpu.id import warp_id as get_warp_id
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout_tensor import LayoutTensorIter
from layout.runtime_layout import RuntimeLayout
from layout.tensor_core_async import TensorCoreAsync, tile_layout_k_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
)
from logger import Logger
from memory import bitcast, stack_allocation
from stdlib.bit import log2_floor

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import MatmulConfig
from ..tile_scheduler import RasterOrder
from ..tile_scheduler_splitk import SplitKTileScheduler
from .loadop import async_load_AB
from .matmul import (
    _get_c_smem_layout,
    cluster_size,
    consumer_main_loop,
    warp_specialized_gemm_output,
)


fn tma_wgmma_warp_specialized_gemm_kernel_persistent_splitk[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    accum_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    c_layout: Layout,
    workspace_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    c_tma_layout: Layout,
    c_smem_layout: Layout,
    cluster_shape: StaticTuple[Int32, 3],
    splits: Int,
    raster_order: RasterOrder,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = True,
    num_threads: Int = 128,
    pipeline_stages: Int = 4,
    partitioned_multicast: Bool = False,
    use_tma_store: Bool = False,
    promotion_frequency: Int = 1,
    pdl_level: PDLLevel = PDLLevel(),
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    reduction_workspace: LayoutTensor[
        accum_type, workspace_layout, MutableAnyOrigin
    ],
    locks_ptr: UnsafePointer[NoneType],
    problem_shape: IndexList[3],
):
    constrained[transpose_b, "Only support transposed B in layout"]()

    alias num_consumer = (num_threads // WARPGROUP_SIZE) - 1
    alias num_consumer_threads = num_consumer * WARPGROUP_SIZE
    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])
    alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

    alias K = b_layout.shape[1].value()
    alias N = b_layout.shape[0].value()
    alias M = a_layout.shape[0].value()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias num_k_iters = K // BK

    alias a_smem_layout = tile_layout_k_major[a_type, BM, BK, a_swizzle]()
    alias b_smem_layout = tile_layout_k_major[b_type, BN, BK, b_swizzle]()

    alias simd_size = simd_width_of[c_type]()

    constrained[a_type == b_type, "A and B must have the same type"]()
    constrained[(K % BK) == 0, "K must be divisible by BK"]()

    constrained[
        not partitioned_multicast
        or a_swizzle.bytes() // size_of[a_type]() == BK,
        (
            "Currently partitioned multi-casting is only supported when BK =="
            " (a_swizzle.bytes // size_of[a_type]"
        ),
    ]()
    constrained[
        not partitioned_multicast
        or b_swizzle.bytes() // size_of[b_type]() == BK,
        (
            "Currently partitioned multi-casting is only supported when BK =="
            " (b_swizzle.bytes // size_of[b_type]"
        ),
    ]()

    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]

    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128

    alias use_cluster = cluster_size[cluster_shape]() > 1

    wgmma_op = TensorCoreAsync[
        accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    var smem = external_memory[
        UInt8, address_space = AddressSpace.SHARED, alignment=8
    ]()

    alias a_smem_size = a_smem_layout.size() * pipeline_stages
    alias b_smem_size = b_smem_layout.size() * pipeline_stages
    alias c_smem_size = c_smem_layout.size()

    alias a_smem_bytes = a_smem_size * size_of[a_type]()
    alias b_smem_bytes = b_smem_size * size_of[b_type]()
    alias c_smem_bytes = c_smem_size * size_of[c_type]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()
    var c_smem = (smem + a_smem_bytes + b_smem_bytes).bitcast[Scalar[c_type]]()
    var smem_pool = (smem + a_smem_bytes + b_smem_bytes + c_smem_bytes).bitcast[
        Int64
    ]()

    var a_smem_iter = LayoutTensorIter[
        a_type,
        a_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=True,
    ](a_smem, a_smem_size)

    var b_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=True,
    ](b_smem, b_smem_size)

    var c_smem_tile = LayoutTensor[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](c_smem)

    var a_mbars_ptr = smem_pool.bitcast[Int64]()
    var b_mbars_ptr = smem_pool.bitcast[Int64]() + pipeline_stages

    full = a_mbars_ptr.bitcast[SharedMemBarrier]()
    empty = b_mbars_ptr.bitcast[SharedMemBarrier]()

    var warp_group_idx = thread_idx.x // UInt(WARPGROUP_SIZE)
    var warp_group_thread_idx = thread_idx.x % UInt(WARPGROUP_SIZE)

    var rank_m = block_id_in_cluster.y
    var rank_n = block_id_in_cluster.x

    var scheduler = SplitKTileScheduler[
        Index(N, K),
        block_tile_shape,
        splits,
        num_consumer,
        pipeline_stages,
        Index(CLUSTER_M, CLUSTER_N),
        raster_order,
    ](
        problem_shape,
        Index(rank_m, rank_n),
        locks_ptr,
    )

    # var SM_NUM = scheduler.get_sm_num()

    @parameter
    if (
        pdl_level > PDLLevel.OFF
        and pdl_level != PDLLevel.NO_WAIT_OVERLAP_AT_END
    ):
        wait_on_dependent_grids()

    var lane_predicate = elect_one_sync()
    if thread_idx.x == 0:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()

        @parameter
        for i in range(pipeline_stages):
            full[i].init(1)
            empty[i].init(num_consumer * CLUSTER_SIZE)

    @parameter
    if cluster_size[cluster_shape]() > 1:
        fence_mbarrier_init()
        cluster_sync_relaxed()
    else:
        barrier()

    var warp_id = get_warp_id()
    if warp_group_idx == 0:
        alias num_regs = 24 if num_consumer <= 2 else 32
        var work_tile_info = scheduler.initial_work_tile_info()

        warpgroup_reg_dealloc[num_regs]()
        if warp_id == 0 and lane_predicate:
            var write_pipeline_states = PipelineState[pipeline_stages]()

            while work_tile_info.is_valid():
                var m_coord = work_tile_info.m * BM
                var n_coord = work_tile_info.n * BN

                alias work_k_tile_count = num_k_iters // splits
                var work_k_tile_start = work_tile_info.get_k_start()

                async_load_AB[
                    block_tile_shape=block_tile_shape,
                    cluster_shape=cluster_shape,
                    partitioned_multicast=partitioned_multicast,
                    num_k_iters=work_k_tile_count,
                ](
                    a_tma_op,
                    b_tma_op,
                    a_smem_iter,
                    b_smem_iter,
                    UInt(m_coord),
                    UInt(n_coord),
                    UInt(work_k_tile_start),
                    rank_n,
                    rank_m,
                    write_pipeline_states,
                    empty,
                    full,
                )

                # Get next work tile
                var next_work_tile_info = scheduler.fetch_next_work(
                    work_tile_info,
                )

                work_tile_info = next_work_tile_info
    else:

        @parameter
        if num_consumer == 1 or num_consumer == 2:
            alias num_regs = 256 if num_consumer == 1 else 240
            warpgroup_reg_alloc[num_regs]()
        else:
            warpgroup_reg_alloc[160]()

        var work_tile_info = scheduler.initial_work_tile_info()

        var local_warp_group_idx = warp_group_idx - 1

        var c_reg_tile = LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        var final_c_reg_tile = LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        @parameter
        if a_type is DType.float8_e4m3fn:
            _ = final_c_reg_tile.fill(0.0)
        else:
            _ = c_reg_tile.fill(0.0)

        @parameter
        for i in range(pipeline_stages):

            @parameter
            if cluster_size[cluster_shape]() > 1:
                if warp_group_thread_idx < CLUSTER_SIZE:
                    _ = empty[i].arrive_cluster(warp_group_thread_idx)
            else:
                if warp_group_thread_idx == 0:
                    _ = empty[i].arrive()

        var read_pipeline_states = PipelineState[pipeline_stages]()

        while work_tile_info.is_valid():
            alias work_k_tile_count = num_k_iters // splits

            consumer_main_loop[
                cluster_shape=cluster_shape,
                promotion_frequency=promotion_frequency,
                num_consumer=num_consumer,
                num_k_iters=work_k_tile_count,
            ](
                final_c_reg_tile,
                c_reg_tile,
                a_smem_iter,
                b_smem_iter,
                read_pipeline_states,
                full,
                empty,
                wgmma_op,
                UInt(local_warp_group_idx),
                UInt(warp_group_thread_idx),
            )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                is DType.float8_e4m3fn else c_reg_tile
            )

            scheduler.reduction(
                reduction_workspace,
                output_reg_tile,
                work_tile_info,
                num_consumer,
                local_warp_group_idx,
            )

            # check if this is the reduction tile
            if scheduler.is_last_split(work_tile_info):
                var block_y = UInt(work_tile_info.m)
                var block_x = UInt(work_tile_info.n)

                warp_specialized_gemm_output[
                    c_tile_shape = Index(BM, BN),
                    c_swizzle=c_swizzle,
                    wgmma_shape=wgmma_shape,
                    num_consumer=num_consumer,
                    use_tma_store=use_tma_store,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                ](
                    c_tma_op,
                    c,
                    c_smem_tile,
                    output_reg_tile,
                    UInt(warp_group_thread_idx),
                    UInt(local_warp_group_idx),
                    thread_idx.x - UInt(WARPGROUP_SIZE),
                    Int(block_y),
                    Int(block_x),
                )

            # Get next work tile
            var next_work_tile_info = scheduler.fetch_next_work(
                work_tile_info,
            )

            work_tile_info = next_work_tile_info

    @parameter
    if pdl_level >= PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()

    # TO ensure SMEM destruction doesn't happen
    @parameter
    if cluster_size[cluster_shape]() > 1:
        cluster_sync()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
    `nvvm.cluster_dim`=cluster_shape,
)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn sm90_warp_specialized_gemm_splitk[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    accum_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    c_tma_layout: Layout,
    c_smem_layout: Layout,
    cluster_shape: StaticTuple[Int32, 3],
    splits: Int,
    raster_order: RasterOrder,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = True,
    num_threads: Int = 128,
    pipeline_stages: Int = 4,
    partitioned_multicast: Bool = False,
    use_tma_store: Bool = False,
    promotion_frequency: Int = 1,
    pdl_level: PDLLevel = PDLLevel(),
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    workspace_buffer: NDBuffer[accum_type, 3, MutableAnyOrigin],
    locks_ptr: UnsafePointer[NoneType],
    problem_shape: IndexList[3],
):
    alias M = c_layout.shape[0].value()
    alias N = c_layout.shape[1].value()
    alias K = a_layout.shape[1].value()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias NUM_TILES = ceildiv(M, BM) * ceildiv(N, BN)

    alias workspace_layout = Layout.row_major(NUM_TILES, BM, BN)

    alias workspace_tensor_type = LayoutTensor[
        accum_type, workspace_layout, MutableAnyOrigin
    ]

    var workspace_tensor = workspace_tensor_type(
        workspace_buffer.data,
        RuntimeLayout[
            workspace_layout,
            element_type = workspace_tensor_type.layout_int_type,
            linear_idx_type = workspace_tensor_type.linear_idx_type,
        ].row_major(
            IndexList[3, element_type = workspace_tensor_type.layout_int_type](
                NUM_TILES, BM, BN
            )
        ),
    )

    tma_wgmma_warp_specialized_gemm_kernel_persistent_splitk[
        a_type,
        b_type,
        c_type,
        accum_type,
        a_layout,
        b_layout,
        a_tile_layout,
        b_tile_layout,
        c_layout,
        workspace_layout,
        block_tile_shape,
        wgmma_shape,
        a_desc_layout,
        b_desc_layout,
        c_desc_layout,
        c_tma_layout,
        c_smem_layout,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cluster_shape=cluster_shape,
        splits=splits,
        raster_order=raster_order,
        transpose_b=transpose_b,
        num_threads=num_threads,
        pipeline_stages=pipeline_stages,
        partitioned_multicast=partitioned_multicast,
        use_tma_store=use_tma_store,
        pdl_level=pdl_level,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
    ](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        c,
        workspace_tensor,
        locks_ptr,
        problem_shape,
    )


fn warp_specialize_gemm_with_multicasting_splitk[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    splits: Int,
    raster_order: RasterOrder,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    var M = c.dim[0]()
    alias N = c_shape.get[1]()
    alias K = a_shape.get[1]()

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

    var logger = Logger()

    logger.info("Executing Split-K Warp Specialized GEMM with Multicasting")
    logger.info("block_tile_shape:", config.block_tile_shape)
    logger.info("cluster_shape:", config.cluster_shape)
    logger.info("mma_shape:", config.mma_shape)

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
        Int(config.num_pipeline_stages),
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

    a_tma_op = create_tma_tile[
        Index(BM // CLUSTER_N, BK) if config.partitioned_multicast else Index(
            BM, BK
        ),
        swizzle_mode=a_swizzle,
    ](ctx, a)
    b_tma_op = create_tma_tile[
        Index(BN // CLUSTER_M, BK) if config.partitioned_multicast else Index(
            BN, BK
        ),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    c_tma_op = create_tma_tile[
        c_smem_tile,
        swizzle_mode=c_swizzle,
        __desc_layout = Layout.row_major(c_smem_tile[0], c_smem_tile[1]),
    ](ctx, c)

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

    alias scheduler = SplitKTileScheduler[
        Index(N, K),
        config.block_tile_shape,
        splits,
        config.num_consumer,
        config.num_pipeline_stages,
        Index(config.cluster_shape[1], config.cluster_shape[0]),
        raster_order,
    ]

    var launch_grid_shape = scheduler.get_grid_shape(
        config.cluster_shape,
        raster_order,
    )

    alias accum_type = DType.float32  # fix this

    var NUM_TILES = scheduler.get_num_tiles(
        Index(M, N, K),
        config.block_tile_shape,
        Index(config.cluster_shape[1], config.cluster_shape[0]),
    )

    var workspace_data = ctx.enqueue_create_buffer[accum_type](
        NUM_TILES * BM * BN
    )
    var reduction_workspace = NDBuffer[accum_type, 3](
        workspace_data.unsafe_ptr(),
        Index(NUM_TILES, BM, BN),
    )

    var locks_buffer_size_bytes = (
        scheduler.get_required_locks_buffer_size_bytes[
            accum_type, config.num_consumer
        ](
            Index(M, N, K),
            config.block_tile_shape,
            Index(CLUSTER_M, CLUSTER_N),
        )
    )

    var locks_ptr = ctx.enqueue_create_buffer[DType.uint8](
        locks_buffer_size_bytes
    )

    ctx.enqueue_memset(locks_ptr, 0)

    alias kernel = sm90_warp_specialized_gemm_splitk[
        a_type,
        b_type,
        c_type,
        accum_type,
        __type_of(a).layout,
        __type_of(b).layout,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c).layout,
        config.block_tile_shape,
        config.mma_shape,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(c_tma_op).desc_layout,
        __type_of(c_tma_op).layout,
        c_smem_layout,
        c_swizzle=c_swizzle,
        cluster_shape=cluster_shape,
        splits=splits,
        raster_order=raster_order,
        transpose_b=True,
        num_threads = Int(num_threads),
        pipeline_stages = Int(config.num_pipeline_stages),
        partitioned_multicast = config.partitioned_multicast,
        use_tma_store=use_tma_store,
        pdl_level = config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
    ]

    ctx.enqueue_function[kernel, dump_asm=False](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        c,
        reduction_workspace,
        locks_ptr,
        Index(M, N, K),
        grid_dim=(
            launch_grid_shape[0],
            launch_grid_shape[1],
            launch_grid_shape[2],
        ),
        block_dim=(num_threads),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
        attributes=pdl_launch_attributes(config.pdl_level()),
    )

    _ = workspace_data^
    _ = locks_ptr^
