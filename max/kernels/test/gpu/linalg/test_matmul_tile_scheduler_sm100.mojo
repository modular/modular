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

from linalg.matmul_tile_scheduler_sm100 import TileScheduler, WorkInfo
from utils.static_tuple import StaticTuple
from utils.index import Index, IndexList
from gpu.host import DeviceContext
from gpu.id import warp_id as get_warp_id
from gpu.id import block_idx, block_id_in_cluster
from gpu.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
)
from layout.tma_async import PipelineState, SharedMemBarrier

from memory import stack_allocation
from gpu.memory import _GPUAddressSpace as AddressSpace, fence_mbarrier_init
from gpu.sync import syncwarp


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
fn test_kernel[
    num_stages: Int, cluster_shape: StaticTuple[Int32, 3]
](cluster_dim: StaticTuple[Int32, 3]):
    var clc_response = stack_allocation[
        num_stages,
        UInt128,
        address_space = AddressSpace.SHARED,
        alignment=16,
    ]()

    var clc_full_mbar = stack_allocation[
        num_stages,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=16,
    ]()

    var clc_empty_mbar = stack_allocation[
        num_stages,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=16,
    ]()

    var clc_throttle_full_mbar = stack_allocation[
        num_stages,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=16,
    ]()

    var clc_throttle_empty_mbar = stack_allocation[
        num_stages,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=16,
    ]()

    alias SCHEDULER_THREADS = 32
    alias TMA_LOAD_THREADS = 32
    alias MMA_THREADS = 32
    alias EPILOGUE_THREADS = 128
    alias CLUSTER_SIZE = cluster_shape[0] * cluster_shape[1]
    alias clc_producer_arv_count = 1
    alias clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    alias clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    alias clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    @parameter
    for i in range(num_stages):
        clc_full_mbar[i].init(clc_producer_arv_count)
        clc_empty_mbar[i].init(clc_consumer_arv_count)
        clc_throttle_full_mbar[i].init(clc_throttle_producer_arv_count)
        clc_throttle_empty_mbar[i].init(clc_throttle_consumer_arv_count)

    fence_mbarrier_init()
    cluster_sync()

    var cta_rank_in_cluster = block_rank_in_cluster()
    var warp_id = get_warp_id()
    var is_first_cta_in_cluster = cta_rank_in_cluster == 0

    # epilogue: 0, 1, 2, 3
    # scheduler: 4
    # tma_load(producer): 5
    # mma(consumer): 6
    var is_epilogue = warp_id < 4
    var is_scheduler = warp_id == 4 and is_first_cta_in_cluster
    var is_tma_load = warp_id == 5
    var is_mma = warp_id == 6

    var scheduler = TileScheduler[
        num_stages=num_stages,
        cluster_shape = Index[dtype = DType.uint32](
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
    ](cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar)

    # thread blocks start with their original cta coordinates
    var work_info = scheduler.initial_work_info()

    var clc_pipe_producer_state = PipelineState[num_stages](0, 1, 0)
    var clc_pipe_consumer_state = PipelineState[num_stages]()

    var clc_throttle_producer_state = PipelineState[num_stages](0, 1, 0)
    var clc_throttle_consumer_state = PipelineState[num_stages]()

    if is_tma_load:
        # This should not be necessary in a regular matmul, but for
        # stream-k, CTAs might be also working on the same coordinates. In
        # that case, we don't need to block the tile scheduler.
        var required_clc_query = True

        while work_info.is_valid():
            # CLC throuttle prevents each CTA from going a few waves ahead.
            if is_first_cta_in_cluster and required_clc_query:
                var index = clc_throttle_producer_state.index()
                var phase = clc_throttle_producer_state.phase()
                clc_throttle_empty_mbar[index].wait(phase)
                _ = clc_throttle_full_mbar[index].arrive()

                _ = clc_throttle_producer_state.step()

            # DO TMA LOAD
            # scheduler fetch next work
            syncwarp()
            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

    if is_scheduler:
        var required_clc_query = True

        while work_info.is_valid():
            if required_clc_query:
                var index = clc_throttle_consumer_state.index()
                var phase = clc_throttle_consumer_state.phase()
                clc_throttle_full_mbar[index].wait(phase)
                _ = clc_throttle_empty_mbar[index].arrive()

                clc_throttle_consumer_state.step()

                # advance to next work
                clc_pipe_producer_state = scheduler.advance_to_next_work(
                    clc_pipe_producer_state
                )
            # scheduler fetch next work
            next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )

            work_info = next_work_info
            clc_pipe_consumer_state.step()

        @parameter
        for i in range(num_stages):
            clc_empty_mbar[clc_pipe_producer_state.index()].wait(
                clc_pipe_producer_state.phase()
            )
            clc_pipe_producer_state.step()

    if is_mma:
        # TMEM alloc
        while work_info.is_valid():
            # DO MMA
            # scheduler fetch next work
            next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )

            work_info = next_work_info
            clc_pipe_consumer_state.step()

    if is_epilogue:
        while work_info.is_valid():
            # WAIT FOR MMA TO FINISH AND STORE RESULT
            # scheduler fetch next work

            next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

            if not work_info.is_valid():
                break


fn test_tile_scheduler(ctx: DeviceContext) raises:
    alias cluster_shape = StaticTuple[Int32, 3](2, 2, 1)
    alias grid_dim = (16, 8, 1)

    alias cluster_dim = StaticTuple[Int32, 3](
        grid_dim[0] // cluster_shape[0],
        grid_dim[1] // cluster_shape[1],
        cluster_shape[2],
    )
    alias kernel = test_kernel[2, cluster_shape]
    ctx.enqueue_function[kernel](
        cluster_dim,
        grid_dim=grid_dim,
        block_dim=(256),
    )


def main():
    with DeviceContext() as ctx:
        test_tile_scheduler(ctx)
