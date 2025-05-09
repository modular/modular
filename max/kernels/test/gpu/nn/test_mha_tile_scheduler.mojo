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

from buffer import NDBuffer
from gpu.host import DeviceContext
from gpu.id import block_idx
from gpu.memory import AddressSpace
from math import ceildiv
from memory import UnsafePointer
from nn.mha_tile_scheduler import (
    WorkInfo,
    TileScheduler,
    MHASchedule,
    MHATileState,
    MHATileSummary,
    MHASchedulerSynchronization,
)
from utils.index import Index, IndexList


fn test_kernel[schedule: MHASchedule]():
    alias scheduler_t = TileScheduler[32, 3, num_ctas=8, schedule=schedule]
    scheduler = scheduler_t()
    valid_length = NDBuffer[DType.uint32, 1, MutableAnyOrigin]()
    tile_summary = MHATileSummary(1, ceildiv(100, 32), valid_length, 0)
    state = scheduler.initial_state(
        UnsafePointer[UInt32, address_space = AddressSpace.SHARED](),
        tile_summary,
    )
    work_info = scheduler.get_current_work_info(tile_summary, state)

    # @parameter
    # @always_inline
    # fn update_work_info(work: WorkInfo):
    #     work_info = work

    # while scheduler.advance[func=update_work_info, producer=True,sync= MHASchedulerSynchronization.DEFAULT](
    #     tile_summary, state, 0
    # ):
    #     print(block_idx.x, work_info)

    while work_info.is_valid():
        print(block_idx.x, work_info)
        work_info = scheduler.fetch_next_work(tile_summary, state)


def test[schedule: MHASchedule](ctx: DeviceContext):
    alias kernel = test_kernel[schedule]

    ctx.enqueue_function[kernel](
        grid_dim=8,
        block_dim=1,
    )

    ctx.synchronize()


def main():
    with DeviceContext() as ctx:
        # CHECK-LABEL: ==== test default schedule
        # CHECK-DAG: 0 (0, 0, 0, True)
        # CHECK-DAG: 0 (0, 2, 0, True)
        # CHECK-DAG: 1 (32, 0, 0, True)
        # CHECK-DAG: 1 (32, 2, 0, True)
        # CHECK-DAG: 2 (64, 0, 0, True)
        # CHECK-DAG: 2 (64, 2, 0, True)
        # CHECK-DAG: 3 (96, 0, 0, True)
        # CHECK-DAG: 3 (96, 2, 0, True)
        # CHECK-DAG: 4 (0, 1, 0, True)
        # CHECK-DAG: 5 (32, 1, 0, True)
        # CHECK-DAG: 6 (64, 1, 0, True)
        # CHECK-DAG: 7 (96, 1, 0, True)
        print("==== test default schedule")
        test[MHASchedule.DEFAULT](ctx)

        # CHECK-LABEL: ==== test prompt rotate schedule
        # CHECK-DAG: 0 (0, 0, 0, True)
        # CHECK-DAG: 0 (0, 2, 0, True)
        # CHECK-DAG: 1 (32, 0, 0, True)
        # CHECK-DAG: 1 (32, 2, 0, True)
        # CHECK-DAG: 2 (64, 0, 0, True)
        # CHECK-DAG: 2 (64, 2, 0, True)
        # CHECK-DAG: 3 (96, 0, 0, True)
        # CHECK-DAG: 3 (96, 2, 0, True)
        # CHECK-DAG: 4 (96, 1, 0, True)
        # CHECK-DAG: 5 (64, 1, 0, True)
        # CHECK-DAG: 6 (32, 1, 0, True)
        # CHECK-DAG: 7 (0, 1, 0, True)
        print("==== test prompt rotate schedule")
        test[MHASchedule.PROMPT_ROTATE](ctx)
