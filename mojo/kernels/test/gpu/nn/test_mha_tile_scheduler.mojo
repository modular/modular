# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host import DeviceContext
from gpu.id import block_idx
from nn.mha_tile_scheduler import MHASchedule, TileScheduler, WorkInfo

from utils.index import Index, IndexList


fn test_kernel[schedule: MHASchedule]():
    scheduler = TileScheduler[32, 3, num_ctas=8, schedule=schedule](1, 100)

    work_info = scheduler.get_current_work_info()

    while work_info.is_valid():
        print(block_idx.x, work_info)
        work_info = scheduler.fetch_next_work()


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
        # CHECK-DAG: 1 (32, 0, 0, True)
        # CHECK-DAG: 0 (0, 0, 0, True)
        # CHECK-DAG: 2 (64, 0, 0, True)
        # CHECK-DAG: 3 (96, 0, 0, True)
        # CHECK-DAG: 4 (0, 1, 0, True)
        # CHECK-DAG: 5 (32, 1, 0, True)
        # CHECK-DAG: 6 (64, 1, 0, True)
        # CHECK-DAG: 7 (96, 1, 0, True)
        # CHECK-DAG: 0 (0, 2, 0, True)
        # CHECK-DAG: 3 (96, 2, 0, True)
        # CHECK-DAG: 1 (32, 2, 0, True)
        # CHECK-DAG: 2 (64, 2, 0, True)
        print("==== test default schedule")
        test[MHASchedule.DEFAULT](ctx)

        # CHECK-LABEL: ==== test prompt rotate schedule
        # CHECK-DAG: 4 (96, 1, 0, True)
        # CHECK-DAG: 7 (0, 1, 0, True)
        # CHECK-DAG: 0 (0, 0, 0, True)
        # CHECK-DAG: 6 (32, 1, 0, True)
        # CHECK-DAG: 5 (64, 1, 0, True)
        # CHECK-DAG: 1 (32, 0, 0, True)
        # CHECK-DAG: 2 (64, 0, 0, True)
        # CHECK-DAG: 3 (96, 0, 0, True)
        # CHECK-DAG: 0 (0, 2, 0, True)
        # CHECK-DAG: 1 (32, 2, 0, True)
        # CHECK-DAG: 2 (64, 2, 0, True)
        # CHECK-DAG: 3 (96, 2, 0, True)
        print("==== test prompt rotate schedule")
        test[MHASchedule.PROMPT_ROTATE](ctx)
