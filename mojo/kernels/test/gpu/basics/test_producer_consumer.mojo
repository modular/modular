# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s

from gpu import WARP_SIZE, barrier, thread_idx
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from layout.tma_async import PipelineState, TMABarrier
from memory import UnsafePointer, stack_allocation
from memory.pointer import _GPUAddressSpace

from utils import StaticTuple


fn producer_consumer_kernel[NUM_THREADS: Int]():
    var warp_id = thread_idx.x // WARP_SIZE
    var mbar = TMABarrier()

    if thread_idx.x == 0:
        mbar.init(NUM_THREADS)

    barrier()

    if warp_id == 0:
        if thread_idx.x == 0:
            print("Producer thread_idx: ", thread_idx.x, "warp_idx: ", warp_id)

        _ = mbar.arrive()
    else:
        mbar.wait(mbar.arrive())
        print("Consumer thread_idx:", thread_idx.x, ", warp_idx: ", warp_id)


def test_producer_consumer_kernel(ctx: DeviceContext):
    var kernel = ctx.compile_function[
        producer_consumer_kernel[64],
        _target = _get_gpu_target["sm_90"](),
    ]()

    ctx.enqueue_function(
        kernel,
        grid_dim=(1),
        block_dim=(64),
    )

    # CHECK-DAG: Producer thread_idx:  0 warp_idx:  0
    # CHECK-DAG: Consumer thread_idx: {{3[2-9]}} , warp_idx:  1
    # CHECK-DAG: Consumer thread_idx: {{[4-5][0-9]}}, warp_idx:  1
    # CHECK-DAG: Consumer thread_idx: {{6[0-3]}}, warp_idx:  1


fn producer_consumer_pipeline_kernel[Q_SIZE: Int](num_iters: Int):
    var k_tile_iters = num_iters

    var producer_mbar = StaticTuple[TMABarrier, Q_SIZE]()
    var consumer_mbar = StaticTuple[TMABarrier, Q_SIZE]()

    @parameter
    for i in range(Q_SIZE):
        producer_mbar[i] = TMABarrier()
        consumer_mbar[i] = TMABarrier()
        if thread_idx.x == 0:
            producer_mbar[i].init(1)
            consumer_mbar[i].init(128)

    barrier()

    var k_tile = 0

    @parameter
    for i in range(Q_SIZE):
        if thread_idx.x == 0:
            # pretend to load into smem tile
            print("prefetch: ", i)
            _ = producer_mbar[i].arrive()
        k_tile += 1
        k_tile_iters -= 1

    var write_state = PipelineState[Q_SIZE]()
    var read_state = PipelineState[Q_SIZE]()

    # producer-consumer loop
    while k_tile_iters > -Q_SIZE:
        var read_idx = read_state.index()
        producer_mbar[read_idx].wait(read_state.phase())

        if thread_idx.x == 0:
            print("consuming: ", read_idx)
        # pretend to do wgmma
        _ = consumer_mbar[read_idx].arrive()
        read_state.step()

        if thread_idx.x == 0:
            var write_idx = write_state.index()
            consumer_mbar[write_idx].wait(write_state.phase())
            print("producing: ", write_idx)
            # pretend to copy into smem tile
            _ = producer_mbar[write_idx].arrive()
            write_state.step()
        k_tile += 1
        k_tile_iters -= 1


def test_producer_consumer_pipeline_kernel(ctx: DeviceContext):
    var kernel = ctx.compile_function[
        producer_consumer_pipeline_kernel[4],
        _target = _get_gpu_target["sm_90"](),
    ]()

    ctx.enqueue_function(
        kernel,
        4,
        grid_dim=(1),
        block_dim=(128),
    )

    # CHECK: prefetch:  0
    # CHECK: prefetch:  1
    # CHECK: prefetch:  2
    # CHECK: prefetch:  3
    # CHECK: consuming:  0
    # CHECK: producing:  0
    # CHECK: consuming:  1
    # CHECK: producing:  1
    # CHECK: consuming:  2
    # CHECK: producing:  2
    # CHECK: consuming:  3
    # CHECK: producing:  3


def main():
    with DeviceContext() as ctx:
        test_producer_consumer_kernel(ctx)
        test_producer_consumer_pipeline_kernel(ctx)
