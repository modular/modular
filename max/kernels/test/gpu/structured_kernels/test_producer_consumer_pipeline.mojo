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
"""Portable stress test for `ProducerConsumerPipeline`.

Exercises the pipeline end-to-end on real hardware through BOTH synchronization
backends, selected by the build target:

- NVIDIA: `NvidiaMbarBackend`  (`mbarrier` objects)
- AMD:    `AmdCounterBackend`  (shared-memory atomic counters)

`AmdCounterBackend` currently has no in-tree caller, so this is the first thing
that actually drives it on hardware.

Design (single block, warp-specialized):

- warp 0 is the sole producer, warp 1 the sole consumer, communicating through
  a `num_stages`-slot ring of `Int32` in shared memory.
- For iteration `i` the producer writes `value(i) = i + 1` into the current
  ring slot, release-fences, then raises `full`. The consumer waits on `full`,
  acquire-fences, copies the slot into `result[i]`, then raises `empty`.
- `n_items` is chosen far larger than `num_stages`, so the ring wraps many
  times. Each wrap advances the pipeline `phase`, so this stresses the
  wrap-around bookkeeping (NVIDIA parity bit) and the counter-threshold math
  (`(phase + 1) * arrivals`) the AMD backend derives from `phase`.
- `value(i)` is strictly monotonic and distinct per iteration, so a stale slot
  read (consumer seeing a previous lap's value in a reused slot), a dropped
  hand-off, or a duplicated one all surface as `result[i] != i + 1`.

Exactly the warp-leader lane signals each hand-off, so both arrive counts are
1. The cross-target `pop.fence` (`std.atomic.fence`) is used for the
data-before-signal ordering that `AmdCounterBackend` delegates to the caller;
`gpu.intrinsics.threadfence` is NVIDIA-only and cannot serve the AMD side.
"""

from std.atomic import Ordering, fence
from max.gpu import barrier, lane_id, thread_idx, WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.sys import has_amd_gpu_accelerator
from std.testing import assert_equal

from structured_kernels.pipeline import ProducerConsumerPipeline
from structured_kernels.pipeline_backend import (
    AmdCounterBackend,
    NvidiaMbarBackend,
    PipelineBackend,
)


@always_inline
def _produced_value(i: Int) -> Int32:
    # Strictly monotonic and distinct per iteration (see module docstring): any
    # mis-ordered, stale, dropped, or duplicated hand-off fails the exact check.
    return Int32(i + 1)


def pingpong_kernel[
    num_stages: Int,
    Backend: PipelineBackend,
](result: UnsafePointer[Scalar[DType.int32], MutAnyOrigin], n_items: Int32):
    """Hand `n_items` values from one producer warp to one consumer warp."""
    comptime n_bar = 2 * num_stages  # full[0..num_stages) then empty[0..)

    var bar = stack_allocation[
        n_bar, Backend.BarrierStorage, address_space=AddressSpace.SHARED
    ]()
    var data = stack_allocation[
        num_stages, Int32, address_space=AddressSpace.SHARED
    ]()

    var pipeline = ProducerConsumerPipeline[num_stages, Backend](bar)

    # Single-thread init of the barrier ring + data slots, published to the
    # whole block by the barrier below. Both arrive counts are 1: only the
    # warp-leader lane signals each hand-off.
    if thread_idx.x == 0:
        pipeline.init_mbars(1, 1)
        for s in range(num_stages):
            data[s] = Int32(0)
    # A plain block barrier publishes the init writes to the whole block before
    # first use.
    barrier()

    var is_leader = lane_id() == 0
    var warp = Int(thread_idx.x) // WARP_SIZE

    if warp == 0:
        for i in range(Int(n_items)):
            pipeline.wait_consumer()
            var slot = pipeline.producer_stage()
            if is_leader:
                data[Int(slot)] = _produced_value(i)
            # Publish the slot write before raising `full`.
            fence[ordering=Ordering.RELEASE, scope=StaticString("")]()
            if is_leader:
                pipeline.backend.arrive_full(slot)
            pipeline.producer_step()
    elif warp == 1:
        for i in range(Int(n_items)):
            pipeline.wait_producer()
            var slot = pipeline.consumer_stage()
            fence[ordering=Ordering.ACQUIRE, scope=StaticString("")]()
            if is_leader:
                result[i] = data[Int(slot)]
            # Retire the read before freeing the slot for the producer's next
            # lap, so the producer cannot overwrite a slot mid-read.
            fence[ordering=Ordering.RELEASE, scope=StaticString("")]()
            if is_leader:
                pipeline.backend.arrive_empty(slot)
            pipeline.consumer_step()


def _run_config[num_stages: Int](ctx: DeviceContext, n_items: Int) raises:
    print("  num_stages=", num_stages, ", n_items=", n_items, sep="")

    var result = ctx.enqueue_create_buffer[DType.int32](n_items)
    ctx.enqueue_memset[DType.int32](result, 0)

    # Exactly two warps: producer (warp 0) + consumer (warp 1).
    comptime block_dim = 2 * WARP_SIZE

    comptime if has_amd_gpu_accelerator():
        ctx.enqueue_function[
            pingpong_kernel[num_stages, AmdCounterBackend[1, 1]]
        ](result, Int32(n_items), grid_dim=1, block_dim=block_dim)
    else:
        ctx.enqueue_function[
            pingpong_kernel[num_stages, NvidiaMbarBackend[num_stages]]
        ](result, Int32(n_items), grid_dim=1, block_dim=block_dim)

    var host = ctx.enqueue_create_host_buffer[DType.int32](n_items)
    ctx.enqueue_copy(host, result)
    ctx.synchronize()

    for i in range(n_items):
        assert_equal(host[i], _produced_value(i))

    _ = result^
    print("    PASSED")


def main() raises:
    print("ProducerConsumerPipeline portable stress test")

    with DeviceContext() as ctx:
        # A large item count against a small ring forces many laps (phase
        # wrap-arounds). num_stages=1 is the strict ping-pong edge case; the
        # rest overlap production and consumption across a deeper ring.
        comptime n_items = 8192
        _run_config[1](ctx, n_items)
        _run_config[2](ctx, n_items)
        _run_config[4](ctx, n_items)
        _run_config[8](ctx, n_items)

    print("ALL PASSED")
