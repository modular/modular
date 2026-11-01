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
"""Loads from read-only memory stay below the PDL wait (NVIDIA SM90+).

A load through an immutable origin becomes `ld.global.nc`, a load of read-only
data through the non-coherent cache. `ptxas` treats that data as invariant and
schedules the load (`LDG.E.CONSTANT`) above `griddepcontrol.wait` (`ACQBULK`)
once its address is known, even though the PTX has it below. With PDL the
predecessor is still writing at that point, so the kernel reads stale data. A
coherent `ld.global` stays below the wait.

The producer releases its dependent at once and writes a slot table and its
payload only after a delay, with no host synchronisation in between. The
consumer reads the table and then the row it names, both after the wait, through
a read-only and through a coherent pointer. A load above the wait sees the
table's initial `-1` and the consumer returns without writing its output.

A pass only counts if the consumer really ran ahead of the producer, which the
driver is free not to do. Before its wait the consumer also reads a probe word
next to the table with a coherent volatile load and keeps the value; `-1` means
the consumer started before the producer's publish. The stale-read check runs on
every launch. Overlap is only needed on some launches: the test keeps launching,
up to a bound, until it has seen `iters` early ones, and reports an environment
that overlaps less often than that as inconclusive, apart from a stale-read
failure. Either way it prints how many early launches it saw. The first launch
of each kernel loads its module lazily and is not early, so it is a warm-up that
is not checked.
"""

from std.sys.intrinsics import llvm_intrinsic

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.testing import assert_equal

comptime NUM_BLOCKS = 64
comptime BLOCK = 256
comptime PAYLOAD = Float32(7)
comptime WEIGHT = Float32(2)
comptime UNWRITTEN = Float32(-1)


def late_producer(
    payload: MutPointer[Float32, MutAnyOrigin],
    table: MutPointer[Int32, MutAnyOrigin],
    delay_us: Int32,
):
    """Releases the dependent grid, sleeps, then publishes its outputs."""
    launch_dependent_grids()
    # nanosleep takes nanoseconds and caps at about 1 ms per call.
    var remaining = Int(delay_us)
    while remaining > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        remaining -= 1000
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, NUM_BLOCKS * BLOCK, stride):
        payload[i] = PAYLOAD
    # `table` holds the slot table, then the consumer's probe words.
    for i in range(tid, 2 * NUM_BLOCKS, stride):
        table[i] = Int32(i % NUM_BLOCKS)


def consumer[
    readonly: Bool
](
    weights: MutPointer[Float32, MutAnyOrigin],
    payload: MutPointer[Float32, MutAnyOrigin],
    table: MutPointer[Int32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    early_slot: MutPointer[Int32, MutAnyOrigin],
):
    """Reads the model weight before the wait, the producer's data after it."""
    var weight = weights.unsafe_offset(Int(thread_idx.x)).unsafe_load()
    var probe = table.unsafe_offset(NUM_BLOCKS + Int(block_idx.x))
    var seen_before_wait = probe.unsafe_load[volatile=True]()
    if thread_idx.x == 0:
        early_slot[Int(block_idx.x)] = seen_before_wait
    wait_on_dependent_grids()

    var table_ptr = table.unsafe_mut_cast[not readonly]()
    var payload_ptr = payload.unsafe_mut_cast[not readonly]()
    var slot = Int(table_ptr.unsafe_offset(Int(block_idx.x)).unsafe_load())
    if slot < 0:
        return
    var x = payload_ptr.unsafe_offset(
        slot * BLOCK + Int(thread_idx.x)
    ).unsafe_load()
    var gid = Int(thread_idx.x + block_idx.x * block_dim.x)
    output.unsafe_offset(gid).unsafe_store(0, weight * x)


def run_chain[
    readonly: Bool
](ctx: DeviceContext, iters: Int, delay_us: Int, max_launches: Int) raises:
    var weights = ctx.enqueue_create_buffer[DType.float32](BLOCK)
    var payload = ctx.enqueue_create_buffer[DType.float32](NUM_BLOCKS * BLOCK)
    var table = ctx.enqueue_create_buffer[DType.int32](2 * NUM_BLOCKS)
    var output = ctx.enqueue_create_buffer[DType.float32](NUM_BLOCKS * BLOCK)
    var early_slot = ctx.enqueue_create_buffer[DType.int32](NUM_BLOCKS)
    var early_slot_host = ctx.enqueue_create_host_buffer[DType.int32](
        NUM_BLOCKS
    )
    var weights_host = ctx.enqueue_create_host_buffer[DType.float32](BLOCK)
    var output_host = ctx.enqueue_create_host_buffer[DType.float32](
        NUM_BLOCKS * BLOCK
    )
    for i in range(BLOCK):
        weights_host[i] = WEIGHT
    ctx.enqueue_copy(weights, weights_host)

    var early_iters = 0
    var launches = 0
    while early_iters < iters and launches <= max_launches:
        var iter = launches
        launches += 1
        ctx.enqueue_memset[DType.float32](payload, 0)
        ctx.enqueue_memset[DType.int32](table, -1)
        ctx.enqueue_memset[DType.float32](output, UNWRITTEN)
        ctx.enqueue_memset[DType.int32](early_slot, 0)
        ctx.enqueue_function[late_producer](
            payload,
            table,
            Int32(delay_us),
            grid_dim=4,
            block_dim=64,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        ctx.enqueue_function[consumer[readonly]](
            weights,
            payload,
            table,
            output,
            early_slot,
            grid_dim=NUM_BLOCKS,
            block_dim=BLOCK,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        ctx.enqueue_copy(output_host, output)
        ctx.enqueue_copy(early_slot_host, early_slot)
        ctx.synchronize()

        var stale = 0
        for i in range(NUM_BLOCKS * BLOCK):
            if output_host[i] != WEIGHT * PAYLOAD:
                stale += 1
        if iter == 0:
            continue
        assert_equal(stale, 0, msg="stale read below the PDL wait")

        for i in range(NUM_BLOCKS):
            if early_slot_host[i] == -1:
                early_iters += 1
                break

    print(
        "early launches:",
        early_iters,
        "/",
        iters,
        "required,",
        launches - 1,
        "checked, readonly =",
        readonly,
    )
    if early_iters < iters:
        raise Error(
            "INCONCLUSIVE: only "
            + String(early_iters)
            + " of "
            + String(launches - 1)
            + " launches had a consumer block start before the producer"
            " published, "
            + String(iters)
            + " needed, so this environment cannot reliably exercise the PDL"
            " wait"
        )


def main() raises:
    # The producer outlives the consumer's launch and prologue by far, so only
    # a load held below the wait can see the published values.
    with DeviceContext() as ctx:
        run_chain[readonly=False](ctx, iters=10, delay_us=3000, max_launches=60)
        run_chain[readonly=True](ctx, iters=10, delay_us=3000, max_launches=60)
