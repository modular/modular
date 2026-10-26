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
"""`broadcast` over slot-backed inputs.

The root's input is handed to `broadcast` as a `TileTensor` on
`StableAddrEngine`: the tile holds the address of a stable address slot on
the root GPU, and the slot holds the data pointer. The same tile goes to
every device's launch, so every rank resolves the root's data through the
slot -- over P2P for the others, locally for the root. The tile is built
once, before the slot is pointed anywhere, and the same tile broadcasts
twice with different slot contents in between. A correct result both times
shows the kernels follow the slot rather than baking in a pointer, which is
the property a device graph relies on to retarget its inputs per replay.

The tuning table never picks the 2-stage kernel for two GPUs, and the
remote runners expose two, so that kernel is additionally driven through
its launcher with a forced 2-stage launch. The in-place variant covers the
2-stage kernel's root self-copy check, which compares the result pointer
against the pointer read from the slot.

Every GPU takes a turn as root so the slot is exercised on both sides of a
P2P read. The slots are filled with a plain copy since no graph builder is
involved; `DeviceGraphBuilder.place_stable()` writes the same one-pointer
buffer.
"""

from std.math import ceildiv
from std.sys import size_of

from comm import Signal
from comm.broadcast import broadcast, broadcast_2stage
from comm.sync import enable_p2p, init_signal_buffer
from extensibility.stable_addr_engine import stable_tile_tensor
from layout import TileTensor, row_major
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    HostBuffer,
    StableAddr,
)
from max.gpu.primitives.grid_controls import PDLLevel
from std.testing import assert_true

comptime dtype = DType.float32

comptime test_lengths = (
    8 * 1024,
    8 * 1024 + 7,
    256 * 1024 + 5,
    1024 * 1024 + 8,
)
comptime test_gpu_counts = (2, 4)


def _slot_over(mut slot: DeviceBuffer[.uint64]) -> StableAddr:
    return StableAddr(
        slot.unsafe_ptr()
        .unsafe_bitcast[Pointer[NoneType, MutUntrackedOrigin]]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def _point_slot_at(
    ctx: DeviceContext,
    slot: DeviceBuffer[.uint64],
    target: DeviceBuffer[dtype],
) raises:
    var value = ctx.enqueue_create_host_buffer[.uint64](1)
    value[0] = UInt64(Int(target.unsafe_ptr()))
    ctx.enqueue_copy(slot, value)


def _input_value(alt: Bool, root: Int, j: Int) -> Scalar[dtype]:
    """Position-based values including the root rank, exactly representable
    in float32. 251 is the largest prime < 256, so the pattern avoids
    power-of-two aliasing; `alt` shifts it so a pass over retargeted slots
    cannot pass on stale results."""
    var base = Scalar[dtype](root + 1) + Scalar[dtype](j % 251)
    if alt:
        base += Scalar[dtype](100000)
    return base


def _check_outputs(
    ctxs: List[DeviceContext],
    out_dev: List[DeviceBuffer[dtype]],
    length: Int,
    alt: Bool,
    root: Int,
) raises:
    for i in range(len(ctxs)):
        var host_output = List(length=length, fill=Scalar[dtype](0))
        ctxs[i].enqueue_copy(host_output, out_dev[i])
        ctxs[i].synchronize()
        for j in range(length):
            var expected = _input_value(alt, root, j)
            if host_output[j] != expected:
                raise Error(
                    "Verification failed at GPU",
                    i,
                    "index",
                    j,
                    "value:",
                    host_output[j],
                    "expected:",
                    expected,
                )


def stable_broadcast_test[
    ngpus: Int, *, force_2stage: Bool, in_place: Bool
](ctxs: List[DeviceContext], length: Int, root: Int) raises:
    var root_ctx = ctxs[root]

    var in_dev = root_ctx.enqueue_create_buffer[dtype](length)
    var in_dev_alt = root_ctx.enqueue_create_buffer[dtype](length)
    var slot = root_ctx.enqueue_create_buffer[.uint64](1)
    var host_buffers = List[HostBuffer[dtype]](capacity=2)

    for alt in range(2):
        var host = root_ctx.enqueue_create_host_buffer[dtype](length)
        for j in range(length):
            host[j] = _input_value(alt == 1, root, j)
        root_ctx.enqueue_copy(in_dev_alt if alt else in_dev, host)
        host_buffers.append(host^)

    # One output per device; the in-place variant aliases root's output to
    # the buffer the slot starts at.
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    for i in range(ngpus):
        if in_place and i == root:
            out_dev.append(in_dev)
        else:
            out_dev.append(ctxs[i].enqueue_create_buffer[dtype](length))

    # Signal buffers carry the staging space the 2-stage kernel's chunks
    # need, so the entry-point and forced launches share them.
    var num_bytes = length * size_of[dtype]()
    var chunk_bytes = ceildiv(num_bytes, ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    for i in range(ngpus):
        signal_buffers.append(
            ctxs[i].create_buffer_sync[.uint8](size_of[Signal]() + chunk_bytes)
        )
        init_signal_buffer(signal_buffers[i], ctxs[i])

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    for i in range(ngpus):
        ctxs[i].synchronize()

    # Built over the slot before it holds a pointer; no data address is
    # involved until a kernel dereferences the tile.
    var in_tile = stable_tile_tensor[dtype](_slot_over(slot), row_major(length))

    comptime OutType = TileTensor[
        dtype, type_of(row_major(length)), MutAnyOrigin
    ]
    var out_tiles = Array[OutType, ngpus](uninitialized=True)
    for i in range(ngpus):
        out_tiles[i] = TileTensor(
            out_dev[i], row_major(length)
        ).as_unsafe_any_origin()

    def run_and_check(alt: Bool) raises {imm}:
        comptime for i in range(ngpus):
            comptime if force_2stage:
                broadcast_2stage[ngpus=ngpus, pdl_level=PDLLevel()](
                    in_tile, out_tiles[i], rank_sigs, ctxs[i], root, rank=i
                )
            else:
                broadcast[ngpus](
                    in_tile, out_tiles[i], rank_sigs, ctxs[i], root, rank=i
                )
        for i in range(ngpus):
            ctxs[i].synchronize()
        _check_outputs(ctxs, out_dev, length, alt, root)

    # Publish the slot before any device reads it. Root's in-place output
    # is the input itself, so the first pass leaves it unpoisoned to keep
    # the data intact.
    _point_slot_at(root_ctx, slot, in_dev)
    for i in range(ngpus):
        if not (in_place and i == root):
            out_dev[i].enqueue_fill(Scalar[dtype](0))
    for i in range(ngpus):
        ctxs[i].synchronize()
    run_and_check(False)

    # Same tile, different slot contents: the broadcast follows the slot.
    _point_slot_at(root_ctx, slot, in_dev_alt)
    for i in range(ngpus):
        out_dev[i].enqueue_fill(Scalar[dtype](0))
    for i in range(ngpus):
        ctxs[i].synchronize()
    run_and_check(True)

    # The kernels reach these only by address and the closures capture them
    # by reference, so without an explicit last use they are destroyed at
    # their last direct mention above and their memory recycled mid-test.
    _ = in_dev^
    _ = in_dev_alt^
    _ = slot^
    _ = out_dev^
    _ = signal_buffers^
    _ = rank_sigs^
    _ = out_tiles^
    _ = host_buffers^


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime ngpus = rebind[Int](test_gpu_counts[gpu_idx])
        if DeviceContext.number_of_devices() < ngpus:
            continue

        var ctxs = List[DeviceContext]()
        for i in range(ngpus):
            ctxs.append(DeviceContext(device_id=i))

        comptime for length_idx in range(len(test_lengths)):
            comptime length = rebind[Int](test_lengths[length_idx])
            for root in range(ngpus):
                print(
                    "====broadcast-stable-addr",
                    dtype,
                    "ngpus",
                    ngpus,
                    "root",
                    root,
                    "nelems",
                    length,
                )
                stable_broadcast_test[
                    ngpus, force_2stage=False, in_place=False
                ](ctxs, length, root)

                print(
                    "====broadcast-stable-addr-2stage",
                    dtype,
                    "ngpus",
                    ngpus,
                    "root",
                    root,
                    "nelems",
                    length,
                )
                stable_broadcast_test[ngpus, force_2stage=True, in_place=False](
                    ctxs, length, root
                )

                # The 2-stage kernel skips root's self-copy when the result
                # aliases the input; with slot-backed inputs that check
                # reads the pointer through the slot.
                print(
                    "====broadcast-stable-addr-2stage-inplace",
                    dtype,
                    "ngpus",
                    ngpus,
                    "root",
                    root,
                    "nelems",
                    length,
                )
                stable_broadcast_test[ngpus, force_2stage=True, in_place=True](
                    ctxs, length, root
                )
