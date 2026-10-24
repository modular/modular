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
"""`allgather` over slot-backed inputs.

Every device's input is handed to `allgather` as a `TileTensor` on
`StableAddrEngine`: the tile holds the address of a stable address slot, and
the slot holds the data pointer. The tiles are built once, before the slots
are pointed anywhere, and the same tiles are gathered twice with different
slot contents in between. A correct result both times shows that every
kernel path resolves its peer reads through the slot rather than baking in
a pointer, which is the property a device graph relies on to retarget its
inputs per replay.

The length set splits the sm10x dispatch inside the P2P path: the
16-byte-aligned lengths take the TMA kernel, the misaligned ones fall
through to the plain P2P kernel. A grouped case (two groups of two)
exercises the group-local slicing of slot-backed inputs, and on CDNA4 the
relay kernel the grouped path routes through.

The slots are filled with a plain copy since no graph builder is involved;
`DeviceGraphBuilder.place_stable()` writes the same one-pointer buffer.
"""

from std.math.uutils import ualign_down
from std.sys import size_of

from comm import Signal
from comm.allgather import allgather
from comm.sync import enable_p2p, init_signal_buffer
from extensibility.stable_addr_engine import (
    StableTileTensor,
    stable_tile_tensor,
)
from layout import TileTensor, row_major
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    HostBuffer,
    StableAddr,
)
from std.testing import assert_equal, assert_true

comptime dtype = DType.float32

# The two 16-byte-aligned lengths take the TMA path on sm10x, the misaligned
# ones fall through to the plain P2P kernel; 256 KiB + 5 also clears the
# relay cutoff for the grouped case.
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


def _input_value(alt: Bool, src_rank: Int, j: Int) -> Scalar[dtype]:
    """Values unique per source and index, exactly representable in float32.
    `alt` shifts the pattern so a pass over retargeted slots cannot pass on
    stale results."""
    var base = Scalar[dtype]((src_rank + 1) * 1000 + j)
    if alt:
        base += Scalar[dtype](1000000)
    return base


def _check_outputs(
    ctxs: List[DeviceContext],
    out_dev: List[DeviceBuffer[dtype]],
    length: Int,
    group_size: Int,
    alt: Bool,
) raises:
    for d in range(len(ctxs)):
        var group_start = ualign_down(d, group_size)
        for l in range(group_size):
            var host_out = ctxs[d].enqueue_create_host_buffer[dtype](length)
            ctxs[d].enqueue_copy(host_out, out_dev[d * group_size + l])
            ctxs[d].synchronize()
            for j in range(length):
                var expected = _input_value(alt, group_start + l, j)
                try:
                    assert_equal(host_out[j], expected)
                except e:
                    print(
                        "Verification failed: device",
                        d,
                        "global source",
                        group_start + l,
                        "index",
                        j,
                    )
                    print("Value:", host_out[j], "Expected:", expected)
                    raise e^


def stable_allgather_test[
    ngpus: Int, group_size: Int
](ctxs: List[DeviceContext], length: Int) raises:
    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var in_dev_alt = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus * group_size)
    var slots = List[DeviceBuffer[.uint64]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var host_buffers = List[HostBuffer[dtype]](capacity=2 * ngpus)

    var temp_buffer_num_bytes = ngpus * size_of[dtype]() * length
    for i in range(ngpus):
        in_dev.append(ctxs[i].enqueue_create_buffer[dtype](length))
        in_dev_alt.append(ctxs[i].enqueue_create_buffer[dtype](length))
        slots.append(ctxs[i].enqueue_create_buffer[.uint64](1))
        signal_buffers.append(
            ctxs[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )
        init_signal_buffer(signal_buffers[i], ctxs[i])

    for k in range(ngpus * group_size):
        var d = k // group_size
        out_dev.append(ctxs[d].enqueue_create_buffer[dtype](length))

    for i in range(ngpus):
        for alt in range(2):
            var host = ctxs[i].enqueue_create_host_buffer[dtype](length)
            for j in range(length):
                host[j] = _input_value(alt == 1, i, j)
            ctxs[i].enqueue_copy(in_dev_alt[i] if alt else in_dev[i], host)
            host_buffers.append(host^)

    for i in range(ngpus):
        ctxs[i].synchronize()

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    # Built over the slots before either slot holds a pointer; no data
    # address is involved until a kernel dereferences the tile.
    comptime InType = StableTileTensor[dtype, type_of(row_major(length))]
    var in_tensors = Array[InType, ngpus](uninitialized=True)
    for i in range(ngpus):
        in_tensors[i] = stable_tile_tensor[dtype](
            _slot_over(slots[i]), row_major(length)
        )

    comptime OutType = TileTensor[
        dtype, type_of(row_major(length)), MutAnyOrigin
    ]

    def out_tile_at(k: Int) {mut out_dev, imm} -> OutType:
        return TileTensor(out_dev[k], row_major(length)).as_unsafe_any_origin()

    var out_tensors = Array[_, ngpus * group_size](fill_with=out_tile_at)

    def run_and_check(alt: Bool) raises {imm}:
        for d in range(ngpus):
            allgather[group_size=group_size](
                in_tensors, out_tensors, rank_sigs, ctxs[d], d
            )
        for i in range(ngpus):
            ctxs[i].synchronize()
        _check_outputs(ctxs, out_dev, length, group_size, alt)

    # Every slot must be published before any device reads a peer's slot.
    for i in range(ngpus):
        _point_slot_at(ctxs[i], slots[i], in_dev[i])
    for k in range(ngpus * group_size):
        out_dev[k].enqueue_fill(Scalar[dtype](0))
    for i in range(ngpus):
        ctxs[i].synchronize()
    run_and_check(False)

    # Same tiles, different slot contents: the gather follows the slots.
    for i in range(ngpus):
        _point_slot_at(ctxs[i], slots[i], in_dev_alt[i])
    for k in range(ngpus * group_size):
        out_dev[k].enqueue_fill(Scalar[dtype](0))
    for i in range(ngpus):
        ctxs[i].synchronize()
    run_and_check(True)

    # The kernels reach these only by address and the closures capture them
    # by reference, so without an explicit last use they are destroyed at
    # their last direct mention above and their memory recycled mid-test.
    _ = in_dev^
    _ = in_dev_alt^
    _ = out_dev^
    _ = slots^
    _ = signal_buffers^
    _ = rank_sigs^
    _ = in_tensors^
    _ = out_tensors^
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
            print(
                t"====allgather-stable-addr-{dtype}-{ngpus}-{length} elements"
            )
            stable_allgather_test[ngpus, group_size=ngpus](ctxs, length)

    # Two groups of two: group-local slicing over slot-backed inputs, plus
    # the relay kernel on CDNA4 where the grouped path pairs groups up.
    if DeviceContext.number_of_devices() >= 4:
        var ctxs = List[DeviceContext]()
        for i in range(4):
            ctxs.append(DeviceContext(device_id=i))
        print("====allgather-stable-addr-grouped-2-groups-of-2")
        stable_allgather_test[4, group_size=2](ctxs, 256 * 1024 + 5)
