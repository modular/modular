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
"""The recording path of `mo.bundled.allreduce.sum` driven by hand into a
`DeviceGraph.create_collective` graph over stable address slots.

Each device's build closure publishes its own input into its stable address
slot, projects every device's slot to a `StableAddrEngine` tile (as the
kernel's `build_graph` projects its `Stable` inputs), and records the `comm`
allreduce into the builder's recording context. The graphs are replayed and
verified, replayed again, and finally rank 0 is rebuilt against an alternate
input so that the other ranks' cached graphs are shown to read rank 0's data
through the slot rather than through an address baked in at record time.
"""

from std.sys import size_of

from comm import Signal
from comm.allreduce import allreduce
from comm.sync import enable_p2p, init_signal_buffer
from extensibility.stable_addr_engine import (
    StableTileTensor,
    stable_tile_tensor,
)
from layout import TileTensor, row_major
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceContextArray,
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
)
from std.testing import assert_almost_equal, assert_true

comptime dtype = DType.float32

comptime test_lengths = (8 * 1024, 8 * 1024 + 7)
comptime test_gpu_counts = (2, 4)


def _source_of(
    mut buf: DeviceBuffer[dtype],
) -> Pointer[NoneType, MutUntrackedOrigin]:
    return (
        buf.unsafe_ptr()
        .unsafe_bitcast[NoneType]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def _check_outputs[
    ngpus: Int
](
    ctxs: Array[DeviceContext[], ngpus],
    out_dev: List[DeviceBuffer[dtype]],
    length: Int,
    expected: Float32,
) raises:
    for i in range(ngpus):
        var host = ctxs[i].enqueue_create_host_buffer[dtype](length)
        ctxs[i].enqueue_copy(host, out_dev[i])
        ctxs[i].synchronize()
        for j in range(length):
            try:
                assert_almost_equal(host[j], expected)
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host[j], "Expected:", expected)
                raise e^


def bundled_allreduce_build_graph_test[
    ngpus: Int, //, length: Int, signal_bytes: Int
](ctxs: Array[DeviceContext[], ngpus]) raises:
    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var in_dev_alt = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)

    for i in range(ngpus):
        in_dev.append(ctxs[i].enqueue_create_buffer[dtype](length))
        in_dev[i].enqueue_fill(Float32(i + 1))
        in_dev_alt.append(ctxs[i].enqueue_create_buffer[dtype](length))
        in_dev_alt[i].enqueue_fill(Float32(10 * (i + 1)))
        out_dev.append(ctxs[i].enqueue_create_buffer[dtype](length))
        out_dev[i].enqueue_fill(Float32(-1))
        signal_buffers.append(ctxs[i].create_buffer_sync[.uint8](signal_bytes))

    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], ctxs[i])
    for i in range(ngpus):
        ctxs[i].synchronize()

    # Which input a graph publishes when it is (re)built. The build closure
    # reads it by reference, so the flip below is visible to later rebuilds.
    var publish_alt = False

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    def build_for[n: Int](mut builder: DeviceGraphBuilder[_]) raises {mut}:
        _ = builder.place_stable(
            n,
            _source_of(in_dev_alt[n]) if publish_alt else _source_of(in_dev[n]),
        )

        # Every peer, this device's included, is read through its slot, as
        # the kernel's `build_graph` reads its `Stable` inputs.
        comptime InType = StableTileTensor[dtype, type_of(row_major(length))]
        var in_tensors = Array[InType, ngpus](uninitialized=True)
        for j in range(ngpus):
            in_tensors[j] = stable_tile_tensor[dtype](
                builder.get_stable(j), row_major(length)
            )

        allreduce[ngpus=ngpus](
            in_tensors,
            TileTensor(out_dev[n], row_major(length)),
            rank_sigs,
            builder.recording_context(),
            my_rank=n,
        )

    def key_for[n: Int]() {imm} -> String:
        return String(t"bundled_allreduce_build_graph_{length}_{n}")

    def key_for_rebuilt_rank0[n: Int]() {imm} -> String:
        comptime if n == 0:
            return String(t"bundled_allreduce_build_graph_{length}_{n}_alt")
        else:
            return String(t"bundled_allreduce_build_graph_{length}_{n}")

    var cache = DeviceGraphCache()
    var stable_devices = List[Int](
        length=ngpus, fill_with=lambda (i: Int) -> Int: i
    )

    def reset_outputs() raises {imm}:
        for i in range(ngpus):
            out_dev[i].enqueue_fill(Float32(-1))
            ctxs[i].synchronize()

    var expected_sum = Float32(0)
    for i in range(ngpus):
        expected_sum += Float32(i + 1)

    var graphs = DeviceGraph.create_collective(
        ctxs,
        build_for,
        key_for,
        cache=Pointer(to=cache),
        stable_devices=stable_devices,
    )

    DeviceGraph.replay_collective(graphs, ctxs)
    _check_outputs(ctxs, out_dev, length, expected_sum)

    reset_outputs()
    DeviceGraph.replay_collective(graphs, ctxs)
    _check_outputs(ctxs, out_dev, length, expected_sum)

    # Rebuild rank 0 alone, publishing its alternate input. The other ranks'
    # graphs are cache hits recorded against rank 0's original buffer; they
    # must pick up the new one through the slot.
    publish_alt = True
    reset_outputs()
    var mixed = DeviceGraph.create_collective(
        ctxs,
        build_for,
        key_for_rebuilt_rank0,
        cache=Pointer(to=cache),
        stable_devices=stable_devices,
    )

    DeviceGraph.replay_collective(mixed, ctxs)
    _check_outputs(ctxs, out_dev, length, expected_sum - 1 + 10)

    # The graphs hold these by address and the build closure captures them by
    # reference, with no other use past this point. Without an explicit last
    # use here they are destroyed as soon as the closure is defined and their
    # memory is recycled underneath the recorded graphs.
    _ = mixed^
    _ = graphs^
    _ = cache^
    _ = in_dev^
    _ = in_dev_alt^
    _ = out_dev^
    _ = rank_sigs^
    _ = signal_buffers^
    _ = publish_alt
    _ = stable_devices^


def _run[ngpus: Int, length: Int](ctxs: Array[DeviceContext[], ngpus]) raises:
    comptime signal_bytes = size_of[Signal]() + 2 * ngpus * size_of[
        dtype
    ]() * length
    bundled_allreduce_build_graph_test[
        length=length, signal_bytes=signal_bytes
    ](ctxs)


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime ngpus = rebind[Int](test_gpu_counts[gpu_idx])
        if DeviceContext.number_of_devices() < ngpus:
            continue

        var ctxs = DeviceContextArray[ngpus]()

        comptime for length_idx in range(len(test_lengths)):
            comptime length = rebind[Int](test_lengths[length_idx])
            print(
                t"====bundled-allreduce-build-graph-{dtype}-{ngpus}-{length}"
                t" elements"
            )
            _run[ngpus, length](ctxs.device_contexts)
