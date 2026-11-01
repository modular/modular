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
"""`allreduce` recorded into a `DeviceGraph.create_collective` graph over
`StableTensor` inputs.

Each device's graph publishes its own input pointer into its stable address
slot and reads every device's input through a `StableTensor` over the
corresponding slot, converted with `to_tile_tensor()` on the host while the
graph records. The graphs are then published, replayed, and verified; replayed
a second time; and finally the collective is rebuilt with a cache key that
changes only for rank 0. Rank 0's new graph publishes a different buffer,
while the other ranks' graphs come straight from the cache, so a correct third
result shows that a cached graph reads a peer's input through the slot rather
than through an address baked in at record time.
"""

from std.sys import size_of

from comm import Signal
from comm.allreduce import allreduce
from comm.sync import enable_p2p, init_signal_buffer
from extensibility import (
    IOSpec,
    StableTensor,
    StableTileTensor,
    get_row_major_tensor_spec_static,
)
from layout import TileTensor, row_major
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
)
from std.testing import assert_almost_equal, assert_true

comptime dtype = DType.float32

# Lengths are compile-time so the `StableTensor` can carry a fully static
# spec, as the graph compiler's specs do. They cover the Lamport path, the
# 1-stage path with a scalar tail, and the bandwidth regime just past the
# four-GPU 1-stage limit: 1-stage on two GPUs, 2-stage on four.
comptime test_lengths = (8 * 1024, 8 * 1024 + 7, 1024 * 1024 + 8)
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
    ctxs: Array[DeviceContext, ngpus],
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


def device_graph_allreduce_test[
    ngpus: Int, length: Int
](ctxs: Array[DeviceContext, ngpus]) raises:
    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var in_dev_alt = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)

    var temp_buffer_num_bytes = ngpus * size_of[dtype]() * length
    for i in range(ngpus):
        in_dev.append(ctxs[i].enqueue_create_buffer[dtype](length))
        in_dev[i].enqueue_fill(Float32(i + 1))
        in_dev_alt.append(ctxs[i].enqueue_create_buffer[dtype](length))
        in_dev_alt[i].enqueue_fill(Float32(10 * (i + 1)))
        out_dev.append(ctxs[i].enqueue_create_buffer[dtype](length))
        out_dev[i].enqueue_fill(Float32(-1))
        signal_buffers.append(
            ctxs[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )

    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], ctxs[i])
    for i in range(ngpus):
        ctxs[i].synchronize()

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    # Which input a graph publishes when it is (re)built. The build closure
    # reads it by reference, so the flip below is visible to later rebuilds.
    var publish_alt = False

    comptime spec = get_row_major_tensor_spec_static[dtype, 1, length]()
    comptime StableIn = StableTensor[io_spec=IOSpec.Input, static_spec=spec]
    comptime InType = StableTileTensor[dtype, StableIn.RuntimeLayout]

    def build_for[n: Int](mut builder: DeviceGraphBuilder[_]) raises {mut}:
        _ = builder.place_stable(
            n,
            _source_of(in_dev_alt[n]) if publish_alt else _source_of(in_dev[n]),
        )

        # Every input, this device's included, is read through its slot. The
        # conversion happens here on the host; the slot is dereferenced only
        # inside the recorded kernels.
        var in_tensors = Array[InType, ngpus](uninitialized=True)
        for j in range(ngpus):
            in_tensors[j] = StableIn(
                builder.get_stable(j), StableIn.RuntimeLayout()
            ).to_tile_tensor()

        var out_tensor = TileTensor(
            out_dev[n], row_major(length)
        ).as_unsafe_any_origin()

        allreduce[ngpus=ngpus](
            in_tensors,
            out_tensor,
            rank_sigs,
            builder.recording_context(),
            my_rank=n,
        )

    def key_for[n: Int]() {imm} -> String:
        return String(t"stable_allreduce_{length}_{n}")

    def key_for_rebuilt_rank0[n: Int]() {imm} -> String:
        comptime if n == 0:
            return String(t"stable_allreduce_{length}_{n}_alt")
        else:
            return String(t"stable_allreduce_{length}_{n}")

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

    # A second replay of the same graphs reuses the signal buffers and, on
    # the Lamport path, advances the generation counters.
    reset_outputs()
    DeviceGraph.replay_collective(graphs, ctxs)
    _check_outputs(ctxs, out_dev, length, expected_sum)

    # Rebuild rank 0 alone, publishing its alternate input. The other ranks'
    # graphs are cache hits and were recorded against rank 0's original
    # buffer; they must pick up the new one through the slot.
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
    _ = signal_buffers^
    _ = rank_sigs^
    _ = publish_alt
    _ = stable_devices^


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime ngpus = rebind[Int](test_gpu_counts[gpu_idx])
        if DeviceContext.number_of_devices() < ngpus:
            continue

        var ctx_list = List[DeviceContext](capacity=ngpus)
        for i in range(ngpus):
            ctx_list.append(DeviceContext(device_id=i))
        var ctxs = Array[DeviceContext, ngpus](
            fill_with=lambda (i: Int) {ref} -> DeviceContext: ctx_list[i]
        )

        comptime for length_idx in range(len(test_lengths)):
            comptime length = rebind[Int](test_lengths[length_idx])
            print(
                t"====allreduce-device-graph-{dtype}-{ngpus}-{length} elements"
            )
            device_graph_allreduce_test[ngpus, length](ctxs)
