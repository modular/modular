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
"""`mo.bundled.allreduce.sum`'s `build_graph` entry point recorded into a
`DeviceGraph.create_collective` graph over `StableInputVariadicTensors`.

Each device's build closure publishes its own input into its stable address
slot, packs every device's slot into a `StableVariadicTensors` (the aggregate
the graph compiler hands the kernel), and records the kernel through its
`build_graph` method. The graphs are replayed and verified, replayed again,
and finally rank 0 is rebuilt against an alternate input so that the other
ranks' cached graphs are shown to read rank 0's data through the slot rather
than through an address baked in at record time.
"""

from std.sys import size_of

from builtin_kernels.kernels import BundledAllReduceSum
from comm import Signal
from comm.sync import enable_p2p, init_signal_buffer
from extensibility import (
    IOSpec,
    ManagedTensorSlice,
    OutputFusion,
    StableVariadicTensors,
    StaticTensorSpecList,
    VariadicTensors,
    get_row_major_tensor_spec_static,
)
from layout import to_index_list
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceContextArray,
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
    StableAddr,
)
from std.testing import assert_almost_equal, assert_equal, assert_true
from std.utils import IndexList, StaticTuple

comptime dtype = DType.float32

# Lengths are compile-time so the packs can carry fully static specs, as the
# graph compiler's specs do.
comptime test_lengths = (8 * 1024, 8 * 1024 + 7)
comptime test_gpu_counts = (2, 4)


@fieldwise_init
struct _StoreOutFusion[fusion_dtype: DType](OutputFusion):
    """Plain store functor for the kernel's fused output: the graph compiler
    always binds one, so the test binds the identity store by hand."""

    var dst: Pointer[Scalar[Self.fusion_dtype], MutUntrackedOrigin]

    def store[
        dtype: DType,
        rank: Int,
        simd_width: SIMDLength,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]):
        self.dst.unsafe_offset(idx[rank - 1]).unsafe_store[
            alignment=element_alignment
        ](rebind[SIMD[Self.fusion_dtype, simd_width]](val))


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


def _check_pack_views[length: Int, ngpus: Int]() raises:
    """`StableVariadicTensors` element views carry the pack's shape and size on
    the host without touching the slots."""
    comptime spec = get_row_major_tensor_spec_static[dtype, 1, length]()
    comptime specs = StaticTensorSpecList[
        ParameterList.of[spec.get_internals(), spec.get_internals()](),
        ParameterList.of[
            to_index_list[1](spec.shape_tuple),
            to_index_list[1](spec.shape_tuple),
        ](),
        ParameterList.of[
            to_index_list[1](spec.strides_tuple),
            to_index_list[1](spec.strides_tuple),
        ](),
    ]()
    comptime SlotPtr = Pointer[
        Pointer[NoneType, MutUntrackedOrigin], ImmUntrackedOrigin
    ]
    var pack = StableVariadicTensors[io_spec=IOSpec.Input, static_specs=specs](
        StaticTuple[StableAddr, 2](
            StableAddr(SlotPtr.unsafe_dangling()),
            StableAddr(SlotPtr.unsafe_dangling()),
        ),
        StaticTuple[IndexList[1], 2](
            IndexList[1](length), IndexList[1](length)
        ),
    )
    assert_equal(len(pack), 2)
    assert_equal(pack[0].size(), length)
    assert_equal(pack[1].shape()[0], length)


def bundled_allreduce_build_graph_test[
    ngpus: Int,
    //,
    length: Int,
    in_specs: StaticTensorSpecList[dtype=dtype, rank=1, ...],
    sig_specs: StaticTensorSpecList[dtype=DType.uint8, rank=1, ...],
    signal_bytes: Int,
](ctxs: Array[DeviceContext, ngpus]) raises:
    comptime in_spec = get_row_major_tensor_spec_static[dtype, 1, length]()

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

    def build_for[n: Int](mut builder: DeviceGraphBuilder[_]) raises {mut}:
        _ = builder.place_stable(
            n,
            _source_of(in_dev_alt[n]) if publish_alt else _source_of(in_dev[n]),
        )

        # Every peer, this device's included, is read through its slot: the
        # pack the graph compiler would hand the kernel.
        var addrs = StaticTuple[StableAddr, ngpus]()
        for j in range(ngpus):
            addrs[j] = builder.get_stable(j)
        var shapes = StaticTuple[IndexList[1], ngpus](IndexList[1](length))
        var inputs = StableVariadicTensors[
            io_spec=IOSpec.Input, static_specs=in_specs
        ](addrs, shapes)

        var sig_ptrs = StaticTuple[
            Pointer[Scalar[DType.uint8], MutUntrackedOrigin], ngpus
        ]()
        for j in range(ngpus):
            sig_ptrs[j] = (
                signal_buffers[j]
                .unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
            )
        var sig_shapes = StaticTuple[IndexList[1], ngpus](
            IndexList[1](signal_bytes)
        )
        var signals = VariadicTensors[
            io_spec=IOSpec.MutableInput, static_specs=sig_specs
        ](sig_ptrs, sig_shapes)

        var out_ptr = (
            out_dev[n].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        )
        var output = ManagedTensorSlice[
            io_spec=IOSpec.Output, static_spec=in_spec
        ](out_ptr, IndexList[1](length))._bind_to_fused_output(
            _StoreOutFusion(out_ptr)
        )

        BundledAllReduceSum.build_graph[dtype, 1, "gpu", "bundled_allreduce"](
            output, inputs, signals, builder
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
    _ = signal_buffers^
    _ = publish_alt
    _ = stable_devices^


def _run[ngpus: Int, length: Int](ctxs: Array[DeviceContext, ngpus]) raises:
    """Builds the static spec lists (one identical spec per peer, spelled the
    way the graph compiler emits them) for the supported peer counts."""
    comptime in_spec = get_row_major_tensor_spec_static[dtype, 1, length]()
    comptime signal_bytes = size_of[Signal]() + 2 * ngpus * size_of[
        dtype
    ]() * length
    comptime sig_spec = get_row_major_tensor_spec_static[
        DType.uint8, 1, signal_bytes
    ]()
    comptime in_shape = to_index_list[1](in_spec.shape_tuple)
    comptime in_strides = to_index_list[1](in_spec.strides_tuple)
    comptime sig_shape = to_index_list[1](sig_spec.shape_tuple)
    comptime sig_strides = to_index_list[1](sig_spec.strides_tuple)

    comptime in_specs = StaticTensorSpecList[
        ParameterList.splat[ngpus, in_spec.get_internals()](),
        ParameterList.splat[ngpus, in_shape](),
        ParameterList.splat[ngpus, in_strides](),
    ]()
    comptime sig_specs = StaticTensorSpecList[
        ParameterList.splat[ngpus, sig_spec.get_internals()](),
        ParameterList.splat[ngpus, sig_shape](),
        ParameterList.splat[ngpus, sig_strides](),
    ]()
    bundled_allreduce_build_graph_test[
        length=length,
        in_specs=in_specs,
        sig_specs=sig_specs,
        signal_bytes=signal_bytes,
    ](ctxs)


def main() raises:
    _check_pack_views[8, 2]()

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
