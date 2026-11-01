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
"""`allreduce` over slot-backed inputs.

Every device's input is handed to `allreduce` as a `TileTensor` on
`StableAddrEngine`: the tile holds the address of a stable address slot, and
the slot holds the data pointer. The tiles are built once, before the slots
are pointed anywhere, and the same tiles are reduced twice with different slot
contents in between. A correct result both times shows that the collective's
flat and sliced views keep resolving through the slot rather than baking in a
pointer, which is the property a device graph relies on to retarget its inputs
per replay.

The tuning tables never pick the 2-stage kernel for two GPUs, and the remote
runners expose two, so that kernel is additionally driven through the P2P
launcher with a forced 2-stage config.

The slots are filled with a plain copy since no graph builder is involved;
`DeviceGraphBuilder.place_stable()` writes the same one-pointer buffer.
"""

from std.sys import simd_width_of, size_of

from comm import Signal, group_end, group_start
from comm.allreduce import (
    AllReduceAlgorithm,
    AllReduceTuningConfig,
    _allreduce_p2p,
    allreduce,
)
from comm.device_query import get_sm_version
from comm.sync import enable_p2p, init_signal_buffer
from extensibility.stable_addr_engine import (
    StableTileTensor,
    stable_tile_tensor,
)
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, StableAddr, get_gpu_target
from max.gpu.primitives.grid_controls import PDLLevel
from std.testing import assert_almost_equal, assert_true
from std.utils import StaticTuple

comptime dtype = DType.float32

# Sizes chosen against the sm_100a rows of `allreduce_tuning_table`: 32 KiB
# takes the Lamport path, the ragged lengths fall through to 1-stage with a
# scalar tail, and 4 MiB + 32 B is the smallest SIMD-multiple size past the
# four-GPU 1-stage limit: 1-stage on two GPUs but 2-stage on four.
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


def _check_outputs(
    ctxs: List[DeviceContext],
    out_dev: List[DeviceBuffer[dtype]],
    length: Int,
    expected: Float32,
) raises:
    for i in range(len(ctxs)):
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


def stable_allreduce_test[
    ngpus: Int, *, force_2stage: Bool
](ctxs: List[DeviceContext], length: Int) raises:
    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var in_dev_alt = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var slots = List[DeviceBuffer[.uint64]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)

    var temp_buffer_num_bytes = ngpus * size_of[dtype]() * length
    for i in range(ngpus):
        in_dev.append(ctxs[i].enqueue_create_buffer[dtype](length))
        in_dev[i].enqueue_fill(Float32(i + 1))
        in_dev_alt.append(ctxs[i].enqueue_create_buffer[dtype](length))
        in_dev_alt[i].enqueue_fill(Float32(10 * (i + 1)))
        out_dev.append(ctxs[i].enqueue_create_buffer[dtype](length))
        out_dev[i].enqueue_fill(Float32(-1))
        slots.append(ctxs[i].enqueue_create_buffer[.uint64](1))
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
    var out_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) {mut out_dev, imm} -> OutType: TileTensor(
            out_dev[i], row_major(length)
        ).as_unsafe_any_origin()
    )

    # The forced 2-stage launch needs an explicit epilogue; `allreduce` builds
    # the equivalent store itself.
    var out_capture = StaticTuple[OutType, ngpus]()
    for i in range(ngpus):
        out_capture[i] = out_tensors[i]

    var two_stage_cfg = AllReduceTuningConfig(
        ngpus=ngpus,
        num_bytes=length * size_of[dtype](),
        sm_version=get_sm_version(),
        num_blocks=216,
        algorithm=AllReduceAlgorithm.TWO_STAGE,
    )

    def run_and_check(expected: Float32) raises {imm}:
        group_start()
        comptime for i in range(ngpus):
            comptime if force_2stage:

                @inline(.always)
                def store_epilogue[
                    _dtype: DType,
                    _width: SIMDLength,
                    *,
                    alignment: Int,
                ](coords: Coord, val: SIMD[_dtype, _width]) {
                    var out_capture
                } -> None:
                    out_capture[i].store[width=_width, alignment=alignment](
                        coords, rebind[SIMD[dtype, _width]](val)
                    )

                _allreduce_p2p[ngpus=ngpus, pdl_level=PDLLevel()](
                    in_tensors,
                    out_tensors[i].layout,
                    store_epilogue,
                    rank_sigs,
                    two_stage_cfg,
                    ctxs[i],
                    i,
                )
            else:
                allreduce[ngpus=ngpus](
                    in_tensors, out_tensors[i], rank_sigs, ctxs[i]
                )
        group_end()
        for i in range(ngpus):
            ctxs[i].synchronize()
        _check_outputs(ctxs, out_dev, length, expected)

    var expected_sum = Float32(0)
    for i in range(ngpus):
        expected_sum += Float32(i + 1)

    # Every slot must be published before any device reads a peer's slot.
    for i in range(ngpus):
        _point_slot_at(ctxs[i], slots[i], in_dev[i])
    for i in range(ngpus):
        ctxs[i].synchronize()
    run_and_check(expected_sum)

    # Same tiles, different slot contents: the reduction follows the slots.
    for i in range(ngpus):
        out_dev[i].enqueue_fill(Float32(-1))
        _point_slot_at(ctxs[i], slots[i], in_dev_alt[i])

    for i in range(ngpus):
        ctxs[i].synchronize()

    run_and_check(10 * expected_sum)

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
    _ = out_capture


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()

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
                t"====allreduce-stable-addr-{dtype}-{ngpus}-{length} elements"
            )
            stable_allreduce_test[ngpus, force_2stage=False](ctxs, length)

            # The 2-stage kernel partitions by whole SIMD vectors and falls
            # back to 1-stage otherwise, so only vector-multiple lengths
            # exercise it.
            comptime if length % simd_width == 0:
                print(
                    t"====allreduce-stable-addr-2stage-{dtype}-{ngpus}-{length}"
                    t" elements"
                )
                stable_allreduce_test[ngpus, force_2stage=True](ctxs, length)
