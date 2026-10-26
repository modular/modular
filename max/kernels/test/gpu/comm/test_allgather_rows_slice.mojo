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
"""Tests `allgather_rows_slice` against `allgather` followed by a row slice.

Target family: NVIDIA sm_100 (B200) and sm_90; the TMA and the plain P2P
paths of `allgather` both sit under the new entry point.

Input shards hold random bit patterns (NaNs, infinities, signed zeros and
denormals included), and the result is compared bit for bit, because the
collective only moves data.
"""

from std.collections import Array
from std.sys import size_of
from std.time import perf_counter_ns

from comm import Signal
from comm.allgather import allgather, allgather_rows_slice
from comm.sync import enable_p2p, init_signal_buffer
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.testing import assert_equal, assert_raises, assert_true


def _fill_random_bits[
    dtype: DType
](host: HostBuffer[dtype], length: Int, seed: Int):
    """Fills `host` with uniformly random bit patterns of the element width."""
    comptime assert size_of[dtype]() == 2, "the generator emits 16-bit words"
    var bits = host.unsafe_ptr().unsafe_bitcast[UInt16]()
    var state = UInt64(seed) * 6364136223846793005 + 1442695040888963407
    for i in range(length):
        state = state * 6364136223846793005 + 1442695040888963407
        bits[unsafe_offset=i] = UInt16(state >> 48)


def rows_slice_test[
    dtype: DType, ngpus: Int
](
    list_of_ctx: List[DeviceContext],
    rows: Int,
    cols: Int,
    iterations: Int = 0,
) raises:
    """Checks the row-sliced gather against the full gather at one shape.

    Args:
        list_of_ctx: One context per device.
        rows: Rows of each device's `[rows, cols]` shard.
        cols: Columns of each shard.
        iterations: When positive, also times that many back-to-back launches
            of each collective and prints the mean per launch.
    """
    var total = rows * cols
    var slice_elems = (rows // ngpus) * cols
    print("  rows", rows, "cols", cols, "-> per-device block", slice_elems)

    var in_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_in = List[HostBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=2 * ngpus)
    # `allgather` and `allgather_rows_slice` each get their own signals, so
    # the two run back to back without sharing barrier state.
    for i in range(ngpus):
        in_bufs.append(list_of_ctx[i].create_buffer_sync[dtype](total))
        var host = list_of_ctx[i].enqueue_create_host_buffer[dtype](total)
        _fill_random_bits(host, total, seed=i + 1)
        list_of_ctx[i].enqueue_copy(in_bufs[i], host)
        host_in.append(host^)
    for i in range(2 * ngpus):
        signal_buffers.append(
            list_of_ctx[i % ngpus].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(signal_buffers[i], list_of_ctx[i % ngpus])

    var ref_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )
    var new_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[ngpus + i])
    )

    # `[device][source]` outputs of the reference (whole shards) and of the
    # new collective (one row block each). Zeroed so an unwritten block fails.
    var ref_outs = List[List[DeviceBuffer[dtype]]](capacity=ngpus)
    var new_outs = List[List[DeviceBuffer[dtype]]](capacity=ngpus)
    for d in range(ngpus):
        var ref_row = List[DeviceBuffer[dtype]](capacity=ngpus)
        var new_row = List[DeviceBuffer[dtype]](capacity=ngpus)
        for j in range(ngpus):
            ref_row.append(list_of_ctx[d].create_buffer_sync[dtype](total))
            new_row.append(
                list_of_ctx[d].create_buffer_sync[dtype](slice_elems)
            )
            list_of_ctx[d].enqueue_memset[dtype](ref_row[j], val=0)
            list_of_ctx[d].enqueue_memset[dtype](new_row[j], val=0)
        ref_outs.append(ref_row^)
        new_outs.append(new_row^)

    comptime InTileType = type_of(
        TileTensor(in_bufs[0], row_major(total)).as_imm().as_unsafe_any_origin()
    )
    var tt_in = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> InTileType: (
            TileTensor(in_bufs[i], row_major(total))
            .as_imm()
            .as_unsafe_any_origin()
        )
    )
    comptime OutTileType = type_of(
        TileTensor(ref_outs[0][0], row_major(total)).as_unsafe_any_origin()
    )

    def tt_ref_out_at(i: Int) {mut ref_outs, imm total} -> OutTileType:
        var d, j = divmod(i, ngpus)
        return TileTensor(
            ref_outs[d][j], row_major(total)
        ).as_unsafe_any_origin()

    def tt_new_out_at(i: Int) {mut new_outs, imm slice_elems} -> OutTileType:
        var d, j = divmod(i, ngpus)
        return TileTensor(
            new_outs[d][j], row_major(slice_elems)
        ).as_unsafe_any_origin()

    var tt_ref_out = Array[_, ngpus * ngpus](fill_with=tt_ref_out_at)
    var tt_new_out = Array[_, ngpus * ngpus](fill_with=tt_new_out_at)

    # A bad output on the last device must prevent every device from launching.
    var invalid_outs = tt_new_out.copy()
    invalid_outs[ngpus * ngpus - 1] = tt_ref_out[ngpus * ngpus - 1]
    for d in range(ngpus):
        with assert_raises(contains="an output buffer is not one row block"):
            allgather_rows_slice(
                tt_in, invalid_outs, new_sigs, cols, list_of_ctx[d], d
            )

    for d in range(ngpus):
        allgather(tt_in, tt_ref_out, ref_sigs, list_of_ctx[d], d)
    for d in range(ngpus):
        allgather_rows_slice(
            tt_in, tt_new_out, new_sigs, cols, list_of_ctx[d], d
        )
    for d in range(ngpus):
        list_of_ctx[d].synchronize()

    # Bit identity against the reference and against the inputs themselves,
    # so a bug shared by both collectives still fails.
    for d in range(ngpus):
        for j in range(ngpus):
            var host_ref = list_of_ctx[d].enqueue_create_host_buffer[dtype](
                total
            )
            var host_new = list_of_ctx[d].enqueue_create_host_buffer[dtype](
                slice_elems
            )
            list_of_ctx[d].enqueue_copy(host_ref, ref_outs[d][j])
            list_of_ctx[d].enqueue_copy(host_new, new_outs[d][j])
            list_of_ctx[d].synchronize()
            var ref_bits = host_ref.unsafe_ptr().unsafe_bitcast[UInt16]()
            var new_bits = host_new.unsafe_ptr().unsafe_bitcast[UInt16]()
            var in_bits = host_in[j].unsafe_ptr().unsafe_bitcast[UInt16]()
            var base = d * slice_elems
            var mismatches = 0
            for k in range(slice_elems):
                var got = new_bits[unsafe_offset=k]
                if (
                    got != ref_bits[unsafe_offset=base + k]
                    or got != in_bits[unsafe_offset=base + k]
                ):
                    mismatches += 1
            if mismatches != 0:
                print(
                    "  MISMATCH device",
                    d,
                    "source",
                    j,
                    "elements:",
                    mismatches,
                    "of",
                    slice_elems,
                )
            assert_equal(mismatches, 0)

    if iterations > 0:
        for _ in range(5):
            for d in range(ngpus):
                allgather(tt_in, tt_ref_out, ref_sigs, list_of_ctx[d], d)
            for d in range(ngpus):
                allgather_rows_slice(
                    tt_in, tt_new_out, new_sigs, cols, list_of_ctx[d], d
                )
        for d in range(ngpus):
            list_of_ctx[d].synchronize()

        var t0 = perf_counter_ns()
        for _ in range(iterations):
            for d in range(ngpus):
                allgather(tt_in, tt_ref_out, ref_sigs, list_of_ctx[d], d)
        for d in range(ngpus):
            list_of_ctx[d].synchronize()
        var ref_us = (
            Float64(perf_counter_ns() - t0) / 1000.0 / Float64(iterations)
        )

        t0 = perf_counter_ns()
        for _ in range(iterations):
            for d in range(ngpus):
                allgather_rows_slice(
                    tt_in, tt_new_out, new_sigs, cols, list_of_ctx[d], d
                )
        for d in range(ngpus):
            list_of_ctx[d].synchronize()
        var new_us = (
            Float64(perf_counter_ns() - t0) / 1000.0 / Float64(iterations)
        )
        print(
            "  timing (mean over",
            iterations,
            "launches): allgather",
            ref_us,
            "us, allgather_rows_slice",
            new_us,
            "us, ratio",
            ref_us / new_us,
        )


def rows_slice_rejects_bad_shapes[
    dtype: DType, ngpus: Int
](list_of_ctx: List[DeviceContext]) raises:
    """Shapes the collective cannot serve raise before any launch."""
    var total = 6 * 8
    var in_buf = list_of_ctx[0].create_buffer_sync[dtype](total)
    var out_buf = list_of_ctx[0].create_buffer_sync[dtype](total)
    var sig = list_of_ctx[0].create_buffer_sync[.uint8](size_of[Signal]())
    init_signal_buffer(sig, list_of_ctx[0])

    comptime InTileType = type_of(
        TileTensor(in_buf, row_major(total)).as_imm().as_unsafe_any_origin()
    )
    comptime OutTileType = type_of(
        TileTensor(out_buf, row_major(total)).as_unsafe_any_origin()
    )
    var tt_in = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> InTileType: (
            TileTensor(in_buf, row_major(total)).as_imm().as_unsafe_any_origin()
        )
    )
    var tt_out = Array[_, ngpus * ngpus](
        fill_with=lambda (i: Int) {ref} -> OutTileType: (
            TileTensor(out_buf, row_major(total)).as_unsafe_any_origin()
        )
    )
    var sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(sig)
    )

    # 6 rows over `ngpus` (a power of two above 2) devices.
    with assert_raises(contains="multiple of the device count"):
        allgather_rows_slice(tt_in, tt_out, sigs, 8, list_of_ctx[0], 0)
    # 8 rows of 3 elements: a 3-element block is 6 bytes, not vector-aligned.
    with assert_raises(contains="16 bytes"):
        allgather_rows_slice(tt_in, tt_out, sigs, 6, list_of_ctx[0], 0)


def main() raises -> None:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    comptime for ngpus_log2 in range(1, 4):
        comptime ngpus = 1 << ngpus_log2
        if DeviceContext.number_of_devices() < ngpus:
            continue
        var ctx = List[DeviceContext]()
        for i in range(ngpus):
            ctx.append(DeviceContext(device_id=i))
        print("Testing", ngpus, "GPUs")

        # Small shapes: one row per device, several, and a block that is not
        # a whole number of TMA chunks.
        rows_slice_test[.bfloat16, ngpus](ctx, ngpus, 16)
        rows_slice_test[.bfloat16, ngpus](ctx, 4 * ngpus, 4096)
        rows_slice_test[.bfloat16, ngpus](ctx, 8 * ngpus, 25008)
        rows_slice_test[.float16, ngpus](ctx, 2 * ngpus, 200)

    # The production shape: the DP8 M3.1 lm_head shard, 64 rows per replica
    # times 8 verify positions, over the 25008-wide vocabulary shard.
    if DeviceContext.number_of_devices() == 8:
        var ctx = List[DeviceContext]()
        for i in range(8):
            ctx.append(DeviceContext(device_id=i))
        print("Testing the M3.1 verify shape on 8 GPUs")
        rows_slice_test[.bfloat16, 8](ctx, 512, 25008, iterations=50)
        rows_slice_rejects_bad_shapes[.bfloat16, 8](ctx)
