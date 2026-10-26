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
"""Tests per-lane global-to-LDS DMA gathering scattered rows."""

from max.gpu import lane_id, WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.sync import s_waitcnt
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

from structured_kernels.amd_tile_io import _global_load_to_lds

# One sparse-MLA latent row: 512 + 64 FP8 bytes, 36 lanes of 16 bytes.
comptime ROW_BYTES = 576
comptime LANE_BYTES = 16
comptime CHUNKS_PER_ROW = ROW_BYTES // LANE_BYTES
comptime NUM_ROWS = 16
comptime NUM_CHUNKS = NUM_ROWS * CHUNKS_PER_ROW
comptime NUM_SRC_ROWS = 1000


def _gather_rows_through_lds(
    src: MutPointer[UInt8, MutAnyOrigin],
    indices: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[UInt8, MutAnyOrigin],
):
    comptime assert NUM_CHUNKS % WARP_SIZE == 0
    var smem = unsafe_stack_allocation[
        NUM_CHUNKS * LANE_BYTES,
        UInt8,
        alignment=16,
        address_space=.SHARED,
    ]()
    var lane = Int(lane_id())
    var src_global = src.unsafe_address_space_cast[.GLOBAL]()
    # A wave instruction spans rows: lane chunks 0..35 come from one row and
    # 36..63 from the next, so every lane needs its own source address.
    comptime for r in range(NUM_CHUNKS // WARP_SIZE):
        var chunk = r * WARP_SIZE + lane
        var slot = chunk // CHUNKS_PER_ROW
        var col = chunk % CHUNKS_PER_ROW
        var row = Int(indices[slot])
        _global_load_to_lds[width=LANE_BYTES](
            src_global.unsafe_offset(row * ROW_BYTES).unsafe_offset(
                col * LANE_BYTES
            ),
            smem.unsafe_offset(r * WARP_SIZE * LANE_BYTES),
        )
    s_waitcnt[vmcnt=0]()
    comptime for r in range(NUM_CHUNKS // WARP_SIZE):
        var offset = (r * WARP_SIZE + lane) * LANE_BYTES
        comptime for i in range(LANE_BYTES):
            dst[offset + i] = smem[offset + i]


def _value(row: Int, byte: Int) -> UInt8:
    return UInt8((row * 7 + byte) % 251)


def main() raises:
    with DeviceContext() as ctx:
        comptime src_count = NUM_SRC_ROWS * ROW_BYTES
        comptime dst_count = NUM_ROWS * ROW_BYTES
        var host_src = ctx.enqueue_create_host_buffer[.uint8](src_count)
        var host_idx = ctx.enqueue_create_host_buffer[.int32](NUM_ROWS)
        var host_dst = ctx.enqueue_create_host_buffer[.uint8](dst_count)
        for row in range(NUM_SRC_ROWS):
            for b in range(ROW_BYTES):
                host_src[row * ROW_BYTES + b] = _value(row, b)
        # Scattered and out of order, including the first and last rows.
        for i in range(NUM_ROWS):
            host_idx[i] = Int32((i * 389 + 17) % NUM_SRC_ROWS)
        host_idx[3] = 0
        host_idx[11] = NUM_SRC_ROWS - 1

        var dev_src = ctx.enqueue_create_buffer[.uint8](src_count)
        var dev_idx = ctx.enqueue_create_buffer[.int32](NUM_ROWS)
        var dev_dst = ctx.enqueue_create_buffer[.uint8](dst_count)
        ctx.enqueue_copy(dev_src, host_src)
        ctx.enqueue_copy(dev_idx, host_idx)
        dev_dst.enqueue_fill(0)

        ctx.enqueue_function[_gather_rows_through_lds](
            dev_src,
            dev_idx,
            dev_dst,
            grid_dim=1,
            block_dim=ctx.default_device_info.warp_size,
        )
        ctx.enqueue_copy(host_dst, dev_dst)
        ctx.synchronize()

        for slot in range(NUM_ROWS):
            var row = Int(host_idx[slot])
            for b in range(ROW_BYTES):
                assert_equal(host_dst[slot * ROW_BYTES + b], _value(row, b))

        _ = dev_src^
        _ = dev_idx^
        _ = dev_dst^
