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
"""GPU probe for `StableAddrEngine`.

A `TileTensor` over a stable address slot is built and tiled on the host, then
passed to a kernel that writes through it. The kernel resolves the slot at the
point of use, so retargeting the slot at a different buffer and relaunching
the very same host-side tile writes the second buffer. That is the property
the engine exists for: host-side views of a `StableTensor` that a device graph
can still redirect per replay.

The slot is a one-element `uint64` device buffer, as the `DeviceGraphCache`
slot table uses; the test writes the target buffer's address into it with a
plain copy since no builder is involved.
"""

from extensibility.stable_addr_engine import (
    StableTileTensor,
    stable_tile_tensor,
)
from layout import TensorLayout, row_major

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, StableAddr

from std.testing import assert_equal

comptime _ROWS = 4
comptime _COLS = 8
comptime _NUM_ELEMENTS = _ROWS * _COLS

# The 2x4 sub-tile at tile coordinate (1, 1) of the 4x8 parent.
comptime _TILE_ROWS = 2
comptime _TILE_COLS = 4
comptime _TILE_ROW0 = 2
comptime _TILE_COL0 = 4


def _value(r: Int, c: Int) -> Float32:
    return Float32(10 * r + c + 1)


def fill_tile_through_slot_kernel[
    LayoutType: TensorLayout,
](tile: StableTileTensor[.float32, LayoutType]):
    """Writes a distinct value at every position of `tile`.

    Each store reads the slot for the data pointer and then applies the
    tile's own layout, so a correct result proves both the slot indirection
    and that the host-computed tile offset survived the launch.
    """
    if global_idx.x != 0:
        return
    var rows = Int(tile.dim[0]())
    var cols = Int(tile.dim[1]())
    for r in range(rows):
        for c in range(cols):
            tile.store((r, c), _value(r, c))


def _slot_over(mut slot: DeviceBuffer[.uint64]) -> StableAddr:
    return StableAddr(
        slot.unsafe_ptr()
        .unsafe_bitcast[Pointer[NoneType, MutUntrackedOrigin]]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def _point_slot_at(
    ctx: DeviceContext,
    slot: DeviceBuffer[.uint64],
    target: DeviceBuffer[.float32],
) raises:
    var value = ctx.enqueue_create_host_buffer[.uint64](1)
    value[0] = UInt64(Int(target.unsafe_ptr()))
    ctx.enqueue_copy(slot, value)


def _check_written(
    ctx: DeviceContext, buf: DeviceBuffer[.float32], written: Bool
) raises:
    var host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    for r in range(_ROWS):
        for c in range(_COLS):
            var in_tile = (
                written
                and _TILE_ROW0 <= r < _TILE_ROW0 + _TILE_ROWS
                and _TILE_COL0 <= c < _TILE_COL0 + _TILE_COLS
            )
            var expected = _value(
                r - _TILE_ROW0, c - _TILE_COL0
            ) if in_tile else Float32(-1)
            assert_equal(host[r * _COLS + c], expected)


def test_host_tile_writes_through_retargeted_slot(
    ctx: DeviceContext,
) raises:
    print("== test_host_tile_writes_through_retargeted_slot")

    var slot = ctx.enqueue_create_buffer[.uint64](1)
    var buf_a = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var buf_b = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    buf_a.enqueue_fill(Float32(-1))
    buf_b.enqueue_fill(Float32(-1))

    # Built and tiled on the host, before the slot has been pointed anywhere
    # the tile will ever be used with. The sub-tile's handle is the slot plus
    # a 20-element offset; no data pointer is involved yet.
    var full = stable_tile_tensor[.float32](
        _slot_over(slot), row_major(_ROWS, _COLS)
    )
    var sub = full.tile[_TILE_ROWS, _TILE_COLS](1, 1)
    assert_equal(sub._storage.offset, _TILE_ROW0 * _COLS + _TILE_COL0)

    comptime kernel = fill_tile_through_slot_kernel[sub.LayoutType]

    _point_slot_at(ctx, slot, buf_a)
    ctx.enqueue_function[kernel](sub, grid_dim=1, block_dim=1)
    _check_written(ctx, buf_a, written=True)
    _check_written(ctx, buf_b, written=False)

    # Same host tile, different slot contents: the write lands in `buf_b`.
    buf_a.enqueue_fill(Float32(-1))
    _point_slot_at(ctx, slot, buf_b)
    ctx.enqueue_function[kernel](sub, grid_dim=1, block_dim=1)
    _check_written(ctx, buf_a, written=False)
    _check_written(ctx, buf_b, written=True)
    # `sub` holds the slot's address, not a reference to the buffer, so without
    # this Mojo frees `slot` after `_point_slot_at` above, before the second
    # launch reads it. On Metal a device-buffer free releases the `MTLBuffer`
    # at once rather than after queued work, so that launch read a released
    # slot and sometimes found zero instead of `buf_b`'s address.
    _ = slot^


def main() raises:
    with DeviceContext() as ctx:
        test_host_tile_writes_through_retargeted_slot(ctx)
