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
"""Host-side checks for `StableAddrEngine`.

Nothing here dereferences a slot: a `StableAddrEngine` tile only reads its
slot on the device. What the host can do is build the tile and derive views
from it, and these tests pin down that every such view keeps the slot and
accumulates a plain element offset, so the slot indirection survives until a
kernel finally dereferences it.
"""

from extensibility.stable_addr_engine import (
    StableAddrEngine,
    stable_tile_tensor,
)
from layout import row_major
from max.gpu.host import StableAddr

from std.testing import TestSuite, assert_equal, assert_true


# `StableAddr` carries no element type, so the dtype and engine are spelled at
# the constructor; only the layout is inferred.
comptime _Engine = StableAddrEngine[element_width=1]


def _fake_slot(mut target: Int) -> StableAddr:
    """Builds a slot over host memory. It is never read, so the contents are
    irrelevant; only the slot's own address matters to these tests."""
    var contents = (
        Pointer(to=target)
        .unsafe_bitcast[NoneType]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var slot = Pointer(to=contents).unsafe_origin_cast[MutUntrackedOrigin]()
    return StableAddr(slot)


def test_constructor_starts_at_offset_zero() raises:
    var payload = 0
    var addr = _fake_slot(payload)
    var tile = stable_tile_tensor[.float32](addr, row_major[4, 8]())

    assert_equal(tile._storage.offset, 0)
    assert_true(tile._storage.addr.ptr == addr.ptr)
    assert_equal(tile.dim[0](), 4)
    assert_equal(tile.dim[1](), 8)


def test_tile_keeps_slot_and_accumulates_offset() raises:
    var payload = 0
    var addr = _fake_slot(payload)
    var tile = stable_tile_tensor[.float32](addr, row_major[4, 8]())

    # Row-major 4x8: tile (1, 1) of 2x4 starts at row 2, col 4 -> 2*8 + 4.
    var sub = tile.tile[2, 4](1, 1)
    assert_true(sub._storage.addr.ptr == addr.ptr)
    assert_equal(sub._storage.offset, 20)

    # A further tile composes: row 1, col 2 of the 2x4 sub-tile -> +8 + 2.
    var inner = sub.tile[1, 2](1, 1)
    assert_true(inner._storage.addr.ptr == addr.ptr)
    assert_equal(inner._storage.offset, 30)


def test_distance_is_offset_difference() raises:
    var payload = 0
    var addr = _fake_slot(payload)
    var tile = stable_tile_tensor[.float32](addr, row_major[4, 8]())
    var sub = tile.tile[2, 4](1, 1)

    var base = tile._unsafe_storage_cast[to_mut=False]()
    var ahead = sub._unsafe_storage_cast[to_mut=False]()
    assert_equal(_Engine.distance(ahead, base), 20)
    assert_equal(_Engine.distance(base, ahead), -20)


def test_immutable_view_keeps_handle() raises:
    var payload = 0
    var addr = _fake_slot(payload)
    var tile = stable_tile_tensor[.float32](addr, row_major[4, 8]())
    var sub = tile.tile[2, 4](1, 1)

    var view: type_of(sub).Immut = sub
    assert_true(view._storage.addr.ptr == addr.ptr)
    assert_equal(view._storage.offset, 20)


def test_type_name_names_engine() raises:
    var payload = 0
    var addr = _fake_slot(payload)
    var tile = stable_tile_tensor[.float32](addr, row_major[4, 8]())

    var name = type_of(tile).get_type_name()
    assert_true("StableAddrEngine[element_size=1]" in name, name)
    assert_true(tile._storage.addr.ptr == addr.ptr)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
