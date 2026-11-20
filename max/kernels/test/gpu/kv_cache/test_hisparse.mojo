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
"""Checks HiSparse resolve and gather against a host model of the hot buffer.

Request 0 decodes three rows over 1,000 tokens whose first six pages live only
in the host mirror. Request 1 is fully resident with no hot region. Two
indexer groups share request 0's hot pages, each through its own layer. Over
several steps every rewritten row must match the model's LRU exactly, and the
bytes behind every selected row must be that token's.
"""

from std.memory.alloc import unsafe_alloc
from std.random import random_ui64, seed
from std.testing import assert_equal, assert_true

from max.gpu.host import DeviceContext

from kv_cache.hisparse import (
    hisparse_gather_group,
    hisparse_resolve,
    hisparse_state_bytes,
)

comptime PAGE = 128
comptime HOT_PAGES = 2
comptime HOT_ROWS = HOT_PAGES * PAGE
comptime GROUPS = 2
comptime MAX_MISSES = 192
comptime HASH_CAPACITY = 512
comptime LAYERS = 3
comptime ROW_BYTES = 80
comptime PAGE_DATA = LAYERS * PAGE * ROW_BYTES
comptime PAGE_STRIDE = PAGE_DATA + 512
comptime NUM_PAGES = 32
comptime HOST_PAGES = 16
comptime LUT_COLS = 8
comptime TOP_K = 64
comptime VALID_K = 48
comptime BATCHES = 2
comptime ROWS = 4
comptime STEPS = 6

comptime CTX0 = 1000
comptime CTX1 = 300
comptime HOT_COLS = HOT_PAGES + 1


def _byte(request: Int, position: Int, layer: Int, k: Int) -> UInt8:
    return UInt8((position * 7 + layer * 31 + k * 3 + request * 101) & 0xFF)


def _fill_page(
    dst: MutPointer[UInt8, _],
    page_bytes: Int,
    page: Int,
    request: Int,
    logical_page: Int,
):
    for layer in range(LAYERS):
        for r in range(PAGE):
            var position = logical_page * PAGE + r
            for k in range(ROW_BYTES):
                dst[
                    unsafe_offset=page * page_bytes
                    + (layer * PAGE + r) * ROW_BYTES
                    + k
                ] = _byte(request, position, layer, k)


struct Model(Copyable, Movable):
    """The hot buffer's LRU for one group of request 0."""

    var order: List[Int]
    var keys: List[Int]
    var hits: Int
    var misses: Int

    def __init__(out self):
        self.order = List[Int]()
        self.keys = List[Int]()
        self.hits = 0
        self.misses = 0
        for s in range(HOT_ROWS):
            self.order.append(s)
            self.keys.append(0)

    def step(
        mut self,
        positions: List[Int],
        host_only: List[Bool],
        hot_pages: List[Int],
    ) -> Dict[Int, Int]:
        """Applies one resolve; returns each host-only position's hot row."""
        var needed = Dict[Int, Bool]()
        for p in positions:
            if p >= 0 and host_only[p // PAGE]:
                needed[p] = True
        var slot_of = Dict[Int, Int]()
        var hit = List[Bool]()
        for s in range(HOT_ROWS):
            var held = self.keys[s] - 1
            var is_hit = held >= 0 and held in needed
            hit.append(is_hit)
            if is_hit:
                slot_of[held] = s
                self.hits += 1
        var misses = List[Int]()
        var queued = Dict[Int, Bool]()
        for p in positions:
            if p >= 0 and host_only[p // PAGE]:
                if p not in slot_of and p not in queued:
                    queued[p] = True
                    misses.append(p)
        var eligible = List[Int]()
        var hits = List[Int]()
        for s in self.order:
            if hit[s]:
                hits.append(s)
            else:
                eligible.append(s)
        var m = len(misses)
        self.misses += m
        var next_order = List[Int]()
        for i in range(m, len(eligible)):
            next_order.append(eligible[i])
        for j in range(m):
            var slot = eligible[j]
            self.keys[slot] = misses[j] + 1
            slot_of[misses[j]] = slot
            next_order.append(slot)
        for s in hits:
            next_order.append(s)
        self.order = next_order^
        var rows = Dict[Int, Int]()
        for entry in slot_of.items():
            var slot = entry.value
            rows[entry.key] = hot_pages[slot // PAGE] * PAGE + slot % PAGE
        return rows^


def main() raises:
    comptime assert (
        hisparse_state_bytes[
            PAGE, HOT_PAGES, GROUPS, MAX_MISSES, HASH_CAPACITY
        ]()
        <= PAGE_DATA
    )
    seed(1695)

    # Request 0: pages 0-5 host only, 6 and 7 resident on 12 and 13.
    var lut = unsafe_alloc[UInt32](BATCHES * LUT_COLS)
    var host_table = unsafe_alloc[Int32](BATCHES * LUT_COLS)
    var hot = unsafe_alloc[UInt32](BATCHES * HOT_COLS)
    var host_only = List[Bool]()
    for p in range(LUT_COLS):
        var resident = p >= 6
        lut[p] = UInt32(12 + p - 6) if resident else 0
        host_table[p] = Int32(3 + p)
        host_only.append(not resident)
        # Request 1: three resident pages, no host copy.
        lut[LUT_COLS + p] = UInt32(14 + p) if p < 3 else 0
        host_table[LUT_COLS + p] = -1
    var hot_pages: List[Int] = [20, 21]
    hot[0] = 20
    hot[1] = 21
    hot[2] = 22
    for c in range(HOT_COLS):
        hot[HOT_COLS + c] = 0
    var row_offsets = unsafe_alloc[UInt32](BATCHES + 1)
    row_offsets[0] = 0
    row_offsets[1] = 3
    row_offsets[2] = 4

    var cache = unsafe_alloc[UInt8](NUM_PAGES * PAGE_STRIDE)
    for i in range(NUM_PAGES * PAGE_STRIDE):
        cache[i] = 0

    for p in range(6, 8):
        _fill_page(cache, PAGE_STRIDE, Int(lut[p]), 0, p)
    for p in range(3):
        _fill_page(cache, PAGE_STRIDE, Int(lut[LUT_COLS + p]), 1, p)

    var logical = unsafe_alloc[Int32](ROWS * TOP_K)
    var physical = unsafe_alloc[Int32](ROWS * TOP_K)
    var resolved = unsafe_alloc[Int32](ROWS * TOP_K)
    var cache_back = unsafe_alloc[UInt8](NUM_PAGES * PAGE_STRIDE)

    var models = List[Model]()
    for _ in range(GROUPS):
        models.append(Model())
    var previous = List[List[Int]]()
    for _ in range(GROUPS):
        previous.append(List[Int]())

    with DeviceContext() as ctx:
        var host_mirror = ctx.enqueue_create_host_buffer[.uint8](
            HOST_PAGES * PAGE_DATA
        )
        ctx.synchronize()
        for p in range(LUT_COLS):
            _fill_page(host_mirror.unsafe_ptr(), PAGE_DATA, 3 + p, 0, p)

        var d_cache = ctx.enqueue_create_buffer[.uint8](NUM_PAGES * PAGE_STRIDE)
        var d_lut = ctx.enqueue_create_buffer[.uint32](BATCHES * LUT_COLS)
        var d_host_table = ctx.enqueue_create_buffer[.int32](BATCHES * LUT_COLS)
        var d_hot = ctx.enqueue_create_buffer[.uint32](BATCHES * HOT_COLS)
        var d_row_offsets = ctx.enqueue_create_buffer[.uint32](BATCHES + 1)
        var d_logical = ctx.enqueue_create_buffer[.int32](ROWS * TOP_K)
        var d_physical = ctx.enqueue_create_buffer[.int32](ROWS * TOP_K)
        ctx.enqueue_copy(d_cache, cache)
        ctx.enqueue_copy(d_lut, lut)
        ctx.enqueue_copy(d_host_table, host_table)
        ctx.enqueue_copy(d_hot, hot)
        ctx.enqueue_copy(d_row_offsets, row_offsets)

        for _ in range(STEPS):
            for group in range(GROUPS):
                # Request 0: three rows, half of each drawn from the group's
                # last selections so the LRU sees reuse.
                var chosen = List[Int]()
                for row in range(ROWS):
                    var ctx_len = CTX0 if row < 3 else CTX1
                    var taken = Dict[Int, Bool]()
                    var valid = VALID_K if row < 3 else TOP_K
                    var k = 0
                    while k < valid:
                        var p: Int
                        var prev = previous[group].copy()
                        if row < 3 and len(prev) > 0 and random_ui64(0, 1) == 0:
                            p = prev[Int(random_ui64(0, UInt64(len(prev) - 1)))]
                        else:
                            p = Int(random_ui64(0, UInt64(ctx_len - 1)))
                        if p in taken:
                            continue
                        taken[p] = True
                        logical[row * TOP_K + k] = Int32(p)
                        if row < 3:
                            chosen.append(p)
                        k += 1
                    for kk in range(valid, TOP_K):
                        logical[row * TOP_K + kk] = -1
                previous[group] = chosen^

                # What the indexer's remap writes: a null page yields a row
                # inside page 0.
                for row in range(ROWS):
                    var b = 0 if row < 3 else 1
                    for k in range(TOP_K):
                        var p = Int(logical[row * TOP_K + k])
                        if p < 0:
                            physical[row * TOP_K + k] = -1
                        else:
                            var page = Int(lut[b * LUT_COLS + p // PAGE])
                            physical[row * TOP_K + k] = Int32(
                                page * PAGE + p % PAGE
                            )

                var request0 = List[Int]()
                for i in range(3 * TOP_K):
                    request0.append(Int(logical[i]))
                var expected_rows = models[group].step(
                    request0, host_only, hot_pages
                )

                ctx.enqueue_copy(d_logical, logical)
                ctx.enqueue_copy(d_physical, physical)
                hisparse_resolve[
                    PAGE, HOT_PAGES, GROUPS, MAX_MISSES, HASH_CAPACITY
                ](
                    d_physical.unsafe_ptr(),
                    d_logical.unsafe_ptr(),
                    d_row_offsets.unsafe_ptr(),
                    d_lut.unsafe_ptr(),
                    d_host_table.unsafe_ptr(),
                    d_hot.unsafe_ptr(),
                    d_cache.unsafe_ptr(),
                    PAGE_STRIDE,
                    PAGE_DATA,
                    BATCHES,
                    LUT_COLS,
                    TOP_K,
                    group,
                    ctx,
                )
                if group == 0:
                    hisparse_gather_group[
                        PAGE, HOT_PAGES, GROUPS, MAX_MISSES, HASH_CAPACITY
                    ](
                        d_cache.unsafe_ptr(),
                        host_mirror.unsafe_ptr(),
                        d_hot.unsafe_ptr(),
                        PAGE_STRIDE,
                        PAGE_DATA,
                        0,
                        ROW_BYTES,
                        2,
                        PAGE * ROW_BYTES,
                        BATCHES,
                        ctx,
                    )
                else:
                    hisparse_gather_group[
                        PAGE, HOT_PAGES, GROUPS, MAX_MISSES, HASH_CAPACITY
                    ](
                        d_cache.unsafe_ptr(),
                        host_mirror.unsafe_ptr(),
                        d_hot.unsafe_ptr(),
                        PAGE_STRIDE,
                        PAGE_DATA,
                        2 * PAGE * ROW_BYTES,
                        ROW_BYTES,
                        1,
                        PAGE * ROW_BYTES,
                        BATCHES,
                        ctx,
                    )
                ctx.enqueue_copy(resolved, d_physical)
                ctx.enqueue_copy(cache_back, d_cache)
                ctx.synchronize()

                for row in range(ROWS):
                    var b = 0 if row < 3 else 1
                    for k in range(TOP_K):
                        var i = row * TOP_K + k
                        var p = Int(logical[i])
                        var got = Int(resolved[i])
                        if p < 0:
                            assert_equal(got, -1, "unused slot changed")
                            continue
                        if b == 0 and host_only[p // PAGE]:
                            assert_equal(
                                got, expected_rows[p], "hot row differs"
                            )
                        else:
                            assert_equal(
                                got,
                                Int(physical[i]),
                                "resident row changed",
                            )
                        var first_layer = 0 if group == 0 else 2
                        var last_layer = 2 if group == 0 else LAYERS
                        for layer in range(first_layer, last_layer):
                            var at = (got // PAGE) * PAGE_STRIDE + (
                                layer * PAGE + got % PAGE
                            ) * ROW_BYTES
                            for kb in range(ROW_BYTES):
                                assert_equal(
                                    cache_back[at + kb],
                                    _byte(b, p, layer, kb),
                                    "row bytes differ",
                                )

                # Nothing may land in a page's padding.
                for page in range(NUM_PAGES):
                    var at = page * PAGE_STRIDE + PAGE_DATA
                    for kb in range(PAGE_STRIDE - PAGE_DATA):
                        assert_equal(cache_back[at + kb], 0, "pad written")

        # Reused selections must have hit and new ones missed, or the test
        # exercised only one path.
        for group in range(GROUPS):
            assert_true(models[group].hits > 0, "no hot-buffer hits")
            assert_true(models[group].misses > HOT_ROWS, "no evictions")

        _ = d_cache^
        _ = d_lut^
        _ = d_host_table^
        _ = d_hot^
        _ = d_row_offsets^
        _ = d_logical^
        _ = d_physical^
        _ = host_mirror^

    lut.unsafe_free()
    host_table.unsafe_free()
    hot.unsafe_free()
    row_offsets.unsafe_free()
    cache.unsafe_free()
    logical.unsafe_free()
    physical.unsafe_free()
    resolved.unsafe_free()
    cache_back.unsafe_free()
