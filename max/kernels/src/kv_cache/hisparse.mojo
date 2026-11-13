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
"""HiSparse hot-buffer kernels for sparse MLA over a partly host-resident cache.

A request in mixed residency has latent pages whose lookup table entry is the
null page 0: their only copy lives in a pinned host mirror. The request also
holds a hot region of `hot_pages` latent pages plus one state page, listed in
its row of the hot table. Sparse MLA reads hot rows like any other cache rows,
because they are rows of the same paged cache.

`hisparse_resolve` runs once per indexer layer, after the indexer has written
each selection's physical row. It rewrites the selections on null pages to hot
rows. A selection already held in a hot slot reads it, and a miss takes the
least recently used slot that no row of this step needs. Misses are numbered by
their first reference in (row, top-k) order, so the plan is deterministic. They
are queued in the state page as `(hot row, host row)` pairs, and
`hisparse_gather_group` copies them for every layer sharing an indexer's
selections in one launch, before the first layer's attention.

Rows on resident pages are left as the indexer wrote them, so the packing that
sparse MLA expects (valid rows first, `-1` after) is unchanged. A request with
no hot region (a zero state page) is skipped.

The state page is zero when leased. Everything the kernels keep across steps
is laid out so that zero reads as empty: slot keys are `position + 1`, slot hit
marks carry a per-call epoch that starts at 1, and a group's LRU order is
initialized on its first call. The hash of one call's selections lives in
shared memory and is cleared at the start of the call, since nothing reads it
afterwards.
"""

from std.atomic import Atomic, Ordering
from std.memory import unsafe_stack_allocation
from std.os import abort
from std.utils import StaticTuple

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.memory import external_memory
from max.gpu.primitives import block
from max.gpu.sync import barrier


@inline(.always)
def _align16(n: Int) -> Int:
    return (n + 15) // 16 * 16


struct HiSparseStateLayout[
    page_size: Int,
    hot_pages: Int,
    num_groups: Int,
    max_misses: Int,
    hash_capacity: Int,
]:
    """Byte layout of a request's HiSparse state page.

    Parameters:
        page_size: Rows per cache page.
        hot_pages: Pages in the hot buffer; it holds
            `hot_pages * page_size` slots.
        num_groups: Indexer layers that keep their own LRU state.
        max_misses: Most distinct host-only selections one request may make
            in one call, summed over its rows.
        hash_capacity: Entries in the per-call hash of those selections; a
            power of two above `max_misses`.
    """

    comptime hot_rows = Self.hot_pages * Self.page_size
    """Slots in the hot buffer."""
    comptime EPOCH = 0
    """`UInt32` call counter, bumped by every resolve."""
    comptime MISS_COUNT = 4
    """`Int32` misses queued by the last resolve."""
    comptime MISS_LIST = 64
    """`Int32[max_misses, 2]` of `(hot row, host row)` pairs."""
    comptime MISS_HASH = _align16(Self.MISS_LIST + 8 * Self.max_misses)
    """`Int32[max_misses]` hash entry of each miss, scratch."""
    comptime VICTIMS = _align16(Self.MISS_HASH + 4 * Self.max_misses)
    """`Int32[max_misses]` slot taken by each miss, scratch."""
    comptime HITS = _align16(Self.VICTIMS + 4 * Self.max_misses)
    """`UInt32[hot_rows]`, the epoch of the call that last hit each slot."""
    comptime GROUPS = _align16(Self.HITS + 4 * Self.hot_rows)
    """Start of the per-group LRU state."""
    comptime GROUP_INIT = 0
    """`UInt32`, nonzero once the group's order is initialized."""
    comptime GROUP_WHICH = 4
    """`UInt32`, which order buffer is current."""
    comptime GROUP_KEYS = 16
    """`Int32[hot_rows]`, the position each slot holds plus one, or 0."""
    comptime GROUP_ORDER0 = Self.GROUP_KEYS + 4 * Self.hot_rows
    """`Int16[hot_rows]` slots, least recently used first."""
    comptime GROUP_ORDER1 = Self.GROUP_ORDER0 + 2 * Self.hot_rows
    """The other order buffer."""
    comptime GROUP_STRIDE = _align16(Self.GROUP_ORDER1 + 2 * Self.hot_rows)
    """Bytes per group."""
    comptime BYTES = Self.GROUPS + Self.num_groups * Self.GROUP_STRIDE
    """Bytes the state needs; must fit one cache page."""


@inline(.always)
def _hash_slot[capacity: Int](position: Int32) -> Int:
    # Knuth multiplicative hash, scaled onto [0, capacity) by a multiply-high.
    var mixed = UInt32(position) * UInt32(2654435761)
    return Int((UInt64(mixed) * UInt64(capacity)) >> 32)


@inline(.always)
def _hash_insert[
    capacity: Int
](
    keys: MutPointer[UInt32, address_space=.SHARED, ...],
    vals: MutPointer[Int32, address_space=.SHARED, ...],
    position: Int32,
) -> Int:
    var key = UInt32(position) + 1
    var h = _hash_slot[capacity](position)
    for _ in range(capacity):
        var current = Atomic.load[ordering=Ordering.RELAXED](
            keys.unsafe_offset(h).as_imm()
        )
        if current == key:
            return h
        if current == 0:
            var expected = UInt32(0)
            if Atomic.compare_exchange[
                success_ordering=Ordering.RELAXED,
                failure_ordering=Ordering.RELAXED,
            ](keys.unsafe_offset(h), expected, key):
                vals[unsafe_offset=h] = -1
                return h
            if expected == key:
                return h
        h = (h + 1) & (capacity - 1)
    abort("hisparse: selection hash is full")


@inline(.always)
def _first_tag(index: Int) -> UInt32:
    return ~UInt32(index)


@inline(.always)
def _is_first(tag: UInt32, index: Int) -> Bool:
    return tag == _first_tag(index)


@inline(.always)
def _hash_find[
    capacity: Int
](
    keys: MutPointer[UInt32, address_space=.SHARED, ...],
    position: Int32,
) -> Int:
    var key = UInt32(position) + 1
    var h = _hash_slot[capacity](position)
    for _ in range(capacity):
        var current = Atomic.load[ordering=Ordering.RELAXED](
            keys.unsafe_offset(h).as_imm()
        )
        if current == key:
            return h
        if current == 0:
            return -1
        h = (h + 1) & (capacity - 1)
    return -1


@inline(.always)
def _paged_row(
    table_row: ImmPointer[UInt32, ImmUntrackedOrigin],
    slot: Int,
    page_size: Int,
) -> Int32:
    return Int32(
        Int(table_row[unsafe_offset=slot // page_size]) * page_size
        + slot % page_size
    )


comptime RESOLVE_BLOCK = 1024
"""Threads per request in `hisparse_resolve`."""


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(RESOLVE_BLOCK))
)
@__name(t"hisparse_resolve_kernel_{page_size}_{hot_pages}_{num_groups}")
def _hisparse_resolve_kernel[
    page_size: Int,
    hot_pages: Int,
    num_groups: Int,
    max_misses: Int,
    hash_capacity: Int,
](
    physical: MutPointer[Int32, MutUntrackedOrigin],
    logical: ImmPointer[Int32, ImmUntrackedOrigin],
    row_offsets: ImmPointer[UInt32, ImmUntrackedOrigin],
    lookup_table: ImmPointer[UInt32, ImmUntrackedOrigin],
    host_table: ImmPointer[Int32, ImmUntrackedOrigin],
    hot_table: ImmPointer[UInt32, ImmUntrackedOrigin],
    cache: MutPointer[UInt8, MutUntrackedOrigin],
    page_stride_bytes_arg: Int64,
    lut_cols_arg: Int32,
    top_k_arg: Int32,
    group_arg: Int32,
):
    comptime L = HiSparseStateLayout[
        page_size, hot_pages, num_groups, max_misses, hash_capacity
    ]
    comptime R = L.hot_rows
    comptime HOT_COLS = hot_pages + 1
    comptime BLOCK = RESOLVE_BLOCK

    var page_stride_bytes = Int(page_stride_bytes_arg)
    var lut_cols = Int(lut_cols_arg)
    var top_k = Int(top_k_arg)
    var group = Int(group_arg)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var hot_row = hot_table.unsafe_offset(b * HOT_COLS)
    var state_page = Int(hot_row[unsafe_offset=hot_pages])
    if state_page == 0:
        return

    var base = cache.unsafe_offset(state_page * page_stride_bytes)
    var epoch_ptr = base.unsafe_offset(L.EPOCH).unsafe_bitcast[UInt32]()
    var miss_count = base.unsafe_offset(L.MISS_COUNT).unsafe_bitcast[Int32]()
    var miss_list = base.unsafe_offset(L.MISS_LIST).unsafe_bitcast[Int32]()
    var miss_hash = base.unsafe_offset(L.MISS_HASH).unsafe_bitcast[Int32]()
    var victims = base.unsafe_offset(L.VICTIMS).unsafe_bitcast[Int32]()
    var hash_mem = external_memory[
        UInt32,
        address_space=.SHARED,
        alignment=16,
        name="hisparse_resolve_hash",
    ]()
    var keys = hash_mem
    var vals = hash_mem.unsafe_offset(hash_capacity).unsafe_bitcast[Int32]()
    var firsts = hash_mem.unsafe_offset(2 * hash_capacity)
    var hits = base.unsafe_offset(L.HITS).unsafe_bitcast[UInt32]()
    var gbase = base.unsafe_offset(L.GROUPS + group * L.GROUP_STRIDE)
    var g_init = gbase.unsafe_offset(L.GROUP_INIT).unsafe_bitcast[UInt32]()
    var g_which = gbase.unsafe_offset(L.GROUP_WHICH).unsafe_bitcast[UInt32]()
    var slot_keys = gbase.unsafe_offset(L.GROUP_KEYS).unsafe_bitcast[Int32]()
    var order0 = gbase.unsafe_offset(L.GROUP_ORDER0).unsafe_bitcast[Int16]()
    var order1 = gbase.unsafe_offset(L.GROUP_ORDER1).unsafe_bitcast[Int16]()

    # [epoch, group initialized, which order, hits, chunk total]
    var shared = unsafe_stack_allocation[8, UInt32, address_space=.SHARED]()
    for h in range(tid, hash_capacity, BLOCK):
        keys[unsafe_offset=h] = 0
        firsts[unsafe_offset=h] = 0
    if tid == 0:
        var epoch = epoch_ptr[unsafe_offset=0] + 1
        if epoch == 0:
            epoch = 1
        epoch_ptr[unsafe_offset=0] = epoch
        shared[unsafe_offset=0] = epoch
        shared[unsafe_offset=1] = g_init[unsafe_offset=0]
        shared[unsafe_offset=2] = g_which[unsafe_offset=0]
        shared[unsafe_offset=3] = 0
    barrier()
    var epoch = shared[unsafe_offset=0]
    var initialized = shared[unsafe_offset=1] != 0
    var which = Int(shared[unsafe_offset=2]) if initialized else 0
    if not initialized:
        for s in range(tid, R, BLOCK):
            order0[unsafe_offset=s] = Int16(s)
            slot_keys[unsafe_offset=s] = 0
    var cur_order = order1 if which == 1 else order0
    var next_order = order0 if which == 1 else order1
    barrier()

    var row_lo = Int(row_offsets[unsafe_offset=b])
    var row_hi = Int(row_offsets[unsafe_offset=b + 1])
    var first = row_lo * top_k
    var n = (row_hi - row_lo) * top_k
    var lut_row = lookup_table.unsafe_offset(b * lut_cols)

    # 1. Collect the distinct selections on host-only pages, and where each
    # is first referenced in (row, top-k) order.
    for i in range(tid, n, BLOCK):
        var position = logical[unsafe_offset=first + i]
        if position < 0:
            continue
        var page = Int(position) // page_size
        if page >= lut_cols or lut_row[unsafe_offset=page] != 0:
            continue
        var h = _hash_insert[hash_capacity](keys, vals, position)
        # Host-only physical rows are rewritten below, so they can hold the
        # hash slot between the insertion and final lookup.
        physical[unsafe_offset=first + i] = Int32(h)
        Atomic.max[ordering=Ordering.RELAXED](
            firsts.unsafe_offset(h), _first_tag(i)
        )
    barrier()

    # 2. Hits: slots whose position this step selects.
    var hit_count = shared.unsafe_offset(3)
    for s in range(tid, R, BLOCK):
        var key = slot_keys[unsafe_offset=s]
        if key > 0:
            var h = _hash_find[hash_capacity](keys, key - 1)
            if h >= 0:
                vals[unsafe_offset=h] = Int32(s)
                hits[unsafe_offset=s] = epoch
                _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
                    hit_count, UInt32(1)
                )
    barrier()

    # 3. Number the misses by first reference, so the plan does not depend
    # on the order threads inserted them.
    var chunk_total = shared.unsafe_offset(4)
    var misses = 0
    for start in range(0, n, BLOCK):
        var i = start + tid
        var is_miss: Int32 = 0
        var h = -1
        if i < n:
            var position = logical[unsafe_offset=first + i]
            if position >= 0:
                var page = Int(position) // page_size
                if page < lut_cols and lut_row[unsafe_offset=page] == 0:
                    h = Int(physical[unsafe_offset=first + i])
                    if vals[unsafe_offset=h] < 0 and _is_first(
                        Atomic.load[ordering=Ordering.RELAXED](
                            firsts.unsafe_offset(h).as_imm()
                        ),
                        i,
                    ):
                        is_miss = 1
        var rank = block.prefix_sum[block_size=BLOCK, exclusive=True](is_miss)
        if is_miss == 1 and misses + Int(rank) < max_misses:
            miss_hash[unsafe_offset=misses + Int(rank)] = Int32(h)
        if tid == BLOCK - 1:
            chunk_total[unsafe_offset=0] = UInt32(rank + is_miss)
        barrier()
        misses += Int(chunk_total[unsafe_offset=0])
        barrier()

    var eligible = R - Int(hit_count[unsafe_offset=0])
    if misses > max_misses or misses > eligible:
        abort("hisparse: more misses than the hot buffer can take")

    # 4. Walk the LRU order: the first `misses` slots no row hit become the
    # misses' slots, and the next order is [untouched | filled | hit].
    var untouched = eligible - misses
    var seen = 0
    for start in range(0, R, BLOCK):
        var i = start + tid
        var is_eligible: Int32 = 0
        var slot = 0
        if i < R:
            slot = Int(cur_order[unsafe_offset=i])
            if hits[unsafe_offset=slot] != epoch:
                is_eligible = 1
        var local = block.prefix_sum[block_size=BLOCK, exclusive=True](
            is_eligible
        )
        if i < R:
            var rank = seen + Int(local)
            var new_pos: Int
            if is_eligible == 1:
                if rank < misses:
                    victims[unsafe_offset=rank] = Int32(slot)
                    new_pos = untouched + rank
                else:
                    new_pos = rank - misses
            else:
                new_pos = untouched + misses + (i - rank)
            next_order[unsafe_offset=new_pos] = Int16(slot)
        if tid == BLOCK - 1:
            chunk_total[unsafe_offset=0] = UInt32(local + is_eligible)
        barrier()
        seen += Int(chunk_total[unsafe_offset=0])
        barrier()

    # 5. Fill the victims and queue their copies.
    var host_row = host_table.unsafe_offset(b * lut_cols)
    for j in range(tid, misses, BLOCK):
        var h = Int(miss_hash[unsafe_offset=j])
        var position = Int32(keys[unsafe_offset=h] - 1)
        var slot = Int(victims[unsafe_offset=j])
        slot_keys[unsafe_offset=slot] = position + 1
        vals[unsafe_offset=h] = Int32(slot)
        var host_page = host_row[unsafe_offset=Int(position) // page_size]
        var src = Int32(-1)
        if host_page >= 0:
            src = host_page * Int32(page_size) + position % Int32(page_size)
        miss_list[unsafe_offset=2 * j] = _paged_row(hot_row, slot, page_size)
        miss_list[unsafe_offset=2 * j + 1] = src
    if tid == 0:
        miss_count[unsafe_offset=0] = Int32(misses)
        g_init[unsafe_offset=0] = 1
        g_which[unsafe_offset=0] = UInt32(1 - which)
    barrier()

    # 6. Point every host-only selection at its slot.
    for i in range(tid, n, BLOCK):
        var position = logical[unsafe_offset=first + i]
        if position < 0:
            continue
        var page = Int(position) // page_size
        if page >= lut_cols or lut_row[unsafe_offset=page] != 0:
            continue
        var h = Int(physical[unsafe_offset=first + i])
        var slot = Int(vals[unsafe_offset=h])
        physical[unsafe_offset=first + i] = _paged_row(hot_row, slot, page_size)


comptime GATHER_BLOCK = 256
"""Threads per gather block; one warp copies one row."""
comptime GATHER_BLOCKS_PER_REQUEST = 16
"""Gather blocks per request; they stride over its misses."""


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GATHER_BLOCK))
)
@__name(t"hisparse_gather_kernel_{page_size}_{hot_pages}_{num_groups}")
def _hisparse_gather_kernel[
    page_size: Int,
    hot_pages: Int,
    num_groups: Int,
    max_misses: Int,
    hash_capacity: Int,
](
    cache: MutPointer[UInt8, MutUntrackedOrigin],
    host: ImmPointer[UInt8, ImmUntrackedOrigin],
    hot_table: ImmPointer[UInt32, ImmUntrackedOrigin],
    page_stride_bytes_arg: Int64,
    host_page_bytes_arg: Int64,
    layer_offset_bytes_arg: Int64,
    row_bytes_arg: Int32,
    layer_count_arg: Int32,
    layer_stride_bytes_arg: Int64,
):
    comptime L = HiSparseStateLayout[
        page_size, hot_pages, num_groups, max_misses, hash_capacity
    ]
    comptime HOT_COLS = hot_pages + 1

    var page_stride_bytes = Int(page_stride_bytes_arg)
    var host_page_bytes = Int(host_page_bytes_arg)
    var layer_offset_bytes = Int(layer_offset_bytes_arg)
    var row_bytes = Int(row_bytes_arg)
    var layer_count = Int(layer_count_arg)
    var layer_stride_bytes = Int(layer_stride_bytes_arg)
    var b = Int(block_idx.x)
    var state_page = Int(hot_table[unsafe_offset=b * HOT_COLS + hot_pages])
    if state_page == 0:
        return
    var base = cache.unsafe_offset(state_page * page_stride_bytes)
    var misses = Int(
        base.unsafe_offset(L.MISS_COUNT).unsafe_bitcast[Int32]()[
            unsafe_offset=0
        ]
    )
    var miss_list = base.unsafe_offset(L.MISS_LIST).unsafe_bitcast[Int32]()

    var warps = Int(block_dim.x) // WARP_SIZE
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var vectors = row_bytes // 16
    var j = Int(block_idx.y) * warps + warp
    var stride = Int(grid_dim.y) * warps
    while j < misses:
        var dst = Int(miss_list[unsafe_offset=2 * j])
        var src = Int(miss_list[unsafe_offset=2 * j + 1])
        if src >= 0:
            var dst_row = cache.unsafe_offset(
                (dst // page_size) * page_stride_bytes
                + layer_offset_bytes
                + (dst % page_size) * row_bytes
            )
            var src_row = host.unsafe_offset(
                (src // page_size) * host_page_bytes
                + layer_offset_bytes
                + (src % page_size) * row_bytes
            )
            for layer in range(layer_count):
                for v in range(lane, vectors, WARP_SIZE):
                    var chunk = src_row.unsafe_offset(
                        layer * layer_stride_bytes + 16 * v
                    ).unsafe_load[width=16, alignment=16]()
                    dst_row.unsafe_offset(
                        layer * layer_stride_bytes + 16 * v
                    ).unsafe_store[width=16, alignment=16](0, chunk)
        j += stride


def hisparse_state_bytes[
    page_size: Int,
    hot_pages: Int,
    num_groups: Int,
    max_misses: Int,
    hash_capacity: Int,
]() -> Int:
    """Returns the bytes a request's state page needs.

    Parameters:
        page_size: Rows per cache page.
        hot_pages: Pages in the hot buffer.
        num_groups: Indexer layers that keep their own LRU state.
        max_misses: Most distinct host-only selections per call.
        hash_capacity: Entries in the per-call selection hash.

    Returns:
        The state page's byte size, which must not exceed a cache page.
    """
    return HiSparseStateLayout[
        page_size, hot_pages, num_groups, max_misses, hash_capacity
    ].BYTES


def hisparse_resolve[
    page_size: Int,
    hot_pages: Int,
    num_groups: Int,
    max_misses: Int,
    hash_capacity: Int,
](
    physical: MutPointer[Int32, _],
    logical: ImmPointer[Int32, _],
    row_offsets: ImmPointer[UInt32, _],
    lookup_table: ImmPointer[UInt32, _],
    host_table: ImmPointer[Int32, _],
    hot_table: ImmPointer[UInt32, _],
    cache: MutPointer[UInt8, _],
    page_stride_bytes: Int,
    page_data_bytes: Int,
    num_batches: Int,
    lut_cols: Int,
    top_k: Int,
    group: Int,
    ctx: DeviceContext,
) raises:
    """Rewrites selections on host-only pages to hot-buffer rows.

    Parameters:
        page_size: Rows per cache page.
        hot_pages: Pages in each request's hot buffer.
        num_groups: Indexer layers that keep their own LRU state.
        max_misses: Most distinct host-only selections one request may make
            per call; covers every row's top-k under speculation.
        hash_capacity: Entries in the per-call selection hash, a power of two
            above `max_misses`.

    Args:
        physical: `[tokens, top_k]` physical rows from the indexer, rewritten
            in place.
        logical: `[tokens, top_k]` logical positions of the same selections.
        row_offsets: `[num_batches + 1]` ragged token offsets.
        lookup_table: `[num_batches, lut_cols]` cache pages; 0 marks a
            host-only page.
        host_table: `[num_batches, lut_cols]` host mirror pages, `-1` where
            a page has no copy.
        hot_table: `[num_batches, hot_pages + 1]` hot pages and, last, the
            state page; a zero state page means no hot region.
        cache: The cache's bytes; pages are `page_stride_bytes` apart.
        page_stride_bytes: Bytes from one cache page to the next.
        page_data_bytes: Bytes of data in a page, which bound the state.
        num_batches: Requests in the batch.
        lut_cols: Columns of the lookup and host tables.
        top_k: Selections per token.
        group: This indexer layer's LRU state index.
        ctx: Device context to enqueue on.
    """
    comptime assert (
        hash_capacity & (hash_capacity - 1)
    ) == 0, "hash_capacity must be a power of two"
    comptime assert hash_capacity > max_misses, "hash must exceed max_misses"
    comptime assert (
        hot_pages * page_size <= 32768
    ), "hot slots are numbered in int16"
    comptime L = HiSparseStateLayout[
        page_size, hot_pages, num_groups, max_misses, hash_capacity
    ]
    if L.BYTES > page_data_bytes:
        raise Error(
            "hisparse state needs ",
            L.BYTES,
            " bytes, more than a cache page's ",
            page_data_bytes,
        )
    if group < 0 or group >= num_groups:
        raise Error("hisparse group ", group, " out of ", num_groups)
    if num_batches == 0:
        return
    comptime kernel = _hisparse_resolve_kernel[
        page_size, hot_pages, num_groups, max_misses, hash_capacity
    ]
    ctx.enqueue_function[kernel](
        physical.unsafe_origin_cast[MutUntrackedOrigin](),
        logical.unsafe_origin_cast[ImmUntrackedOrigin](),
        row_offsets.unsafe_origin_cast[ImmUntrackedOrigin](),
        lookup_table.unsafe_origin_cast[ImmUntrackedOrigin](),
        host_table.unsafe_origin_cast[ImmUntrackedOrigin](),
        hot_table.unsafe_origin_cast[ImmUntrackedOrigin](),
        cache.unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(page_stride_bytes),
        Int32(lut_cols),
        Int32(top_k),
        Int32(group),
        grid_dim=num_batches,
        block_dim=RESOLVE_BLOCK,
        shared_mem_bytes=3 * hash_capacity * 4,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(3 * hash_capacity * 4)
        ),
    )


def hisparse_gather_group[
    page_size: Int,
    hot_pages: Int,
    num_groups: Int,
    max_misses: Int,
    hash_capacity: Int,
](
    cache: MutPointer[UInt8, _],
    host: ImmPointer[UInt8, _],
    hot_table: ImmPointer[UInt32, _],
    page_stride_bytes: Int,
    host_page_bytes: Int,
    first_layer_offset_bytes: Int,
    row_bytes: Int,
    layer_count: Int,
    layer_stride_bytes: Int,
    num_batches: Int,
    ctx: DeviceContext,
) raises:
    """Copies the last resolve's misses for consecutive layers in one launch.

    Parameters:
        page_size: Rows per cache page.
        hot_pages: Pages in each request's hot buffer.
        num_groups: Indexer layers that keep their own LRU state.
        max_misses: As passed to `hisparse_resolve`.
        hash_capacity: As passed to `hisparse_resolve`.

    Args:
        cache: The cache's bytes; pages are `page_stride_bytes` apart.
        host: The host mirror's bytes, readable by the device; pages are
            `host_page_bytes` apart and laid out like cache pages.
        hot_table: `[num_batches, hot_pages + 1]`, as passed to the resolve.
        page_stride_bytes: Bytes from one cache page to the next.
        host_page_bytes: Bytes from one host page to the next.
        first_layer_offset_bytes: Byte offset of the first layer's rows within
            a page.
        row_bytes: Bytes per row, a multiple of 16.
        layer_count: How many consecutive layers to copy.
        layer_stride_bytes: Bytes from one layer's rows to the next within a
            page, a multiple of 16.
        num_batches: Requests in the batch.
        ctx: Device context to enqueue on.
    """
    if row_bytes % 16 != 0 or layer_stride_bytes % 16 != 0:
        raise Error(
            "hisparse rows and layer stride must be multiples of 16 bytes"
        )
    if layer_count <= 0 or num_batches == 0:
        return
    comptime kernel = _hisparse_gather_kernel[
        page_size, hot_pages, num_groups, max_misses, hash_capacity
    ]
    ctx.enqueue_function[kernel](
        cache.unsafe_origin_cast[MutUntrackedOrigin](),
        host.unsafe_origin_cast[ImmUntrackedOrigin](),
        hot_table.unsafe_origin_cast[ImmUntrackedOrigin](),
        Int64(page_stride_bytes),
        Int64(host_page_bytes),
        Int64(first_layer_offset_bytes),
        Int32(row_bytes),
        Int32(layer_count),
        Int64(layer_stride_bytes),
        grid_dim=(num_batches, GATHER_BLOCKS_PER_REQUEST),
        block_dim=GATHER_BLOCK,
    )
