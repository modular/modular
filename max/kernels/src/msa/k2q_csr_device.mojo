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
"""Device (GPU) reverse-CSR builder for KV-block-major sparse MHA.

GPU port of host `k2q_csr.build_k2q_csr` (its oracle).  Inverts the query-major
selection `q2k [head_kv, total_q, topK]` into the KV-block-major CSR + schedule
the block-major forward/combine kernels consume, emitting the SAME contract
tensors the host builder produces directly into device buffers (no host round-
trip).

Five stages:
  row_map     round-robin (batch, kv_block) -> row_linear + row_coords
  hist        per-(CTA,warp) unit histograms -> tile_counts + row_counts
  row_prefix  one block per head: row_counts -> row_ptr, emit scheduler_metadata
  tile_prefix scan tile_counts along the (CTA,warp) unit axis -> per-unit base
  scatter     per-unit q-sequential write of qsplit / split_counts

The hist/scatter grid is `(g, head_kv)`: heads run as parallel CTAs (grid.y) and
the q-range is tiled across `g` CTAs x `kwarps` warps (`g_total` units), each
owning a contiguous q-sub-range -- so `g*head_kv` CTAs spread the q*topk edge
stream across the SMs (a single under-gridded CTA serializes it on one SM).
Per-row slots are reserved by an exclusive prefix scan over the units (PR + PT),
so scatter writes without cross-CTA atomics and the per-unit ranges concatenate
to a globally q-ascending row, byte-identical to the host's sequential writer.

SMEM histogram/cursor entries are one `Int32` per (warp,row) (no int16 bit-pack):
no per-warp count cap, 2x the per-warp SMEM; `kwarps` is picked so two CTAs still
fit per SM at the BF16/non-paged row counts.  `q_per_cta` chunking: each non-
empty row -> ceil(row_count/q_per_cta) work items, default 128 = the fwd CTA
query cap (BM).
"""

from std.math import ceildiv
from std.sys import size_of
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
    lane_id,
    warp_id as get_warp_id,
    WARP_SIZE,
)
from std.gpu import barrier
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.memory import AddressSpace, external_memory
from std.gpu.sync import syncwarp
import std.gpu.primitives.warp as warp
from std.atomic import Atomic
from std.bit import count_trailing_zeros, pop_count
from std.sys._assembly import inlined_assembly


@always_inline
def _match_any_u32(value: Int32) -> UInt32:
    """Lanes in the warp whose `value` equals this lane's (PTX match.any.sync).

    Returns a 32-bit mask with bit `l` set for every active lane `l` holding the
    same `value`.  Lets one leader lane fold a group of equal edges into a single
    non-atomic SMEM add instead of one `atom.shared` per edge.
    """
    return inlined_assembly[
        "match.any.sync.b32 $0, $1, $2;",
        UInt32,
        constraints="=r,r,r",
        has_side_effect=False,
    ](value, UInt32(0xFFFFFFFF))


comptime _UNUSED: Int32 = -1
"""Padding / unused-slot sentinel in `qsplit_indices`."""

# Largest warps-per-CTA we will pick.  Picked down (in `k2q_csr_sizes`) so two
# CTAs fit per SM in the per-warp SMEM histogram/cursor budget.
comptime _MAX_KWARPS = 4
# SM SMEM budget (B200 = 228 KB).  Used only to pick `kwarps` / occupancy.
comptime _SM_SMEM_BYTES = 228 * 1024
# Smallest q-slice a warp processes.  A 256-q/CTA floor would pin G=2 at prefill
# total_q=512 (2 CTAs, 146/148 SMs idle); flooring per-warp instead lets the
# q-range fan out across ~num_sms CTAs while each warp still does enough work to
# amortize its SMEM clear + row_counts atomic.
comptime _MIN_Q_PER_WARP = 8


@always_inline
def _batch_of_dev(
    cu_q: UnsafePointer[Int32, MutAnyOrigin], batch: Int, q_abs: Int
) -> Int:
    """Batch owning global query token `q_abs` (cu_q[b] <= q_abs < cu_q[b+1]).
    """
    var b = 0
    while b < batch and Int(cu_q[b + 1]) <= q_abs:
        b += 1
    return b


# ===-------------------------------------------------------------------=== #
# Stage M: round-robin row map.  Grid (max_kv_blocks,1,1); thread 0 of each
# block owns one kv-block level.
# ===-------------------------------------------------------------------=== #
def _k2q_row_map(
    cu_k: UnsafePointer[Int32, MutAnyOrigin],
    row_map: UnsafePointer[Int32, MutAnyOrigin],  # [batch, max_kv_blocks]
    row_coords: UnsafePointer[Int32, MutAnyOrigin],  # [total_rows, 2]
    batch: Int,
    max_kv_blocks: Int,
    blk_kv: Int,
):
    var level = Int(block_idx.x)
    if level >= max_kv_blocks or thread_idx.x != 0:
        return
    var rows_before = 0
    for b in range(batch):
        var rb = ceildiv(Int(cu_k[b + 1] - cu_k[b]), blk_kv)
        rows_before += min(rb, level)
    var active_before = 0
    for b in range(batch):
        var rb = ceildiv(Int(cu_k[b + 1] - cu_k[b]), blk_kv)
        if rb > level:
            var row_linear = rows_before + active_before
            row_map[b * max_kv_blocks + level] = Int32(row_linear)
            row_coords[row_linear * 2] = Int32(b)
            row_coords[row_linear * 2 + 1] = Int32(level)
            active_before += 1
        else:
            row_map[b * max_kv_blocks + level] = _UNUSED


# ===-------------------------------------------------------------------=== #
# Stage H: per-(CTA,warp) histogram.  Grid (g, head_kv); CTA (c,h) owns head `h`
# and q-range [c*q_per_cta, ...), warp `w` owns its q_per_warp slice.  Counts
# edges into its own SMEM row bucket (match.any.sync folds equal-row edges into
# one leader add -- no per-edge atomic), writes tile_counts[c*kwarps+w, h, row]
# and atomically accumulates row_counts[h,row].  Per-warp SMEM is dynamic (no cap).
# ===-------------------------------------------------------------------=== #
def _k2q_hist[
    topk: Int
](
    q2k: UnsafePointer[Int32, MutAnyOrigin],
    cu_q: UnsafePointer[Int32, MutAnyOrigin],
    row_map: UnsafePointer[Int32, MutAnyOrigin],
    row_counts: UnsafePointer[Int32, MutAnyOrigin],  # [head_kv, total_rows]
    tile_counts: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [g_total, head_kv, total_rows]
    head_kv: Int,
    batch: Int,
    total_q: Int,
    total_rows: Int,
    max_kv_blocks: Int,
    kwarps: Int,
    q_per_cta: Int,
    q_per_warp: Int,
):
    var smem = external_memory[
        Int32,
        address_space=AddressSpace.SHARED,
        alignment=16,
        name="k2q_hist_smem",
    ]()  # [kwarps, total_rows]
    var tid = Int(thread_idx.x)
    var warp_id = get_warp_id[broadcast=True]()
    var lane = Int(lane_id())
    var threads = kwarps * WARP_SIZE
    var c = Int(block_idx.x)
    var h = Int(block_idx.y)  # one head per CTA (grid.y = head_kv)
    var q_start_cta = c * q_per_cta
    var q_end_cta = min(q_start_cta + q_per_cta, total_q)
    var q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta)
    var q_end_warp = min(q_start_warp + q_per_warp, q_end_cta)
    var my_hist = smem + warp_id * total_rows

    var i = lane
    while i < total_rows:
        my_hist[i] = 0
        i += WARP_SIZE
    barrier()

    if q_start_warp < q_end_warp:
        var q2k_head = q2k + h * total_q * topk
        var lower_lane_mask = UInt32(0) if lane == 0 else (
            (UInt32(1) << UInt32(lane)) - 1
        )
        # Warp-uniform wave count so all 32 lanes hit match.any.sync converged;
        # lanes past the q-range carry row=-1 and drop out of every group.
        var n_waves = ceildiv(q_end_warp - q_start_warp, WARP_SIZE)
        for wave in range(n_waves):
            var qi = q_start_warp + wave * WARP_SIZE + lane
            var rmap = row_map
            if qi < q_end_warp:
                rmap += _batch_of_dev(cu_q, batch, qi) * max_kv_blocks
            comptime for t in range(topk):
                var row = -1
                if qi < q_end_warp:
                    var kvb = Int(q2k_head[qi * topk + t])
                    if 0 <= kvb < max_kv_blocks:
                        var r = Int(rmap[kvb])
                        if 0 <= r < total_rows:
                            row = r
                # Fold the lanes hitting `row` into one non-atomic add: the
                # lowest lane in the group writes popcount(group) to my_hist.
                var group = _match_any_u32(Int32(row))
                if row >= 0 and (group & lower_lane_mask) == 0:
                    my_hist[row] += Int32(pop_count(Int(group)))
    barrier()

    # Each warp publishes its own tile_counts slice (unit = c*kwarps+w).
    var my_tile = (
        tile_counts + ((c * kwarps + warp_id) * head_kv + h) * total_rows
    )
    var ti = lane
    while ti < total_rows:
        my_tile[ti] = my_hist[ti]
        ti += WARP_SIZE
    barrier()

    # Reduce across warps -> row_counts (global atomic; many CTAs contribute).
    var head_rc = row_counts + h * total_rows
    var ri = tid
    while ri < total_rows:
        var s = 0
        for w in range(kwarps):
            s += Int(smem[w * total_rows + ri])
        if s > 0:
            _ = Atomic.fetch_add(head_rc + ri, Int32(s))
        ri += threads


# ===-------------------------------------------------------------------=== #
# Stage PR: row prefix.  One block per head; exclusive scan of row_counts ->
# row_ptr, and emit ceil(row_count/q_per_cta) q-chunk work items per non-empty
# row.
# ===-------------------------------------------------------------------=== #
def _k2q_row_prefix[
    threads: Int
](
    row_counts: UnsafePointer[Int32, MutAnyOrigin],
    row_ptr: UnsafePointer[Int32, MutAnyOrigin],  # [head_kv, total_rows+1]
    row_coords: UnsafePointer[Int32, MutAnyOrigin],
    scheduler_metadata: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [work_capacity, 6]
    work_count: UnsafePointer[Int32, MutAnyOrigin],  # [1]
    total_rows: Int,
    work_capacity: Int,
    q_per_cta: Int,
):
    var h = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var scan_buf = external_memory[
        Int32,
        address_space=AddressSpace.SHARED,
        alignment=16,
        name="k2q_rowprefix_smem",
    ]()  # [threads]

    var head_counts = row_counts + h * total_rows
    var head_rowptr = row_ptr + h * (total_rows + 1)
    var chunk = ceildiv(total_rows, threads)
    var lo = tid * chunk
    var hi = min(lo + chunk, total_rows)

    var local_sum = 0
    for i in range(lo, hi):
        local_sum += Int(head_counts[i])
    scan_buf[tid] = Int32(local_sum)
    barrier()

    # Hillis-Steele inclusive scan over the per-thread partial sums.
    var off = 1
    while off < threads:
        var add = Int(scan_buf[tid - off]) if tid >= off else 0
        barrier()
        scan_buf[tid] = Int32(Int(scan_buf[tid]) + add)
        barrier()
        off <<= 1

    var running = Int(scan_buf[tid]) - local_sum
    if tid == 0:
        head_rowptr[0] = 0
    for i in range(lo, hi):
        var row_count = Int(head_counts[i])
        running += row_count
        head_rowptr[i + 1] = Int32(running)
        if row_count > 0:
            # Reserve a contiguous block of work slots for this row's chunks,
            # then emit (q_begin, q_count) per chunk.
            var num_chunks = ceildiv(row_count, q_per_cta)
            var base = Int(Atomic.fetch_add(work_count, Int32(num_chunks)))
            var batch_idx = row_coords[i * 2]
            var kv_block_idx = row_coords[i * 2 + 1]
            for c in range(num_chunks):
                var work_idx = base + c
                if work_idx < work_capacity:
                    var q_begin = c * q_per_cta
                    var q_count = min(q_per_cta, row_count - q_begin)
                    var meta = scheduler_metadata + work_idx * 6
                    meta[0] = Int32(h)
                    meta[1] = Int32(i)
                    meta[2] = Int32(q_begin)
                    meta[3] = Int32(q_count)
                    meta[4] = batch_idx
                    meta[5] = kv_block_idx


# ===-------------------------------------------------------------------=== #
# Stage PT: tile prefix.  Scan tile_counts along the unit (c*kwarps+w) axis,
# fusing row_ptr into the base, so tile_counts[unit,h,row] becomes the absolute
# slot where that unit starts writing row `row`.  Each block stages
# `rows_per_block` rows x g_total units for one head into SMEM (coalesced
# load), per-warp scans one row over the unit axis, stores back.  Grid
# (head_kv * blocks_per_h, 1, 1).
# ===-------------------------------------------------------------------=== #
def _k2q_tile_prefix[
    rows_per_block: Int
](
    tile_counts: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [g_total, head_kv, total_rows]
    row_ptr: UnsafePointer[Int32, MutAnyOrigin],
    head_kv: Int,
    total_rows: Int,
    g_total: Int,
):
    var smem = external_memory[
        Int32,
        address_space=AddressSpace.SHARED,
        alignment=16,
        name="k2q_tileprefix_smem",
    ]()  # [rows_per_block, g_total]
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp_id = get_warp_id()
    var threads = Int(block_dim.x)

    var blocks_per_h = ceildiv(total_rows, rows_per_block)
    var h = Int(block_idx.x) // blocks_per_h
    var b_in_h = Int(block_idx.x) - h * blocks_per_h
    if h >= head_kv:
        return
    var base_r = b_in_h * rows_per_block
    if base_r >= total_rows:
        return
    var actual_m = min(rows_per_block, total_rows - base_r)

    # tile_counts[g, h, base_r + r_off]; the unit stride along g is head_kv*total_rows.
    var stride_g = head_kv * total_rows
    var base_ptr = tile_counts + h * total_rows + base_r
    var total_elems = g_total * actual_m

    # Coalesced load: i -> (r_off = i % actual_m, g = i / actual_m).
    var i = tid
    while i < total_elems:
        var r_off = i % actual_m
        var g = i // actual_m
        smem[r_off * g_total + g] = base_ptr[g * stride_g + r_off]
        i += threads
    barrier()

    # Per-warp scan: warp w scans row (base_r + w) over the unit axis, offset by
    # row_ptr[h, base_r + w].  Exclusive scan -> each unit's start slot.
    if warp_id < actual_m:
        var abs_r = base_r + warp_id
        var running = Int(row_ptr[h * (total_rows + 1) + abs_r])
        var my_smem = smem + warp_id * g_total
        var g0 = 0
        while g0 < g_total:
            var g = g0 + lane
            var v = my_smem[g] if g < g_total else Int32(0)
            var x = v
            var sh = UInt32(1)
            while sh < UInt32(WARP_SIZE):
                var nbr = warp.shuffle_up(x, sh)
                if UInt32(lane) >= sh:
                    x += nbr
                sh <<= 1
            var excl = Int32(running) + x - v  # exclusive prefix
            if g < g_total:
                my_smem[g] = excl
            running += Int(warp.shuffle_idx(x, UInt32(WARP_SIZE - 1)))
            g0 += WARP_SIZE
    barrier()

    # Coalesced store back.
    i = tid
    while i < total_elems:
        var r_off = i % actual_m
        var g = i // actual_m
        base_ptr[g * stride_g + r_off] = smem[r_off * g_total + g]
        i += threads


# ===-------------------------------------------------------------------=== #
# Stage S: scatter.  Grid (g, head_kv); CTA (c,h) owns head `h` and
# [c*q_per_cta, ...), warp `w` its
# q_per_warp slice (unit = c*kwarps+w).  Lanes cover `kq_per_iter` queries x `topk`
# slots in lockstep; warp ballot ranks valid edges (split_slot) and a per-warp
# SMEM cursor (one Int32/row) hands out q-ascending slots within the unit's
# absolute base.
# ===-------------------------------------------------------------------=== #
def _k2q_scatter[
    topk: Int
](
    q2k: UnsafePointer[Int32, MutAnyOrigin],
    cu_q: UnsafePointer[Int32, MutAnyOrigin],
    row_map: UnsafePointer[Int32, MutAnyOrigin],
    abs_base: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # tile_counts (post tile-prefix)
    qsplit_idx: UnsafePointer[Int32, MutAnyOrigin],  # [head_kv, total_q*topk]
    split_counts: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [batch, max_seqlen_q, head_kv]
    head_kv: Int,
    batch: Int,
    total_q: Int,
    total_rows: Int,
    max_kv_blocks: Int,
    max_seqlen_q: Int,
    kwarps: Int,
    q_per_cta: Int,
    q_per_warp: Int,
):
    comptime kq_per_iter = WARP_SIZE // topk if WARP_SIZE // topk > 0 else 1
    var smem = external_memory[
        Int32,
        address_space=AddressSpace.SHARED,
        alignment=16,
        name="k2q_scatter_smem",
    ]()  # [kwarps, total_rows] cursor
    var tid = Int(thread_idx.x)
    var warp_id = get_warp_id[broadcast=True]()
    var lane = Int(lane_id())
    var c = Int(block_idx.x)
    var h = Int(block_idx.y)  # one head per CTA (grid.y = head_kv)
    var q_start_cta = c * q_per_cta
    var q_end_cta = min(q_start_cta + q_per_cta, total_q)
    var q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta)
    var q_end_warp = min(q_start_warp + q_per_warp, q_end_cta)

    var q_in_iter = lane // topk
    var slot_in_q = lane % topk
    var lane_active = lane < kq_per_iter * topk
    var my_cursor = smem + warp_id * total_rows

    # group_mask selects the topk lanes of this lane's query within the warp.
    comptime group_lo = UInt32((1 << topk) - 1) if topk < 32 else UInt32(
        0xFFFFFFFF
    )
    var group_mask = UInt32(0xFFFFFFFF) if topk == 32 else (
        group_lo << UInt32(q_in_iter * topk)
    )
    var lower_lane_mask = UInt32(0) if lane == 0 else (
        (UInt32(1) << UInt32(lane)) - 1
    )

    var i = lane
    while i < total_rows:
        my_cursor[i] = 0
        i += WARP_SIZE
    syncwarp()

    if q_start_warp < q_end_warp:
        var q2k_head = q2k + h * total_q * topk
        var my_abs = (
            abs_base + ((c * kwarps + warp_id) * head_kv + h) * total_rows
        )
        var head_qsplit = qsplit_idx + h * total_q * topk

        var qi_base = q_start_warp
        while qi_base < q_end_warp:
            var my_qi = qi_base + q_in_iter
            var valid_q = (my_qi < q_end_warp) and lane_active
            var kvb = -1
            var qloc = 0
            var bi = 0
            if valid_q:
                bi = _batch_of_dev(cu_q, batch, my_qi)
                qloc = my_qi - Int(cu_q[bi])
                kvb = Int(q2k_head[my_qi * topk + slot_in_q])
            var rmap = row_map + bi * max_kv_blocks
            var row = -1
            if valid_q and 0 <= kvb < max_kv_blocks:
                row = Int(rmap[kvb])
            var valid_edge = 0 <= row < total_rows

            var valid_mask = warp.vote[DType.uint32](valid_edge)
            var split_slot = pop_count(
                Int(valid_mask & group_mask & lower_lane_mask)
            )
            var valid_count = pop_count(Int(valid_mask & group_mask))
            if valid_q and slot_in_q == 0:
                split_counts[(bi * max_seqlen_q + qloc) * head_kv + h] = Int32(
                    valid_count
                )
            # Fold the lanes hitting `row` into one non-atomic cursor bump: the
            # group leader reads the base and advances by popcount(group), so the
            # per-edge atom.shared membar is gone.  The cursor is per-warp-private
            # and waves run sequentially, so a plain ld/st RMW yields the same
            # base the atomic would.  base + rank recreates the lane-ordered slots.
            var match_row = Int32(row) if valid_edge else Int32(-1)
            var group = _match_any_u32(match_row)
            var rank = pop_count(Int(group & lower_lane_mask))
            var leader_lane = count_trailing_zeros(Int(group))
            var base = 0
            if valid_edge and lane == leader_lane:
                base = Int(my_cursor[row])
                my_cursor[row] = Int32(base + pop_count(Int(group)))
            base = Int(warp.shuffle_idx(Int32(base), UInt32(leader_lane)))
            if valid_edge:
                var out_pos = Int(my_abs[row]) + base + rank
                head_qsplit[out_pos] = Int32(qloc | ((split_slot & 0xFF) << 24))
            qi_base += kq_per_iter


# ===-------------------------------------------------------------------=== #
# Sizing + dispatch.
# ===-------------------------------------------------------------------=== #


@fieldwise_init
struct K2qCsrDeviceSizes(Copyable, Movable):
    """Host-computed sizing for the device CSR (allocated by the caller)."""

    var batch: Int
    var total_rows: Int
    var max_kv_blocks: Int
    var work_capacity: Int
    var g: Int
    """CTAs over the q-range (hist/scatter grid.x)."""
    var kwarps: Int
    """Warps per CTA; each owns a contiguous q-sub-range."""
    var g_total: Int
    """Number of units = g * kwarps (the tile_counts unit axis length)."""
    var q_per_cta: Int
    """Queries per CTA (ceil(total_q / g))."""
    var q_per_warp: Int
    """Queries per warp (ceil(q_per_cta / kwarps))."""

    def tile_counts_len(self, head_kv: Int) -> Int:
        """Length of the tile_counts scratch buffer (the caller allocates it).
        """
        return max(self.g_total * head_kv * self.total_rows, 1)


def k2q_csr_sizes(
    cu_seqlens_k: List[Int32],
    head_kv: Int,
    blk_kv: Int,
    max_seqlen_k: Int,
    total_q: Int,
    topk: Int,
    num_sms: Int,
    q_per_cta_chunk: Int = 128,
) raises -> K2qCsrDeviceSizes:
    """Returns the device-CSR sizing (matches the host builder's formulas).

    `num_sms` (e.g. `ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)`)
    sizes the multi-CTA hist/scatter grid.  `q_per_cta_chunk` is the scheduler
    q-chunk cap (= the fwd CTA BM), distinct from the hist/scatter `q_per_cta`.
    """
    var batch = len(cu_seqlens_k) - 1
    var total_rows = 0
    for b in range(batch):
        total_rows += ceildiv(
            Int(cu_seqlens_k[b + 1] - cu_seqlens_k[b]), blk_kv
        )
    var max_kv_blocks = ceildiv(max(max_seqlen_k, blk_kv), blk_kv)
    # Q-chunked work_capacity (matches the host builder).
    var work_capacity = total_rows * head_kv + ceildiv(
        total_q * topk * head_kv, q_per_cta_chunk
    )
    # Pick (g, kwarps, q_per_cta).  The hist/scatter grid is (g, head_kv): the
    # head loop runs as parallel CTAs (grid.y) and `g` tiles the q-range over
    # CTAs (grid.x), so g*head_kv CTAs spread the q*topk edge stream across the
    # SMs.  These CTAs are tiny (~1KB SMEM, 36 regs -> ncu: ~12 fit per SM), so
    # one CTA/SM leaves the GPU ~90% idle (waves/SM ~0.09).  Oversubscribe to
    # ~4 waves (g*head_kv ~ 4*num_sms) to fill the SMs -- measured ~2x on
    # hist+scatter at the 8K SOT shape.  Bounded by how finely the q-range can
    # split while each warp still owns >= _MIN_Q_PER_WARP queries.
    var g = 1
    var kwarps = 1
    var q_per_cta = max(total_q, 1)
    if total_rows > 0 and total_q > 0:
        # Push parallelism into `g` (head_kv already eats grid.y), capped by how
        # finely the q-range splits at >= _MIN_Q_PER_WARP queries/warp.
        var max_g_for_q = max(ceildiv(total_q, _MIN_Q_PER_WARP), 1)
        var target_g = max((4 * num_sms) // max(head_kv, 1), 1)
        g = max(min(min(target_g, max_g_for_q), total_q), 1)
        q_per_cta = ceildiv(total_q, g)
        g = ceildiv(total_q, q_per_cta)
        # Spread each CTA's queries over warps for memory-level parallelism,
        # holding the per-warp floor.  SMEM (kwarps Int32/row) fits 2 CTAs/SM.
        var per_warp_smem = total_rows * size_of[Int32]()  # one Int32 per row
        kwarps = min(_MAX_KWARPS, max(q_per_cta // _MIN_Q_PER_WARP, 1))
        while kwarps > 1 and (kwarps * per_warp_smem) * 2 > _SM_SMEM_BYTES:
            kwarps >>= 1
    var q_per_warp = ceildiv(q_per_cta, kwarps)
    var g_total = g * kwarps
    return K2qCsrDeviceSizes(
        batch,
        total_rows,
        max_kv_blocks,
        work_capacity,
        g,
        kwarps,
        g_total,
        q_per_cta,
        q_per_warp,
    )


def build_k2q_csr_device[
    topk: Int
](
    # Device inputs.
    q2k: UnsafePointer[Int32, MutAnyOrigin],  # [head_kv, total_q, topk]
    cu_seqlens_q: UnsafePointer[Int32, MutAnyOrigin],  # [batch+1]
    cu_seqlens_k: UnsafePointer[Int32, MutAnyOrigin],  # [batch+1]
    # Device outputs (caller-allocated to the k2q_csr_sizes sizing).
    row_ptr: UnsafePointer[Int32, MutAnyOrigin],  # [head_kv, total_rows+1]
    qsplit_indices: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [head_kv, total_q*topk]
    scheduler_metadata: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [work_capacity, 6]
    work_count: UnsafePointer[Int32, MutAnyOrigin],  # [1]
    split_counts: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [batch, max_sq, head_kv]
    # Scratch (caller-allocated).
    row_map: UnsafePointer[Int32, MutAnyOrigin],  # [batch, max_kv_blocks]
    row_coords: UnsafePointer[Int32, MutAnyOrigin],  # [total_rows, 2]
    row_counts: UnsafePointer[Int32, MutAnyOrigin],  # [head_kv, total_rows]
    tile_counts: UnsafePointer[
        Int32, MutAnyOrigin
    ],  # [g_total, head_kv, total_rows]
    # Sizing.
    head_kv: Int,
    total_q: Int,
    blk_kv: Int,
    max_seqlen_q: Int,
    sizes: K2qCsrDeviceSizes,
    ctx: DeviceContext,
    q_per_cta: Int = 128,
) raises:
    """Builds the reverse-CSR + schedule on the device into caller buffers.

    Emits the identical contract tensors as host `build_k2q_csr` for the same
    `q2k`: `row_ptr` byte-identical, `qsplit_indices` q-ascending within each
    row, `scheduler_metadata`/`work_count`, `split_counts`.
    The outputs and scratch must be pre-zeroed where the host pre-zeroes
    (`row_counts`, `work_count`); `qsplit_indices` must be -1 filled
    (unused tail); `split_counts` zero-filled.  `q_per_cta` here is the fwd-CTA
    q-chunk cap (the scheduler chunking), distinct from the hist/scatter
    `sizes.q_per_cta` that tiles the q-range across CTAs.
    """
    comptime assert topk <= 32, "topk packs into the qsplit high byte (<=255)"
    var batch = sizes.batch
    var total_rows = sizes.total_rows
    var max_kv_blocks = sizes.max_kv_blocks

    if total_q == 0 or total_rows == 0 or head_kv == 0 or max_kv_blocks == 0:
        return  # caller-prezeroed outputs are already the empty-build answer

    var g = sizes.g
    var kwarps = sizes.kwarps
    var g_total = sizes.g_total
    var csr_threads = kwarps * WARP_SIZE
    var per_warp_smem = total_rows * size_of[Int32]()
    var hs_smem = UInt32(kwarps * per_warp_smem)

    # M: row map.
    ctx.enqueue_function[_k2q_row_map](
        cu_seqlens_k,
        row_map,
        row_coords,
        batch,
        max_kv_blocks,
        blk_kv,
        grid_dim=(max_kv_blocks, 1, 1),
        block_dim=(WARP_SIZE, 1, 1),
    )

    # H: per-warp histogram + tile_counts + row_counts (g*head_kv CTAs).
    ctx.enqueue_function[_k2q_hist[topk]](
        q2k,
        cu_seqlens_q,
        row_map,
        row_counts,
        tile_counts,
        head_kv,
        batch,
        total_q,
        total_rows,
        max_kv_blocks,
        kwarps,
        sizes.q_per_cta,
        sizes.q_per_warp,
        grid_dim=(g, head_kv, 1),
        block_dim=(csr_threads, 1, 1),
        shared_mem_bytes=Int(hs_smem),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(hs_smem),
    )

    # PR: row prefix + schedule.
    comptime rp_threads = 256
    ctx.enqueue_function[_k2q_row_prefix[rp_threads]](
        row_counts,
        row_ptr,
        row_coords,
        scheduler_metadata,
        work_count,
        total_rows,
        sizes.work_capacity,
        q_per_cta,
        grid_dim=(head_kv, 1, 1),
        block_dim=(rp_threads, 1, 1),
        shared_mem_bytes=rp_threads * size_of[Int32](),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(rp_threads * size_of[Int32]())
        ),
    )

    # PT: tile prefix (tile_counts -> per-unit absolute base) over the unit axis.
    comptime pt_threads = 256
    comptime pt_rows_per_block = 8
    var blocks_per_h = ceildiv(total_rows, pt_rows_per_block)
    var pt_smem = UInt32(pt_rows_per_block * g_total * size_of[Int32]())
    ctx.enqueue_function[_k2q_tile_prefix[pt_rows_per_block]](
        tile_counts,
        row_ptr,
        head_kv,
        total_rows,
        g_total,
        grid_dim=(head_kv * blocks_per_h, 1, 1),
        block_dim=(pt_threads, 1, 1),
        shared_mem_bytes=Int(pt_smem),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(pt_smem),
    )

    # S: scatter (g*head_kv CTAs).
    ctx.enqueue_function[_k2q_scatter[topk]](
        q2k,
        cu_seqlens_q,
        row_map,
        tile_counts,
        qsplit_indices,
        split_counts,
        head_kv,
        batch,
        total_q,
        total_rows,
        max_kv_blocks,
        max_seqlen_q,
        kwarps,
        sizes.q_per_cta,
        sizes.q_per_warp,
        grid_dim=(g, head_kv, 1),
        block_dim=(csr_threads, 1, 1),
        shared_mem_bytes=Int(hs_smem),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(hs_smem),
    )
