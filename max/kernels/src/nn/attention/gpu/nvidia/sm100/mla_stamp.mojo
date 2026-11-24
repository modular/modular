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
"""Debug-only per-warp clock stamps for sparse MLA decode (`-D MLA_STAMP=1`).

Lane 0 of a warp writes one `clock64` word to
`buf[((cta * STAMP_WARPS) + warp) * STAMP_SLOTS + slot]` for CTAs with a
linear index below `STAMP_MAX_CTAS`. The buffer address is parked in the last
16 bytes of dynamic smem (the CTA record below it), which the launch reserves
only when stamping is on, so role functions need no extra argument. Off, every
call elaborates to nothing.

Warp 8 (the Q load) carries no role stamps, so its row holds the CTA-level
stamps thread 0 writes at the CTA's exit: entry at `ST_START`, exit at
`ST_END`, the walk's key count at `ST_NTILES`, the block and grid indices at
`ST_BLOCK`/`ST_GRID`, and the prologue and epilogue sync points as tile-0
events (`CTA_*`). They are on whenever `MLA_STAMP` is, whatever the role mask.
"""

from std.math import align_up
from std.sys import get_defined_int, llvm_intrinsic
from std.time import global_perf_counter_ns
from std.sys._assembly import inlined_assembly
from max.gpu import block_idx, grid_dim, lane_id, thread_idx, warp_id
from max.gpu.sync import syncwarp
from max.gpu.memory import external_memory
from nn.attention.gpu.nvidia.sm100.attention_utils import SharedMemPointer

comptime MLA_STAMP: Bool = get_defined_int["MLA_STAMP", 0]() != 0
comptime STAMP_ROLES = get_defined_int["MLA_STAMP_ROLES", 0x7F]()
"""Bit r enables role r: 0 softmax, 1 correction, 2 gather, 3 QK, 4 PV,
5 index, 6 re-swizzle."""
comptime STAMP_WARPS = 16
comptime STAMP_SLOTS = get_defined_int["MLA_STAMP_SLOTS", 256]()
comptime STAMP_MAX_CTAS = get_defined_int["MLA_STAMP_MAX_CTAS", 1024]()
comptime STAMP_SMX: Bool = get_defined_int["MLA_STAMP_SMX", 0]() != 0
"""Softmax (role 0) stamps that sum in shared memory (`-D MLA_STAMP_SMX=1`).

One elected lane of each softmax warp adds the 32-bit clock of tile event
`e` to word `e` of its row (`red.shared.add`), so each tile costs four
fire-and-forget adds and no global store; the per-tile mean of interval
`e -> e + 1` is (word `e + 1` - word `e`) / tiles. The exit copies the row
to the warp's stamp slots `ST_TILE0 + e`, with `ST_START` left 0 (see
`stamp_finish`). The per-tile global stores, and their 64-bit base held
across the loop, perturb the softmax too much to time it. The rows take
the CTA record's event words (the Layout G carve has no room for more), so
the CTA events are off under this knob."""
comptime _SMX_WORDS = 4
"""One word per tile event (`ST_EVENTS`)."""
comptime _SMX_WARPS = 4
comptime STAMP_SMEM_BYTES = 128 if MLA_STAMP else 0

comptime ST_START = 0
comptime ST_END = 1
comptime ST_SMID = 2
comptime ST_NTILES = 3
comptime ST_START_NS = 4
comptime ST_END_NS = 5
comptime ST_BLOCK = 6
"""CTA row only: `block_idx` x | y << 16 | z << 40; `ST_GRID` packs
`grid_dim` the same way."""
comptime ST_GRID = 7
comptime ST_TILE0 = 8
"""Event `e` of tile `t` lands at `ST_TILE0 + t * ST_EVENTS + e`."""
comptime ST_EVENTS = 4
comptime ST_MAX_TILES = (STAMP_SLOTS - ST_TILE0) // ST_EVENTS

comptime STAMP_CTA_WARP = 8
comptime CTA_SETUP = 0
"""After the CTA barrier that publishes the barrier inits and TMEM base."""
comptime CTA_SYNC_IN = 1
"""After the pair's entry cluster sync."""
comptime CTA_ROLES_DONE = 2
"""After the CTA barrier every role reaches when its walk is done."""
comptime CTA_OFFSETS = 3
"""Thread 0, once the offsets are known and the early exits are passed."""
comptime CTA_INITS = 4
"""The barrier-init thread, after its inits, fence and prefetches."""
comptime CTA_ALLOC = 5
"""The TMEM-alloc warp, after its alloc."""
comptime CTA_ZEROED = 6
"""Thread 0, after its share of the NoPE-tail zeroing."""
comptime CTA_INDEX_START = 7
"""The index warp, at its role's start."""
comptime CTA_INITS_UNFENCED = 8
"""The barrier-init thread, after its inits, before their fence."""
comptime _CTA_RECORD_EVENTS = 9
comptime CTA_LATE_INBOUND = 9
"""Written by warp 9 itself after thread 0's exit: after the pair's tail
waits."""
comptime CTA_LATE_DEALLOC = 10
"""Written by warp 9 itself after thread 0's exit: after the TMEM dealloc."""
comptime CTA_LATE_RELEASED = 11
"""Written by warp 9 itself after thread 0's exit: after the allocation-permit
release, before the TMEM dealloc."""


@inline(.always)
def _clock64() -> UInt64:
    return UInt64(
        llvm_intrinsic[
            "llvm.nvvm.read.ptx.sreg.clock64", Int64, has_side_effect=True
        ]()
    )


@inline(.always)
def _clock32() -> UInt32:
    # The low half of `clock64` (CS2R, fixed latency): the 32-bit `%clock`
    # lowers to S2R, whose variable latency stalls the add that consumes it.
    return UInt32(_clock64())


@inline(.always)
def _slot_ptr() -> SharedMemPointer[UInt64]:
    var base = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=128,
        name="mha_dynamic_shared_memory",
    ]()
    var size = UInt32(
        llvm_intrinsic[
            "llvm.nvvm.read.ptx.sreg.dynamic_smem_size",
            Int32,
            # Re-read at each use rather than carried in a register.
            has_side_effect=True,
        ]()
    )
    # The reserve is the last 16 B, but the carve before it need not end on
    # an 8 B boundary; round up so the `UInt64` slot is aligned and stays
    # inside the reserve (`base` is 128 B aligned).
    return (
        (base.unsafe_offset(align_up(Int(size) - 16, 8)))
        .unsafe_bitcast[UInt64]()
        .as_unsafe_any_origin()
    )


@inline(.always)
def stamp_park(buf_addr: UInt64):
    """Thread 0 parks the buffer address; a CTA barrier must follow."""
    comptime if MLA_STAMP:
        if thread_idx.x == 0:
            _slot_ptr()[] = buf_addr


@inline(.always)
def stamp_begin[role: Int]() -> UInt64:
    """Stamps the role start and returns this warp's stamp base address.

    The base is 0 off lane 0, past `STAMP_MAX_CTAS`, or with no buffer, so
    later stamps are one predicated store. Under `MLA_STAMP_SMX` the
    softmax's base is its shared-memory row, the same in every lane.
    """
    comptime if (
        MLA_STAMP and (STAMP_ROLES >> role) & 1 == 1 and role == 0 and STAMP_SMX
    ):
        # The row address is warp-uniform, so the adds need no per-lane
        # base register and no branch (see `stamp_tile`).
        var row = _smx_row(Int(warp_id[broadcast=True]()) % _SMX_WARPS)
        if lane_id() == 0:
            comptime for k in range(_SMX_WORDS):
                row[k] = 0
        syncwarp()
        return UInt64(UInt32(Int(row)))
    elif MLA_STAMP and (STAMP_ROLES >> role) & 1 == 1:
        var addr = _slot_ptr()[]
        var cta = UInt64(block_idx.x) + UInt64(grid_dim.x) * (
            UInt64(block_idx.y) + UInt64(grid_dim.y) * UInt64(block_idx.z)
        )
        if lane_id() != 0 or addr == 0 or cta >= UInt64(STAMP_MAX_CTAS):
            return 0
        var warp = UInt64(thread_idx.x // 32)
        var base = addr + (cta * UInt64(STAMP_WARPS) + warp) * UInt64(
            STAMP_SLOTS * 8
        )
        _put(base, ST_START, _clock64())
        _put(base, ST_START_NS, global_perf_counter_ns())
        return base
    else:
        return 0


@inline(.always)
def _put(base: UInt64, slot: Int, v: UInt64):
    if base != 0:
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=Int(base + UInt64(slot * 8))
        )[] = v


@inline(.always)
def stamp_tile[role: Int](base: UInt64, tile: Int, event: Int):
    comptime if (
        MLA_STAMP and (STAMP_ROLES >> role) & 1 == 1 and role == 0 and STAMP_SMX
    ):
        inlined_assembly[
            """{
            .reg .pred p;
            elect.sync _|p, 0xffffffff;
            @p red.shared.add.u32 [$0], $1;
            }""",
            NoneType,
            constraints="r,r",
            has_side_effect=True,
        ](UInt32(base) + UInt32(event * 4), _clock32())
    elif MLA_STAMP and (STAMP_ROLES >> role) & 1 == 1:
        if tile < ST_MAX_TILES:
            _put(base, ST_TILE0 + tile * ST_EVENTS + event, _clock64())


@inline(.always)
def stamp_finish[role: Int](base: UInt64, num_tiles: Int):
    comptime if (
        MLA_STAMP and (STAMP_ROLES >> role) & 1 == 1 and role == 0 and STAMP_SMX
    ):
        var t = _clock32()
        # The elected lane made the adds; order them before lane 0's reads.
        syncwarp()
        if lane_id() != 0:
            return
        var row = SharedMemPointer[UInt32](
            unsafe_from_address=Int(UInt32(base))
        )
        var gbase = _warp_row_global(Int(thread_idx.x // 32))
        if gbase == 0:
            return
        _put(gbase, ST_END, UInt64(t))
        _put(gbase, ST_NTILES, UInt64(num_tiles))
        _put(
            gbase,
            ST_SMID,
            UInt64(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.smid",
                    Int32,
                    has_side_effect=False,
                ]()
            ),
        )
        comptime for k in range(ST_EVENTS):
            _put(gbase, ST_TILE0 + k, UInt64(row[k]))
    elif MLA_STAMP and (STAMP_ROLES >> role) & 1 == 1:
        _put(base, ST_END, _clock64())
        _put(base, ST_END_NS, global_perf_counter_ns())
        _put(base, ST_NTILES, UInt64(num_tiles))
        _put(
            base,
            ST_SMID,
            UInt64(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.smid",
                    Int32,
                    has_side_effect=False,
                ]()
            ),
        )


@inline(.always)
def _pack3(x: Int, y: Int, z: Int) -> UInt64:
    return UInt64(x) | (UInt64(y) << 16) | (UInt64(z) << 40)


@inline(.always)
def _cta_rec_ptr() -> SharedMemPointer[UInt64]:
    # The CTA record sits below the parked buffer address, inside the
    # reserve. Thread 0 keeps its CTA stamps here until the exit: a global
    # store before the pair's cluster syncs would make their release fence
    # wait for it, which inflates exactly the phases being measured.
    var base = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=128,
        name="mha_dynamic_shared_memory",
    ]()
    var size = UInt32(
        llvm_intrinsic[
            "llvm.nvvm.read.ptx.sreg.dynamic_smem_size",
            Int32,
            # Re-read at each use rather than carried in a register.
            has_side_effect=True,
        ]()
    )
    return (
        (base.unsafe_offset(align_up(Int(size) - STAMP_SMEM_BYTES, 8)))
        .unsafe_bitcast[UInt64]()
        .as_unsafe_any_origin()
    )


@inline(.always)
def _is_thread0() -> Bool:
    # Re-read at each CTA stamp: a thread index the compiler keeps from the
    # entry stamp is one more register live across the walk.
    return (
        llvm_intrinsic[
            "llvm.nvvm.read.ptx.sreg.tid.x", Int32, has_side_effect=True
        ]()
        == 0
    )


comptime _REC_START = 0
comptime _REC_START_NS = 1
comptime _REC_BLOCK = 2
comptime _REC_GRID = 3
comptime _REC_KEYS = 4
comptime _REC_EVENT0 = 5
comptime _REC_WORDS = _REC_EVENT0 + _CTA_RECORD_EVENTS


@inline(.always)
def _smx_row(warp: Int) -> SharedMemPointer[UInt32]:
    # Softmax rows follow the CTA record inside the reserve, below the
    # parked buffer address.
    return (
        _cta_rec_ptr()
        .unsafe_bitcast[UInt32]()
        .unsafe_offset(_REC_EVENT0 * 2 + warp * _SMX_WORDS)
    )


@inline(.always)
def _warp_row_global(warp: Int) -> UInt64:
    """This CTA's global stamp row for `warp`, or 0 past `STAMP_MAX_CTAS`
    or with no buffer; the CTA index comes from the record."""
    var addr = _slot_ptr()[]
    var rec = _cta_rec_ptr()
    var blk = rec[_REC_BLOCK]
    var grd = rec[_REC_GRID]
    var cta = (blk & 0xFFFF) + (grd & 0xFFFF) * (
        ((blk >> 16) & 0xFFFFFF) + ((grd >> 16) & 0xFFFFFF) * (blk >> 40)
    )
    if addr == 0 or cta >= UInt64(STAMP_MAX_CTAS):
        return 0
    return addr + (cta * UInt64(STAMP_WARPS) + UInt64(warp)) * UInt64(
        STAMP_SLOTS * 8
    )


@inline(.always)
def stamp_cta_entry():
    """Thread 0 records the CTA entry; `stamp_cta_exit` writes it out."""
    comptime if MLA_STAMP:
        comptime assert (
            _REC_WORDS * 8 + 16 <= STAMP_SMEM_BYTES
        ), "the CTA record overlaps the parked buffer address"
        comptime assert (
            _SMX_WARPS * _SMX_WORDS * 4 <= _CTA_RECORD_EVENTS * 8
        ), "the softmax rows overrun the CTA record's event words"
        var t = _clock64()
        if _is_thread0():
            var rec = _cta_rec_ptr()
            rec[_REC_START] = t
            rec[_REC_START_NS] = global_perf_counter_ns()
            rec[_REC_BLOCK] = _pack3(block_idx.x, block_idx.y, block_idx.z)
            rec[_REC_GRID] = _pack3(grid_dim.x, grid_dim.y, grid_dim.z)
            comptime for e in range(_REC_WORDS - _REC_EVENT0):
                rec[_REC_EVENT0 + e] = 0


@inline(.always)
def stamp_cta_event(event: Int):
    comptime if MLA_STAMP and not STAMP_SMX:
        var t = _clock64()
        if _is_thread0():
            _cta_rec_ptr()[_REC_EVENT0 + event] = t


@inline(.always)
def stamp_cta_event_here(event: Int):
    """Records a CTA event from the calling thread; one thread calls it."""
    comptime if MLA_STAMP and not STAMP_SMX:
        _cta_rec_ptr()[_REC_EVENT0 + event] = _clock64()


@inline(.always)
def stamp_cta_event_warp(event: Int):
    """Records a CTA event from lane 0 of the one warp that calls it."""
    comptime if MLA_STAMP and not STAMP_SMX:
        var t = _clock64()
        if lane_id() == 0:
            _cta_rec_ptr()[_REC_EVENT0 + event] = t


@inline(.always)
def stamp_cta_late(event: Int):
    """Lane 0 writes a CTA-row event straight to global memory.

    For events after thread 0's exit write; the CTA index comes from the
    record, which the setup barrier published."""
    comptime if MLA_STAMP and not STAMP_SMX:
        var t = _clock64()
        if lane_id() != 0:
            return
        var addr = _slot_ptr()[]
        var rec = _cta_rec_ptr()
        var blk = rec[_REC_BLOCK]
        var grd = rec[_REC_GRID]
        var cta = (blk & 0xFFFF) + (grd & 0xFFFF) * (
            ((blk >> 16) & 0xFFFFFF) + ((grd >> 16) & 0xFFFFFF) * (blk >> 40)
        )
        if addr == 0 or cta >= UInt64(STAMP_MAX_CTAS):
            return
        var base = addr + (
            cta * UInt64(STAMP_WARPS) + UInt64(STAMP_CTA_WARP)
        ) * UInt64(STAMP_SLOTS * 8)
        _put(base, ST_TILE0 + event, t)


@inline(.always)
def stamp_cta_keys(num_keys: Int):
    """Records the CTA's walk length; called once the offsets are known."""
    comptime if MLA_STAMP:
        if _is_thread0():
            _cta_rec_ptr()[_REC_KEYS] = UInt64(num_keys)


@inline(.always)
def stamp_cta_exit():
    """Thread 0 writes the CTA row: its record, then the exit stamps.

    Needs the buffer address parked (`stamp_park` and its barrier)."""
    comptime if MLA_STAMP:
        var t = _clock64()
        var t_ns = global_perf_counter_ns()
        if not _is_thread0():
            return
        var addr = _slot_ptr()[]
        var rec = _cta_rec_ptr()
        # The CTA index comes from the record, not the special registers:
        # the compiler would otherwise carry the entry's block and grid
        # reads across the walk, and the softmax warpgroup has no
        # registers to spare.
        var blk = rec[_REC_BLOCK]
        var grd = rec[_REC_GRID]
        var cta = (blk & 0xFFFF) + (grd & 0xFFFF) * (
            ((blk >> 16) & 0xFFFFFF) + ((grd >> 16) & 0xFFFFFF) * (blk >> 40)
        )
        if addr == 0 or cta >= UInt64(STAMP_MAX_CTAS):
            return
        var base = addr + (
            cta * UInt64(STAMP_WARPS) + UInt64(STAMP_CTA_WARP)
        ) * UInt64(STAMP_SLOTS * 8)
        _put(base, ST_START, rec[_REC_START])
        _put(base, ST_START_NS, rec[_REC_START_NS])
        _put(base, ST_BLOCK, blk)
        _put(base, ST_GRID, grd)
        _put(base, ST_NTILES, rec[_REC_KEYS])
        comptime if not STAMP_SMX:
            comptime for e in range(_REC_WORDS - _REC_EVENT0):
                _put(base, ST_TILE0 + e, rec[_REC_EVENT0 + e])
        _put(base, ST_END, t)
        _put(base, ST_END_NS, t_ns)
        _put(
            base,
            ST_SMID,
            UInt64(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.smid",
                    Int32,
                    has_side_effect=False,
                ]()
            ),
        )
