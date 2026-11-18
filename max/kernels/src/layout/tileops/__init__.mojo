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
"""Defines TileOps: tile-level operations built from `layout.atoms`.

A TileOp moves or computes a whole tile by issuing atoms. It derives the
memory tile's placement, a `BitLayout` from logical coordinate bits to
element offsets, and matches it against the atom's operand maps, converted
from the operand's address encoding to the engine's element offsets: every
index bit the atom fixes on the memory side must be played by one logical
bit that reaches the same offset.

The `threads` that issue together split the tile: each bit of a thread's
index plays one logical bit. A bit the atom fixes, such as a `tcgen05.ld`
lane or warp, plays the logical bit at the offset the atom puts it at;
every other one takes the free logical bit with the smallest memory offset,
so adjacent threads move adjacent memory. The remaining logical bits, in
order, are this thread's register tile, whose placement must put the
atom's value bits on the registers its register map names. Logical bits no
index bit plays are issued over: the TileOp issues the atom once per value
of them. Each issue passes the memory at those bits and at this thread's
index bits, except the lanes an instruction adds itself, as `tcgen05.ld`
does.

A TileOp never waits. `tiled_load` and `tiled_store` issue their atoms and
return; the caller waits with the atom's completion, such as
`TMemLoadWait.wait()`, when it needs the registers or the memory, so that
other work can run while the issues are in flight.

You can import these APIs from the `layout.tileops` package:

```mojo
from layout.tileops import (
    can_tiled_load,
    can_tiled_store,
    tiled_load,
    tiled_store,
)
```
"""

from layout import ImmTileTensor, MutTileTensor, TileTensor
from layout.atoms import (
    INDEX_LANE,
    INDEX_WARP,
    TMEM_COLUMN,
    TMEM_LANE,
    AddressEncoding,
    BarrierCompletion,
    BatchCompletion,
    HandleCompletion,
    LoadAtom,
    OperandMap,
    StoreAtom,
    same_engine,
)
from layout.bit_layout import BitLayout, bit, bits, is_bit_linear, placement
from layout.tensor_engine import TensorEngine
from layout.coord import ComptimeInt, Coord, DynamicCoord
from layout.layout import Layout, TensorLayout
from layout.tmem_engine import TMEM_NUM_LANES, TMemEngine

from std.bit import count_trailing_zeros, pop_count
from std.sys import size_of

comptime _NO_MATCH = ~UInt64(0)

comptime _MAX_THREAD_BITS = 16
"""The most thread index bits a TileOp splits a tile over."""

comptime _WARP_LANE_BITS = 5
"""The lane bits of a thread's index. Only the TMEM atoms fix lanes, and a
`tcgen05` warp has 32 lanes; `INDEX_WARP` leaves room for 64."""


def _log2(n: Int) -> Int:
    return Int(count_trailing_zeros(UInt64(n)))


def _logical_bits[L: TensorLayout]() -> Int:
    return _log2(L.static_product)


@fieldwise_init
struct _Plan(TrivialRegisterPassable):
    """How a TileOp issues an atom over a memory tile and a register tile."""

    var ok: Bool
    var free_mem: UInt64
    """The memory tile's logical bits that no index bit covers. The TileOp
    issues the atom once for each value of these bits."""
    var free_reg: UInt64
    """The register tile's bits that match `free_mem`, in the same order."""
    var thread_play: SIMD[DType.int64, _MAX_THREAD_BITS]
    """For each bit of a thread's index, the memory logical bit it selects,
    or -1."""
    var base_threads: UInt64
    """The thread index bits each thread adds to its issues' memory base:
    all of them but the lanes the instruction adds itself."""
    var values: Int
    """The elements one issue moves per thread."""
    var value_stride: Int
    """The memory offset between adjacent values of one issue."""
    var value_offsets: UInt64
    """The memory offset bits that the values of one issue cover. The atom
    adds these offsets to the issue's base, so the base must keep them
    clear."""


def _no_plan() -> _Plan:
    return {False, 0, 0, SIMD[DType.int64, _MAX_THREAD_BITS](-1), 0, 0, 0, 0}


def _offset_map[
    E: TensorEngine
](encoding: AddressEncoding, eb: Int) -> BitLayout:
    """Returns the map from an address encoding's bits to engine `E`'s
    element offset bits, or the empty map when they do not fit."""
    var m = BitLayout()
    if encoding == AddressEncoding.BYTES:
        var shift = _log2(eb)
        for a in range(shift, 64):
            m = m.put(a, bit(a - shift))
    elif encoding == AddressEncoding.TMEM and same_engine[E, TMemEngine]():
        for i in range(_log2(TMEM_NUM_LANES)):
            m = m.put(TMEM_LANE + i, bit(i))
        for c in range(TMEM_LANE - TMEM_COLUMN):
            m = m.put(TMEM_COLUMN + c, bit(_log2(TMEM_NUM_LANES) + c))
    return m


def _image(m: BitLayout, x: UInt64) -> UInt64:
    """Returns `m(x)`, or all ones when `x` has bits outside `m`."""
    if x & ~m.dom != 0:
        return _NO_MATCH
    return m(x)


def _rank_in(mask: UInt64, b: Int) -> Int:
    """Returns how many set bits of `mask` are below bit `b`."""
    return Int(pop_count(mask & bits(0, b)))


def _plan[
    MemMap: OperandMap,
    RegMap: OperandMap,
    LM: TensorLayout,
    LR: TensorLayout,
    EM: TensorEngine,
    eb: Int,
    threads: Int,
]() -> _Plan:
    comptime if is_bit_linear[LM]() and is_bit_linear[LR]():
        return _linear_plan[MemMap, RegMap, LM, LR, EM, eb, threads]()
    else:
        return _no_plan()


def _linear_plan[
    MemMap: OperandMap,
    RegMap: OperandMap,
    LM: TensorLayout,
    LR: TensorLayout,
    EM: TensorEngine,
    eb: Int,
    threads: Int,
]() -> _Plan:
    if RegMap.encoding != AddressEncoding.REGISTERS:
        return _no_plan()
    var conv = _offset_map[EM](MemMap.encoding, eb)
    if conv.dom == 0:
        return _no_plan()
    var mem = placement[LM]()
    var regs = placement[LR]()
    var n = _log2(LM.static_product)
    var thread_index = bits(INDEX_LANE, 64 - INDEX_LANE)
    # Register-only index bits, other than the lanes that pick the thread,
    # would need a register the memory side cannot name.
    if (RegMap.fixed.dom & ~thread_index) & ~MemMap.fixed.dom != 0:
        return _no_plan()
    var plays = SIMD[DType.int64, 64](-1)
    var played: UInt64 = 0
    var index = MemMap.fixed.dom
    while index != 0:
        var d = Int(count_trailing_zeros(index))
        index &= index - 1
        var want = _image(conv, MemMap.fixed.img[d])
        if want == _NO_MATCH:
            return _no_plan()
        var found = -1
        for b in range(n):
            if played & bit(b) == 0 and mem.img[b] == want:
                found = b
                break
        if found < 0:
            return _no_plan()
        plays[d] = Int64(found)
        played |= bit(found)
    var nt = _log2(threads)
    if threads != 1 << nt or nt > _MAX_THREAD_BITS:
        return _no_plan()
    var avoid = _image(conv, MemMap.avoid & conv.dom)
    var thread_play = SIMD[DType.int64, _MAX_THREAD_BITS](-1)
    var lanes_added: UInt64 = 0
    for d in range(INDEX_LANE, 64):
        if plays[d] < 0:
            continue
        if d < INDEX_WARP and d - INDEX_LANE >= _WARP_LANE_BITS:
            return _no_plan()
        var t = (
            d - INDEX_LANE if d
            < INDEX_WARP else _WARP_LANE_BITS + d - INDEX_WARP
        )
        if t >= nt or thread_play[t] >= 0:
            return _no_plan()
        thread_play[t] = plays[d]
        if d < INDEX_WARP:
            lanes_added |= bit(t)
    for t in range(nt):
        if thread_play[t] >= 0:
            continue
        var best = -1
        for b in range(n):
            if played & bit(b) != 0 or mem.img[b] & avoid != 0:
                continue
            if best < 0 or mem.img[b] < mem.img[best]:
                best = b
        if best < 0:
            return _no_plan()
        thread_play[t] = Int64(best)
        played |= bit(best)
    var by_thread: UInt64 = 0
    for t in range(nt):
        by_thread |= bit(Int(thread_play[t]))
    var own = bits(0, n) & ~by_thread
    if Int(pop_count(own)) != _log2(LR.static_product):
        return _no_plan()
    var values = 1
    var value_stride = 0
    var value_offsets: UInt64 = 0
    for d in range(INDEX_LANE):
        if plays[d] < 0:
            continue
        var b = Int(plays[d])
        var r = _rank_in(own, b)
        if regs.img[r] != RegMap.fixed.img[d]:
            return _no_plan()
        values *= 2
        value_offsets |= mem.img[b]
        var step = Int(mem.img[b])
        if d == 0:
            value_stride = step
        elif step != value_stride << d:
            return _no_plan()
    var free_mem = own & ~played
    var free_reg: UInt64 = 0
    var rest = free_mem
    while rest != 0:
        var b = Int(count_trailing_zeros(rest))
        rest &= rest - 1
        if mem.img[b] & avoid != 0:
            return _no_plan()
        var r = _rank_in(own, b)
        if regs.img[r] & RegMap.avoid != 0:
            return _no_plan()
        free_reg |= bit(r)
    return {
        True,
        free_mem,
        free_reg,
        thread_play,
        bits(0, nt) & ~lanes_added,
        values,
        value_stride,
        value_offsets,
    }


def can_tiled_load[
    A: LoadAtom,
    LR: TensorLayout,
    LM: TensorLayout,
    EM: TensorEngine,
    *,
    threads: Int = 1,
]() -> Bool:
    """Returns whether `tiled_load` can move a memory tile of layout `LM` in
    engine `EM` into register tiles of layout `LR` with atom `A`.

    Parameters:
        A: The load atom.
        LR: Each thread's register tile layout.
        LM: The memory tile's layout.
        EM: The memory tile's engine.
        threads: The threads that issue together.

    Returns:
        True when both layouts are F2-linear, every index bit `A` fixes is
        played by a logical bit, `threads` covers the lanes and warps `A`
        fixes, and the register tile holds exactly the elements left to each
        thread.
    """
    comptime plan = _plan[
        A.mem_map, A.reg_map, LM, LR, EM, size_of[A.dtype](), threads
    ]()
    return plan.ok


def can_tiled_store[
    A: StoreAtom,
    LM: TensorLayout,
    EM: TensorEngine,
    LR: TensorLayout,
    *,
    threads: Int = 1,
]() -> Bool:
    """Returns whether `tiled_store` can move register tiles of layout `LR`
    into a memory tile of layout `LM` in engine `EM` with atom `A`.

    Parameters:
        A: The store atom.
        LM: The memory tile's layout.
        EM: The memory tile's engine.
        LR: Each thread's register tile layout.
        threads: The threads that issue together.

    Returns:
        True when both layouts are F2-linear, every index bit `A` fixes is
        played by a logical bit, `threads` covers the lanes and warps `A`
        fixes, and the register tile holds exactly the elements left to each
        thread.
    """
    comptime plan = _plan[
        A.mem_map, A.reg_map, LM, LR, EM, size_of[A.dtype](), threads
    ]()
    return plan.ok


@always_inline
def _deposit(j: Int, free: UInt64) -> UInt64:
    """Spreads the bits of `j` over the set bits of `free`, low bit first."""
    var x: UInt64 = 0
    var rest = free
    var k = 0
    while rest != 0:
        var b = count_trailing_zeros(rest)
        rest &= rest - 1
        if (j >> k) & 1 != 0:
            x |= UInt64(1) << b
        k += 1
    return x


@always_inline
def _thread_bits(plan: _Plan, thread: Int) -> UInt64:
    """Returns the memory logical bits this thread's index adds to its
    issues' base."""
    var x: UInt64 = 0
    comptime for t in range(_MAX_THREAD_BITS):
        if (
            plan.thread_play[t] >= 0
            and (plan.base_threads >> UInt64(t)) & 1 != 0
            and (thread >> t) & 1 != 0
        ):
            x |= bit(Int(plan.thread_play[t]))
    return x


def _mode_shift[L: TensorLayout](i: Int) -> Int:
    """Returns the lowest logical bit of mode `i`."""
    var shift = 0
    comptime for m in range(L.rank):
        if m < i:
            shift += _log2(L.static_shape[m])
    return shift


@always_inline
def _coord[L: TensorLayout, i: Int](x: UInt64) -> Int:
    """Returns mode `i`'s coordinate of logical point `x`."""
    comptime shift = _mode_shift[L](i)
    return Int((x >> UInt64(shift)) & UInt64(L.static_shape[i] - 1))


@always_inline
def _offset_of(t: TileTensor, x: UInt64) -> UInt64:
    """Returns the offset of `t`'s logical point `x`, from its own layout, so
    that a swizzled view's XOR base is applied."""
    comptime L = t.LayoutType
    var c = DynamicCoord[.int64, L.rank]()
    comptime for i in range(L.rank):
        Pointer(to=c[i]).write(
            rebind[type_of(c[i])](Scalar[.int64](_coord[L, i](x)))
        )
    return UInt64(Int(t.layout[linear_idx_type=t.linear_idx_type](c)))


def _base_clear(plan: _Plan, t: TileTensor) -> Bool:
    """Returns whether `t`'s XOR base keeps the bits that the values of one
    issue cover clear. The plan checks the layout's bit images, which
    exclude that runtime base."""
    return _offset_of(t, 0) & plan.value_offsets == 0


@always_inline
def _at[
    L: TensorLayout
](t: TileTensor, offset: UInt64, layout: L) -> t.OffsetViewType[
    TypeList.of[Scalar[t.linear_idx_type]](), L
]:
    """Returns the view of `t` from `offset` elements, with layout `L`."""
    return {t._offset_storage(Scalar[t.linear_idx_type](offset)), layout}


def _check_completion[C: AnyType]():
    comptime assert not (
        conforms_to(C, BarrierCompletion)
        or conforms_to(C, BatchCompletion)
        or conforms_to(C, HandleCompletion)
    ), "per-thread TileOps support synchronous and counter completions"


comptime _IssueLayout[values: Int, stride: Int] = Layout[
    Coord[ComptimeInt[1], ComptimeInt[values]].element_types,
    Coord[ComptimeInt[1], ComptimeInt[stride]].element_types,
]
"""One issue's view of an operand: `values` elements `stride` apart."""


@always_inline
def tiled_load[
    A: LoadAtom,
    T: DType,
    LR: TensorLayout,
    ER: TensorEngine,
    LM: TensorLayout,
    EM: TensorEngine,
    *,
    threads: Int = 1,
](
    atom: A,
    dst: MutTileTensor[T, LR, Engine=ER, ...],
    src: ImmTileTensor[T, LM, Engine=EM, ...],
    thread: Int = 0,
):
    """Loads a memory tile into the issuing threads' register tiles with
    `atom`.

    Parameters:
        A: The load atom.
        T: The element type.
        LR: Each thread's register tile layout.
        ER: The register tile's engine.
        LM: The memory tile's layout.
        EM: The memory tile's engine.
        threads: The threads that issue together.

    Args:
        atom: The atom to issue.
        dst: This thread's registers.
        src: The memory tile the issuing threads load together.
        thread: This thread's index among the `threads`, in
            `[0, threads)`. When `A` fixes lanes or warps, its lane bits are
            the lane id and the bits above them the warp id modulo
            `threads / 32`.

    Constraints:
        `can_tiled_load[A, LR, LM, EM, threads=threads]()`, and `A`'s
        completion is synchronous or a counter. `tiled_load` does not wait:
        for a counter, the caller waits with `A.Completion.wait()` before it
        reads the registers.
    """
    comptime assert T == A.dtype, "tiled_load: operand dtype"
    _check_completion[A.Completion]()
    comptime plan = _plan[
        A.mem_map, A.reg_map, LM, LR, EM, size_of[A.dtype](), threads
    ]()
    comptime assert plan.ok, "tiled_load: the atom does not fit the tiles"
    comptime IssueM = _IssueLayout[plan.values, plan.value_stride]
    comptime IssueR = _IssueLayout[plan.values, 1]
    debug_assert(0 <= thread < threads, "tiled_load: thread out of range")
    debug_assert(
        _base_clear(plan, src), "tiled_load: the XOR base splits one access"
    )
    var w = _thread_bits(plan, thread)
    comptime for j in range(1 << Int(pop_count(plan.free_mem))):
        _ = atom(
            _at(dst, _offset_of(dst, _deposit(j, plan.free_reg)), IssueR()),
            _at(src, _offset_of(src, _deposit(j, plan.free_mem) | w), IssueM()),
        )


@always_inline
def tiled_store[
    A: StoreAtom,
    T: DType,
    LM: TensorLayout,
    EM: TensorEngine,
    LR: TensorLayout,
    ER: TensorEngine,
    *,
    threads: Int = 1,
](
    atom: A,
    dst: MutTileTensor[T, LM, Engine=EM, ...],
    src: ImmTileTensor[T, LR, Engine=ER, ...],
    thread: Int = 0,
):
    """Stores the issuing threads' register tiles into a memory tile with
    `atom`.

    Parameters:
        A: The store atom.
        T: The element type.
        LM: The memory tile's layout.
        EM: The memory tile's engine.
        LR: Each thread's register tile layout.
        ER: The register tile's engine.
        threads: The threads that issue together.

    Args:
        atom: The atom to issue.
        dst: The memory tile the issuing threads store together.
        src: This thread's registers.
        thread: This thread's index among the `threads`, in
            `[0, threads)`. When `A` fixes lanes or warps, its lane bits are
            the lane id and the bits above them the warp id modulo
            `threads / 32`.

    Constraints:
        `can_tiled_store[A, LM, EM, LR, threads=threads]()`, and `A`'s
        completion is synchronous or a counter. `tiled_store` does not wait:
        for a counter, the caller waits with `A.Completion.wait()` before it
        reuses the registers.
    """
    comptime assert T == A.dtype, "tiled_store: operand dtype"
    _check_completion[A.Completion]()
    comptime plan = _plan[
        A.mem_map, A.reg_map, LM, LR, EM, size_of[A.dtype](), threads
    ]()
    comptime assert plan.ok, "tiled_store: the atom does not fit the tiles"
    comptime IssueM = _IssueLayout[plan.values, plan.value_stride]
    comptime IssueR = _IssueLayout[plan.values, 1]
    debug_assert(0 <= thread < threads, "tiled_store: thread out of range")
    debug_assert(
        _base_clear(plan, dst), "tiled_store: the XOR base splits one access"
    )
    var w = _thread_bits(plan, thread)
    comptime for j in range(1 << Int(pop_count(plan.free_mem))):
        _ = atom(
            _at(dst, _offset_of(dst, _deposit(j, plan.free_mem) | w), IssueM()),
            _at(src, _offset_of(src, _deposit(j, plan.free_reg)), IssueR()),
        )
