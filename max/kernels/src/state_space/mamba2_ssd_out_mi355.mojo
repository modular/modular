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

"""The chunked Mamba-2 scan's output stage on MI355 (gfx950).

  Y = diag(exp2(cs)) C h^T + ((C B^T) .* M) (dt X) + D X,
  M[i, j] = (j <= i) exp2(cs_i - cs_j)

per chunk slot and head, on the chunked scan's varlen contract
(`mamba2_ssd_chunked`): x, B and C rows of the packed conv output at the
slot's first token, fp32 dt and log2 cumsum `[H, slots, CHUNK]`, the bf16
state entering each slot `[slots, H, P, N]`, bf16 D `[H]` or none, and y
`[T, H, P]` written only on each slot's own rows. P is 64, N 128.

A persistent workgroup walks work items of the used slots: a slot, two heads
of one group (they share C B^T), and one half of the slot's rows. Two load
waves stream the item's per-key vectors (cs, dt, the factored decay V and
D), the entering state h, and 32-token key blocks (B's rows and both heads'
x rows; rows past the slot's end are zeros) into LDS rings; eight MFMA waves
each own 16 rows. A wave starts Y^T with the carry exp2(cs_i) C_i h^T, then
per 32-key group at or below its rows computes S^T = B C^T with 16x16x32
bf16 MFMAs. A lane then holds S for its row at keys {4q.., 16 + 4q..},
exactly an MFMA B operand in that key order, so P = S times the decay (the
exact one in the rows' own group with D on the diagonal, the factored one
below) goes to Y^T += X^T P^T from the registers, X^T read transposed
(`ds_read_tr16_b64`) in the same order. Y leaves through a per-wave LDS
tile as whole 256-byte rows.

This is the Moka component `Kernels/test/gpu/Moka/amd/moka_amd/kernels/
ssd_varlen.mojo` (`ssd_out_amd_plan`), statement for statement, with no
Moka in it: its rings are LDS counters, as the component binds them.
"""

from std.atomic import Atomic, Ordering
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import exp2
from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.utils import StaticTuple

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu.intrinsics import ds_read_tr16_b64
from max.gpu.memory import AddressSpace, external_memory
from max.gpu.sync import barrier, s_waitcnt

comptime L = 256
"""Chunk length: the chunked scan's CHUNK."""
comptime P = 64
comptime N = 128
comptime KB = 32
"""Tokens of one key block, the unit the load waves stream."""
comptime NKB = L // KB
comptime GROUP = 32
"""Keys of one decay group: one MFMA k step."""
comptime LOAD_WAVES = 2
comptime MMA_WAVES = 8
comptime WAVES = LOAD_WAVES + MMA_WAVES
comptime THREADS = WAVES * WARP_SIZE
comptime LOAD_THREADS = LOAD_WAVES * WARP_SIZE
comptime KEY_DEPTH = 4
comptime KEY_CHUNKS = 32 * KB
"""16-byte chunks of a key block: B's rows, then both heads' x rows."""
comptime B_CHUNKS = 16 * KB

comptime Frag = SIMD[DType.bfloat16, 8]
comptime Acc = SIMD[DType.float32, 4]
comptime F4 = SIMD[DType.float32, 4]

comptime LPtr[T: AnyType] = Pointer[
    T, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]
comptime Lds = LPtr[UInt8]
comptime GPtr[T: AnyType] = Pointer[
    T, ImmUntrackedOrigin, address_space=AddressSpace.GLOBAL
]
comptime GMutPtr[T: AnyType] = Pointer[
    T, MutUntrackedOrigin, address_space=AddressSpace.GLOBAL
]
comptime LdsAtomic = Atomic[UInt32, scope="workgroup"]

# The output's LDS. A B row is read 16 bytes a lane, 16 rows at once (272 =
# 68 words: the rows start 4 banks apart); x rows are read transposed, 4
# rows x 32 bytes at once (144 = 36 words: no two of the 4 rows share a
# bank).
comptime OB_PITCH = 2 * N + 16
comptime OX_PITCH = 2 * P + 16
comptime OX_OFF = KB * OB_PITCH
comptime OKEY_BYTES = OX_OFF + 2 * KB * OX_PITCH
# The per-item vectors, [3][2][L] fp32 (V, the log2 cumsum, dt of each
# head), then D of each head.
comptime LOC_V = 0
comptime LOC_CS = 1
comptime LOC_DT = 2
comptime LOC_D = 3 * 2 * L
comptime VEC_BYTES = 4 * (LOC_D + 4)
comptime VEC_DEPTH = 2
# Both heads' entering state, [2 P] rows of N.
comptime H_PITCH = OB_PITCH
comptime H_BYTES = 2 * P * H_PITCH
# Each MFMA wave stages its 16 rows of Y (both heads, 256 bytes a row).
comptime Y_PITCH = 2 * 2 * P + 16
comptime Y_BYTES = 16 * Y_PITCH

comptime OFF_KEYS = 0
comptime OFF_VEC = OFF_KEYS + KEY_DEPTH * OKEY_BYTES
comptime OFF_H = OFF_VEC + VEC_DEPTH * VEC_BYTES
comptime OFF_Y = OFF_H + H_BYTES
comptime OFF_WORDS = OFF_Y + MMA_WAVES * Y_BYTES
# Ring counters: per slot a FULL count then an EMPTY count.
comptime W_KEYS = 0
comptime W_VEC = W_KEYS + 2 * KEY_DEPTH
comptime W_H = W_VEC + 2 * VEC_DEPTH
comptime NWORDS = W_H + 2
comptime OUT_LDS = OFF_WORDS + 4 * NWORDS


@fieldwise_init
struct SsdMi355Params(DevicePassable, ImplicitlyCopyable):
    """Addresses and uniform scalars of both MI355 stages."""

    var slots: Int64
    var heads: Int64
    var heads_per_group: Int64
    var batch: Int64
    var x_ld: Int64
    var bc_ld: Int64
    var x: Int64
    var b: Int64
    var c: Int64
    """C's address (the output; 0 for the state)."""
    var d: Int64
    """D's address (bf16 [H]) or 0."""
    var dt: Int64
    var cs: Int64
    var h: Int64
    """The bf16 entering states (the output; 0 for the state)."""
    var out: Int64
    """The output y, or the fp32 states for the state stage."""
    var info: Int64

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SsdMi355Params"


@fieldwise_init
struct Item(TrivialRegisterPassable):
    """A work item: chunk slot `slot`, heads `h0` and `h0 + 1` of group `g`,
    the slot's first token and length (0: unused), and the output's half of
    the slot's rows (0 for the state)."""

    var slot: Int
    var h0: Int
    var g: Int
    var start: Int
    var len: Int
    var half: Int


struct Table(Movable):
    """A wave's walk over its items, with the chunk table entry of its next
    item in flight: `fetch` uses it when the tags match, then prefetches
    the following one."""

    var info: GPtr[Int32]
    var slots: Int
    var heads: Int
    var hpg: Int
    var halves: Int
    var next_start: Int32
    var next_len: Int32
    var next_tag: Int
    """The prefetched item plus 1; 0 for none."""

    def __init__(out self, p: SsdMi355Params, halves: Int):
        self.info = GPtr[Int32](unsafe_from_address=Int(p.info))
        self.slots = Int(p.slots)
        self.heads = Int(p.heads)
        self.hpg = Int(p.heads_per_group)
        self.halves = halves
        self.next_start = 0
        self.next_len = 0
        self.next_tag = 0

    @always_inline
    def items(self, batch: Int) -> Int:
        """This workgroup's trips over the used slots' items."""
        var used = Int(
            self.info[unsafe_offset=3 * self.slots + batch - 1]
        ) + Int(self.info[unsafe_offset=3 * self.slots + 2 * batch - 1])
        var per = self.heads // 2 * self.halves
        var g = Int(grid_dim.x)
        return max((used * per - Int(block_idx.x) + g - 1) // g, 0)

    @always_inline
    def decode(self, item: Int, start: Int, n: Int) -> Item:
        var pairs = self.heads // 2
        var slot, rest = divmod(item, pairs * self.halves)
        var hp, half = divmod(rest, self.halves)
        var len = max(n, 0)
        return Item(
            slot,
            2 * hp,
            2 * hp // self.hpg,
            start if len > 0 else 0,
            len,
            half,
        )

    @always_inline
    def fetch(mut self, item: Int) -> Item:
        var slot = item // (self.heads // 2 * self.halves)
        var start: Int
        var n: Int
        if self.next_tag == item + 1:
            n = Int(self.next_len)
            start = Int(self.next_start)
        else:
            n = Int(self.info[unsafe_offset=self.slots + slot])
            start = Int(self.info[unsafe_offset=slot])
        var it = self.decode(item, start, n)
        var nxt = item + Int(grid_dim.x)
        if nxt < self.slots * (self.heads // 2) * self.halves:
            var ns = nxt // (self.heads // 2 * self.halves)
            self.next_start = self.info[unsafe_offset=ns]
            self.next_len = self.info[unsafe_offset=self.slots + ns]
            self.next_tag = nxt + 1
        return it


@always_inline
def lds_at[T: AnyType](base: Lds, offset: Int) -> LPtr[T]:
    return base.unsafe_offset(offset).unsafe_bitcast[T]()


@always_inline
def gfrag(addr: Int) -> Frag:
    """16 bytes of device memory; a global (not generic) access, so an LDS
    wait never drains it."""
    return GPtr[Frag](unsafe_from_address=addr)[]


@always_inline
def tr_frag(base: Lds, row0: Int, row1: Int, pitch: Int, col: Int) -> Frag:
    """An MFMA operand from a token-major LDS tile: lane l of quad q gets
    column `col + l % 16` of rows `row0 + 4 q`.. and `row1 + 4 q`.. (4 each)
    for its 8 k values. `ds_read_tr16_b64` hands lane s of each 16-lane
    group column s of the 4 x 16 block its lanes address, lane s pointing
    at row s / 4, columns 4 (s % 4).."""
    var l = lane_id()
    var s, q = l % 16, l // 16
    var off = 2 * (col + 4 * (s % 4))
    var lo = ds_read_tr16_b64(
        lds_at[BFloat16](base, (row0 + 4 * q + s // 4) * pitch + off)
    )
    var hi = ds_read_tr16_b64(
        lds_at[BFloat16](base, (row1 + 4 * q + s // 4) * pitch + off)
    )
    return lo.join(hi)


# ===----------------------------------------------------------------------=== #
# Rings
# ===----------------------------------------------------------------------=== #


@always_inline
def wait_count(word: LPtr[UInt32], target: Int):
    """Spins until the count reaches `target`, sleeping between polls so a
    waiting wave leaves the LDS and the issue slots to the working ones."""
    while Int(LdsAtomic.load[ordering=Ordering.ACQUIRE](word)) < target:
        llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))


@always_inline
def arrive(word: LPtr[UInt32]):
    """The wave's LDS operations complete, then its leader adds 1. What a
    ring slot carries is LDS that threads wrote or read, and LDS runs a
    wave's operations in order, so device loads and stores stay in flight."""
    s_waitcnt[lgkmcnt=0]()
    if lane_id() == 0:
        _ = LdsAtomic.fetch_add[ordering=Ordering.RELEASE](word, UInt32(1))


struct Ring[depth: Int, producers: Int, consumers: Int](Movable):
    """One side's cursor over a ring of `depth` slots whose FULL count each
    of `producers` waves and EMPTY count each of `consumers` waves add 1 to
    per trip."""

    var words: LPtr[UInt32]
    var trip: Int

    def __init__(out self, words: LPtr[UInt32], word: Int):
        self.words = words.unsafe_offset(word)
        self.trip = 0

    @always_inline
    def acquire(mut self) -> Int:
        """The producer's slot for this trip, once its consumers freed it."""
        var gen, slot = divmod(self.trip, Self.depth)
        if gen > 0:
            wait_count(
                self.words.unsafe_offset(Self.depth + slot),
                Self.consumers * gen,
            )
        return slot

    @always_inline
    def commit(mut self):
        arrive(self.words.unsafe_offset(self.trip % Self.depth))
        self.trip += 1

    @always_inline
    def wait(mut self) -> Int:
        """The consumer's slot for this trip, once its producers filled it."""
        var gen, slot = divmod(self.trip, Self.depth)
        wait_count(self.words.unsafe_offset(slot), Self.producers * (gen + 1))
        return slot

    @always_inline
    def release(mut self):
        arrive(self.words.unsafe_offset(Self.depth + self.trip % Self.depth))
        self.trip += 1


@always_inline
def load_keys[
    b_pitch: Int, x_pitch: Int
](p: SsdMi355Params, it: Item, t0: Int, live: Int, slot: Lds):
    """Key block rows [t0, t0 + KB) of the item into `slot`: B's rows at
    `b_pitch`, then each head's x rows at `x_pitch`; rows from `live` on are
    zeros. Every load issues before the first LDS write."""
    comptime per = KEY_CHUNKS // LOAD_THREADS
    comptime assert per * LOAD_THREADS == KEY_CHUNKS, "chunks split unevenly"
    comptime x_off = KB * b_pitch
    var xld = Int(p.x_ld)
    var bcld = Int(p.bc_ld)
    var me = Int(thread_idx.x)
    var vals = StaticTuple[Frag, per]()
    comptime for k in range(per):
        var c = me + k * LOAD_THREADS
        vals[k] = Frag(0)
        if c < B_CHUNKS:
            var row, q = divmod(c, 16)
            if row < live:
                vals[k] = gfrag(
                    Int(p.b)
                    + 2 * ((it.start + t0 + row) * bcld + it.g * N + 8 * q)
                )
        else:
            var hh, r = divmod(c - B_CHUNKS, 8 * KB)
            var row, q = divmod(r, 8)
            if row < live:
                vals[k] = gfrag(
                    Int(p.x)
                    + 2
                    * ((it.start + t0 + row) * xld + (it.h0 + hh) * P + 8 * q)
                )
    comptime for k in range(per):
        var c = me + k * LOAD_THREADS
        if c < B_CHUNKS:
            var row, q = divmod(c, 16)
            lds_at[Frag](slot, row * b_pitch + 16 * q)[] = vals[k]
        else:
            var hh, r = divmod(c - B_CHUNKS, 8 * KB)
            var row, q = divmod(r, 8)
            lds_at[Frag](
                slot, x_off + (hh * KB + row) * x_pitch + 16 * q
            )[] = vals[k]


@always_inline
def zero_words(words: LPtr[UInt32], nwords: Int):
    """Every ring count to 0, before any wave touches a ring."""
    var t = Int(thread_idx.x)
    if t < nwords:
        words[unsafe_offset=t] = 0
    barrier()


def resident_workgroups[
    kernel: def(SsdMi355Params) thin -> None
](ctx: DeviceContext, lds: Int) raises -> Int:
    """The workgroups of `kernel` the device holds at once."""
    var f = ctx.compile_function[kernel]()
    return f.occupancy_max_active_blocks_per_multiprocessor(
        THREADS, lds
    ) * ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)


# ===----------------------------------------------------------------------=== #
# Output
# ===----------------------------------------------------------------------=== #


@always_inline
def _row_tile(wave: Int, half: Int) -> Int:
    """MFMA wave `wave`'s row tile in half `half` of the slot's rows."""
    return MMA_WAVES * half + wave


@always_inline
def _key_blocks(half: Int) -> Int:
    """The key blocks half `half`'s rows reach."""
    return (half + 1) * (NKB // 2)


@always_inline
def _out_vec(p: SsdMi355Params, it: Item, sv_t: Lds):
    """The item's per-key vectors for each head: the log2 cumsum, dt, and
    V_j = exp2(cs[32 (k + 1)] - cs_j) dt_j for key j of group k, the decay
    to the next group's first key (the last group's is never used), and D
    of each head."""
    comptime per = 2 * L // 4 // LOAD_THREADS
    var sv = sv_t.unsafe_bitcast[Float32]()
    var cs = GPtr[Float32](unsafe_from_address=Int(p.cs))
    var slots = Int(p.slots)
    var me = Int(thread_idx.x)
    # four consecutive keys a thread: every load issues before the math
    var c4 = StaticTuple[F4, per]()
    var d4 = StaticTuple[F4, per]()
    var a1 = StaticTuple[Float32, per]()
    comptime for k in range(per):
        var hh, j = divmod(4 * (me + k * LOAD_THREADS), L)
        var v = ((it.h0 + hh) * slots + it.slot) * L
        c4[k] = GPtr[F4](unsafe_from_address=Int(p.cs) + 4 * (v + j))[]
        d4[k] = GPtr[F4](unsafe_from_address=Int(p.dt) + 4 * (v + j))[]
        var g1 = min(j // GROUP + 1, L // GROUP - 1)
        a1[k] = cs[unsafe_offset=v + GROUP * g1]
    var da = Int(p.d)
    var dh = Float32(0)
    if me < 2 and da != 0:
        dh = GPtr[BFloat16](unsafe_from_address=da)[
            unsafe_offset=it.h0 + me
        ].cast[DType.float32]()
    comptime for k in range(per):
        var hh, j = divmod(4 * (me + k * LOAD_THREADS), L)
        var f = exp2(F4(a1[k]) - c4[k])
        f *= d4[k]
        if j // GROUP + 1 >= L // GROUP:
            f = F4(0)
        lds_at[F4](sv_t, 4 * ((2 * LOC_V + hh) * L + j))[] = f
        lds_at[F4](sv_t, 4 * ((2 * LOC_CS + hh) * L + j))[] = c4[k]
        lds_at[F4](sv_t, 4 * ((2 * LOC_DT + hh) * L + j))[] = d4[k]
    if me < 2:
        sv[unsafe_offset=LOC_D + me] = dh


@always_inline
def _out_h(p: SsdMi355Params, it: Item, dst: Lds):
    """The bf16 state entering the item's slot, both heads' [P, N], into LDS
    rows."""
    comptime per = 2 * P * N * 2 // 16 // LOAD_THREADS
    # both heads' rows are contiguous: [2 P] rows of N
    var src = Int(p.h) + 2 * ((it.slot * Int(p.heads) + it.h0) * P * N)
    var me = Int(thread_idx.x)
    var vals = StaticTuple[Frag, per]()
    comptime for k in range(per):
        vals[k] = gfrag(src + 16 * (me + k * LOAD_THREADS))
    comptime for k in range(per):
        var row, q = divmod(me + k * LOAD_THREADS, 16)
        lds_at[Frag](dst, row * H_PITCH + 16 * q)[] = vals[k]


@always_inline
def _load_c(p: SsdMi355Params, it: Item, wave: Int) -> StaticTuple[Frag, 4]:
    """C's rows of the wave's row tile as MFMA B fragments; rows past the
    slot's end read its last row (finite, never stored)."""
    var l = lane_id()
    var s, q = l % 16, l // 16
    var i = min(16 * _row_tile(wave, it.half) + s, it.len - 1)
    var cf = StaticTuple[Frag, 4]()
    comptime for nk in range(4):
        cf[nk] = gfrag(
            Int(p.c)
            + 2 * ((it.start + i) * Int(p.bc_ld) + it.g * N + 32 * nk + 8 * q)
        )
    return cf


@always_inline
def _out_load(p: SsdMi355Params, lds: Lds, mut tab: Table):
    var words = lds_at[UInt32](lds, OFF_WORDS)
    var vec = Ring[VEC_DEPTH, LOAD_WAVES, MMA_WAVES](words, W_VEC)
    var hst = Ring[1, LOAD_WAVES, MMA_WAVES](words, W_H)
    var keys = Ring[KEY_DEPTH, LOAD_WAVES, MMA_WAVES](words, W_KEYS)
    var T = tab.items(Int(p.batch))
    for k in range(T):
        var it = tab.fetch(Int(block_idx.x) + k * Int(grid_dim.x))
        var v = vec.acquire()
        if it.len > 0:
            _out_vec(p, it, lds.unsafe_offset(OFF_VEC + v * VEC_BYTES))
        vec.commit()
        _ = hst.acquire()
        if it.len > 0:
            _out_h(p, it, lds.unsafe_offset(OFF_H))
        hst.commit()
        for kb in range(NKB):
            var s = keys.acquire()
            var t0 = kb * KB
            if t0 < it.len and kb < _key_blocks(it.half):
                load_keys[OB_PITCH, OX_PITCH](
                    p,
                    it,
                    t0,
                    it.len - t0,
                    lds.unsafe_offset(OFF_KEYS + s * OKEY_BYTES),
                )
            keys.commit()


@always_inline
def _out_init(
    it: Item,
    wave: Int,
    sv_t: Lds,
    ht: Lds,
    cf: StaticTuple[Frag, 4],
    mut y: StaticTuple[Acc, 8],
):
    """The carry exp2(cs_i) C_i h^T into Y^T for the wave's row tile and
    both heads."""
    var l = lane_id()
    var s, q = l % 16, l // 16
    var sv = sv_t.unsafe_bitcast[Float32]()
    var r = _row_tile(wave, it.half)
    comptime for hh in range(2):
        var f = exp2(sv[unsafe_offset=(2 * LOC_CS + hh) * L + 16 * r + s])
        comptime for pt in range(4):
            var acc = Acc(0)
            # h's rows 16 pt.. of the head: the A operand
            var hrow = (P * hh + 16 * pt + s) * H_PITCH
            comptime for nk in range(4):
                var hf = lds_at[Frag](ht, hrow + 2 * (32 * nk + 8 * q))[]
                mma(acc, hf, cf[nk], acc)
            acc *= f
            y[hh * 4 + pt] = acc


@always_inline
def _out_step(
    it: Item,
    wave: Int,
    kb: Int,
    slot: Lds,
    sv: Lds,
    cf: StaticTuple[Frag, 4],
    dd: StaticTuple[Float32, 2],
    mut y: StaticTuple[Acc, 8],
):
    """Key block `kb` into Y^T for the wave's row tile: per 32-key group at
    or below the row tile's own, S^T = B C^T, P = S times the decay (the
    exact one in the rows' own group, with D on the diagonal; the factored
    one below), then Y^T += X^T P^T with P straight from the registers."""
    var l = lane_id()
    var s, q = l % 16, l // 16
    var r = _row_tile(wave, it.half)
    if not (16 * r + 15 >= kb * KB and 16 * r < it.len):
        return
    var i = 16 * r + s
    var gi = r // 2
    var ci = StaticTuple[Float32, 2]()
    comptime for hh in range(2):
        ci[hh] = lds_at[Float32](sv, 4 * ((2 * LOC_CS + hh) * L + i))[]
    comptime for gg in range(KB // GROUP):
        var g = (KB // GROUP) * kb + gg
        if g <= gi:
            comptime jl = GROUP * gg
            var s0 = Acc(0)
            var s1 = Acc(0)
            comptime for nk in range(4):
                var a0 = lds_at[Frag](
                    slot, (jl + s) * OB_PITCH + 2 * (32 * nk + 8 * q)
                )[]
                var a1 = lds_at[Frag](
                    slot, (jl + 16 + s) * OB_PITCH + 2 * (32 * nk + 8 * q)
                )[]
                mma(s0, a0, cf[nk], s0)
                mma(s1, a1, cf[nk], s1)
            var j0 = GROUP * g + 4 * q
            comptime for hh in range(2):
                var pa: F4
                var pb: F4
                if g < gi:
                    var u = exp2(
                        ci[hh]
                        - lds_at[Float32](
                            sv, 4 * ((2 * LOC_CS + hh) * L + GROUP * (g + 1))
                        )[]
                    )
                    var vo = (2 * LOC_V + hh) * L + j0
                    pa = s0 * lds_at[F4](sv, 4 * vo)[] * u
                    pb = s1 * lds_at[F4](sv, 4 * (vo + 16))[] * u
                else:
                    var co = (2 * LOC_CS + hh) * L + j0
                    var do = (2 * LOC_DT + hh) * L + j0
                    var c4a = lds_at[F4](sv, 4 * co)[]
                    var c4b = lds_at[F4](sv, 4 * (co + 16))[]
                    var d4a = lds_at[F4](sv, 4 * do)[]
                    var d4b = lds_at[F4](sv, 4 * (do + 16))[]
                    pa = s0 * exp2(F4(ci[hh]) - c4a) * d4a
                    pb = s1 * exp2(F4(ci[hh]) - c4b) * d4b
                    var dh = dd[hh]
                    comptime for m in range(4):
                        var ja = j0 + m
                        var jb = j0 + 16 + m
                        if ja == i:
                            pa[m] += dh
                        if jb == i:
                            pb[m] += dh
                        if ja > i:
                            pa[m] = 0
                        if jb > i:
                            pb[m] = 0
                var pf = pa.join(pb).cast[DType.bfloat16]()
                comptime for pt in range(4):
                    var xa = tr_frag(
                        slot.unsafe_offset(OX_OFF + hh * KB * OX_PITCH),
                        jl,
                        jl + 16,
                        OX_PITCH,
                        16 * pt,
                    )
                    mma(y[hh * 4 + pt], xa, pf, y[hh * 4 + pt])


@always_inline
def _out_store(
    p: SsdMi355Params, it: Item, wave: Int, stage: Lds, y: StaticTuple[Acc, 8]
):
    """Y in bf16 for the slot's live rows: the wave writes its Y^T (4
    consecutive p of one row a lane) into its staging tile, then stores
    whole rows, both heads' 256 contiguous bytes, 16 bytes a lane."""
    var l = lane_id()
    var s, q = l % 16, l // 16
    var heads = Int(p.heads)
    # the wave reads back only what it wrote, and LDS keeps one wave's
    # accesses in order
    var r = _row_tile(wave, it.half)
    comptime for hh in range(2):
        comptime for pt in range(4):
            lds_at[SIMD[DType.bfloat16, 4]](
                stage, s * Y_PITCH + 2 * (P * hh + 16 * pt + 4 * q)
            )[] = y[hh * 4 + pt].cast[DType.bfloat16]()
    comptime for k in range(16 * 16 // WARP_SIZE):
        var c = l + k * WARP_SIZE
        var s16, q16 = divmod(c, 16)
        var i = 16 * r + s16
        var v = lds_at[Frag](stage, s16 * Y_PITCH + 16 * q16)[]
        if i < it.len:
            GMutPtr[Frag](
                unsafe_from_address=Int(p.out)
                + 2 * ((it.start + i) * heads * P + it.h0 * P + 8 * q16)
            )[] = v


@always_inline
def _out_mma(p: SsdMi355Params, lds: Lds, mut tab: Table, wave: Int):
    var words = lds_at[UInt32](lds, OFF_WORDS)
    var vec = Ring[VEC_DEPTH, LOAD_WAVES, MMA_WAVES](words, W_VEC)
    var hst = Ring[1, LOAD_WAVES, MMA_WAVES](words, W_H)
    var keys = Ring[KEY_DEPTH, LOAD_WAVES, MMA_WAVES](words, W_KEYS)
    var stage = lds.unsafe_offset(OFF_Y + wave * Y_BYTES)
    var T = tab.items(Int(p.batch))
    var y = StaticTuple[Acc, 8](fill=Acc(0))
    var cf = StaticTuple[Frag, 4](fill=Frag(0))
    # the item plus 1 whose C rows `cf` holds, prefetched by the previous
    # item's store
    var ctag = 0
    for k in range(T):
        var item = Int(block_idx.x) + k * Int(grid_dim.x)
        var it = tab.fetch(item)
        var v = vec.wait()
        var sv = lds.unsafe_offset(OFF_VEC + v * VEC_BYTES)
        _ = hst.wait()
        var dd = StaticTuple[Float32, 2](fill=0)
        if it.len > 0:
            comptime for hh in range(2):
                dd[hh] = lds_at[Float32](sv, 4 * (LOC_D + hh))[]
            if ctag != item + 1:
                cf = _load_c(p, it, wave)
            _out_init(it, wave, sv, lds.unsafe_offset(OFF_H), cf, y)
        hst.release()
        for kb in range(NKB):
            var s = keys.wait()
            if kb * KB < it.len:
                _out_step(
                    it,
                    wave,
                    kb,
                    lds.unsafe_offset(OFF_KEYS + s * OKEY_BYTES),
                    sv,
                    cf,
                    dd,
                    y,
                )
            keys.release()
        vec.release()
        if it.len > 0:
            _out_store(p, it, wave, stage, y)
            # the next item's C rows, from its prefetched table entry
            var nxt = item + Int(grid_dim.x)
            if tab.next_tag == nxt + 1:
                var nit = tab.decode(
                    nxt, Int(tab.next_start), Int(tab.next_len)
                )
                if nit.len > 0:
                    cf = _load_c(p, nit, wave)
                    ctag = nxt + 1


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
def ssd_out_mi355_kernel(params: SsdMi355Params):
    """Persistent: workgroup c takes items c, c + grid, ... Waves 0-1 load,
    2-9 compute."""
    var lds = external_memory[
        UInt8, address_space=AddressSpace.SHARED, alignment=128
    ]()
    zero_words(lds_at[UInt32](lds, OFF_WORDS), NWORDS)
    var tab = Table(params, 2)
    var wave = Int(thread_idx.x) // WARP_SIZE
    if wave < LOAD_WAVES:
        _out_load(params, lds, tab)
    else:
        _out_mma(params, lds, tab, wave - LOAD_WAVES)


def ssd_out_mi355_supported(
    nheads: Int, ngroups: Int, head_dim: Int, dstate: Int
) -> Bool:
    """P 64, N 128, and an even number of heads per group."""
    if head_dim != P or dstate != N or ngroups <= 0 or nheads % ngroups != 0:
        return False
    return (nheads // ngroups) % 2 == 0


def ssd_out_mi355(
    ctx: DeviceContext,
    batch: Int,
    slots: Int,
    nheads: Int,
    ngroups: Int,
    x_addr: Int,
    b_addr: Int,
    c_addr: Int,
    x_ld: Int,
    bc_ld: Int,
    d_addr: Int,
    dt_addr: Int,
    cs_addr: Int,
    h_addr: Int,
    y_addr: Int,
    info_addr: Int,
) raises:
    """Y for every used chunk slot's rows into y ([T, H, P] bf16). x, B and C
    start at their first element, rows `x_ld` and `bc_ld` apart; `h` is the
    bf16 state entering each slot ([slots, H, P, N]) and `d_addr` D's
    address (bf16 [H]) or 0."""
    var params = SsdMi355Params(
        Int64(slots),
        Int64(nheads),
        Int64(nheads // ngroups),
        Int64(batch),
        Int64(x_ld),
        Int64(bc_ld),
        Int64(x_addr),
        Int64(b_addr),
        Int64(c_addr),
        Int64(d_addr),
        Int64(dt_addr),
        Int64(cs_addr),
        Int64(h_addr),
        Int64(y_addr),
        Int64(info_addr),
    )
    var grid = min(
        slots * nheads,
        resident_workgroups[ssd_out_mi355_kernel](ctx, OUT_LDS),
    )
    ctx.enqueue_function[ssd_out_mi355_kernel](
        params, grid_dim=grid, block_dim=THREADS, shared_mem_bytes=OUT_LDS
    )
