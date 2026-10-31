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

"""The chunked Mamba-2 scan's output stage on SM100 (B200).

For each chunk slot of `CHUNK` tokens and head, with `a` the log2 decay
cumsum `cs` of the slot and `h` the state entering it:

  y_i = exp2(cs_i) C_i h^T + sum_{j <= i} (C_i . B_j) exp2(cs_i - cs_j) dt_j x_j
        + D x_i

on the chunked scan's varlen contract (`mamba2_ssd_chunked`): x, B and C are
rows of the packed conv output read by TMA at the slot's first token, dt and
cs are its fp32 `[H, slots, CHUNK]` vectors, `h` is bf16 `[slots, H, P, N]`,
and y is written only on the slot's own rows.

A persistent CTA walks work items of one slot and two heads of one group,
which share C, B and S = C B^T. Five warp roles hand slots to each other
through shared-memory barriers:

- load (1 warp): TMA copies in the order the MMA uses them (C0, h, B0, C1,
  B1, X0, X1) through a ring of six 32 KB slots, and the item's dt and cs.
- mma (1 warp): the carry C [hA; hB]^T into Y, S = C B^T per 128 x 128
  tile, and Y += P X with P read from tensor memory.
- mask (8 warps): each S tile into both heads' P (bf16) in the tile's own
  TMEM columns, with D on the diagonal. Below a row's own 32-key group the
  decay factors as exp2(cs_i - cs_r) exp2(cs_r - cs_j), r the next group's
  first key, both at most 1, so one exp2 a group and a per-key factor
  replace an exp2 per element.
- epilogue (4 warps): the carry's row scale exp2(cs_i) in place before P X
  accumulates, then Y to bf16 and out by TMA, or row by row from registers
  for a slot's partial last 128 rows so a chunk never writes the next
  sequence's rows.

Loads past a sequence's end read the next sequence's rows (or TMA's zero
fill): they reach only rows never stored and keys the causal mask zeroes.

This is the Moka component `Kernels/test/gpu/Moka/kernels/mamba2/
compute_out_varlen_plan.mojo` (after Joe Todd's `compute_out_plan.mojo`),
statement for statement, with no Moka in it.
"""

from std.math import ceildiv, exp2
from std.memory import bitcast
from std.utils import StaticTuple
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.utils.coord import Idx
from std.sys._assembly import inlined_assembly

from layout.tma_async import SharedMemBarrier
from linalg.arch.sm100.mma import MmaOpSM100_SS, _create_mma_desc_k_major
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
    mma_arrive,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import (
    TMADescriptor,
    TensorMapSwizzle,
    create_tma_descriptor,
    prefetch_tma_descriptor,
)
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import (
    AddressSpace,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.sync import (
    barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    named_barrier,
    named_barrier_arrive,
    syncwarp,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import mul_ftz, sub_ftz

comptime SPtr[T: AnyType] = Pointer[
    T, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]

comptime L = 256
"""Tokens of a chunk slot."""
comptime HP = 2
"""Heads of one group a work item takes."""
comptime P = 64
"""Head dim: one head's x, y and state rows are one 128-byte swizzle span."""
comptime N = 128
"""State dim: C, B and the state are two 64-column spans along it."""
comptime ROWS = 128
"""Tokens of a row block: one TMEM lane each."""
comptime NRB = L // ROWS
comptime NS = 6
"""Operand ring slots of 32 KB."""
comptime THREADS = 512
comptime SMS = 148

# shared memory, from the component's dump
comptime OFF_T = 0
comptime SLOT = 32768
comptime OFF_DT = 196608
comptime OFF_CS = 201728
comptime VEC_SLOT = 2560
comptime VEC_ROW = 320
"""The fp32 elements of one head's row in a vector slot: two copy boxes."""
comptime VEC_BOX = 160
comptime OFF_Y = 206848
comptime OFF_V = 223232
comptime OFF_BARS = 229376
comptime OFF_TMEM = 229632
comptime SMEM_BYTES = 229648

# barriers: each ring's full ones, then its empty ones
comptime B_OP = 0
comptime B_DC = 12
comptime B_S = 16
comptime B_P = 20
comptime B_Y = 24
comptime B_TFREE = 28
comptime B_XF = 29
"""The x0 / x1 tiles land here; the fixup then publishes them on the ring."""
comptime NBARS = 31

# named barriers
comptime NB_OSTORE = 1
comptime NB_TPTR = 2
comptime NB_MASK = 3
comptime NB_EPI = 4
comptime NB_FIX = 5

# tensor memory columns
comptime TMEM_COLS = 512
comptime TM_S = 0
"""S (and P in its columns), two stages of 128 columns."""
comptime TM_Y = 256
"""Y, two stages of 64 HP columns."""

# per-thread registers after setmaxnreg: mask, epilogue, the rest
comptime REGS_MASK = 192
comptime REGS_EPI = 88
comptime REGS_REST = 40

# the mask block's scratch: V, then cs and dt, each [HP, L] fp32
comptime LOC_V = 0
comptime LOC_CS = 1
comptime LOC_DT = 2
comptime LOC_N = HP * L


@fieldwise_init
struct SsdOutParams(DevicePassable, ImplicitlyCopyable):
    """TMA descriptors and uniform scalars, passed as one grid constant."""

    var c_desc: TMADescriptor
    var b_desc: TMADescriptor
    var x_desc: TMADescriptor
    var h_desc: TMADescriptor
    var dt_desc: TMADescriptor
    var cs_desc: TMADescriptor
    var y_desc: TMADescriptor
    var slots: Int32
    var batch: Int32
    var heads: Int32
    var heads_per_group: Int32
    var head_pairs: Int32
    var d_addr: Int64
    """D, bf16 [H], or 0 for none."""
    var info_addr: Int64
    var y_addr: Int64

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SsdOutParams"


comptime ParamsPtr = Pointer[SsdOutParams, ImmUntrackedOrigin]


@inline(.always)
def _desc[k: Int](p: ParamsPtr) -> OpaquePointer[ImmUntrackedOrigin]:
    comptime if k == 0:
        return Pointer(to=p[].c_desc).unsafe_bitcast[NoneType]()
    elif k == 1:
        return Pointer(to=p[].b_desc).unsafe_bitcast[NoneType]()
    elif k == 2:
        return Pointer(to=p[].x_desc).unsafe_bitcast[NoneType]()
    elif k == 3:
        return Pointer(to=p[].h_desc).unsafe_bitcast[NoneType]()
    elif k == 4:
        return Pointer(to=p[].dt_desc).unsafe_bitcast[NoneType]()
    elif k == 5:
        return Pointer(to=p[].cs_desc).unsafe_bitcast[NoneType]()
    return Pointer(to=p[].y_desc).unsafe_bitcast[NoneType]()


# ===----------------------------------------------------------------------=== #
# Rings
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Cursor(TrivialRegisterPassable):
    """One side's place in a ring: the next slot and its phase parity."""

    var slot: Int32
    var phase: UInt32

    @staticmethod
    @inline(.always)
    def producer() -> Self:
        """A producer's first wait on an empty slot passes: parity 1."""
        return Self(0, 1)

    @staticmethod
    @inline(.always)
    def consumer() -> Self:
        return Self(0, 0)

    @inline(.always)
    def step[depth: Int](mut self):
        var slot = self.slot + 1
        if slot >= Int32(depth):
            slot = 0
            self.phase ^= 1
        self.slot = slot


struct Bars(TrivialRegisterPassable):
    var base: SPtr[SharedMemBarrier]

    @inline(.always)
    def __init__(out self, smem: SPtr[UInt8], off: Int):
        self.base = smem.unsafe_offset(off).unsafe_bitcast[SharedMemBarrier]()

    @inline(.always)
    def full[ring: Int](self, slot: Int32) -> SPtr[SharedMemBarrier]:
        return self.base.unsafe_offset(ring + Int(slot))

    @inline(.always)
    def empty[
        ring: Int, depth: Int
    ](self, slot: Int32) -> SPtr[SharedMemBarrier]:
        return self.base.unsafe_offset(ring + depth + Int(slot))

    @inline(.always)
    def wait_full[ring: Int, depth: Int](self, mut cu: Cursor) -> Int32:
        var s = cu.slot
        self.full[ring](s)[].wait(cu.phase)
        cu.step[depth]()
        return s

    @inline(.always)
    def wait_empty[ring: Int, depth: Int](self, mut cu: Cursor) -> Int32:
        var s = cu.slot
        self.empty[ring, depth](s)[].wait(cu.phase)
        cu.step[depth]()
        return s

    @inline(.always)
    def init_ring[base: Int, depth: Int, full: Int, empty: Int](self):
        comptime for k in range(depth):
            comptime if full > 0:
                self.base[unsafe_offset=base + k].init(Int32(full))
            comptime if empty > 0:
                self.base[unsafe_offset=base + depth + k].init(Int32(empty))


@inline(.always)
def _arrive_warp(bar: SPtr[SharedMemBarrier]):
    """A release one lane per warp makes, after the warp's lanes meet."""
    syncwarp()
    if lane_id() == 0:
        _ = bar[].arrive()


# ===----------------------------------------------------------------------=== #
# Work items
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Item(TrivialRegisterPassable):
    var slot: Int
    var hp: Int


@inline(.always)
def _item(p: ParamsPtr, it: Int) -> Item:
    """Work item `it`: head pairs innermost, then slots."""
    var nhp = UInt32(p[].head_pairs)
    var q = UInt32(it) // nhp
    return Item(Int(q), Int(UInt32(it) - q * nhp))


@inline(.always)
def _info(p: ParamsPtr, k: Int) -> Int:
    return Int(
        Pointer[Int32, ImmUntrackedOrigin](
            unsafe_from_address=Int(p[].info_addr)
        )[unsafe_offset=k]
    )


@inline(.always)
def _slot_rows(p: ParamsPtr, slot: Int) -> Tuple[Int, Int]:
    """The slot's first token and length (0 and 0 for an unused slot)."""
    var n = _info(p, Int(p[].slots) + slot)
    if n <= 0:
        return (0, 0)
    return (_info(p, slot), n)


@inline(.always)
def _items_of_cta(p: ParamsPtr) -> Int:
    """Items cta, cta + grid, ... below used slots x head pairs: the used
    slots are the last sequence's first slot plus its count."""
    var slots = Int(p[].slots)
    var nb = Int(p[].batch)
    var used = _info(p, 3 * slots + nb - 1) + _info(p, 3 * slots + 2 * nb - 1)
    var n = UInt32(min(used, slots) * Int(p[].head_pairs))
    var g = UInt32(grid_dim.x)
    var c = UInt32(block_idx.x)
    return Int((n - c + g - 1) // g) if c < n else 0


@inline(.always)
def _vec_start(p: ParamsPtr, w: Item, hh: Int) -> Int:
    """The element of [H, slots, L] where head hh's slot starts."""
    return (w.hp * HP + hh) * (Int(p[].slots) * L) + w.slot * L


# ===----------------------------------------------------------------------=== #
# Ops
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _load_vecs[
    desc_k: Int
](p: ParamsPtr, dst: SPtr[UInt8], w: Item, bar: SPtr[SharedMemBarrier]):
    """Each head's L fp32 of dt or cs from the 16-byte boundary at or below
    its slot's first element, in whole boxes."""
    comptime for hh in range(HP):
        var e = _vec_start(p, w, hh)
        var e0 = e - Int(UInt32(e) % 4)
        comptime for k in range(VEC_ROW // VEC_BOX):
            var d = dst.unsafe_offset((hh * VEC_ROW + k * VEC_BOX) * 4)
            cp_async_bulk_tensor_shared_cluster_global(
                d, _desc[desc_k](p), bar, (e0 + k * VEC_BOX,)
            )


@inline(.always)
def _zero_rows[threads: Int](dst: SPtr[UInt8], live: Int, end: Int, tid: Int):
    """Zeroes rows `[live, end)` of both 64-column spans of a 32 KB slot,
    `threads` threads. A 128-byte row stays one row under the swizzle."""
    var n = (end - live) * 2 * 8
    for k in range(tid, n, threads):
        var r = live + k // 16
        var z = (k // 8) % 2
        var c = k % 8
        dst.unsafe_offset(16384 * z + r * 128 + c * 16).unsafe_bitcast[
            SIMD[DType.uint8, 16]
        ]()[] = SIMD[DType.uint8, 16](0)


@inline(.always)
def _tile(smem: SPtr[UInt8], slot: Int32) -> SPtr[UInt8]:
    return smem.unsafe_offset(OFF_T + Int(slot) * SLOT)


@inline(.always)
def _umma_ss(
    d: UInt32,
    a_lo: SPtr[UInt8],
    b_lo: SPtr[UInt8],
):
    """Computes acc[128 x 128] = A B^T over K 128 as two 64-column spans
    16 KB apart, both K-major 128-byte swizzled; the first MMA overwrites."""
    comptime op = MmaOpSM100_SS[
        DType.float32,
        DType.bfloat16,
        DType.bfloat16,
        (128, 128, 64),
        (128, 128, 16),
        accum_type=DType.float32,
        cta_group=1,
        cluster_shape=(1, 1, 1),
        a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
        b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
        transpose_b=True,
    ]
    comptime for i in range(2):
        op().mma_from_desc(
            _create_mma_desc_k_major[
                DType.bfloat16, TensorMapSwizzle.SWIZZLE_128B
            ](a_lo.unsafe_offset(16384 * i).unsafe_bitcast[BFloat16]()),
            _create_mma_desc_k_major[
                DType.bfloat16, TensorMapSwizzle.SWIZZLE_128B
            ](b_lo.unsafe_offset(16384 * i).unsafe_bitcast[BFloat16]()),
            d,
            i == 0,
        )


@inline(.always)
def _p_col(hh: Int, j: Int) -> Int:
    """TMEM column, from the tile's first, of K step j (keys 16 j .. 16 j +
    15) of head hh's P: key group j / 2 keeps head 0's 16 packed columns,
    then head 1's."""
    return 32 * (j // 2) + 16 * hh + 8 * (j % 2)


@inline(.always)
def _umma_px(y: UInt32, pt: UInt32, x_lo: SPtr[UInt8], ksteps: Int):
    """Y += P X over the first 16 `ksteps` of 128 keys for each head: M 128
    tokens, N 64, K 16 an instruction, P (bf16) from TMEM, X an MN-major
    [128 keys x 64] span. Keys past a sequence's end have P = 0, so the
    steps holding only those are skipped."""
    comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        DType.float32,
        DType.bfloat16,
        DType.bfloat16,
        (UInt32(ROWS), UInt32(P)),
        transpose_b=False,
    ]()
    comptime for hh in range(HP):
        var xd = MMASmemDescriptor.create[
            1024, 16384, TensorMapSwizzle.SWIZZLE_128B
        ](x_lo.unsafe_offset(16384 * hh))
        comptime for j in range(ROWS // 16):
            if j < ksteps:
                mma(
                    pt + UInt32(_p_col(hh, j)),
                    xd + j * 16 * 128,
                    y + UInt32(P * hh),
                    idesc,
                    UInt32(1),
                )


@inline(.always)
def _pk(v: SIMD[DType.float32, 2]) -> UInt32:
    return bitcast[DType.uint32, 1](v.cast[DType.bfloat16]())


@inline(.always)
def _prep(
    p: ParamsPtr,
    dt0: SPtr[Float32],
    cs0: SPtr[Float32],
    loc: SPtr[Float32],
    w: Item,
    rt: Int,
) -> StaticTuple[Float32, HP]:
    """The item's per-key factors and aligned copies of its vectors: for key
    j of key group k, V = exp2(cs[32 (k + 1)] - cs_j) dt_j. Returns D of each
    head."""
    comptime NT = 256
    comptime for hh in range(HP):
        var e = _vec_start(p, w, hh)
        var sk = Int(UInt32(e) % 4)
        var dt = dt0.unsafe_offset(hh * VEC_ROW + sk)
        var cs = cs0.unsafe_offset(hh * VEC_ROW + sk)
        comptime for k in range(L // NT):
            var j = k * NT + rt
            var g1 = j // 32 + 1
            var cj = cs[unsafe_offset=j]
            var dj = dt[unsafe_offset=j]
            var f = Float32(0)
            if g1 < L // 32:
                f = exp2(cs[unsafe_offset=32 * g1] - cj) * dj
            var o = hh * L + j
            loc[unsafe_offset=LOC_V * LOC_N + o] = f
            loc[unsafe_offset=LOC_CS * LOC_N + o] = cj
            loc[unsafe_offset=LOC_DT * LOC_N + o] = dj
    var dd = StaticTuple[Float32, HP](fill=0)
    if p[].d_addr != 0:
        var d = Pointer[BFloat16, ImmUntrackedOrigin](
            unsafe_from_address=Int(p[].d_addr)
        )
        comptime for hh in range(HP):
            dd[hh] = d[unsafe_offset=w.hp * HP + hh].cast[DType.float32]()
    return dd


@inline(.always)
def _mask_group[
    n: Int
](
    s: Array[Float32, n],
    q: Int32,
    kq: Int,
    diag: Bool,
    wq: Int32,
    lane: Int32,
    ci: StaticTuple[Float32, HP],
    dd: StaticTuple[Float32, HP],
    loc: SPtr[Float32],
) -> Array[Scalar[DType.uint32], 16 * HP]:
    """Key group q of a tile for each head, packed: the exact decay in the
    row's own group with D on the diagonal, the factored one below it, else
    zeros."""
    comptime F2 = SIMD[DType.float32, 2]
    var o = Array[Scalar[DType.uint32], 16 * HP](fill=0)
    if diag and q == wq:
        comptime for hh in range(HP):
            var cr = loc.unsafe_offset(LOC_CS * LOC_N + hh * L + 32 * kq)
            var dr = loc.unsafe_offset(LOC_DT * LOC_N + hh * L + 32 * kq)
            var c2 = F2(ci[hh])
            comptime for e in range(0, 32, 4):
                var c4 = cr.unsafe_load[width=4, alignment=16](e)
                var d4 = dr.unsafe_load[width=4, alignment=16](e)
                comptime for h in range(2):
                    comptime e2 = e + 2 * h
                    var x = sub_ftz(c2, F2(c4[2 * h], c4[2 * h + 1]))
                    var f = exp2(x)
                    var pr = mul_ftz(
                        mul_ftz(F2(s[e2], s[e2 + 1]), f),
                        F2(d4[2 * h], d4[2 * h + 1]),
                    )
                    var a = pr[0] + (
                        dd[hh] if Int32(e2) == lane else Float32(0)
                    )
                    var c = pr[1] + (
                        dd[hh] if Int32(e2 + 1) == lane else Float32(0)
                    )
                    a = a if Int32(e2) <= lane else Float32(0)
                    c = c if Int32(e2 + 1) <= lane else Float32(0)
                    o[16 * hh + e2 // 2] = _pk(F2(a, c))
    elif not diag or q < wq:
        comptime for hh in range(HP):
            var u = exp2(
                ci[hh]
                - loc[unsafe_offset=LOC_CS * LOC_N + hh * L + 32 * (kq + 1)]
            )
            var u2 = F2(u)
            var vr = loc.unsafe_offset(LOC_V * LOC_N + hh * L + 32 * kq)
            comptime for e in range(0, 32, 4):
                var v4 = vr.unsafe_load[width=4, alignment=16](e)
                var p0 = mul_ftz(
                    mul_ftz(F2(s[e], s[e + 1]), F2(v4[0], v4[1])), u2
                )
                var p1 = mul_ftz(
                    mul_ftz(F2(s[e + 2], s[e + 3]), F2(v4[2], v4[3])), u2
                )
                o[16 * hh + e // 2] = _pk(p0)
                o[16 * hh + e // 2 + 1] = _pk(p1)
    return o^


@inline(.always)
def _mask_tile(
    sb: UInt32,
    pb: UInt32,
    loc: SPtr[Float32],
    t: Int32,
    rt: Int32,
    dd: StaticTuple[Float32, HP],
):
    """P for one 128 x 128 S tile and each head, into the tile's own
    columns. Tile t of an item is row block (t + 1) / 2 and key block t / 2:
    (0, 0), (1, 0), (1, 1). Each of the two warpgroups takes every other key
    group of the same 128 rows."""
    comptime NG = 2
    var rb = (t + 1) >> 1
    var cb = t >> 1
    var diag = rb == cb
    var wg = rt >> 7
    var r = rt & 127
    var wq = r >> 5
    var lane = r & 31
    var i = rb * ROWS + r
    var ci = StaticTuple[Float32, HP](fill=0)
    comptime for hh in range(HP):
        ci[hh] = loc[unsafe_offset=LOC_CS * LOC_N + hh * L + Int(i)]
    comptime ldx = tcgen05_ld[
        datapaths=32, bits=32, repeat=32, dtype=DType.float32, pack=False
    ]
    for jq in range(_opaque_i32(Int32(4 // NG))):
        var q = wg + Int32(NG) * jq
        var s = ldx(sb + UInt32(32 * q))
        tcgen05_load_wait()
        var o = _mask_group(s, q, Int(cb * 4 + q), diag, wq, lane, ci, dd, loc)
        tcgen05_st[datapaths=32, bits=32, repeat=16 * HP, pack=False](
            pb + UInt32(32 * q), o
        )
    tcgen05_store_wait()


@inline(.always)
def _opaque_i32(v: Int32) -> Int32:
    """A value the compiler cannot see through: a loop bound it cannot
    unroll keeps one copy of the body in the instruction cache."""
    return inlined_assembly[
        "mov.b32 $0, $1;", Int32, constraints="=r,r", has_side_effect=True
    ](v)


@inline(.always)
def _epi_scale(
    y: UInt32, cs0: SPtr[Float32], p: ParamsPtr, w: Item, rb: Int, rt: Int
):
    """The carry's row scale in place: row i of each head's C h^T times
    exp2(cs_i), before P X accumulates on it."""
    var i = rb * ROWS + rt
    comptime for hh in range(HP):
        var r = tcgen05_ld[
            datapaths=32, bits=32, repeat=P, dtype=DType.float32, pack=False
        ](y + UInt32(P * hh))
        var e = _vec_start(p, w, hh)
        var cs = cs0.unsafe_offset(hh * VEC_ROW + Int(UInt32(e) % 4))
        var f = exp2(cs[unsafe_offset=i])
        tcgen05_load_wait()
        comptime for k in range(P):
            r[k] = r[k] * f
        tcgen05_st[datapaths=32, bits=32, repeat=P, pack=False](
            y + UInt32(P * hh), r
        )
    tcgen05_store_wait()


@inline(.always)
def _stage_row[
    b: Int, n: Int, //
](dst: SPtr[UInt8], v: Array[Float32, n], row: Int):
    """32 columns (half b of a 64-column row) as four 16-byte chunks; chunk
    j of the 128-byte row lands at j ^ (row mod 8)."""
    comptime for q in range(4):
        var x = SIMD[DType.bfloat16, 8]()
        comptime for e in range(8):
            x[e] = v[8 * q + e].cast[DType.bfloat16]()
        var j = 4 * b + q
        var off = row * 128 + ((j ^ (row % 8)) * 16)
        dst.unsafe_offset(off).unsafe_bitcast[BFloat16]().unsafe_store[
            alignment=16
        ](0, x)


@inline(.always)
def _epi_stage(
    dst: SPtr[UInt8],
    y: UInt32,
    hh: Int,
    p: ParamsPtr,
    w: Item,
    rb: Int,
    row: Int,
):
    """One head's 64 columns of Y to bf16: into the swizzled staging tile for
    a whole row block, or straight to y from the registers for a partial
    one's live rows."""
    comptime ld = tcgen05_ld[
        datapaths=32, bits=32, repeat=32, dtype=DType.float32, pack=False
    ]
    var a = y + UInt32(P * hh)
    var v0 = ld(a)
    var v1 = ld(a + UInt32(32))
    tcgen05_load_wait()
    var sl = _slot_rows(p, w.slot)
    var live = sl[1] - rb * ROWS
    if live >= ROWS:
        _stage_row[b=0](dst, v0, row)
        _stage_row[b=1](dst, v1, row)
    elif row < live:
        var ld_y = Int(p[].heads) * P
        var o = (sl[0] + rb * ROWS + row) * ld_y + (w.hp * HP + hh) * P
        var yp = Pointer[BFloat16, MutUntrackedOrigin](
            unsafe_from_address=Int(p[].y_addr)
        ).unsafe_offset(o)
        comptime for q in range(4):
            var x0 = SIMD[DType.bfloat16, 8]()
            var x1 = SIMD[DType.bfloat16, 8]()
            comptime for e in range(8):
                x0[e] = v0[8 * q + e].cast[DType.bfloat16]()
                x1[e] = v1[8 * q + e].cast[DType.bfloat16]()
            yp.unsafe_store[alignment=16](8 * q, x0)
            yp.unsafe_store[alignment=16](32 + 8 * q, x1)


# ===----------------------------------------------------------------------=== #
# The kernel
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _load(p: ParamsPtr, smem: SPtr[UInt8], bars: Bars, T: Int):
    var c_op = Cursor.producer()
    var c_dc = Cursor.producer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)

    @inline(.always)
    def acquire_op() {mut c_op, var bars} -> Int32:
        var s = bars.wait_empty[B_OP, NS](c_op)
        fence_async_view_proxy()
        return s

    for k in range(T):
        var w = _item(p, cta + k * grid)
        var sl = _slot_rows(p, w.slot)
        var g = Int(UInt32(w.hp * HP) // UInt32(p[].heads_per_group))
        var d = bars.wait_empty[B_DC, 2](c_dc)
        fence_async_view_proxy()
        if elect_one_sync():
            var bar = bars.full[B_DC](d)
            bar[].expect_bytes(Int32(2 * HP * VEC_ROW * 4))
            _load_vecs[4](
                p, smem.unsafe_offset(OFF_DT + Int(d) * VEC_SLOT), w, bar
            )
            _load_vecs[5](
                p, smem.unsafe_offset(OFF_CS + Int(d) * VEC_SLOT), w, bar
            )

        # C0, the state, B0, C1, B1, X0, X1, in the MMA's order
        comptime for step in range(7):
            var s = acquire_op()
            if elect_one_sync():
                # x0 / x1 land on B_XF for the fixup, which publishes them
                var bar = bars.full[B_OP](s)
                comptime if step >= 5:
                    bar = bars.base.unsafe_offset(B_XF + step - 5)
                bar[].expect_bytes(Int32(SLOT))
                var dst = _tile(smem, s)
                comptime if step == 1:
                    comptime for hh in range(HP):
                        comptime for half in range(2):
                            cp_async_bulk_tensor_shared_cluster_global(
                                dst.unsafe_offset(8192 * (2 * half + hh)),
                                _desc[3](p),
                                bar,
                                (
                                    half * 64,
                                    (w.slot * Int(p[].heads) + w.hp * HP + hh)
                                    * P,
                                ),
                            )
                elif step == 5 or step == 6:
                    var row = sl[0] + (step - 5) * ROWS
                    comptime for z in range(HP):
                        cp_async_bulk_tensor_shared_cluster_global(
                            dst.unsafe_offset(16384 * z),
                            _desc[2](p),
                            bar,
                            ((w.hp * HP + z) * P, row, 0),
                        )
                else:
                    # C0 (0), B0 (2), C1 (3), B1 (4)
                    comptime is_c = step == 0 or step == 3
                    comptime a = 0 if step <= 2 else 1
                    var row = sl[0] + a * ROWS
                    comptime for z in range(2):
                        cp_async_bulk_tensor_shared_cluster_global(
                            dst.unsafe_offset(16384 * z),
                            _desc[0 if is_c else 1](p),
                            bar,
                            (g * N + z * 64, row, 0),
                        )
    # the last phases: every slot back before the CTA exits
    comptime for _ in range(NS):
        _ = bars.wait_empty[B_OP, NS](c_op)
    comptime for _ in range(2):
        _ = bars.wait_empty[B_DC, 2](c_dc)


@inline(.always)
def _mma(p: ParamsPtr, smem: SPtr[UInt8], bars: Bars, T: Int):
    var slot_ptr = smem.unsafe_offset(OFF_TMEM).unsafe_bitcast[UInt32]()
    tcgen05_alloc[1](slot_ptr, UInt32(TMEM_COLS))
    syncwarp()
    var tmem = slot_ptr[]
    named_barrier_arrive[Int32(32 + 256 + 128)](Int32(NB_TPTR))
    var c_op = Cursor.consumer()
    var c_y = Cursor.producer()
    var c_s = Cursor.producer()
    var c_p = Cursor.consumer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)

    @inline(.always)
    def s_col(s: Int32) {var tmem} -> UInt32:
        return tmem + UInt32(TM_S + 128 * Int(s))

    @inline(.always)
    def y_col(s: Int32) {var tmem} -> UInt32:
        return tmem + UInt32(TM_Y + 64 * HP * Int(s))

    @inline(.always)
    def commit(bar: SPtr[SharedMemBarrier]):
        if elect_one_sync():
            mma_arrive[1](bar)

    @inline(.always)
    def acq_y() {mut c_y, var bars} -> Int32:
        var s = bars.wait_empty[B_Y, 2](c_y)
        tcgen05_fence_after()
        return s

    @inline(.always)
    def acq_s() {mut c_s} -> Int32:
        # implied empty: the MMA's program order frees S
        var s = c_s.slot
        c_s.step[2]()
        tcgen05_fence_after()
        return s

    @inline(.always)
    def wait_p() {mut c_p, var bars} -> Int32:
        var s = bars.wait_full[B_P, 2](c_p)
        tcgen05_fence_after()
        return s

    @inline(.always)
    def wait_op() {mut c_op, var bars} -> Int32:
        return bars.wait_full[B_OP, NS](c_op)

    for k in range(T):
        # 16-key steps of x0 and x1 that hold a live key; x1's are needed only
        # when the slot has rows past 128 (else its P x output rows are dead)
        var sl = _slot_rows(p, _item(p, cta + k * grid).slot)
        var k0 = min(ceildiv(sl[1], 16), ROWS // 16)
        var k1 = min(ceildiv(max(sl[1] - ROWS, 0), 16), ROWS // 16)
        var c0 = wait_op()
        var hs = wait_op()
        var y0 = acq_y()
        if elect_one_sync():
            _umma_ss(y_col(y0), _tile(smem, c0), _tile(smem, hs))
        commit(bars.full[B_Y](y0))
        var b0 = wait_op()
        var s0 = acq_s()
        if elect_one_sync():
            _umma_ss(s_col(s0), _tile(smem, c0), _tile(smem, b0))
        commit(bars.full[B_S](s0))
        commit(bars.empty[B_OP, NS](c0))
        var c1 = wait_op()
        var y1 = acq_y()
        if elect_one_sync():
            _umma_ss(y_col(y1), _tile(smem, c1), _tile(smem, hs))
        commit(bars.full[B_Y](y1))
        commit(bars.empty[B_OP, NS](hs))
        var s1 = acq_s()
        if elect_one_sync():
            _umma_ss(s_col(s1), _tile(smem, c1), _tile(smem, b0))
        commit(bars.full[B_S](s1))
        commit(bars.empty[B_OP, NS](b0))
        var b1 = wait_op()
        var x0 = wait_op()
        var p0 = wait_p()
        var y0b = acq_y()
        if elect_one_sync():
            _umma_px(y_col(y0b), s_col(p0), _tile(smem, x0), k0)
        tcgen05_fence_before()
        commit(bars.full[B_Y](y0b))
        var s2 = acq_s()
        if elect_one_sync():
            _umma_ss(s_col(s2), _tile(smem, c1), _tile(smem, b1))
        commit(bars.full[B_S](s2))
        commit(bars.empty[B_OP, NS](c1))
        commit(bars.empty[B_OP, NS](b1))
        var p1 = wait_p()
        var y1b = acq_y()
        if elect_one_sync():
            _umma_px(y_col(y1b), s_col(p1), _tile(smem, x0), k0)
        tcgen05_fence_before()
        commit(bars.empty[B_OP, NS](x0))
        var x1 = wait_op()
        var p2 = wait_p()
        if elect_one_sync():
            _umma_px(y_col(y1b), s_col(p2), _tile(smem, x1), k1)
        tcgen05_fence_before()
        commit(bars.full[B_Y](y1b))
        commit(bars.empty[B_OP, NS](x1))
    tcgen05_release_allocation_lock[1]()
    bars.full[B_TFREE](0)[].wait(0)
    tcgen05_fence_after()
    tcgen05_dealloc[1](tmem, UInt32(TMEM_COLS))


@inline(.always)
def _fix(p: ParamsPtr, smem: SPtr[UInt8], bars: Bars, T: Int, ft: Int):
    """Publishes each x tile on the operand ring, after zeroing its rows past
    the sequence's end inside the last 16-key step the MMA reads (it skips
    the steps past that).

    Those rows belong to the next sequence. P is 0 there, but 0 * a
    non-finite x is NaN. The tiles land on B_XF, which only these two warps
    wait on, in order; the MMA waits on the operand ring as for any tile.
    """
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)
    var phase = UInt32(0)
    for k in range(T):
        var sl = _slot_rows(p, _item(p, cta + k * grid).slot)
        comptime for rb in range(NRB):
            bars.base.unsafe_offset(B_XF + rb)[].wait((phase >> UInt32(rb)) & 1)
            phase ^= UInt32(1) << UInt32(rb)
            var xs = Int32((k * 7 + 5 + rb) % NS)
            var live = sl[1] - rb * ROWS
            if live > 0 and live < ROWS and live % 16 != 0:
                _zero_rows[64](
                    _tile(smem, xs), live, ceildiv(live, 16) * 16, ft
                )
                fence_async_view_proxy()
            named_barrier[Int32(64)](Int32(NB_FIX))
            if ft == 0:
                _ = bars.full[B_OP](xs)[].arrive()


@inline(.always)
def _mask(p: ParamsPtr, smem: SPtr[UInt8], bars: Bars, T: Int, rt: Int):
    named_barrier[Int32(32 + 256 + 128)](Int32(NB_TPTR))
    var tmem = smem.unsafe_offset(OFF_TMEM).unsafe_bitcast[UInt32]()[]
    var loc = smem.unsafe_offset(OFF_V).unsafe_bitcast[Float32]()
    var c_dc = Cursor.consumer()
    var c_s = Cursor.consumer()
    var c_p = Cursor.producer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)
    for k in range(T):
        var w = _item(p, cta + k * grid)
        var d = bars.wait_full[B_DC, 2](c_dc)
        var dd = _prep(
            p,
            smem.unsafe_offset(OFF_DT + Int(d) * VEC_SLOT).unsafe_bitcast[
                Float32
            ](),
            smem.unsafe_offset(OFF_CS + Int(d) * VEC_SLOT).unsafe_bitcast[
                Float32
            ](),
            loc,
            w,
            rt,
        )
        _arrive_warp(bars.empty[B_DC, 2](d))
        named_barrier[Int32(256)](Int32(NB_MASK))
        for t in range(NRB * (NRB + 1) // 2):
            var s = bars.wait_full[B_S, 2](c_s)
            tcgen05_fence_after()
            # implied empty: P's slot is S's, freed by the MMA's order
            var ps = c_p.slot
            c_p.step[2]()
            tcgen05_fence_after()
            var col = tmem + UInt32(TM_S + 128 * Int(s))
            _mask_tile(
                col,
                tmem + UInt32(TM_S + 128 * Int(ps)),
                loc,
                Int32(t),
                Int32(rt),
                dd,
            )
            tcgen05_fence_before()
            tcgen05_fence_before()
            _arrive_warp(bars.full[B_P](ps))
        named_barrier[Int32(256)](Int32(NB_MASK))


@inline(.always)
def _epi(p: ParamsPtr, smem: SPtr[UInt8], bars: Bars, T: Int, rt: Int):
    named_barrier[Int32(32 + 256 + 128)](Int32(NB_TPTR))
    var tmem = smem.unsafe_offset(OFF_TMEM).unsafe_bitcast[UInt32]()[]
    var stage = smem.unsafe_offset(OFF_Y)
    var c_dc = Cursor.consumer()
    var c_y = Cursor.consumer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)
    for k in range(T):
        var w = _item(p, cta + k * grid)
        var d = bars.wait_full[B_DC, 2](c_dc)
        var cs0 = smem.unsafe_offset(OFF_CS + Int(d) * VEC_SLOT).unsafe_bitcast[
            Float32
        ]()
        for rb in range(NRB):
            var y = bars.wait_full[B_Y, 2](c_y)
            tcgen05_fence_after()
            _epi_scale(
                tmem + UInt32(TM_Y + 64 * HP * Int(y)), cs0, p, w, rb, rt
            )
            tcgen05_fence_before()
            _arrive_warp(bars.empty[B_Y, 2](y))
        _arrive_warp(bars.empty[B_DC, 2](d))
        var sl = _slot_rows(p, w.slot)
        for rb in range(NRB):
            var y = bars.wait_full[B_Y, 2](c_y)
            tcgen05_fence_after()
            var full = sl[1] - rb * ROWS >= ROWS
            comptime for hh in range(HP):
                # the staging slot is free once the store before read it
                if rt == 0:
                    cp_async_bulk_wait_group[0]()
                named_barrier[Int32(128)](Int32(NB_OSTORE))
                _epi_stage(
                    stage,
                    tmem + UInt32(TM_Y + 64 * HP * Int(y)),
                    hh,
                    p,
                    w,
                    rb,
                    rt,
                )
                comptime if hh == HP - 1:
                    tcgen05_fence_before()
                    _arrive_warp(bars.empty[B_Y, 2](y))
                fence_async_view_proxy()
                named_barrier[Int32(128)](Int32(NB_EPI))
                if rt == 0:
                    if full:
                        cp_async_bulk_tensor_global_shared_cta(
                            stage,
                            _desc[6](p),
                            ((w.hp * HP + hh) * P, sl[0] + rb * ROWS, 0),
                        )
                    cp_async_bulk_commit_group()
    if rt == 0:
        cp_async_bulk_wait_group[0]()
    tcgen05_fence_before()
    _ = bars.full[B_TFREE](0)[].arrive()


@__llvm_arg_metadata(params, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
# without a CTA count ptxas cannot size the launch's register pool and drops
# every `setmaxnreg`
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
def ssd_out_sm100_kernel(params: SsdOutParams):
    """Grid min(SMS, items), block THREADS: warps 0-7 mask, 8-11 epilogue, 12
    mma, 13 load, 14-15 tail fixup."""
    var smem = external_memory[
        UInt8, address_space=AddressSpace.SHARED, alignment=1024
    ]()
    var p = Pointer(to=params).unsafe_origin_cast[ImmUntrackedOrigin]()
    var bars = Bars(smem, OFF_BARS)
    var tid = Int(thread_idx.x)
    if tid == 0:
        comptime for g in range(7):
            prefetch_tma_descriptor(_desc[g](p))
        bars.init_ring[B_OP, NS, 1, 1]()
        bars.init_ring[B_DC, 2, 1, 12]()
        bars.init_ring[B_S, 2, 1, 0]()
        bars.init_ring[B_P, 2, 8, 0]()
        bars.init_ring[B_Y, 2, 1, 4]()
        bars.init_ring[B_TFREE, 1, 128, 0]()
        bars.init_ring[B_XF, 2, 1, 0]()
        fence_mbarrier_init()
    var T = _items_of_cta(p)
    barrier()
    var w = tid // WARP_SIZE
    if w < 8:
        warpgroup_reg_alloc[REGS_MASK]()
        _mask(p, smem, bars, T, tid)
    elif w < 12:
        warpgroup_reg_dealloc[REGS_EPI]()
        _epi(p, smem, bars, T, tid - 256)
    else:
        warpgroup_reg_dealloc[REGS_REST]()
        if w == 12:
            _mma(p, smem, bars, T)
        elif w == 13:
            _load(p, smem, bars, T)
        else:
            _fix(p, smem, bars, T, tid - 448)


# ===----------------------------------------------------------------------=== #
# Host side
# ===----------------------------------------------------------------------=== #


def ssd_out_sm100_supported(
    nheads: Int, ngroups: Int, head_dim: Int, dstate: Int
) -> Bool:
    """Returns whether a launch fits the kernel: P 64, N 128, and an even
    number of heads per group."""
    return (
        head_dim == P
        and dstate == N
        and nheads % ngroups == 0
        and (nheads // ngroups) % HP == 0
    )


def _bf16_buf(
    ctx: DeviceContext, addr: Int, n: Int
) raises -> DeviceBuffer[DType.bfloat16]:
    return DeviceBuffer(
        ctx,
        Pointer[BFloat16, MutAnyOrigin](unsafe_from_address=addr),
        n,
        owning=False,
    )


def _f32_buf(
    ctx: DeviceContext, addr: Int, n: Int
) raises -> DeviceBuffer[DType.float32]:
    return DeviceBuffer(
        ctx,
        Pointer[Float32, MutAnyOrigin](unsafe_from_address=addr),
        n,
        owning=False,
    )


def _rows_desc(
    buf: DeviceBuffer[DType.bfloat16], rows: Int, cols: Int, ld: Int
) raises -> TMADescriptor:
    """Returns a rank 3 descriptor over `rows` rows of `cols` elements `ld`
    apart, a box of 128 rows x 64 columns, 128-byte swizzle."""
    return create_tma_descriptor[
        DType.bfloat16, swizzle_mode=TensorMapSwizzle.SWIZZLE_128B
    ](
        buf,
        (1, rows, cols),
        (rows * ld, ld, Idx[1]),
        (Idx[1], Idx[ROWS], Idx[64]),
    )


def ssd_out_sm100(
    ctx: DeviceContext,
    total_len: Int,
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
    """Computes y for every used chunk slot and head.

    Args:
        ctx: The device context.
        total_len: Tokens of the packed batch.
        batch: Sequences.
        slots: Chunk slots of the table (`chunked_slots`).
        nheads: Heads.
        ngroups: B and C groups.
        x_addr: The first element of x, [T, H, P] bf16 with rows `x_ld` apart.
        b_addr: B's first element, [T, G, N] bf16 with rows `bc_ld` apart.
        c_addr: C's first element, as B.
        x_ld: Row stride of x in elements (a multiple of 8).
        bc_ld: Row stride of B and C in elements (a multiple of 8).
        d_addr: D, bf16 [H], or 0 for none.
        dt_addr: The softplus dt, fp32 [H, slots, CHUNK].
        cs_addr: The log2 decay cumsum, fp32 [H, slots, CHUNK].
        h_addr: The state entering each slot, bf16 [slots, H, P, N].
        y_addr: The output, bf16 [T, H, P], contiguous.
        info_addr: The chunk table, int32 [3 slots + 2 batch].
    """
    var nv = nheads * slots * L
    var params = SsdOutParams(
        _rows_desc(
            _bf16_buf(ctx, c_addr, total_len * bc_ld),
            total_len,
            ngroups * N,
            bc_ld,
        ),
        _rows_desc(
            _bf16_buf(ctx, b_addr, total_len * bc_ld),
            total_len,
            ngroups * N,
            bc_ld,
        ),
        _rows_desc(
            _bf16_buf(ctx, x_addr, total_len * x_ld),
            total_len,
            nheads * P,
            x_ld,
        ),
        create_tma_descriptor[
            DType.bfloat16, swizzle_mode=TensorMapSwizzle.SWIZZLE_128B
        ](
            _bf16_buf(ctx, h_addr, slots * nheads * P * N),
            (slots * nheads * P, N),
            (N, Idx[1]),
            (Idx[64], Idx[64]),
        ),
        create_tma_descriptor[DType.float32](
            _f32_buf(ctx, dt_addr, nv), (nv,), (Idx[1],), (Idx[VEC_BOX],)
        ),
        create_tma_descriptor[DType.float32](
            _f32_buf(ctx, cs_addr, nv), (nv,), (Idx[1],), (Idx[VEC_BOX],)
        ),
        _rows_desc(
            _bf16_buf(ctx, y_addr, total_len * nheads * P),
            total_len,
            nheads * P,
            nheads * P,
        ),
        Int32(slots),
        Int32(batch),
        Int32(nheads),
        Int32(nheads // ngroups),
        Int32(nheads // HP),
        Int64(d_addr),
        Int64(info_addr),
        Int64(y_addr),
    )
    var items = slots * (nheads // HP)
    ctx.enqueue_function[ssd_out_sm100_kernel](
        params,
        grid_dim=min(SMS, items),
        block_dim=THREADS,
        shared_mem_bytes=SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(SMEM_BYTES)
        ),
    )
