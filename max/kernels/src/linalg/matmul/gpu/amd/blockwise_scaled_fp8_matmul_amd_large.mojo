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
"""Blockwise-scaled FP8 dense matmul for gfx950 (MI355X) at large M.

Computes `C[m, n] = sum_k A[m, k] * B[n, k] * a_scale[k // 128, m] *
b_scale[n // 128, k // 128]` with e4m3 operands, fp32 scales per (1, 128,
128) block, `transpose_b` (B is `[N, K]`) and a bf16 `C`: the contract of
`blockwise_scaled_fp8_matmul_amd`, for the prefill shapes where that
kernel's one-wave 64 x 64 tiles leave the matrix cores mostly idle.

Structure. One workgroup of eight waves per CU (two a SIMD) computes a
128 x 256 output tile; wave `w` owns columns `32 w ..< 32 w + 32` of all
128 rows. The K loop walks 256-byte slots, two `16x16x128` MFMA K steps (two
scale blocks) each. A goes to an LDS double buffer by `buffer_load ... lds`,
B to a slot of LDS of each wave's own, both by DMAs of whole rows (4 rows a
DMA, each lane fetching the 16-byte granule an XOR-16 swizzle puts at its
address), beside the slot's fp32 A scales. Every lane then reads its MFMA
fragments from LDS; B's for the next slot are read right after the barrier
that ends a slot, which frees the wave's B slot for the next fill. One
barrier per slot orders the fill of A's next half against the reads of
this one.

Why both operands go through LDS. The vector memory path a CU's waves share
is the bottleneck. A load in the MFMA's lane layout reads 64 bytes of each
of 16 rows, and that path spends about three times as long on it as on a
DMA of whole lines; B through LDS costs LDS bandwidth, which has room.

Scales. The hardware block-scaled MFMA only takes E8M0 scales, so the fp32
scales are applied in software: each MFMA (one K step, exactly one scale
block) writes a fresh temporary, and `acc += tmp * a_s * b_s` folds it into
the fp32 accumulator `FOLD_LAG` MFMAs later, so the fold's VALU work
overlaps the matrix core instead of waiting on it. The MFMA's output lane
holds four columns of one row, so the fold multiplies by one per-lane
scalar. B's scale is wave-uniform (32 columns lie in one 128-column block)
and comes through the scalar cache. A's rows are stored permuted in LDS,
MFMA row `r` of row tile `m` being tile row `8 r + m`, so a lane's eight
row scales are contiguous: two `ds_read_b128`s instead of one read per row
tile.

Every lane-derived offset reads the lane id through `fresh_lane()`: as a
plain intrinsic it is hoisted out of the loop and everything derived from it
spilled, and each spill reload drains every load in flight (gfx9 has one
in-order vmcnt).
"""

from std.math import ceildiv
from std.memory import bitcast
from std.sys import inlined_assembly, llvm_intrinsic
from std.utils import StaticTuple

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import AMDBufferResource, permlane_swap, readfirstlane
from max.gpu.memory import AddressSpace, external_memory
from max.gpu.sync import barrier, s_waitcnt
from structured_kernels.amd_tile_io import _load_from_lds, _load_to_lds

comptime F8 = Scalar[DType.float8_e4m3fn]
comptime Frag = SIMD[DType.uint8, 32]
comptime Acc = SIMD[DType.float32, 4]
comptime LdsBytes = Pointer[
    UInt8, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]

comptime BM = 128
"""Rows of a tile."""
comptime BK = 256
"""K bytes of a slot: two `16x16x128` K steps, two scale blocks."""
comptime SCALE_BLOCK = 128
comptime KS = BK // SCALE_BLOCK
comptime WAVES = 8
comptime WN = 32
"""Columns of a wave: two 16-column MFMA tiles, inside one 128-column scale
block."""
comptime BN = WAVES * WN
comptime MT = BM // 16
comptime NT = WN // 16
comptime THREADS = WAVES * WARP_SIZE
comptime DMA_BYTES = WARP_SIZE * 16
"""Bytes one `buffer_load ... lds` fills: 16 a lane, 4 rows of a slot."""
comptime DMA_ROWS = DMA_BYTES // BK
comptime A_DMAS = BM // DMA_ROWS // WAVES
"""A DMAs a wave issues per slot."""
comptime B_DMAS = WN // DMA_ROWS
"""B DMAs a wave issues per slot."""
comptime A_SLOT = BM * BK
"""LDS bytes of one half of A's double buffer."""
comptime B_SLOT = WN * BK
"""LDS bytes of a wave's B slot."""
comptime SA_SLOT = KS * BM * 4
"""LDS bytes of one half's A scales: `[KS][BM]` fp32, in tile-row order."""
comptime SA_WAVES = SA_SLOT // (4 * WARP_SIZE)
"""Waves that each fill 64 of a slot's A scales with one 4-byte DMA."""
comptime B_OFF = 2 * A_SLOT
comptime SA_OFF = B_OFF + WAVES * B_SLOT
comptime LDS_BYTES = SA_OFF + 2 * SA_SLOT
comptime FOLD_LAG = 4
"""MFMAs between one writing its temporary and the fold reading it: enough
issue slots that the fold never waits out the MFMA's result latency."""
comptime XCDS = 8
comptime GROUP_M = 8
"""Row tiles a group of the tile walk spans (see the kernel)."""
comptime MAX_UNROLL = 24
"""Slots of K the loop unrolls fully; longer K loops over slot pairs."""


@inline(.always)
def fresh_lane() -> Int:
    """The lane id, read where it is used: as a plain intrinsic it is hoisted
    out of the loop, and the offsets derived from it are spilled and reloaded
    one at a time, each reload draining every load in flight."""
    var lo = inlined_assembly[
        "v_mbcnt_lo_u32_b32 $0, -1, 0",
        UInt32,
        constraints="=v",
        has_side_effect=True,
    ]()
    return Int(
        inlined_assembly[
            "v_mbcnt_hi_u32_b32 $0, -1, $1",
            UInt32,
            constraints="=v,v",
            has_side_effect=True,
        ](lo)
    )


@inline(.always)
def sched_fence():
    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))


@inline(.always)
def read_frag[s: Int, m: Int](base: LdsBytes, r: Int, quad: Int) -> Frag:
    """Row tile `m`'s fragment for K step `s` of a slot at `base`: the bytes a
    lane feeds the MFMA, granules `8 s + q` and `8 s + 4 + q` of row `16 m +
    r`, stored under the XOR of the row mod 16 so the reads spread over the
    banks."""
    var row = 16 * m + r
    var p = base.unsafe_offset(row * BK)
    var lo = _load_from_lds[width=16](
        p.unsafe_offset(16 * ((8 * s + quad) ^ r)).unsafe_bitcast[F8]()
    )
    var hi = _load_from_lds[width=16](
        p.unsafe_offset(16 * ((8 * s + 4 + quad) ^ r)).unsafe_bitcast[F8]()
    )
    return bitcast[DType.uint8, 16](lo).join(bitcast[DType.uint8, 16](hi))


@inline(.always)
def read_a_scales[
    s: Int
](lds: LdsBytes, half: Int, r: Int) -> SIMD[DType.float32, MT]:
    """The A scales of tile rows `8 r ..< 8 r + 8` for K step `s`: the row
    scale of each of the lane's row tiles."""
    var p = lds.unsafe_offset(SA_OFF + half * SA_SLOT + s * BM * 4 + 4 * MT * r)
    # Two `ds_read_b128`s; TileIO's alias-scoped read takes them as bytes.
    return bitcast[DType.float32, MT](
        _load_from_lds[width=32](p.unsafe_bitcast[F8]())
    )


@inline(.always)
def mfma_fp8(b: Frag, a: Frag) -> Acc:
    """One `16x16x128` e4m3 MFMA into a zero accumulator: output lane
    `(r, q)` holds columns `4 q ..< 4 q + 4` of B's fragment for row `r` of
    A's. Zero scale operands select the unscaled instruction."""
    return llvm_intrinsic["llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4", Acc](
        bitcast[DType.int32, 8](b),
        bitcast[DType.int32, 8](a),
        Acc(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
    )


@inline(.always)
def fold(tmp: Acc, scale: Float32, acc: Acc) -> Acc:
    """`acc + tmp * scale`."""
    return tmp.fma(Acc(scale), acc)


@inline(.always)
def tile_gemm[
    N: Int, K: Int, ACTIVE: Bool
](
    lds: LdsBytes,
    mut acc: StaticTuple[Acc, MT * NT],
    a_base: Int,
    a_rows: Int,
    a_scales: Int,
    sa_stride: Int,
    m_tile_row: Int,
    b: Int,
    b_scales: Int,
    n_wave: Int,
    wave: Int,
):
    """One tile's K into the wave's accumulators: rows `m_tile_row ..<
    m_tile_row + BM` of A, of which the first `a_rows` exist (the rest read
    as zeros), columns `n_wave ..< n_wave + WN` of B. `sa_stride` is the row
    stride of A's scales. `ACTIVE == False`: the wave owns no columns of
    this tile (past `N`), and only fills its share of A and meets the
    barriers."""
    comptime KT = K // BK
    comptime KB = K // SCALE_BLOCK
    comptime assert K % (2 * BK) == 0
    comptime assert DMA_ROWS == 4 and A_DMAS == 4 and SA_WAVES <= WAVES

    # Per-lane DMA source offsets, formed once. A lane fills granule `l % 16`
    # of LDS row `4 j' + l / 16`, which holds the row's granule `l % 16 ^
    # row % 16`. DMA `j` of wave `w` fills A's LDS rows `4 (4 w + j) ..< + 4`,
    # i.e. tile rows `32 j + 8 (l / 16) + w` (MFMA row `r` of row tile `m`,
    # LDS row `16 m + r`, is tile row `8 r + m`). The whole row term is in
    # the lane's offset, so the descriptor's size drops rows past A's end
    # however the hardware treats the scalar offset, which carries only the
    # K position. B's slot is A's layout without the row permutation: DMA
    # `j` fills its rows `4 j ..< 4 j + 4`, all of them inside B.
    var lrow, lgran = divmod(fresh_lane(), BK // 16)
    var voff = StaticTuple[Int32, A_DMAS]()
    var bvoff = StaticTuple[Int32, 4]()
    comptime for q in range(4):
        var g = 16 * (lgran ^ (4 * q + lrow))
        voff[q] = Int32(K * (32 * q + MT * lrow + wave) + g)
        bvoff[q] = Int32(K * lrow + g)

    # Wave `w < SA_WAVES` fills scales of K step `w // PER_STEP`, tile rows
    # `64 (w % PER_STEP) ..< + 64`: their address in K block 0, how many of
    # them exist, and the bytes between K blocks.
    comptime PER_STEP = BM // WARP_SIZE
    var sa_row0 = WARP_SIZE * (wave % PER_STEP)
    var sa_base = Int(
        readfirstlane(
            Pointer[Float32, ImmUntrackedOrigin](
                unsafe_from_address=a_scales
                + 4 * ((wave // PER_STEP) * sa_stride + m_tile_row + sa_row0)
            )
        )
    )
    var sa_step = Int(readfirstlane(Int32(4 * sa_stride)))
    var sa_records = Int(
        readfirstlane(Int32(max(min(a_rows - sa_row0, WARP_SIZE), 0)))
    )

    @inline(.always)
    def fill_a(
        k0: Int, half: Int
    ) {
        imm lds,
        imm a_base,
        imm a_rows,
        imm sa_base,
        imm sa_step,
        imm sa_records,
        imm wave,
        imm voff,
    }:
        # TileIO's DMA carries the alias scope its `_load_from_lds` reads are
        # noalias against, so reads of the other half need no `vmcnt(0)`; the
        # counted wait and the barrier are the fence. Descriptors and LDS
        # bases (M0) are rebuilt from wave-uniform values at each use: one
        # held across the loop can land in VGPRs under register pressure, and
        # the compiler then wraps every DMA in a waterfall loop.
        var a_rsrc = AMDBufferResource(
            readfirstlane(
                Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=a_base)
            ),
            Int(readfirstlane(Int32(a_rows * K))),
        )
        var base = readfirstlane(lds.unsafe_offset(half * A_SLOT))
        comptime for j in range(A_DMAS):
            _load_to_lds[width=16](
                a_rsrc,
                voff[j],
                base.unsafe_offset((wave * A_DMAS + j) * DMA_BYTES),
                scalar_offset=Int32(k0),
            )
        # The slot's A scales, `[KS][BM]` fp32 in tile-row order: wave `w <
        # SA_WAVES` fills 64 of them, from a descriptor over just those rows
        # of one K block, so rows past A's end read as zeros (their A is
        # zero too).
        if wave < SA_WAVES:
            var s_rsrc = AMDBufferResource(
                readfirstlane(
                    Pointer[Float32, ImmUntrackedOrigin](
                        unsafe_from_address=sa_base
                        + (k0 // SCALE_BLOCK) * sa_step
                    )
                ),
                sa_records,
            )
            var sbase = readfirstlane(
                lds.unsafe_offset(SA_OFF + half * SA_SLOT + wave * 256)
            ).unsafe_bitcast[Float32]()
            _load_to_lds[width=1](s_rsrc, Int32(fresh_lane()), sbase)

    comptime if not ACTIVE:
        for kp in range(KT // 2):
            comptime for h in range(2):
                fill_a((2 * kp + h) * BK, h)
                s_waitcnt[vmcnt=0]()
                barrier()
        return

    var quad, r = divmod(fresh_lane(), 16)
    var bs_base = readfirstlane(Int32((n_wave // SCALE_BLOCK) * KB))

    @inline(.always)
    def b_slot() {imm lds, imm wave} -> LdsBytes:
        return readfirstlane(lds.unsafe_offset(B_OFF + wave * B_SLOT))

    @inline(.always)
    def fill_b(k0: Int) {imm b, imm n_wave, imm bvoff, imm b_slot}:
        var rsrc = AMDBufferResource(
            readfirstlane(
                Pointer[UInt8, ImmUntrackedOrigin](
                    unsafe_from_address=b + n_wave * K
                )
            ),
            Int(readfirstlane(Int32(WN * K))),
        )
        var base = b_slot()
        comptime for j in range(B_DMAS):
            _load_to_lds[width=16](
                rsrc,
                bvoff[j % 4],
                base.unsafe_offset(j * DMA_BYTES),
                scalar_offset=Int32(k0 + DMA_ROWS * j * K),
            )

    @inline(.always)
    def read_b() {imm b_slot, imm r, imm quad} -> StaticTuple[Frag, NT * KS]:
        var base = b_slot()
        var res = StaticTuple[Frag, NT * KS]()
        comptime for s in range(KS):
            comptime for t in range(NT):
                res[t * KS + s] = read_frag[s, t](base, r, quad)
        return res

    @inline(.always)
    def ld_b_scales(
        k0: Int,
    ) {imm b_scales, imm bs_base} -> SIMD[DType.float32, 2]:
        # Wave-uniform: a constant-address-space load is a scalar load, off
        # the vector memory path.
        var p = Pointer[
            Float32, ImmUntrackedOrigin, address_space=AddressSpace.CONSTANT
        ](
            unsafe_from_address=Int(
                readfirstlane(
                    Pointer[Float32, ImmUntrackedOrigin](
                        unsafe_from_address=b_scales
                    )
                )
            )
        ).unsafe_offset(
            Int(bs_base) + k0 // SCALE_BLOCK
        )
        return SIMD[DType.float32, 2](p[unsafe_offset=0], p[unsafe_offset=1])

    fill_b(0)
    fill_a(0, 0)
    var bsc = ld_b_scales(0)
    s_waitcnt[vmcnt=0]()
    barrier()
    var bf = read_b()

    # One MFMA stream over the tile: MFMA `i` writes temporary `i % FOLD_LAG`
    # and is folded into its accumulator just before MFMA `i + FOLD_LAG`
    # reuses that temporary, so each fold's VALU issues in the shadow of the
    # MFMAs before it. A K step's last FOLD_LAG MFMAs (its last TAIL row
    # tiles) are folded in the next step with the scales they were computed
    # under; before the first step those are zeros.
    comptime STEP = MT * NT
    comptime TAIL = FOLD_LAG // NT
    comptime assert STEP >= FOLD_LAG and FOLD_LAG % NT == 0
    var tmp = StaticTuple[Acc, FOLD_LAG](fill=Acc(0))
    var sm_tail = SIMD[DType.float32, TAIL](0)

    @inline(.always)
    def slot[
        half: Int, REFILL: Bool
    ](k0: Int) {
        mut acc,
        mut tmp,
        mut sm_tail,
        mut bsc,
        mut bf,
        imm fill_a,
        imm fill_b,
        imm read_b,
        imm ld_b_scales,
        imm lds,
        imm r,
        imm quad,
    }:
        var bsc_next = bsc
        comptime if REFILL:
            # A's other half is free since the last barrier; this slot's B
            # was read into registers right after it, and the wave's B slot
            # is free once those reads return.
            fill_a(k0 + BK, 1 - half)
            s_waitcnt[lgkmcnt=0]()
            fill_b(k0 + BK)
            bsc_next = ld_b_scales(k0 + BK)
        var a_slot = lds.unsafe_offset(half * A_SLOT)
        comptime for s in range(KS):
            var sm = read_a_scales[s](lds, half, r) * bsc[s]
            var a_cur = read_frag[s, 0](a_slot, r, quad)
            comptime for m in range(MT):
                var a_next = a_cur
                comptime if m + 1 < MT:
                    a_next = read_frag[s, m + 1](a_slot, r, quad)
                comptime for t in range(NT):
                    comptime i = m * NT + t
                    sched_fence()
                    comptime if i >= FOLD_LAG:
                        comptime j = i - FOLD_LAG
                        acc[j] = fold(tmp[j % FOLD_LAG], sm[j // NT], acc[j])
                    else:
                        comptime j = STEP - FOLD_LAG + i
                        acc[j] = fold(
                            tmp[j % FOLD_LAG],
                            sm_tail[j // NT - (MT - TAIL)],
                            acc[j],
                        )
                    tmp[i % FOLD_LAG] = mfma_fp8(bf[t * KS + s], a_cur)
                a_cur = a_next
            sched_fence()
            sm_tail = sm.slice[TAIL, offset=MT - TAIL]()
        comptime if REFILL:
            s_waitcnt[vmcnt=0]()
            barrier()
            bf = read_b()
            bsc = bsc_next

    comptime if KT > MAX_UNROLL:
        # A loop over slot pairs (the two halves of A's double buffer):
        # unrolled over a long K, the compiler hoists per-slot values across
        # the whole of it and spills.
        for kp in range(KT // 2 - 1):
            slot[0, True](2 * kp * BK)
            slot[1, True]((2 * kp + 1) * BK)
        slot[0, True](K - 2 * BK)
        slot[1, False](K - BK)
    else:
        comptime for kk in range(KT):
            slot[kk % 2, kk + 1 < KT](kk * BK)
    comptime for j in range(STEP - FOLD_LAG, STEP):
        acc[j] = fold(tmp[j % FOLD_LAG], sm_tail[j // NT - (MT - TAIL)], acc[j])


@inline(.always)
def store_c[
    N: Int
](acc: StaticTuple[Acc, MT * NT], c: Int, rows: Int, n_wave: Int):
    """The tile's first `rows` rows (at most BM) of the wave's 32 columns as
    bf16; `c` addresses the tile's first row."""
    var quad, r = divmod(fresh_lane(), 16)
    var col0 = readfirstlane(Int32(n_wave))
    # The tile's live rows end the descriptor, so rows past them drop. Sized
    # to the tile, not the rest of C: 32 bits of bytes would wrap.
    var out_rsrc = AMDBufferResource(
        readfirstlane(
            Pointer[BFloat16, MutUntrackedOrigin](unsafe_from_address=c)
        ),
        rows * N,
    )
    comptime assert NT == 2
    comptime for m in range(MT):
        # Quad q holds 8 bytes of tile t at row bytes 32 t + 8 q. A 2 x 4
        # transpose of those units among the row's four lanes gives quad q
        # the row's bytes [16 q, 16 q + 16): one store.
        var u0 = bitcast[DType.uint32, 2](acc[m * NT].cast[DType.bfloat16]())
        var u1 = bitcast[DType.uint32, 2](
            acc[m * NT + 1].cast[DType.bfloat16]()
        )

        @inline(.always)
        def swap[
            stride: Int
        ](x: SIMD[DType.uint32, 2], y: SIMD[DType.uint32, 2]) -> StaticTuple[
            SIMD[DType.uint32, 2], 2
        ]:
            # VALU lane swaps, not LDS-crossbar shuffles: permlane32_swap is
            # the xor-32 exchange of (x, y), permlane16_swap the xor-16 one.
            var p = permlane_swap[stride](x[0], y[0])
            var q = permlane_swap[stride](x[1], y[1])
            var res = StaticTuple[SIMD[DType.uint32, 2], 2]()
            res[0] = SIMD[DType.uint32, 2](p[0], q[0])
            res[1] = SIMD[DType.uint32, 2](p[1], q[1])
            return res

        # After the xor-32 swap the quads hold (t0 q0, t0 q1, t1 q0, t1 q1)
        # and (t0 q2, t0 q3, t1 q2, t1 q3); the xor-16 swap pairs them up.
        var g = swap[32](u0, u1)
        var h = swap[16](g[0], g[1])
        var row = Int32(MT * r + m)
        var off = (row * Int32(N) + col0) // 2 + Int32(4 * quad)
        out_rsrc.store[DType.uint32, 4](off, h[0].join(h[1]))


# Without the launch's size the compiler assumes 1024 threads, caps a lane at
# 128 VGPRs and spills. Two waves a SIMD, one workgroup per CU. The 32-column
# wave tile keeps the fold's temporaries in VGPRs: with 64 columns a wave
# needs more than 256 registers, and at one wave a SIMD the compiler gives
# the MFMA results AGPRs, which costs four `v_accvgpr_read`s per fold.
@__name(t"blockwise_scaled_fp8_matmul_amd_large_N{N}_K{K}")
@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(2))
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
def blockwise_scaled_fp8_matmul_amd_large_kernel[
    N: Int, K: Int
](
    c: Int64,
    a: Int64,
    a_scales: Int64,
    b: Int64,
    b_scales: Int64,
    m_rows: Int32,
    a_scale_stride: Int32,
):
    """One workgroup's 128 x 256 output tile. See the module docstring."""
    comptime N_TILES = ceildiv(N, BN)
    var M = Int(m_rows)
    var m_tiles = ceildiv(M, BM)
    var tiles = m_tiles * N_TILES
    # Workgroups go to the XCDs round robin; give each XCD a contiguous run
    # of tiles. Within it, groups of GROUP_M row tiles walk all column tiles
    # (row tile inner), so an XCD's concurrent tiles share both A rows and B
    # columns in its L2.
    var grid = ceildiv(tiles, XCDS) * XCDS
    var cta = Int(block_idx.x)
    var t_idx = (cta % XCDS) * (grid // XCDS) + cta // XCDS
    if t_idx >= tiles:
        return
    var group, in_group = divmod(t_idx, GROUP_M * N_TILES)
    var g_rows = min(GROUP_M, m_tiles - group * GROUP_M)
    var n_tile, m_in = divmod(in_group, g_rows)
    var m_tile = group * GROUP_M + m_in
    var m_row = m_tile * BM
    var wave = Int(readfirstlane(Int32(thread_idx.x // WARP_SIZE)))
    var n_wave = n_tile * BN + wave * WN
    var lds = external_memory[
        UInt8, address_space=AddressSpace.SHARED, alignment=128
    ]()
    var a_base = Int(a) + m_row * K
    # The tile's rows that exist; descriptors span the tile, not the rest of
    # the matrix, whose size in bytes can pass 32 bits.
    var rows = min(M - m_row, BM)
    var acc = StaticTuple[Acc, MT * NT](fill=Acc(0))
    if n_wave < N:
        tile_gemm[N, K, True](
            lds,
            acc,
            a_base,
            rows,
            Int(a_scales),
            Int(a_scale_stride),
            m_row,
            Int(b),
            Int(b_scales),
            n_wave,
            wave,
        )
        store_c[N](acc, Int(c) + 2 * m_row * N, rows, n_wave)
    else:
        tile_gemm[N, K, False](
            lds,
            acc,
            a_base,
            rows,
            Int(a_scales),
            Int(a_scale_stride),
            m_row,
            Int(b),
            Int(b_scales),
            n_wave,
            wave,
        )


def blockwise_scaled_fp8_matmul_amd_large_supported[
    c_type: DType, N: Int, K: Int
]() -> Bool:
    """Whether the large-M kernel covers a (bf16 `C`, N, K): whole waves of
    32 columns and whole pairs of 256-byte K slots."""
    return c_type == DType.bfloat16 and N % WN == 0 and K % (2 * BK) == 0


def blockwise_scaled_fp8_matmul_amd_large[
    N: Int, K: Int
](
    ctx: DeviceContext,
    c: Int,
    a: Int,
    a_scales: Int,
    b: Int,
    b_scales: Int,
    M: Int,
    a_scale_stride: Int,
) raises:
    """Enqueues the large-M blockwise FP8 matmul.

    Parameters:
        N: Columns of `C` (rows of B); a multiple of 32.
        K: Contraction length; a multiple of 512.

    Args:
        ctx: Device context.
        c: Address of `C`, `[M, N]` bf16 row major.
        a: Address of A, `[M, K]` e4m3 row major.
        a_scales: Address of A's scales, `[K // 128, M]` fp32, rows
            `a_scale_stride` elements apart.
        b: Address of B, `[N, K]` e4m3 row major.
        b_scales: Address of B's scales, `[ceildiv(N, 128), K // 128]` fp32
            row major.
        M: Rows of `C` and A.
        a_scale_stride: Row stride of `a_scales` in elements, at least `M`.
    """
    comptime assert N % WN == 0, "N must be a multiple of 32"
    comptime assert K % (2 * BK) == 0, "K must be a multiple of 512"
    if M == 0:
        return
    var tiles = ceildiv(M, BM) * ceildiv(N, BN)
    comptime kernel = blockwise_scaled_fp8_matmul_amd_large_kernel[N, K]
    ctx.enqueue_function[kernel](
        Int64(c),
        Int64(a),
        Int64(a_scales),
        Int64(b),
        Int64(b_scales),
        Int32(M),
        Int32(a_scale_stride),
        grid_dim=ceildiv(tiles, XCDS) * XCDS,
        block_dim=THREADS,
        shared_mem_bytes=LDS_BYTES,
    )
