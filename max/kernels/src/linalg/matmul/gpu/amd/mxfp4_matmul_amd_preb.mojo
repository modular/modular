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
"""MXFP4 block-scaled matmul on AMD CDNA4 with preshuffled B + scales + direct VGPR loads.

Variant of `MXFP4MatmulAMD` that skips LDS staging for both B and the
A/B scales. B is preshuffled host-side via `Shuffler.preshuffle_b_5d`
so each lane's 16-byte fragment lives at a known DRAM offset and is
read with a single `buffer_load_dwordx4`. Scales are addressed by
`Shuffler.scale_4d_byte_off` — each lane reads one Int32 covering a
(mn_pack=2, k_pack=2) cell that feeds 4 sub-MMAs via the MFMA's
OPSEL byte selector.

Only suitable when `num_warps_m == 1` (BM == WM) — otherwise B would be
read multiply across the warps in the M direction without LDS reuse.

Tile constraints:
  * `BM == 16 or BM % 32 == 0`. BM=16 uses one sub-MMA per CTA along M;
    the scale i32's mn_pack=1 byte is rotated into OPSEL byte 0/2 with
    `shrui` (see `BlockScaledMmaOp_PreB.mma`).
  * `WN == 16 or WN % 32 == 0`. Same logic per-warp along N.
  * `num_k_mmas` must be even (k_pack=2 cell halves).
  * `N` must be a multiple of 32 (= 16 * mn_pack) for B-scale cell alignment.
"""

from std.math import ceildiv
from std.math.uutils import udivmod
from std.sys import simd_width_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import DeviceContext

from layout import TensorLayout, TileTensor
from layout.tile_layout import row_major, col_major
from layout.tile_tensor import stack_allocation

from std.utils import IndexList, StaticTuple
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)
from structured_kernels.amd_tile_io import RegTileLoader, RegTileWriter

from .mxfp4_matmul_amd import MX_BLOCK_SIZE
from .mxfp4_preshuffle_loaders import PreshuffledBLoader, PreshuffledScaleLoader


# ===----------------------------------------------------------------------=== #
# BlockScaledMmaOp_PreB — preb-specific MFMA op with preshuffled-scale loads.
# ===----------------------------------------------------------------------=== #
#
# Sibling of `BlockScaledMmaOp` (the SMEM-scale variant in mxfp4_matmul_amd.mojo).
# Same A/B/C register storage and fragment loaders; differs only in:
#   * Scale storage shape: `[ceildiv(num_*_mmas, 2), num_k_mmas / 2]` — one
#     Int32 cell per (mn_pair, k_pair), covering up to 4 sub-MMAs at OPSEL
#     byte indices `(mma_k_idx % 2) * 2 + (m % 2)` (A) and same with `n` (B).
#     When num_*_mmas is odd (WM/WN=16), the last cell's mn_pack=1 byte is
#     unused; per-CTA `shrui` rotates the i32 in `mma()` so OPSEL byte 0/2
#     still picks the right scale.
#   * Scale loads come from `PreshuffledScaleLoader` (direct DRAM → VGPR),
#     not SMEM.
#
# Kept separate from `BlockScaledMmaOp` for readability while the
# preshuffled-scales path is being developed; can consolidate later.


struct BlockScaledMmaOp_PreB[
    mma_shape: IndexList[3],  # (16, 16, 128) for MXFP4
    warp_tile: IndexList[3],  # (WM, WN, BK_ELEMS) in MFMA-native element units
    num_b_slots: Int = 1,
]:
    """Per-warp register state + MFMA dispatch for the preb (preshuffled-B,
    preshuffled-scale) kernel.

    `warp_tile` is the (M, N, K) region this warp computes per outer-K
    iteration, in the same element units as `mma_shape`. Per-warp MFMA
    counts are derived as `warp_tile[i] // mma_shape[i]`.

    Asserted in `__init__`: `warp_tile[i] % mma_shape[i] == 0` per axis,
    and `num_k_mmas % 2 == 0` (k_pack=2 cell halves). `num_m_mmas` /
    `num_n_mmas` may be odd; the constructor rotates the scale i32 per
    CTA so OPSEL keeps the same comptime formula. See module-level
    comment for the scale-cell byte ordering.
    """

    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]

    comptime num_m_mmas = Self.warp_tile[0] // Self.MMA_M
    comptime num_n_mmas = Self.warp_tile[1] // Self.MMA_N
    comptime num_k_mmas = Self.warp_tile[2] // Self.MMA_K

    comptime MMA_K_BYTES = Self.MMA_K // 2  # 64 bytes / MFMA along K
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE  # 4
    comptime mma_frag_width_bytes: Int = 16

    comptime _a_reg_layout = row_major[
        Self.num_k_mmas,
        Self.num_m_mmas,
        Self.mma_frag_width_bytes,
    ]()
    # `_b_reg` holds B fragments for the current + (optionally) prefetched
    # outer-K iter. Stored 4D: [slot, mma_k_idx, n_mma, frag_bytes]. Slot
    # is outermost so the prefetch ring just toggles the leading index.
    comptime _b_reg_layout = row_major[
        Self.num_b_slots,
        Self.num_k_mmas,
        Self.num_n_mmas,
        Self.mma_frag_width_bytes,
    ]()
    comptime _c_reg_layout = row_major[
        Self.num_m_mmas,
        Self.num_n_mmas * Self.c_frag_size,
    ]()

    # 2x2 (mn_pack × k_pack) cell packing: one Int32 per (mi_pair, k_pair).
    # ceildiv so odd num_m_mmas/num_n_mmas (e.g. WM=16) still allocate a cell;
    # the unused mn_pack=1 byte is loaded but never OPSEL'd.
    comptime _a_scale_layout = row_major[
        ceildiv(Self.num_m_mmas, 2), Self.num_k_mmas // 2
    ]()
    comptime _b_scale_layout = row_major[
        ceildiv(Self.num_n_mmas, 2), Self.num_k_mmas // 2
    ]()

    var _a_reg: TileTensor[
        DType.uint8,
        type_of(Self._a_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _b_reg: TileTensor[
        DType.uint8,
        type_of(Self._b_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _c_reg: TileTensor[
        DType.float32,
        type_of(Self._c_reg_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _a_scale_packed: TileTensor[
        DType.int32,
        type_of(Self._a_scale_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _b_scale_packed: TileTensor[
        DType.int32,
        type_of(Self._b_scale_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    # Per-kernel runtime parity shifts for BM=16 / WN=16 cell-straddle.
    # 0 or 8 bits. Only consulted when warp_tile[0]==16 / warp_tile[1]==16;
    # otherwise the comptime-gated shift is eliminated.
    var _a_scale_shift: UInt32
    var _b_scale_shift: UInt32

    @always_inline
    def __init__(out self, warp_m_off: Int, warp_n_off: Int):
        comptime assert (
            Self.warp_tile[0] % Self.MMA_M == 0
        ), "warp_tile[0] (M) must be a multiple of mma_shape[0]"
        comptime assert (
            Self.warp_tile[1] % Self.MMA_N == 0
        ), "warp_tile[1] (N) must be a multiple of mma_shape[1]"
        comptime assert (
            Self.warp_tile[2] % Self.MMA_K == 0
        ), "warp_tile[2] (K) must be a multiple of mma_shape[2]"
        comptime assert (
            Self.num_k_mmas % 2 == 0
        ), "preb scale path requires num_k_mmas % 2 == 0 (k_pack=2)"

        self._a_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            Self._a_reg_layout
        )
        self._b_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            Self._b_reg_layout
        )
        self._c_reg = stack_allocation[DType.float32, AddressSpace.LOCAL](
            Self._c_reg_layout
        )
        _ = self._c_reg.fill(Scalar[DType.float32](0))

        self._a_scale_packed = stack_allocation[
            DType.int32, AddressSpace.LOCAL
        ](Self._a_scale_layout)

        self._b_scale_packed = stack_allocation[
            DType.int32, AddressSpace.LOCAL
        ](Self._b_scale_layout)

        # WM=16 / WN=16 cell-straddle: when warp_*_off // 16 is odd, the CTA's
        # m=0 / n=0 maps to the cell's mn_pack=1 byte. shrui by 8 in `mma()`
        # brings that byte to OPSEL position 0. For WM>=32 / WN>=32 parity is
        # always 0 and the shift is comptime-eliminated.
        self._a_scale_shift = UInt32(((warp_m_off >> 4) & 1) << 3)  # 0 or 8
        self._b_scale_shift = UInt32(((warp_n_off >> 4) & 1) << 3)

    @always_inline
    def accum_tile(self) -> ref[self._c_reg] type_of(self._c_reg):
        return self._c_reg

    @always_inline
    def load_a_frag_from_smem[
        mma_k_idx: Int
    ](
        self,
        a_smem_warp: TileTensor[
            DType.uint8, _, _, address_space=AddressSpace.SHARED, ...
        ],
    ):
        """Load A fragment for MFMA-K position `mma_k_idx` from row-major SMEM.
        """
        comptime lane_layout = col_major[
            Self.MMA_M, WARP_SIZE // Self.MMA_M
        ]()  # 16x4 col major

        var a_reg_v = self._a_reg.vectorize[1, 1, Self.mma_frag_width_bytes]()
        comptime for i in range(Self.num_m_mmas):
            var a_frag = (
                a_smem_warp.tile[Self.MMA_M, Self.MMA_K_BYTES](i, mma_k_idx)
                .vectorize[1, Self.mma_frag_width_bytes]()
                .distribute[lane_layout](lane_id())
            )
            a_reg_v[mma_k_idx, i, 0] = a_frag[0, 0]

    @always_inline
    def load_b_frag_preshuffled[
        mma_k_idx: Int, slot: Int = 0
    ](
        self,
        b_loader: PreshuffledBLoader[_, _],
        warp_n_off: Int,
        k_byte_base: Int,
    ):
        """Load B fragments direct from preshuffled DRAM into b_reg slot `slot`.
        """
        comptime assert slot < Self.num_b_slots, "slot out of range"

        var lane_klane, lane_nlane = udivmod(lane_id(), Self.MMA_N)

        var b_reg_v = self._b_reg.vectorize[
            1, 1, 1, Self.mma_frag_width_bytes
        ]()
        comptime for i in range(Self.num_n_mmas):
            # the logical n row in the expert we will be loading from
            # the warp_n_offset is the tile base position for this warp block,
            # i * Self.MMA_N shifts down by 16 based on what MMA this warp is
            # processing in the warp tile, then we add the specific lane in n
            # this tile is responsible for.
            var n_log = warp_n_off + i * Self.MMA_N + lane_nlane

            # K_byte_base the starting byte offset based on the Kth block tile we are on.
            # mma_k_idx * Self.MMA_K_BYTES, shifts that based on the mma_k tile we are
            # processing in that block. Finally we add the lane's klane offset within that K tile,
            # This is usally a multiple of 16

            var k_byte_log = (
                k_byte_base
                + mma_k_idx * Self.MMA_K_BYTES
                + lane_klane * Self.mma_frag_width_bytes
            )

            # we pass in the logical Nth row, and K byte to get the shuffled
            # coordiante we are loading from
            b_reg_v[slot, mma_k_idx, i, 0] = b_loader.load_fragment(
                n_log, k_byte_log
            )

    @always_inline
    def load_a_scales_preshuffled[
        k_pair: Int
    ](
        mut self,
        a_scale_loader: PreshuffledScaleLoader[_, _],
        warp_m_off: Int,
        k_pair_idx: Int,
    ):
        """Issue per-lane i32 scale loads for A at one k_pair slot.

        Caller provides the absolute `k_pair_idx` (= `k_iter *
        (num_k_mmas / 2) + k_pair`); each step advances by 8 K-scales
        (= 2 MFMAs along K). One i32 per (mi_pair, k_pair) per lane.
        """
        comptime assert k_pair < Self.num_k_mmas // 2, "k_pair out of range"

        var lane_klane, lane_mn = udivmod(lane_id(), Self.MMA_M)

        var k_scale_idx = k_pair_idx * 8 + lane_klane

        comptime for m_pack_idx in range(ceildiv(Self.num_m_mmas, 2)):
            var mn_log = warp_m_off + m_pack_idx * 32 + lane_mn
            self._a_scale_packed[
                m_pack_idx, k_pair
            ] = a_scale_loader.load_packed(mn_log, k_scale_idx)

    @always_inline
    def load_b_scales_preshuffled[
        k_pair: Int
    ](
        mut self,
        b_scale_loader: PreshuffledScaleLoader[_, _],
        warp_n_off: Int,
        k_pair_idx: Int,
    ):
        """Mirror of `load_a_scales_preshuffled` along N."""
        comptime assert k_pair < Self.num_k_mmas // 2, "k_pair out of range"

        var lane_klane, lane_mn = udivmod(lane_id(), Self.MMA_N)

        var k_scale_idx = k_pair_idx * 8 + lane_klane

        comptime for n_pack_idx in range(ceildiv(Self.num_n_mmas, 2)):
            var mn_log = warp_n_off + n_pack_idx * 32 + lane_mn
            self._b_scale_packed[
                n_pack_idx, k_pair
            ] = b_scale_loader.load_packed(mn_log, k_scale_idx)

    @always_inline
    def mma[mma_k_idx: Int, slot: Int = 0](self):
        """Execute block-scaled MFMA at MFMA-K position `mma_k_idx` using B from `slot`.

        B→src_a, A→src_b (AMD MFMA convention).

        OPSEL byte selection picks the right byte from the 2x2 cell:
            a_byte = (mma_k_idx % 2) * 2 + (m % 2)
            b_byte = (mma_k_idx % 2) * 2 + (n % 2)
        Scale dword lives at `_*_scale_packed[mn // 2, mma_k_idx // 2]`.

        WM=16 / WN=16 case: every CTA only ever sees `m=0` / `n=0`, so
        OPSEL is fixed at byte 0 (or 2 for k_pack=1). The constructor
        records a `shrui` amount (`_a_scale_shift` / `_b_scale_shift`)
        that rotates the i32 right by 0 or 8 bits so the byte OPSEL
        selects is the one for this CTA's half of the cell.
        """
        comptime assert slot < Self.num_b_slots, "slot out of range"
        var a_reg_v = self._a_reg.vectorize[1, 1, Self.mma_frag_width_bytes]()
        var b_reg_v = self._b_reg.vectorize[
            1, 1, 1, Self.mma_frag_width_bytes
        ]()
        var c_reg_v = self._c_reg.vectorize[1, Self.c_frag_size]()

        comptime for m in range(Self.num_m_mmas):
            # A-side state — invariant across the inner n loop. The
            # cdna4_block_scaled_mfma wrapper expects A/B fragments
            # sized to FLOAT4_E2M1.simd_width() = 16 — pass the 16-byte
            # `*_data` lanes directly (do NOT pad to 32).
            var a_frag = a_reg_v[mma_k_idx, m, 0]

            comptime a_byte = (mma_k_idx % 2) * 2 + (m % 2)
            var a_scale = rebind[Int32](
                self._a_scale_packed[m // 2, mma_k_idx // 2]
            )

            comptime if Self.warp_tile[0] == 16:
                a_scale = Int32(UInt32(a_scale) >> self._a_scale_shift)

            comptime for n in range(Self.num_n_mmas):
                var b_frag = b_reg_v[slot, mma_k_idx, n, 0]

                var c_frag = c_reg_v[m, n]

                comptime b_byte = (mma_k_idx % 2) * 2 + (n % 2)
                var b_scale = rebind[Int32](
                    self._b_scale_packed[n // 2, mma_k_idx // 2]
                )
                comptime if Self.warp_tile[1] == 16:
                    b_scale = Int32(UInt32(b_scale) >> self._b_scale_shift)

                cdna4_block_scaled_mfma[
                    Int32(b_byte),
                    Int32(a_byte),
                    CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                    CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                ](
                    c_frag,
                    b_frag,
                    a_frag,
                    b_scale,
                    a_scale,
                )

                c_reg_v[m, n] = c_frag


struct MXFP4MatmulAMD_PreB[
    BM: Int = 64,
    BN: Int = 128,
    BK_ELEMS: Int = 512,
    WN: Int = 64,
    B_PREFETCH: Bool = False,
]:
    """Preshuffled-B variant of `MXFP4MatmulAMD`.

    The preb path requires `num_warps_m == 1` (no LDS staging for B = no
    cross-warp M-direction B reuse), so `WM` is structurally fixed to `BM`.

    When `B_PREFETCH=True`, runs a depth-2 outer-K software pipeline: while
    the current iter's MFMAs execute, the next iter's B fragments stream
    from DRAM into the alternate b_reg slot. Doubles `_b_reg` size (extra
    VGPRs) but hides DRAM B latency across the inner MFMA chain. Targets
    K-heavy shapes (e.g. gate/up, K=7168) where outer-iter serialization
    dominates.
    """

    # WM is locked to BM — single warp along M for the preb (no-LDS-B) path.
    comptime WM = Self.BM

    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 128

    comptime num_b_slots = 2 if Self.B_PREFETCH else 1

    comptime MmaOpType = BlockScaledMmaOp_PreB[
        mma_shape=IndexList[3](Self.MMA_M, Self.MMA_N, Self.MMA_K),
        warp_tile=IndexList[3](Self.WM, Self.WN, Self.BK_ELEMS),
        num_b_slots=Self.num_b_slots,
    ]

    comptime num_m_mmas = Self.MmaOpType.num_m_mmas
    comptime num_n_mmas = Self.MmaOpType.num_n_mmas
    comptime num_k_mmas = Self.MmaOpType.num_k_mmas
    comptime MMA_K_BYTES = Self.MmaOpType.MMA_K_BYTES
    comptime c_frag_size = Self.MmaOpType.c_frag_size

    comptime BK_BYTES = Self.BK_ELEMS // 2

    comptime num_warps_m = 1
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_warps = Self.num_warps_n
    comptime num_threads = Self.num_warps * WARP_SIZE

    comptime simd_width = simd_width_of[DType.uint8]()

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        )
    )
    @staticmethod
    def run[
        out_dtype: DType,
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_pre_layout: TensorLayout,
        sfa_layout: TensorLayout,
        sfb_layout: TensorLayout,
        N: Int,
        K_BYTES: Int,
    ](
        c: TileTensor[out_dtype, c_layout, MutAnyOrigin],
        a: TileTensor[DType.uint8, a_layout, ImmutAnyOrigin],
        b_pre: TileTensor[DType.uint8, b_pre_layout, ImmutAnyOrigin],
        sfa: TileTensor[DType.float8_e8m0fnu, sfa_layout, ImmutAnyOrigin],
        sfb: TileTensor[DType.float8_e8m0fnu, sfb_layout, ImmutAnyOrigin],
        n_tile_idx: Int,
        m_tile_idx: Int,
    ):
        comptime assert (
            K_BYTES % Self.BK_BYTES == 0
        ), "K_BYTES must be a multiple of BK_BYTES"

        comptime assert (
            Self.BM == 16 or Self.BM % 32 == 0
        ), "preshuffled scales require BM == 16 or BM % 32 == 0"
        comptime assert (
            Self.WN == 16 or Self.WN % 32 == 0
        ), "preshuffled scales require WN == 16 or WN % 32 == 0"
        comptime assert Self.num_k_mmas % 2 == 0, (
            "preshuffled scales require num_k_mmas % 2 == 0 (BK_ELEMS % 256"
            " == 0)"
        )
        comptime assert (
            N % 32 == 0
        ), "N must be a multiple of 32 (= 16 * mn_pack) for preshuffled scales"

        # K_SCALES = K / 32; K_BYTES = K / 2 → K_SCALES = K_BYTES / 16.
        comptime K_SCALES = K_BYTES // 16
        # Number of (k_pack=2) scale-dwords needed per outer-K iter.
        comptime mma_k_pair_per_tile = Self.num_k_mmas // 2
        # MN_padded is only used by the layout for shape bookkeeping —
        # the byte-offset math is shape-agnostic (only K_SCALES enters).
        # See address-math notes in mxfp4_preshuffle_layouts.mojo.
        comptime MN_PADDED_PLACEHOLDER = 32

        var M = Int(a.dim[0]())

        var warp_id = warp_id()
        var warp_m, warp_n = divmod(warp_id, Self.num_warps_n)

        # SMEM for A only — B and scales come direct from preshuffled DRAM.
        var a_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.BM, Self.BK_BYTES]()
        )

        var b_loader = PreshuffledBLoader[N=N, K_BYTES=K_BYTES](b_pre)
        # Bitcast scales' float8_e8m0fnu to uint8 — same byte representation,
        # the PreshuffledScaleLoader expects uint8 buffers.
        var sfa_u8 = TileTensor(
            sfa.ptr.bitcast[Scalar[DType.uint8]](), sfa.layout
        )
        var sfb_u8 = TileTensor(
            sfb.ptr.bitcast[Scalar[DType.uint8]](), sfb.layout
        )
        var a_scale_loader = PreshuffledScaleLoader[
            MN_padded=MN_PADDED_PLACEHOLDER, K_SCALES=K_SCALES
        ](sfa_u8)
        var b_scale_loader = PreshuffledScaleLoader[
            MN_padded=MN_PADDED_PLACEHOLDER, K_SCALES=K_SCALES
        ](sfb_u8)

        comptime load_thread_cols = Self.BK_BYTES // Self.simd_width
        # Cap rows at BM so small BM (e.g. 16) with many warps doesn't ask
        # the load layout for more rows than the A tile has.
        comptime load_thread_rows = min(
            Self.num_threads // load_thread_cols, Self.BM
        )
        comptime load_layout = row_major[load_thread_rows, load_thread_cols]()
        comptime a_loads_per_tile = Self.BM // load_thread_rows
        comptime load_active_threads = load_thread_rows * load_thread_cols
        comptime a_reg_elems = Self.BM * Self.BK_BYTES // load_active_threads

        # this is of size BM x The entire matrix row
        var a_blockrow = a.tile[Self.BM, K_BYTES](m_tile_idx, 0)

        var a_load_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            row_major[1, a_reg_elems]()
        )

        var a_loader = RegTileLoader[DType.uint8, load_layout](
            a_blockrow,
            bounds_from=a,
        )

        var warp_m_off_global = m_tile_idx * Self.BM
        var warp_n_off_global = n_tile_idx * Self.BN + warp_n * Self.WN

        var mma_op = Self.MmaOpType(warp_m_off_global, warp_n_off_global)

        var c_writer = RegTileWriter[
            out_dtype, Self.MMA_M, WARP_SIZE // Self.MMA_M
        ](c)

        var k_counter = 0

        @always_inline
        @parameter
        def load_a_tile_from_dram():
            var a_block = a_blockrow.tile[Self.BM, Self.BK_BYTES](0, k_counter)
            # Idle threads past the A load layout (only matters when BM is
            # small enough that load_thread_rows is capped at BM, e.g.
            # BM=16 with many warps_n).
            if Int(thread_idx.x) < load_active_threads:
                a_loader.load(
                    a_load_reg, a_block.vectorize[1, Self.simd_width]()
                )
            k_counter += 1

        @always_inline
        @parameter
        def copy_a_tile_to_smem():
            if Int(thread_idx.x) < load_active_threads:
                var a_smem_dist = a_smem.vectorize[
                    1, Self.simd_width
                ]().distribute[load_layout](thread_idx.x)
                comptime for v in range(a_loads_per_tile):
                    a_smem_dist[v, 0] = a_load_reg.raw_load[
                        width=Self.simd_width
                    ](v * Self.simd_width)

        @always_inline
        @parameter
        def load_scales_for_iter(k_pair_base: Int):
            """Issue all A+B preshuffled scale-dword loads for one outer-K iter.

            `k_pair_base = k_iter * mma_k_pair_per_tile` is the absolute
            scale-pack offset; each k_pair advances by 1 (S_K_BLOCK = 8
            K-scales = 2 k_tiles).
            """
            comptime for k_pair in range(mma_k_pair_per_tile):
                mma_op.load_a_scales_preshuffled[k_pair=k_pair](
                    a_scale_loader,
                    warp_m_off_global,
                    k_pair_base + k_pair,
                )
                mma_op.load_b_scales_preshuffled[k_pair=k_pair](
                    b_scale_loader,
                    warp_n_off_global,
                    k_pair_base + k_pair,
                )

        comptime num_tiles = K_BYTES // Self.BK_BYTES

        # TODO use comptime pipeline scheduler

        comptime if Self.B_PREFETCH:
            # Depth-2 outer-K software pipeline.
            #
            # Prologue: load A (smem), all B fragments (slot 0), and the
            # iter-0 scale dwords into VGPRs.
            load_a_tile_from_dram()
            copy_a_tile_to_smem()
            comptime for k in range(Self.num_k_mmas):
                mma_op.load_b_frag_preshuffled[k, slot=0](
                    b_loader, warp_n_off_global, 0
                )
            load_scales_for_iter(0)
            barrier()

            # Steady state: for each i in [0, num_tiles-1), MFMA iter i
            # from `cur_slot` while prefetching iter i+1's B into
            # `nxt_slot`. A SMEM is refilled before the barrier so iter
            # i+1's MFMAs can read it next pass. Scales are reloaded
            # post-barrier from the preshuffled tensor (no SMEM).
            comptime for i in range(num_tiles - 1):
                comptime cur_slot = i % 2
                comptime nxt_slot = (i + 1) % 2
                var nxt_k_byte_base = (i + 1) * Self.BK_BYTES

                comptime for k in range(Self.num_k_mmas):
                    mma_op.load_b_frag_preshuffled[k, slot=nxt_slot](
                        b_loader, warp_n_off_global, nxt_k_byte_base
                    )

                var a_warp = a_smem.tile[Self.WM, Self.BK_BYTES](warp_m, 0)
                comptime for k in range(Self.num_k_mmas):
                    mma_op.load_a_frag_from_smem[k](a_warp)
                    mma_op.mma[k, slot=cur_slot]()

                load_a_tile_from_dram()
                barrier()
                copy_a_tile_to_smem()
                load_scales_for_iter((i + 1) * mma_k_pair_per_tile)
                barrier()

            # Epilogue: MFMA the last iter from its slot.
            comptime last_slot = (num_tiles - 1) % 2
            var a_warp = a_smem.tile[Self.WM, Self.BK_BYTES](warp_m, 0)
            comptime for k in range(Self.num_k_mmas):
                mma_op.load_a_frag_from_smem[k](a_warp)
                mma_op.mma[k, slot=last_slot]()
            barrier()
        else:
            for k_iter in range(num_tiles):
                load_a_tile_from_dram()
                copy_a_tile_to_smem()
                load_scales_for_iter(k_iter * mma_k_pair_per_tile)
                barrier()

                var a_warp = a_smem.tile[Self.WM, Self.BK_BYTES](warp_m, 0)
                var k_byte_base = k_iter * Self.BK_BYTES

                comptime for k in range(Self.num_k_mmas):
                    mma_op.load_a_frag_from_smem[k](a_warp)
                    mma_op.load_b_frag_preshuffled[k](
                        b_loader, warp_n_off_global, k_byte_base
                    )
                    mma_op.mma[k]()
                barrier()

        var c_reg = mma_op.accum_tile()
        var c_block = c.tile[Self.BM, Self.BN](m_tile_idx, n_tile_idx)
        var c_warp = c_block.tile[Self.WM, Self.WN](warp_m, warp_n)

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                c_writer.store(
                    c_warp.tile[Self.MMA_M, Self.MMA_N](m_mma, n_mma).vectorize[
                        1, Self.c_frag_size
                    ](),
                    c_reg.tile[1, Self.c_frag_size](m_mma, n_mma),
                )
