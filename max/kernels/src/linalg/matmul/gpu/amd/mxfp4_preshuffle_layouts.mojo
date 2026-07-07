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
"""Host-side MXFP4 preshuffle layouts for AMD CDNA4 grouped MoE matmul.

`Shuffler` bundles two layout transforms required by the FP4 MoE matmul
kernel. Both run once on the host at weight-load time (per the load-time-
prep convention).

`Shuffler.preshuffle_b_5d`:
    [E, N, K_BYTES] (row-major, packed FP4) -> flat byte buffer indexed as
    `(E, N0, K0, KLane=4, NLane=16, KPack=16)`. Each lane's 16-byte MFMA
    fragment lands at a contiguous DRAM address, so B reads go straight
    DRAM -> VGPR via `buffer_load_dwordx4` with no LDS round-trip.

`Shuffler.preshuffle_scale_4d`:
    [E, MN, K_SCALES] (row-major, E8M0 bytes) -> flat byte buffer indexed
    as `(E, MN1, K1, XdlKThread=4, XdlMNThread=16, KXdlPack=2,
    MNXdlPack=2)`. One i32 lane-load fetches 4 E8M0 scales packed in
    (k_pack, mn_pack) order, feeding 4 sub-MMAs via the MFMA opsel byte
    selectors.

`mxfp4_preshuffle_b_5d_gpu`:
    GPU equivalent of `Shuffler.preshuffle_b_5d`. LDS-staged so both HBM
    reads and writes are wave-coalesced. Constant-folded by the graph
    compiler when called via `mo.mxfp4.preshuffle.b.5d` on a `Constant`
    weight, so the shuffle runs once at session.load instead of every
    forward pass.

Layout reference (canonical):
    composable_kernel/example/ck_tile/18_flatmm/mxgemm/mx_flatmm_arch_traits.hpp:73-167
        — `preShuffleWeight` (B 5D) and `preShuffleScale` (scale 4D).
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceContext, HostBuffer
from std.math import align_up
from std.math.uutils import udivmod, uceildiv
from std.memory import bitcast

from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import Layout, TensorLayout, col_major
from layout.tile_tensor import stack_allocation

from std.utils import StaticTuple


struct Shuffler[E: Int]:
    """MXFP4 preshuffle layouts and helpers for AMD CDNA4.

    Parameters:
        E: Number of groups (experts / sort-blocks) the shuffler operates on.
            Use `Shuffler[1]` for single-group consumers.
    """

    # ---- 16x16x128 f8f6f4 MFMA hardware constants — not B-specific. ----
    # 16 rows per MFMA across the matrix MN axis (handled by 16 threads).
    comptime MFMA_MN_LANES: Int = 16
    # That 16-lane cluster is repeated 4 times across K (64 lanes / 16 = 4).
    comptime MFMA_K_LANES: Int = 4
    # Each lane supplies 16 bytes (= 32 FP4 elements) of operand per MFMA.
    # FP4-specific: 128 K elements / 4 K-lanes = 32 elts/lane = 16 bytes.
    comptime MFMA_LANE_BYTES: Int = 16
    # Total K extent per MFMA tile in bytes: 4 K-lanes x 16 bytes = 64 bytes.
    comptime MFMA_K_BYTES: Int = Self.MFMA_K_LANES * Self.MFMA_LANE_BYTES  # 64
    # 64-thread wave, one atom per lane; covers a full 16x4 atom dst tile.
    comptime NUM_THREADS: Int = Self.MFMA_MN_LANES * Self.MFMA_K_LANES  # 64

    # ---- Byte strides for the B 5D layout ----
    # leaf-to-root: KPack=1, NLane=16, KLane=NLane*KPack=256,
    # K0=KLane*256=1024, N0=K0*1024 (runtime).
    comptime B_STRIDE_LANE_BYTES: Int = 1
    comptime B_STRIDE_MN_LANE: Int = Self.MFMA_LANE_BYTES  # 16
    comptime B_STRIDE_K_LANE: Int = (
        Self.MFMA_MN_LANES * Self.MFMA_LANE_BYTES
    )  # 256
    comptime B_STRIDE_K0: Int = Self.MFMA_K_LANES * Self.B_STRIDE_K_LANE  # 1024

    # ---- Scale 4D layout (FP4-specific) ----
    # Each i32 cell = 2x2 = 4 E8M0 bytes. Lane counts come from MFMA
    # hardware constants above.
    comptime S_MN_PACK: Int = 2
    comptime S_K_PACK: Int = 2
    comptime S_MN_BLOCK: Int = Self.MFMA_MN_LANES * Self.S_MN_PACK  # 32
    comptime S_K_BLOCK: Int = Self.MFMA_K_LANES * Self.S_K_PACK  # 8

    # One packed-scale cell = 4 bytes (mn_pack=2 × k_pack=2 E8M0 bytes,
    # arranged with mn_pack stride 1 and k_pack stride 2 within the cell).
    comptime packed_scale_bytes: Int = 4

    # ---- B 5D grouped layout ----
    #
    # Access pattern — per-MFMA, single 64-lane wave:
    #
    #   lane_id ∈ [0, 64)
    #   nlane = lane_id % 16   (which of 16 matrix rows on B's N axis)
    #   klane = lane_id / 16   (which of 4 K-segments)
    #
    #   Lane issues `buffer_load_dwordx4` (16 bytes) from:
    #     addr = base + nlane*B_STRIDE_MN_LANE      # 16
    #                 + klane*B_STRIDE_K_LANE       # 256
    #                 + [0..16)*B_STRIDE_LANE_BYTES # 1 (within the dwordx4)
    #
    # Coalescing: lanes 0..15 → [0..256), 16..31 → [256..512),
    # 32..47 → [512..768), 48..63 → [768..1024). The wave's 64 simultaneous
    # 16-byte loads cover exactly [0..1024) = one MFMA tile of B = K0 stride,
    # contiguous, no gaps. AMD memory subsystem merges this into ~16 cache
    # lines (1024 / 64 B per line) with no wasted bandwidth.
    #
    # Worked example — N=64, K_BYTES=128 (K0_count=2, N0_count=4). Logical
    # coords each lane passes to b_5d_grouped_layout, and the byte returned:
    #
    # (n_iter=0, k_iter=0)  warp_n_base = 0   k_byte_base = 0
    #   lane  0 (nl=0,  kl=0):  Coord(Idx[0], Idx[0],  0..15)   → bytes    0..15
    #   lane  1 (nl=1,  kl=0):  Coord(Idx[0], Idx[1],  0..15)   → bytes   16..31
    #   lane 15 (nl=15, kl=0):  Coord(Idx[0], Idx[15], 0..15)   → bytes  240..255
    #   lane 16 (nl=0,  kl=1):  Coord(Idx[0], Idx[0],  16..31)  → bytes  256..271
    #   lane 32 (nl=0,  kl=2):  Coord(Idx[0], Idx[0],  32..47)  → bytes  512..527
    #   lane 48 (nl=0,  kl=3):  Coord(Idx[0], Idx[0],  48..63)  → bytes  768..783
    #   lane 63 (nl=15, kl=3):  Coord(Idx[0], Idx[15], 48..63)  → bytes 1008..1023
    #
    # (n_iter=0, k_iter=1)  k_byte_base = 64
    #   lane 0:  Coord(Idx[0], Idx[0], 64..79)                  → bytes 1024..1039
    #                                                                  (1 * B_STRIDE_K0)
    #
    # (n_iter=1, k_iter=0)  warp_n_base = 16
    #   lane 0:  Coord(Idx[0], Idx[16], 0..15)                  → bytes 2048..2063
    #                                                                  (1 * K0_count * B_STRIDE_K0)
    #
    # (n_iter=1, k_iter=1)  warp_n_base = 16,  k_byte_base = 64
    #   lane  0 (nl=0, kl=0):  Coord(Idx[0], Idx[16], 64..79)   → bytes 3072..3087
    #                                                                  (2048 N-stride + 1024 K-stride)
    #   lane 17 (nl=1, kl=1):  Coord(Idx[0], Idx[17], 80..95)   → bytes 3344..3359
    #                                                                  (16 + 2048 + 256 + 1024 = 3344)
    #
    # The logical E (group / expert) axis is prepended with stride =
    # bytes-per-group, so a single TileTensor view spans all groups and host
    # preshuffle iterates `(e, n, k_byte)` with no per-group pointer math at
    # the call site. Single-group consumers pass E=1 and Idx[0] for e.
    comptime b_5d_grouped_layout[N: Int, K_BYTES: Int] = Layout(
        Coord(
            Idx[Self.E],
            Coord(Idx[Self.MFMA_MN_LANES], Idx[N // Self.MFMA_MN_LANES]),
            Coord(
                Idx[Self.MFMA_LANE_BYTES],
                Idx[Self.MFMA_K_LANES],
                Idx[K_BYTES // Self.MFMA_K_BYTES],
            ),
        ),
        Coord(
            Idx[N * K_BYTES],
            Coord(
                Idx[Self.B_STRIDE_MN_LANE],
                Idx[(K_BYTES // Self.MFMA_K_BYTES) * Self.B_STRIDE_K0],
            ),
            Coord(
                Idx[Self.B_STRIDE_LANE_BYTES],
                Idx[Self.B_STRIDE_K_LANE],
                Idx[Self.B_STRIDE_K0],
            ),
        ),
    )

    @staticmethod
    @always_inline
    def scale_4d_byte_off[
        K_SCALES: Int, packed_mode: Bool = False
    ](mn: Int, k_scale: Int) -> Int:
        # packed_mode changes the element granularity. In packed mode we provide
        # the byte index to the next packed scale. Otherwise its to the scale.

        # K_SCALES is the number of K_SCALES in the K dimension of the matrix

        # One Scale Atom is a 16 x 4 tile consisting of packed scales
        # Each packed scale contains 4 scales, 2 scales across the mn mode
        # spaced out by 16 rows (rows in one mfma), and 2 scales across the
        # k mode spaced out by 4 scales (columns in one mfma)

        # The 16x4 atom is column major with the M scales in the packed
        # dimension increasing the fastest.

        # The Scale Atom's are tiled in row major format

        # column in the packed scale
        var packed_m_pos = (mn // Self.MFMA_MN_LANES) % Self.S_MN_PACK

        # row in the packed scale
        var packed_k_pos = (
            (k_scale // Self.MFMA_K_LANES) % Self.S_K_PACK
        ) * Self.S_MN_PACK

        var packed_byte_off = packed_m_pos + packed_k_pos

        # What row in the Scale Atom
        var mn_lane = mn % Self.MFMA_MN_LANES

        # MN Scale Atom tile
        var mn_scale_tile = mn // Self.S_MN_BLOCK

        # What column in the Scale Atom
        var k_lane = k_scale % Self.MFMA_K_LANES

        # K Scale Atom tile
        var k_scale_tile = k_scale // Self.S_K_BLOCK

        comptime scale_atom_bytes_per_column: Int = (
            Self.MFMA_MN_LANES * Self.packed_scale_bytes
        )

        comptime bytes_per_atom: Int = Self.MFMA_MN_LANES * Self.packed_scale_bytes * Self.MFMA_K_LANES
        comptime atoms_per_row: Int = K_SCALES // Self.S_K_BLOCK
        comptime bytes_per_row: Int = bytes_per_atom * atoms_per_row

        var atom_byte_off = (
            mn_lane * Self.packed_scale_bytes
            + k_lane * scale_atom_bytes_per_column
        )
        var tile_byte_off = (
            mn_scale_tile * bytes_per_row + k_scale_tile * bytes_per_atom
        )

        comptime if packed_mode:
            return tile_byte_off + atom_byte_off
        else:
            return packed_byte_off + tile_byte_off + atom_byte_off

    @staticmethod
    @always_inline
    def scale_4d_slot_byte_off[
        K_SCALES: Int, packed_mode: Bool = False
    ](expert_slot: Int, mn: Int, k_scale: Int, max_padded_M: Int) -> Int:
        """Byte offset of an E8M0 scale within the per-expert `scale_4d` slot.

        Single source of truth for the offset shared by (1) the standalone
        `_preshuffle_grouped_scale_4d_kernel`, (2) the `fused_silu` KS64 fold,
        and (3) the `ep_wait` KS224 fold. Each expert owns a fixed-stride slot
        of `max_padded_M * K_SCALES` bytes; within it the scale lands at
        `scale_4d_byte_off(mn, k_scale)`.

        Parameters:
            K_SCALES: Number of E8M0 scales along K (`K // 32`).
            packed_mode: Byte index of the next packed scale (used by the
                standalone preshuffle's i32-cell gather); otherwise the byte
                index of the scale itself.

        Args:
            expert_slot: Per-expert slot index (`expert_id + shared_offset`).
            mn: Local row within the expert (token row, 0-based).
            k_scale: Scale index along K.
            max_padded_M: Per-expert slot stride in rows (= `align_up(max, 32)`).

        Returns:
            Byte offset into the flat `scale_4d` buffer.
        """
        return expert_slot * max_padded_M * K_SCALES + Self.scale_4d_byte_off[
            K_SCALES=K_SCALES, packed_mode=packed_mode
        ](mn, k_scale)

    # ---- Wrapped TileTensor types — what the preshuffle functions return ----
    comptime BTileTensor[N: Int, K_BYTES: Int] = TileTensor[
        mut=True,
        DType.uint8,
        type_of(Self.b_5d_grouped_layout[N=N, K_BYTES=K_BYTES]),
        MutAnyOrigin,
    ]

    # Scale-buffer callers use a plain `TileTensor[DType.uint8, ...]` or a
    # raw `HostBuffer[DType.uint8]` — the byte layout is fully expressed by
    # `scale_4d_byte_off`, no type-system marker needed.

    # ---- Helpers ----

    @staticmethod
    @always_inline
    def scale_padded_mn(MN: Int) -> Int:
        """Padded MN dim used by the 4D scale layout: MN rounded up to 32."""
        return align_up(MN, Self.S_MN_BLOCK)

    # ---- B (weight) preshuffle (GPU) ----
    #
    # Each dst tile is a column-major 16×4 grid of 16-byte atoms: NLane
    # is the fast atom-axis (stride 16 B), KLane is the slow atom-axis
    # (stride 256 B). The wave maps one atom per lane and uses a
    # col_major thread layout on the dst side so the 64 lanes' writes
    # hit the tile's 1024 contiguous bytes in N-fast order — a single
    # coalesced wave write. Bank conflicts on LDS ignored for now;
    # revisit if profiling shows LDS is the bottleneck.

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__name("mxfp4_preshuffle_b_5d_kernel")
    def _preshuffle_b_5d_kernel[
        N: Int,
        K_BYTES: Int,
        RawLayout: TensorLayout,
        DstLayout: TensorLayout,
    ](
        raw: TileTensor[DType.uint8, RawLayout, ImmutAnyOrigin],
        dst: TileTensor[DType.uint8, DstLayout, MutAnyOrigin],
    ):
        """LDS-staged per-tile B 5D preshuffle on AMD GPU.

        One CTA per `(e, n0, k0)` 16×64-byte tile. Phase 1 reads the
        tile from `raw` coalesced and stages it in LDS at the literal
        `(nlane, klane)` slot. Phase 2 reads the LDS transposed (N-fast
        across the wave) and writes coalesced into `dst`'s column-major
        tile.
        """
        var tid = Int(thread_idx.x)
        var e = Int(block_idx.z)
        var n0 = Int(block_idx.y)
        var k0 = Int(block_idx.x)

        # raw: per-CTA [1, 16, 4] atom tile from the row-major 3D view.
        var raw_tile = raw.tile[1, Self.MFMA_MN_LANES, Self.MFMA_K_BYTES](
            e, n0, k0
        ).vectorize[1, 1, Self.MFMA_LANE_BYTES]()

        # dst: derive the tile's base byte offset from the block coords
        # (tiles are 1024 B each, contiguous in the 5D layout), then
        # wrap a flat per-tile view with (NLane, KLane, KPack) strides
        # matching b_5d_grouped_layout's intra-tile pattern (NLane stride
        # 16 B, KLane 256 B, KPack 1 B). Vectorizing KPack absorbs the
        # 16-byte atom into element_size, giving a clean rank-3 [16, 4,
        # 1] col-major-in-atoms tile that .distribute can consume.
        comptime TILE_BYTES = Self.MFMA_MN_LANES * Self.MFMA_K_BYTES
        var dst_tile_base = dst.ptr + (
            e * N * K_BYTES  # skip to current expert
            + n0
            * K_BYTES
            * Self.MFMA_MN_LANES  # skip past N rows in current expert in units of N_Tile length
            + k0 * TILE_BYTES  # go to the Kth tile of this row
        )

        var dst_tile = TileTensor[mut=True](
            dst_tile_base,
            Layout(
                Coord(
                    Idx[Self.MFMA_MN_LANES],
                    Idx[Self.MFMA_K_LANES],
                    Idx[Self.MFMA_LANE_BYTES],
                ),
                Coord(
                    Idx[Self.B_STRIDE_MN_LANE],
                    Idx[Self.B_STRIDE_K_LANE],
                    Idx[Self.B_STRIDE_LANE_BYTES],
                ),
            ),
        ).vectorize[1, 1, Self.MFMA_LANE_BYTES]()

        # LDS staging: 16 NLane × 4 KLane atoms in their literal slot.
        var smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.MFMA_MN_LANES, Self.MFMA_K_BYTES]()
        )
        var smem_atoms = smem.vectorize[1, Self.MFMA_LANE_BYTES]()

        # raw is row-major. row_major thread layout (K-fast) puts 4
        # lanes per N row sweeping 64 contiguous K bytes -> coalesced
        # HBM read; same layout on the LDS write lands atoms in their
        # literal `(nlane, klane)` slot.
        comptime row_major_thread_layout_3d = row_major[
            1, Self.MFMA_MN_LANES, Self.MFMA_K_LANES
        ]()
        comptime row_major_thread_layout_2d = row_major[
            Self.MFMA_MN_LANES, Self.MFMA_K_LANES
        ]()
        var v = raw_tile.distribute[row_major_thread_layout_3d](tid)[0, 0, 0]
        smem_atoms.distribute[row_major_thread_layout_2d](tid)[0, 0] = v

        barrier()

        # dst tile is column-major in atoms (NLane stride = 1 atom). The
        # col_major thread layout maps tid -> (nl=tid%16, kl=tid//16);
        # the same layout on SMEM (transposed read) and on dst (the
        # natural intra-tile order) -> 64 lanes write 1024 contiguous
        # bytes per CTA in a single coalesced wave.
        comptime col_major_thread_layout_3d = col_major[
            Self.MFMA_MN_LANES, Self.MFMA_K_LANES, 1
        ]()
        comptime col_major_thread_layout_2d = col_major[
            Self.MFMA_MN_LANES, Self.MFMA_K_LANES
        ]()
        var w = smem_atoms.distribute[col_major_thread_layout_2d](tid)[0, 0]
        dst_tile.distribute[col_major_thread_layout_3d](tid)[0, 0, 0] = w

    @staticmethod
    def preshuffle_b_5d[
        N: Int,
        K_BYTES: Int,
    ](
        raw: TileTensor[
            mut=False, DType.uint8, address_space=AddressSpace.GENERIC, ...
        ],
        dst: TileTensor[
            mut=True, DType.uint8, address_space=AddressSpace.GENERIC, ...
        ],
        ctx: DeviceContext,
    ) raises:
        """Launch the GPU MXFP4 B 5D preshuffle.

        Invoked eagerly from model weight adapters (one-shot graph) so
        the shuffle runs once at session.load instead of the ~hours-long
        numpy CPU path. Mirrors `block_scales_interleave`'s origin
        handling pattern (accept any origin, cast to any-origin for the
        kernel).

        Parameters:
            N: Per-expert N (must be a multiple of 16).
            K_BYTES: Per-expert FP4-packed K (must be a multiple of 64).

        Args:
            raw: Row-major source weights `[E, N, K_BYTES]`.
            dst: Destination buffer (same byte footprint; bytes get
                written in `b_5d_grouped_layout` order).
            ctx: AMD device context.
        """
        comptime assert (
            N % Self.MFMA_MN_LANES == 0
        ), "preshuffle_b_5d: N must be a multiple of 16"
        comptime assert (
            K_BYTES % Self.MFMA_K_BYTES == 0
        ), "preshuffle_b_5d: K_BYTES must be a multiple of 64"

        var raw_any = raw.as_unsafe_any_origin()
        var dst_any = dst.as_unsafe_any_origin()
        comptime kernel = Self._preshuffle_b_5d_kernel[
            N,
            K_BYTES,
            type_of(raw_any).LayoutType,
            type_of(dst_any).LayoutType,
        ]
        ctx.enqueue_function[kernel](
            raw_any,
            dst_any,
            grid_dim=(
                K_BYTES // Self.MFMA_K_BYTES,
                N // Self.MFMA_MN_LANES,
                Self.E,
            ),
            block_dim=Self.NUM_THREADS,
        )

    # ---- Scale preshuffle (works for both A and B scales; same layout) ----

    @staticmethod
    def preshuffle_scale_4d[
        MN: Int,
        K_SCALES: Int,
        SrcLayout: TensorLayout,
    ](
        src: TileTensor[DType.uint8, SrcLayout, MutAnyOrigin],
        mut dst: HostBuffer[DType.uint8],
    ):
        # shuffles the scale layout on CPU, and pads the layout
        # to align the scales with the atom

        comptime assert (
            K_SCALES % Self.S_K_BLOCK == 0
        ), "preshuffle_scale_4d: K_SCALES must be a multiple of 8"

        comptime MN_padded = Self.scale_padded_mn(MN)
        comptime group_bytes = MN_padded * K_SCALES

        comptime for e in range(Self.E):
            comptime e_off = e * group_bytes
            for mn in range(MN):
                for k_scale in range(K_SCALES):
                    var byte_off = e_off + Self.scale_4d_byte_off[
                        K_SCALES=K_SCALES
                    ](mn, k_scale)
                    dst[byte_off] = src[Coord(e, mn, k_scale)]

            for mn in range(MN, MN_padded):
                for k_scale in range(K_SCALES):
                    var byte_off = e_off + Self.scale_4d_byte_off[
                        K_SCALES=K_SCALES
                    ](mn, k_scale)
                    dst[byte_off] = UInt8(0)

    # ---- Grouped A-scale GPU preshuffle (per-step transient) ----
    #
    # Fixed-stride slot layout. Each expert e occupies a slot of
    # `max_padded_M` rows starting at `e * max_padded_M`, regardless of
    # its actual `num_tokens[e]`. Real preshuffled scales fill the first
    # `num_tokens[e]` rows of the slot; the kernel does NOT zero the rows
    # past `align_up(num_tokens[e], 32)` (they are left uninitialized) —
    # the matmul's tight per-expert V# bound clamps those OOB reads to 0.
    #
    # No metadata array: the matmul dispatcher derives the per-expert
    # start as `expert_slot * max_padded_M` from a single runtime int.

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__name(t"mxfp4_preshuffle_grouped_scale_4d_kernel_KS{K_SCALES}")
    def _preshuffle_grouped_scale_4d_kernel[
        K_SCALES: Int,
        SrcLayout: TensorLayout,
        DstLayout: TensorLayout,
        AOffsetsLayout: TensorLayout,
    ](
        sfa_raw: TileTensor[DType.uint8, SrcLayout, ImmutAnyOrigin],
        sfa_pre: TileTensor[mut=True, DType.uint8, DstLayout, MutAnyOrigin],
        a_offsets: TileTensor[DType.uint32, AOffsetsLayout, ImmutAnyOrigin],
        num_active_experts: Int,
        max_padded_M: Int,
        total_wg: Int,
    ):
        """Persistent grid-strided per-expert scale preshuffle.

        Grid: `(total_wg,)`; block: 64 threads (one warp). Each CTA
        grid-strides over the global tile counter, where a tile is one
        `(expert, m_block, k_block)` triple writing 64 i32 cells.
        `m_blocks` per expert is `uceildiv(num_tokens, S_MN_BLOCK)` —
        only REAL M is iterated, so pad rows past `align_up(num_tokens,
        32)` are never written.

        That last fact is safe because the matmul constructs its sfa V#
        with bound `align_up(num_tokens[e], 32) * K_SCALES`, so OOB
        scale reads past real data clamp to 0 in hardware. Avoiding the
        zero-fill on trailing m_blocks is the whole point of this
        kernel — it's the dominant cost when `max_padded_M >>
        num_tokens` (worst-case ~256x over-provisioning at decode).

        Slot stride remains `max_padded_M * K_SCALES` bytes per expert,
        so the matmul reads expert `e` at offset `e * max_padded_M *
        K_SCALES` unchanged.
        """
        var tid = thread_idx.x
        var linear_wg = block_idx.x

        var k_lane, mn_lane = udivmod(tid, Self.MFMA_MN_LANES)  # 0..3, 0..15

        var k_blocks = K_SCALES // Self.S_K_BLOCK
        var target_tile = linear_wg
        var current_tile = 0

        for expert_slot in range(num_active_experts):
            var token_start = Int(a_offsets[Coord(expert_slot)])
            var num_tokens = (
                Int(a_offsets[Coord(expert_slot + 1)]) - token_start
            )

            # Empty slot — contributes zero tiles to the global counter,
            # so the invariant `current_tile` holds without update.
            if num_tokens == 0:
                continue

            var m_blocks = uceildiv(num_tokens, Self.S_MN_BLOCK)
            var expert_work = m_blocks * k_blocks
            var expert_end = current_tile + expert_work

            # Target tile is past this expert — bump current_tile to the
            # next expert's start and continue. No work for this WG here.
            if target_tile >= expert_end:
                current_tile = expert_end
                continue

            # Slot base in bytes — fixed-stride per expert. Trailing
            # m_blocks past uceildiv(num_tokens, 32) are NOT written; the
            # matmul's tight per-expert V# clamps OOB reads to 0. The
            # slot + cell offset is folded into `scale_4d_slot_byte_off`
            # (shared with the fused_silu / ep_wait folds).

            while target_tile < expert_end:
                var local_tile = target_tile - current_tile
                var m_block, k_block = udivmod(local_tile, k_blocks)

                var cell_mn_base = m_block * Self.S_MN_BLOCK + mn_lane
                var cell_k_base = k_block * Self.S_K_BLOCK + k_lane

                # Gather 4 source bytes into the i32 cell. Each cell holds
                # bytes at (mn_pack, k_pack) ∈ {0,1}², packed col-major:
                #   byte 0 = (mn_pack=0, k_pack=0), byte 1 = (mn_pack=1, k_pack=0),
                #   byte 2 = (mn_pack=0, k_pack=1), byte 3 = (mn_pack=1, k_pack=1).
                # OOB rows past num_tokens stay zero in the cell (= the
                # last partial m_block's pad rows). Higher m_blocks are
                # not iterated at all.
                var cell_bytes = SIMD[DType.uint8, 4](0)

                comptime for k_pack in range(Self.S_K_PACK):
                    comptime for mn_pack in range(Self.S_MN_PACK):
                        var src_mn = cell_mn_base + mn_pack * Self.MFMA_MN_LANES
                        var src_k = cell_k_base + k_pack * Self.MFMA_K_LANES
                        if src_mn < num_tokens:
                            cell_bytes[
                                k_pack * Self.S_MN_PACK + mn_pack
                            ] = sfa_raw[Coord((token_start + src_mn), src_k)]

                var cell_byte_off = Self.scale_4d_slot_byte_off[
                    K_SCALES=K_SCALES, packed_mode=True
                ](expert_slot, cell_mn_base, cell_k_base, max_padded_M)
                var dst_ptr = (sfa_pre.ptr + cell_byte_off).bitcast[
                    Scalar[DType.int32]
                ]()
                dst_ptr[0] = bitcast[DType.int32, 1](cell_bytes)[0]

                target_tile += total_wg

            current_tile = expert_end

    @staticmethod
    def preshuffle_grouped_scale_4d_gpu[
        K_SCALES: Int,
        SfaRawLayout: TensorLayout,
        SfaPreLayout: TensorLayout,
        AOffsetsLayout: TensorLayout,
    ](
        sfa_raw: TileTensor[
            mut=False,
            DType.uint8,
            SfaRawLayout,
            address_space=AddressSpace.GENERIC,
            ...,
        ],
        sfa_pre: TileTensor[
            mut=True,
            DType.uint8,
            SfaPreLayout,
            address_space=AddressSpace.GENERIC,
            ...,
        ],
        a_offsets: TileTensor[
            mut=False,
            DType.uint32,
            AOffsetsLayout,
            address_space=AddressSpace.GENERIC,
            ...,
        ],
        num_active_experts: Int,
        max_num_tokens_per_expert: Int,
        total_wg: Int,
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            K_SCALES % Self.S_K_BLOCK == 0
        ), "preshuffle_grouped_scale_4d_gpu: K_SCALES must be a multiple of 8"

        var raw_any = sfa_raw.as_unsafe_any_origin()
        var pre_any = sfa_pre.as_unsafe_any_origin()
        var a_off_any = a_offsets.as_unsafe_any_origin()

        var max_padded_M = align_up(max_num_tokens_per_expert, Self.S_MN_BLOCK)

        comptime kernel = Self._preshuffle_grouped_scale_4d_kernel[
            K_SCALES,
            type_of(raw_any).LayoutType,
            type_of(pre_any).LayoutType,
            type_of(a_off_any).LayoutType,
        ]
        ctx.enqueue_function[kernel](
            raw_any,
            pre_any,
            a_off_any,
            num_active_experts,
            max_padded_M,
            total_wg,
            grid_dim=(total_wg,),
            block_dim=Self.NUM_THREADS,
        )
