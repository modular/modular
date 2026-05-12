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

Layout reference (canonical):
    composable_kernel/example/ck_tile/18_flatmm/mxgemm/mx_flatmm_arch_traits.hpp:73-167
        — `preShuffleWeight` (B 5D) and `preShuffleScale` (scale 4D).
"""

from std.gpu.host import HostBuffer
from std.math import ceildiv

from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import Layout, TensorLayout


struct Shuffler[E: Int]:
    """Host-side MXFP4 preshuffle layouts and helpers for AMD CDNA4.

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

    # Byte strides for the scale 4D layout. Within-cell: mn_pack=1, k_pack=2.
    # Between-cell: mn_lane=4 (one i32), k_lane=4*16=64, k0=64*4=256, n0=K0*256.
    comptime S_STRIDE_MN_PACK: Int = 1
    comptime S_STRIDE_K_PACK: Int = 2
    comptime S_STRIDE_MN_LANE: Int = 4
    comptime S_STRIDE_K_LANE: Int = (
        Self.MFMA_MN_LANES * Self.S_STRIDE_MN_LANE
    )  # 64
    comptime S_STRIDE_K0: Int = Self.MFMA_K_LANES * Self.S_STRIDE_K_LANE  # 256

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
    #   lane  0 (nl=0,  kl=0):  Coord(Idx(0), Idx(0),  Idx(0..15))   → bytes    0..15
    #   lane  1 (nl=1,  kl=0):  Coord(Idx(0), Idx(1),  Idx(0..15))   → bytes   16..31
    #   lane 15 (nl=15, kl=0):  Coord(Idx(0), Idx(15), Idx(0..15))   → bytes  240..255
    #   lane 16 (nl=0,  kl=1):  Coord(Idx(0), Idx(0),  Idx(16..31))  → bytes  256..271
    #   lane 32 (nl=0,  kl=2):  Coord(Idx(0), Idx(0),  Idx(32..47))  → bytes  512..527
    #   lane 48 (nl=0,  kl=3):  Coord(Idx(0), Idx(0),  Idx(48..63))  → bytes  768..783
    #   lane 63 (nl=15, kl=3):  Coord(Idx(0), Idx(15), Idx(48..63))  → bytes 1008..1023
    #
    # (n_iter=0, k_iter=1)  k_byte_base = 64
    #   lane 0:  Coord(Idx(0), Idx(0), Idx(64..79))                  → bytes 1024..1039
    #                                                                  (1 * B_STRIDE_K0)
    #
    # (n_iter=1, k_iter=0)  warp_n_base = 16
    #   lane 0:  Coord(Idx(0), Idx(16), Idx(0..15))                  → bytes 2048..2063
    #                                                                  (1 * K0_count * B_STRIDE_K0)
    #
    # (n_iter=1, k_iter=1)  warp_n_base = 16,  k_byte_base = 64
    #   lane  0 (nl=0, kl=0):  Coord(Idx(0), Idx(16), Idx(64..79))   → bytes 3072..3087
    #                                                                  (2048 N-stride + 1024 K-stride)
    #   lane 17 (nl=1, kl=1):  Coord(Idx(0), Idx(17), Idx(80..95))   → bytes 3344..3359
    #                                                                  (16 + 2048 + 256 + 1024 = 3344)
    #
    # The logical E (group / expert) axis is prepended with stride =
    # bytes-per-group, so a single TileTensor view spans all groups and host
    # preshuffle iterates `(e, n, k_byte)` with no per-group pointer math at
    # the call site. Single-group consumers pass E=1 and Idx(0) for e.
    comptime b_5d_grouped_layout[N: Int, K_BYTES: Int] = Layout(
        Coord(
            Idx[Self.E](),
            Coord(Idx[Self.MFMA_MN_LANES](), Idx[N // Self.MFMA_MN_LANES]()),
            Coord(
                Idx[Self.MFMA_LANE_BYTES](),
                Idx[Self.MFMA_K_LANES](),
                Idx[K_BYTES // Self.MFMA_K_BYTES](),
            ),
        ),
        Coord(
            Idx[N * K_BYTES](),
            Coord(
                Idx[Self.B_STRIDE_MN_LANE](),
                Idx[(K_BYTES // Self.MFMA_K_BYTES) * Self.B_STRIDE_K0](),
            ),
            Coord(
                Idx[Self.B_STRIDE_LANE_BYTES](),
                Idx[Self.B_STRIDE_K_LANE](),
                Idx[Self.B_STRIDE_K0](),
            ),
        ),
    )

    # ---- Scale 4D grouped layout ----
    #
    # Used for both A scales (matrix-axis = M) and B scales (matrix-axis =
    # N) — the layout is symmetric, "MN" is generic for the matrix-axis lane.
    #
    # Access pattern — per-MFMA, single 64-lane wave:
    #
    #   mn_lane = lane_id % 16   (matrix-axis lane; M for A, N for B)
    #   k_lane  = lane_id / 16   (K-segment within the MFMA tile)
    #
    #   One i32 lane-load (4 bytes = 4 E8M0 scales) from:
    #     addr = base + mn_lane * S_STRIDE_MN_LANE     # 4
    #                 + k_lane  * S_STRIDE_K_LANE      # 64
    #   Within that i32, the 4 bytes are arranged as a (k_pack, mn_pack) 2x2 cell:
    #     byte 0 (mn_pack=0, k_pack=0) → MMA sub-tile (m=0, k=0)
    #     byte 1 (mn_pack=1, k_pack=0) → MMA sub-tile (m=1, k=0)
    #     byte 2 (mn_pack=0, k_pack=1) → MMA sub-tile (m=0, k=1)
    #     byte 3 (mn_pack=1, k_pack=1) → MMA sub-tile (m=1, k=1)
    #   The MFMA opsel byte selector picks the right byte per sub-MMA — one i32
    #   lane-load amortizes across pack_K * num_*_mmas = 4 sub-MMAs per side.
    #
    # Coalescing: lanes 0..15 (k_lane=0) → [0..64), 16..31 → [64..128),
    # 32..47 → [128..192), 48..63 → [192..256). The wave's 64 simultaneous i32
    # loads cover exactly [0..256) = one K-block of scales = 4 MFMA tiles
    # (2 mn × 2 k pack) worth of scales, contiguous, no gaps. 256 bytes = 4
    # cache lines.
    #
    # Worked example — MN_padded=64, K_SCALES=16 (MN0_count=2, K0_count=2).
    # Each lane's i32 lane-load — coord passed and the 4-byte range it covers:
    #
    #   lane  0 (ml=0,  kl=0):  Coord(Idx(0), Idx(0),  Idx(0))   → bytes   0..3
    #   lane  1 (ml=1,  kl=0):  Coord(Idx(0), Idx(1),  Idx(0))   → bytes   4..7
    #   lane 15 (ml=15, kl=0):  Coord(Idx(0), Idx(15), Idx(0))   → bytes  60..63
    #   lane 16 (ml=0,  kl=1):  Coord(Idx(0), Idx(0),  Idx(1))   → bytes  64..67
    #   lane 32 (ml=0,  kl=2):  Coord(Idx(0), Idx(0),  Idx(2))   → bytes 128..131
    #   lane 48 (ml=0,  kl=3):  Coord(Idx(0), Idx(0),  Idx(3))   → bytes 192..195
    #   lane 63 (ml=15, kl=3):  Coord(Idx(0), Idx(15), Idx(3))   → bytes 252..255
    comptime scale_4d_grouped_layout[MN_padded: Int, K_SCALES: Int] = Layout(
        Coord(
            Idx[Self.E](),
            Coord(
                Idx[Self.MFMA_MN_LANES](),
                Idx[Self.S_MN_PACK](),
                Idx[MN_padded // Self.S_MN_BLOCK](),
            ),
            Coord(
                Idx[Self.MFMA_K_LANES](),
                Idx[Self.S_K_PACK](),
                Idx[K_SCALES // Self.S_K_BLOCK](),
            ),
        ),
        Coord(
            Idx[MN_padded * K_SCALES](),
            Coord(
                Idx[Self.S_STRIDE_MN_LANE](),
                Idx[Self.S_STRIDE_MN_PACK](),
                Idx[(K_SCALES // Self.S_K_BLOCK) * Self.S_STRIDE_K0](),
            ),
            Coord(
                Idx[Self.S_STRIDE_K_LANE](),
                Idx[Self.S_STRIDE_K_PACK](),
                Idx[Self.S_STRIDE_K0](),
            ),
        ),
    )

    # ---- Wrapped TileTensor types — what the preshuffle functions return ----
    comptime BTileTensor[N: Int, K_BYTES: Int] = TileTensor[
        mut=True,
        DType.uint8,
        type_of(Self.b_5d_grouped_layout[N=N, K_BYTES=K_BYTES]),
        MutAnyOrigin,
    ]

    comptime ScaleTileTensor[MN: Int, K_SCALES: Int] = TileTensor[
        mut=True,
        DType.uint8,
        type_of(
            Self.scale_4d_grouped_layout[
                MN_padded=Self.scale_padded_mn(MN),
                K_SCALES=K_SCALES,
            ]
        ),
        MutAnyOrigin,
    ]

    # ---- Helpers ----

    @staticmethod
    @always_inline
    def scale_padded_mn(MN: Int) -> Int:
        """Padded MN dim used by the 4D scale layout: MN rounded up to 32."""
        return ceildiv(MN, Self.S_MN_BLOCK) * Self.S_MN_BLOCK

    # ---- B (weight) preshuffle ----

    @staticmethod
    def preshuffle_b_5d[
        N: Int,
        K_BYTES: Int,
        SrcLayout: TensorLayout,
    ](
        src: TileTensor[DType.uint8, SrcLayout, MutAnyOrigin],
        mut dst: HostBuffer[DType.uint8],
    ) -> Self.BTileTensor[N, K_BYTES]:
        """Preshuffle B from `[E, N, K_BYTES]` row-major to the 5D byte layout.

        `src` is a 3D `(E, N, K_BYTES)` row-major tensor; `dst` is a flat
        host buffer of size `E*N*K_BYTES` bytes. Returns `dst` wrapped as
        a TileTensor with `Shuffler.b_5d_grouped_layout[E, N, K_BYTES]`,
        ready for `buffer_load_dwordx4` direct-to-VGPR reads on the kernel
        side.

        `K_BYTES` is the FP4-packed K dim (logical K // 2). N must be a
        multiple of 16, K_BYTES a multiple of 64.
        """
        comptime assert (
            N % Self.MFMA_MN_LANES == 0
        ), "preshuffle_b_5d: N must be a multiple of 16"
        comptime assert (
            K_BYTES % Self.MFMA_K_BYTES == 0
        ), "preshuffle_b_5d: K_BYTES must be a multiple of 64"

        var dst_tt = TileTensor(
            dst, Self.b_5d_grouped_layout[N=N, K_BYTES=K_BYTES]
        )
        for e in range(Self.E):
            for n in range(N):
                for k_byte in range(K_BYTES):
                    dst_tt[Coord(Idx(e), Idx(n), Idx(k_byte))] = src[
                        Coord(Idx(e), Idx(n), Idx(k_byte))
                    ]
        return dst_tt

    # ---- Scale preshuffle (works for both A and B scales; same layout) ----

    @staticmethod
    def preshuffle_scale_4d[
        MN: Int,
        K_SCALES: Int,
        SrcLayout: TensorLayout,
    ](
        src: TileTensor[DType.uint8, SrcLayout, MutAnyOrigin],
        mut dst: HostBuffer[DType.uint8],
    ) -> Self.ScaleTileTensor[MN, K_SCALES]:
        """Preshuffle E8M0 scales from `[E, MN, K_SCALES]` to the 4D layout.

        `src` is a 3D `(E, MN, K_SCALES)` row-major tensor; `dst` is a flat
        host buffer of size `E * Shuffler.scale_padded_mn(MN) * K_SCALES`
        bytes. Returns `dst` wrapped as a TileTensor with the 4D scale
        layout. `K_SCALES` is `K // 32` (one E8M0 byte per 32 FP4 elements).
        K_SCALES must be a multiple of 8. Per-group MN is padded up to 32;
        pad rows are zero-filled.
        """
        comptime assert (
            K_SCALES % Self.S_K_BLOCK == 0
        ), "preshuffle_scale_4d: K_SCALES must be a multiple of 8"

        comptime MN_padded = Self.scale_padded_mn(MN)
        comptime zero = SIMD[DType.uint8, 1](0)

        var dst_tt = TileTensor(
            dst,
            Self.scale_4d_grouped_layout[
                MN_padded=MN_padded, K_SCALES=K_SCALES
            ],
        )
        for e in range(Self.E):
            for mn in range(MN):
                for k_scale in range(K_SCALES):
                    dst_tt[Coord(Idx(e), Idx(mn), Idx(k_scale))] = src[
                        Coord(Idx(e), Idx(mn), Idx(k_scale))
                    ]
            for mn in range(MN, MN_padded):
                for k_scale in range(K_SCALES):
                    dst_tt[Coord(Idx(e), Idx(mn), Idx(k_scale))] = zero
        return dst_tt
