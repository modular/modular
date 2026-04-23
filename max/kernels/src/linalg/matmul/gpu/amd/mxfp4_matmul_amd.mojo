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
"""Native MXFP4 block-scaled matmul on AMD CDNA4 via f8f6f4 MFMA.

Computes C = (A * scale_a) @ (B * scale_b)^T where A and B are packed
MXFP4 (E2M1) in uint8 with per-block E8M0 scaling factors. Uses the
CDNA4 mfma.scale.f32.16x16x128.f8f6f4 instruction which natively
consumes MXFP4 operands with E8M0 scale words — no dequantization needed.

Structure mirrors AMDMatmul: TileTensor throughout, RegTileLoader for
DRAM→regs, row-major SMEM (no blocked-product or swizzle — the FP4
MFMA expects a simple row-major lane-to-data mapping unlike BF16/FP8),
schedule-driven pipeline.

MXFP4 data layout:
  A: [M, K//2] uint8 (two MXFP4 nibbles packed per byte), row-major
  B: [N, K//2] uint8, row-major (transposed: each row is one output column)
  scale_a: [M, K//32] float8_e8m0fnu (one scale per 32 MXFP4 elements)
  scale_b: [N, K//32] float8_e8m0fnu

MFMA lane-to-data mapping for 16x16x128 FP4:
  Each lane loads 16 contiguous bytes from its assigned matrix row.
  lane_row = lane_id % MMA_M, lane_chunk = lane_id / MMA_M.
  Offset = lane_row * row_stride + lane_chunk * 16.
  The 16 bytes are zero-extended to SIMD[uint8, 32] for the MFMA operand.

MFMA scale model (16x16x128):
  Each lane holds 16x128/64 = 32 FP4 elements and one E8M0 scale.
  This matches the MX format exactly: one scale per 32 elements.
  The 64 scale values (16 rows x 4 K-groups = 64) come from 64
  lanes, each contributing one byte.

  Lane mapping: lane_row = lane % 16 (matrix row), lane_k_group =
  lane / 16 (which 32-element K-group within the row, 0..3).
  Each lane loads scale_ptr[row * stride + base_k + lane_k_group].

  The scale byte is placed in byte 0 of an Int32 word passed to
  the MFMA intrinsic (byte_index=0 / OPSEL=0).

Entry point: mxfp4_block_scaled_matmul_amd()
"""

from std.math import ceildiv
from std.memory import bitcast
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
from std.gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
)
from layout import TensorLayout, TileTensor
from layout.tile_layout import row_major, col_major
from layout.tile_tensor import stack_allocation

from std.utils import IndexList, StaticTuple
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)
from structured_kernels.amd_tile_io import RegTileWriter
from .amd_matmul_schedule import build_default_matmul_schedule
from .amd_matmul_schedule import (
    DefaultMatmulOps,
    COMPUTE,
    LOAD_DRAM,
    LOAD_FRAG,
    STORE_SMEM,
)
from pipeline.pipeline_dsl import ScheduleEntry

# MXFP4: 32 MXFP4 elements per E8M0 scale.
comptime MX_BLOCK_SIZE = 32

comptime SCHED_MASK_DS_READ = 0
comptime SCHED_MASK_DS_WRITE = 1
comptime SCHED_MASK_VMEM_READ = 2
comptime SCHED_MASK_MFMA = 3


# ===----------------------------------------------------------------------=== #
# BlockScaledMmaOp — MFMA with inline scale application
# ===----------------------------------------------------------------------=== #


struct BlockScaledMmaOp[
    mma_shape: IndexList[3],  # (16, 16, 128) for MXFP4
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_k_tiles: Int,
]:
    """Register ownership + block-scaled MFMA execution.

    Loads packed uint8 A/B fragments from SMEM or GMEM and executes
    cdna4_block_scaled_mfma with per-lane E8M0 scale values.

    Scale operand model:
      Each lane holds 32 FP4 elements and one E8M0 scale byte,
      matching the MX format's per-32-element granularity exactly.
      For 16x16x128: 64 lanes cover 16 rows x 4 K-groups.
        lane_row = lane_id % 16   (matrix row)
        lane_k_group = lane_id / 16  (K-group 0..3)

      Scale packing: 4 spatial MMA tiles' scale bytes are packed into
      one Int32 VGPR — byte i holds the scale for m_mma=i (A) or
      n_mma=i (B). The MFMA byte-index selector (OP_SEL) picks the
      correct byte for each MMA tile, so one scale load covers all
      4 m_mma or n_mma positions with zero overhead.
    """

    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]

    comptime packed_k_per_mma = Self.MMA_K // 2  # bytes consumed per MFMA
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE  # 4

    # Per-lane data width: 16 bytes of FP4 data, zero-extended to 32
    # for the MFMA operand VGPR.
    comptime mma_frag_width: Int = 16

    # Scales: 4 E8M0 bytes per MFMA call (128 MXFP4 / 32 per scale = 4).
    comptime scales_per_mma = Self.MMA_K // MX_BLOCK_SIZE  # 4

    comptime _a_reg_layout = row_major[
        Self.num_m_mmas * Self.num_k_tiles,
        Self.mma_frag_width,
    ]()
    comptime _b_reg_layout = row_major[
        Self.num_n_mmas * Self.num_k_tiles,
        Self.mma_frag_width,
    ]()
    comptime _c_reg_layout = row_major[
        Self.num_m_mmas,
        Self.num_n_mmas * Self.c_frag_size,
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

    # Packed scale VGPRs: one Int32 for A, one for B. Byte i holds the
    # scale for spatial MMA tile i (m_mma for A, n_mma for B).
    var _a_scale_packed: Int32
    var _b_scale_packed: Int32

    @always_inline
    def __init__(out self):
        comptime assert (
            Self.num_m_mmas <= 4
        ), "num_m_mmas must be <= 4 for packed scales"
        comptime assert (
            Self.num_n_mmas <= 4
        ), "num_n_mmas must be <= 4 for packed scales"
        self._a_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            Self._a_reg_layout
        )
        self._b_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            Self._b_reg_layout
        )
        self._c_reg = stack_allocation[DType.float32, AddressSpace.LOCAL](
            Self._c_reg_layout
        )
        comptime num_c = Self.num_m_mmas * Self.num_n_mmas * Self.c_frag_size
        comptime for i in range(num_c):
            self._c_reg.raw_store(i, Scalar[DType.float32](0))
        self._a_scale_packed = Int32(0)
        self._b_scale_packed = Int32(0)

    @always_inline
    def accum_tile(self) -> ref[self._c_reg] type_of(self._c_reg):
        return self._c_reg

    @always_inline
    def load_frag_from_smem[
        k_tile_idx: Int
    ](
        self,
        a_smem_warp: TileTensor[
            DType.uint8, _, _, address_space=AddressSpace.SHARED, ...
        ],
        b_smem_warp: TileTensor[
            DType.uint8, _, _, address_space=AddressSpace.SHARED, ...
        ],
    ):
        """Load MXFP4 A/B fragments from row-major SMEM for k-tile k_tile_idx.

        Uses tile to extract the [MMA_M, packed_k_per_mma] sub-tile,
        vectorize groups 64 bytes into 4 x 16-byte elements, and
        distribute with col_major[MMA_M, 4] assigns each lane its
        16-byte fragment matching the MFMA native lane mapping.
        """
        comptime frag_w = Self.mma_frag_width  # 16
        comptime mma_k_bytes = Self.packed_k_per_mma  # 64
        comptime lane_layout = col_major[Self.MMA_M, WARP_SIZE // Self.MMA_M]()

        # B fragments first (for ds_read scheduling).
        comptime for i in range(Self.num_n_mmas):
            var b_idx = k_tile_idx * Self.num_n_mmas + i
            var b_frag = (
                b_smem_warp.tile[Self.MMA_N, mma_k_bytes](i, k_tile_idx)
                .vectorize[1, frag_w]()
                .distribute[lane_layout](lane_id())
            )
            self._b_reg.vectorize[1, frag_w]()[b_idx, 0] = b_frag[0, 0]

        # A fragments.
        comptime for i in range(Self.num_m_mmas):
            var a_idx = k_tile_idx * Self.num_m_mmas + i
            var a_frag = (
                a_smem_warp.tile[Self.MMA_M, mma_k_bytes](i, k_tile_idx)
                .vectorize[1, frag_w]()
                .distribute[lane_layout](lane_id())
            )
            self._a_reg.vectorize[1, frag_w]()[a_idx, 0] = a_frag[0, 0]

    @always_inline
    def load_scales_from_smem(
        mut self,
        a_scale_smem_warp: TileTensor[
            DType.uint8, address_space=AddressSpace.SHARED, ...
        ],
        b_scale_smem_warp: TileTensor[
            DType.uint8, address_space=AddressSpace.SHARED, ...
        ],
    ):
        """Load packed scale VGPRs from SMEM.

        Packs num_m_mmas (A) or num_n_mmas (B) scale bytes into one
        Int32 each. Byte i holds the scale for spatial MMA tile i.

        The scale SMEM tile is [WM/WN, scales_per_mma] uint8 row-major.
        For each spatial tile i, this lane's scale byte is at:
          row = i * MMA_M + lane_row, col = lane_k_group
        We read one byte per tile and pack them into bytes 0..3.

        The MFMA byte-index selector (a_scale_byte_index=m_mma,
        b_scale_byte_index=n_mma) picks the correct byte — no shifts
        or masks at consumption time.
        """
        var lane = Int(lane_id())
        var lane_row = lane % Self.MMA_M
        var lane_k_group = lane // Self.MMA_M

        var a_packed = Int32(0)
        comptime for m_mma in range(Self.num_m_mmas):
            var smem_off = (
                m_mma * Self.MMA_M + lane_row
            ) * Self.scales_per_mma + lane_k_group
            var byte_val = UInt32(a_scale_smem_warp.ptr[smem_off])
            a_packed = a_packed | rebind[Scalar[DType.int32]](
                byte_val << UInt32(m_mma * 8)
            )
        self._a_scale_packed = a_packed

        var b_packed = Int32(0)
        comptime for n_mma in range(Self.num_n_mmas):
            var smem_off = (
                n_mma * Self.MMA_N + lane_row
            ) * Self.scales_per_mma + lane_k_group
            var byte_val = UInt32(b_scale_smem_warp.ptr[smem_off])
            b_packed = b_packed | rebind[Scalar[DType.int32]](
                byte_val << UInt32(n_mma * 8)
            )
        self._b_scale_packed = b_packed

    @always_inline
    def mma[k_tile_idx: Int](self):
        """Execute block-scaled MFMA for k-tile k_tile_idx.

        B→src_a, A→src_b (AMD MFMA convention).
        The packed scale VGPRs hold one byte per spatial MMA tile.
        a_scale_byte_index=m selects byte m from _a_scale_packed,
        b_scale_byte_index=n selects byte n from _b_scale_packed.
        """
        comptime for m in range(Self.num_m_mmas):
            comptime for n in range(Self.num_n_mmas):
                var a_row = k_tile_idx * Self.num_m_mmas + m
                var b_row = k_tile_idx * Self.num_n_mmas + n
                var c_off = (
                    m * Self.num_n_mmas * Self.c_frag_size
                    + n * Self.c_frag_size
                )

                var a_data = self._a_reg.raw_load[width=Self.mma_frag_width](
                    a_row * Self.mma_frag_width
                )
                var b_data = self._b_reg.raw_load[width=Self.mma_frag_width](
                    b_row * Self.mma_frag_width
                )

                var a_frag = SIMD[DType.uint8, 32](0)
                var b_frag = SIMD[DType.uint8, 32](0)
                a_frag = a_frag.insert[offset=0](a_data)
                b_frag = b_frag.insert[offset=0](b_data)

                var c_frag = self._c_reg.raw_load[width=Self.c_frag_size](c_off)

                cdna4_block_scaled_mfma[
                    Int32(n),
                    Int32(m),
                    CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                    CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                ](
                    c_frag,
                    b_frag,
                    a_frag,
                    self._b_scale_packed,
                    self._a_scale_packed,
                )

                self._c_reg.raw_store[width=Self.c_frag_size](c_off, c_frag)


# ===----------------------------------------------------------------------=== #
# MXFP4MatmulAMD — kernel struct
# ===----------------------------------------------------------------------=== #

comptime MXFP4_MMA_M = 16
comptime MXFP4_MMA_N = 16
comptime MXFP4_MMA_K = 128

comptime MXFP4_BM = 128
comptime MXFP4_BN = 128
comptime MXFP4_BK_ELEMS = 128
comptime MXFP4_BK_BYTES = MXFP4_BK_ELEMS // 2  # 64 packed bytes

comptime MXFP4_WM = 64
comptime MXFP4_WN = 64

comptime MXFP4_NUM_WARPS_M = MXFP4_BM // MXFP4_WM  # 2
comptime MXFP4_NUM_WARPS_N = MXFP4_BN // MXFP4_WN  # 2
comptime MXFP4_NUM_WARPS = MXFP4_NUM_WARPS_M * MXFP4_NUM_WARPS_N  # 4
comptime MXFP4_NUM_THREADS = MXFP4_NUM_WARPS * WARP_SIZE  # 256

comptime MXFP4_NUM_M_MMAS = MXFP4_WM // MXFP4_MMA_M  # 4
comptime MXFP4_NUM_N_MMAS = MXFP4_WN // MXFP4_MMA_N  # 4


struct MXFP4MatmulAMD:
    """Native MXFP4 block-scaled matmul for AMD CDNA4.

    Uses cdna4_block_scaled_mfma with FLOAT4_E2M1 format directly.
    Single-buffer pipeline with schedule-driven prologue/kernel/epilogue.
    SMEM is plain row-major (no blocked-product, no swizzle).
    """

    comptime BM = MXFP4_BM
    comptime BN = MXFP4_BN
    comptime BK_BYTES = MXFP4_BK_BYTES
    comptime BK_ELEMS = MXFP4_BK_ELEMS

    comptime WM = MXFP4_WM
    comptime WN = MXFP4_WN

    comptime MMA_M = MXFP4_MMA_M
    comptime MMA_N = MXFP4_MMA_N
    comptime MMA_K = MXFP4_MMA_K

    comptime num_warps_m = MXFP4_NUM_WARPS_M
    comptime num_warps_n = MXFP4_NUM_WARPS_N
    comptime num_threads = MXFP4_NUM_THREADS

    comptime num_m_mmas = MXFP4_NUM_M_MMAS
    comptime num_n_mmas = MXFP4_NUM_N_MMAS

    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE  # 4
    comptime packed_k_per_mma = Self.MMA_K // 2  # 64 bytes per MFMA
    comptime num_k_tiles: Int = 1  # one MFMA per BK tile

    # DRAM→regs loading constants.
    comptime simd_width = simd_width_of[DType.uint8]()  # 16
    comptime k_tile_size = Self.BK_BYTES  # 64 bytes

    # Scale tile: [BM, scales_per_mma] uint8 per BK iteration.
    # scales_per_mma = MMA_K / 32 = 4 bytes per row.
    comptime scales_per_mma = Self.MMA_K // MX_BLOCK_SIZE  # 4

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(MXFP4_NUM_THREADS)
        )
    )
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        sfa_layout: TensorLayout,
        sfb_layout: TensorLayout,
    ](
        c: TileTensor[DType.float32, c_layout, MutAnyOrigin],
        a: TileTensor[DType.uint8, a_layout, ImmutAnyOrigin],
        b: TileTensor[DType.uint8, b_layout, ImmutAnyOrigin],
        sfa: TileTensor[DType.float8_e8m0fnu, sfa_layout, ImmutAnyOrigin],
        sfb: TileTensor[DType.float8_e8m0fnu, sfb_layout, ImmutAnyOrigin],
    ):
        """MXFP4 block-scaled GEMM kernel with SMEM pipeline."""
        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK_BYTES = Self.BK_BYTES
        comptime WM = Self.WM
        comptime WN = Self.WN
        comptime MMA_M = Self.MMA_M
        comptime MMA_N = Self.MMA_N
        comptime num_m_mmas = Self.num_m_mmas
        comptime num_n_mmas = Self.num_n_mmas
        comptime c_frag_size = Self.c_frag_size
        comptime num_k_tiles = Self.num_k_tiles
        comptime simd_width = Self.simd_width
        comptime num_threads = Self.num_threads

        comptime K_BYTES = type_of(a).static_shape[1]  # K//2
        comptime N = type_of(b).static_shape[0]
        comptime assert N > 0, "N must be known at compile time"
        comptime assert K_BYTES > 0, "K (packed) must be known at compile time"

        comptime K_SCALES = type_of(sfa).static_shape[1]  # K//32

        var _warp_id = warp_id()
        var warp_m, warp_n = divmod(_warp_id, Self.num_warps_n)

        # === GMEM views ===
        var a_gmem = TileTensor(a.ptr.bitcast[Scalar[DType.uint8]](), a.layout)
        var b_gmem = TileTensor(b.ptr.bitcast[Scalar[DType.uint8]](), b.layout)

        # === SMEM tiles (row-major, no swizzle) ===
        var a_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[BM, BK_BYTES]()
        )
        var b_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[BN, BK_BYTES]()
        )

        # Scale SMEM: [BM, scales_per_mma] and [BN, scales_per_mma] uint8.
        # Each row's 4 scale bytes will be read as one coalesced Int32.
        comptime scales_per_mma = Self.scales_per_mma
        var sfa_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[BM, scales_per_mma]()
        )
        var sfb_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[BN, scales_per_mma]()
        )

        # === DRAM→regs→SMEM loading ===
        # Two-phase: LOAD_DRAM loads to register buffers, STORE_SMEM
        # copies registers to SMEM. This keeps the schedule's barrier
        # placement correct (barrier between STORE_SMEM and LOAD_FRAG).
        #
        # Thread distribution: row_major[load_rows, load_cols] maps each
        # thread to a (row, col) position. Each thread loads loads_per_tile
        # vector-width chunks, covering the full [BM, BK_BYTES] tile.
        comptime load_thread_cols = BK_BYTES // simd_width
        comptime load_thread_rows = num_threads // load_thread_cols
        comptime load_layout = row_major[load_thread_rows, load_thread_cols]()
        comptime loads_per_tile = BM // load_thread_rows
        comptime reg_elems = BM * BK_BYTES // num_threads

        # Block-row tiles spanning the full K dimension for tile-based indexing.
        var a_blockrow = a_gmem.tile[BM, K_BYTES](block_idx.y, 0)
        var b_blockrow = b_gmem.tile[BN, K_BYTES](block_idx.x, 0)

        # Register buffers for DRAM loads (one per matrix).
        var a_load_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            row_major[1, reg_elems]()
        )
        var b_load_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            row_major[1, reg_elems]()
        )

        # === MMA operator ===
        var mma_op = BlockScaledMmaOp[
            mma_shape=IndexList[3](MMA_M, MMA_N, Self.MMA_K),
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            num_k_tiles=num_k_tiles,
        ]()

        # === Output writer ===
        var c_writer = RegTileWriter[DType.float32, MMA_M, WARP_SIZE // MMA_M](
            c
        )

        # === Pipeline helpers ===
        var k_counter = 0
        var k_scale_counter = 0

        @always_inline
        @parameter
        def load_tiles_from_dram():
            """Load one BK-wide tile from DRAM to register buffers."""
            var a_block = a_blockrow.tile[BM, BK_BYTES](0, k_counter)
            var b_block = b_blockrow.tile[BN, BK_BYTES](0, k_counter)
            var a_dist = a_block.vectorize[1, simd_width]().distribute[
                load_layout
            ](thread_idx.x)
            var b_dist = b_block.vectorize[1, simd_width]().distribute[
                load_layout
            ](thread_idx.x)
            comptime for v in range(loads_per_tile):
                a_load_reg.raw_store[width=simd_width](
                    v * simd_width, a_dist[v, 0]
                )
                b_load_reg.raw_store[width=simd_width](
                    v * simd_width, b_dist[v, 0]
                )
            k_counter += 1

        @always_inline
        @parameter
        def copy_tiles_to_smem():
            """Copy register buffers to SMEM in row-major order."""
            var a_smem_dist = a_smem.vectorize[1, simd_width]().distribute[
                load_layout
            ](thread_idx.x)
            var b_smem_dist = b_smem.vectorize[1, simd_width]().distribute[
                load_layout
            ](thread_idx.x)
            comptime for v in range(loads_per_tile):
                a_smem_dist[v, 0] = a_load_reg.raw_load[width=simd_width](
                    v * simd_width
                )
                b_smem_dist[v, 0] = b_load_reg.raw_load[width=simd_width](
                    v * simd_width
                )

        @always_inline
        @parameter
        def load_scales_to_smem():
            """Cooperatively load scale tiles from GMEM to SMEM.

            Scale tile per BK iteration: [BM, scales_per_mma] for A and
            [BN, scales_per_mma] for B, both uint8. Each row is 4 bytes.
            Threads 0..BM-1 load A scales, threads BM..BM+BN-1 load B.
            Each active thread loads one 4-byte row as an Int32, giving
            coalesced 4-byte aligned GMEM reads.
            """
            var tid = Int(thread_idx.x)
            var base_scale_k = k_scale_counter * scales_per_mma
            var a_base_row = Int(block_idx.y) * BM
            var b_base_row = Int(block_idx.x) * BN

            if tid < BM:
                var row = a_base_row + tid
                var src_off = row * K_SCALES + base_scale_k
                var src_word = sfa.ptr.bitcast[Scalar[DType.int32]]()[
                    src_off // scales_per_mma
                ]
                sfa_smem.ptr.bitcast[Scalar[DType.int32]]()[tid] = src_word
            if tid < BN:
                var row = b_base_row + tid
                var src_off = row * K_SCALES + base_scale_k
                var src_word = sfb.ptr.bitcast[Scalar[DType.int32]]()[
                    src_off // scales_per_mma
                ]
                sfb_smem.ptr.bitcast[Scalar[DType.int32]]()[tid] = src_word

            k_scale_counter += 1

        # === Schedule-driven pipeline ===
        # The schedule prologue pre-loads 2 tiles, so we need at least 2
        # K-iterations. For K with only 1 tile, fall back to a simple loop.
        comptime a_loads_per_thread = BM // load_thread_rows
        comptime b_loads_per_thread = BN // load_thread_rows

        @always_inline
        @parameter
        def simple_k_loop():
            """Fallback for small K where schedule prologue doesn't fit."""
            for k_iter in range(K_BYTES // BK_BYTES):
                load_tiles_from_dram()
                load_scales_to_smem()
                copy_tiles_to_smem()
                barrier()

                var a_warp = a_smem.tile[WM, BK_BYTES](warp_m, 0)
                var b_warp = b_smem.tile[WN, BK_BYTES](warp_n, 0)
                mma_op.load_frag_from_smem[0](a_warp, b_warp)

                var sfa_warp = sfa_smem.tile[WM, scales_per_mma](warp_m, 0)
                var sfb_warp = sfb_smem.tile[WN, scales_per_mma](warp_n, 0)
                mma_op.load_scales_from_smem(sfa_warp, sfb_warp)

                mma_op.mma[0]()
                barrier()

        @always_inline
        @parameter
        def scheduled_k_loop():
            """Pipelined K-loop via build_default_matmul_schedule."""
            comptime schedule = build_default_matmul_schedule[
                num_k_tiles=num_k_tiles,
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                num_k_mmas=1,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                a_loads_per_thread=a_loads_per_thread,
                b_loads_per_thread=b_loads_per_thread,
            ]()

            @parameter
            @always_inline
            def _bind[entry: ScheduleEntry]():
                comptime if entry.op.tag == LOAD_DRAM:
                    load_tiles_from_dram()
                elif entry.op.tag == STORE_SMEM:
                    copy_tiles_to_smem()
                    load_scales_to_smem()
                elif entry.op.tag == LOAD_FRAG:
                    var a_warp = a_smem.tile[WM, BK_BYTES](warp_m, 0)
                    var b_warp = b_smem.tile[WN, BK_BYTES](warp_n, 0)
                    mma_op.load_frag_from_smem[entry.op.subtile](a_warp, b_warp)
                    var sfa_warp = sfa_smem.tile[WM, scales_per_mma](warp_m, 0)
                    var sfb_warp = sfb_smem.tile[WN, scales_per_mma](warp_n, 0)
                    mma_op.load_scales_from_smem(sfa_warp, sfb_warp)
                elif entry.op.tag == COMPUTE:
                    mma_op.mma[entry.op.subtile]()
                elif entry.op.tag == DefaultMatmulOps.BARRIER.value:
                    barrier()
                elif entry.op.tag == DefaultMatmulOps.SCHEDULE_BARRIER.value:
                    schedule_barrier()
                elif entry.op.tag == (
                    DefaultMatmulOps.SCHED_GROUP_BARRIER.value
                ):
                    comptime sub = entry.op.subtile
                    comptime wait = entry.op.wait_value
                    comptime if sub == SCHED_MASK_DS_READ:
                        schedule_group_barrier(
                            AMDScheduleBarrierMask.DS_READ, Int32(wait), 0
                        )
                    elif sub == SCHED_MASK_DS_WRITE:
                        schedule_group_barrier(
                            AMDScheduleBarrierMask.DS_WRITE, Int32(wait), 0
                        )
                    elif sub == SCHED_MASK_VMEM_READ:
                        schedule_group_barrier(
                            AMDScheduleBarrierMask.VMEM_READ, Int32(wait), 0
                        )
                    elif sub == SCHED_MASK_MFMA:
                        schedule_group_barrier(
                            AMDScheduleBarrierMask.MFMA, Int32(wait), 0
                        )

            # Prologue.
            comptime for i in range(len(schedule.prologue)):
                _bind[schedule.prologue[i]]()

            # Main K-loop.
            for _ in range(2, K_BYTES // BK_BYTES):
                comptime for i in range(len(schedule.kernel)):
                    _bind[schedule.kernel[i]]()

            # Epilogue.
            comptime for i in range(len(schedule.epilogue)):
                _bind[schedule.epilogue[i]]()

        if K_BYTES // BK_BYTES < 2:
            simple_k_loop()
        else:
            scheduled_k_loop()

        # === Output store ===
        var c_reg = mma_op.accum_tile()
        var c_block = c.tile[BM, BN](block_idx.y, block_idx.x)
        var c_warp = c_block.tile[WM, WN](warp_m, warp_n)

        comptime for m_mma in range(num_m_mmas):
            comptime for n_mma in range(num_n_mmas):
                c_writer.store(
                    c_warp.tile[MMA_M, MMA_N](m_mma, n_mma).vectorize[
                        1, c_frag_size
                    ](),
                    c_reg.tile[1, c_frag_size](m_mma, n_mma),
                )


# ===----------------------------------------------------------------------=== #
# Public entry point
# ===----------------------------------------------------------------------=== #


def mxfp4_block_scaled_matmul_amd(
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    ctx: DeviceContext,
) raises:
    """Launch native MXFP4 block-scaled matmul on AMD CDNA4.

    Uses cdna4_block_scaled_mfma with FLOAT4_E2M1 directly — no
    dequantization to FP8. Both A and B must be packed uint8 with
    E8M0 scaling factors.

    Args:
        c: Output [M, N] float32.
        a: Packed A [M, K//2] uint8 (two MXFP4 elements per byte).
        b: Packed B [N, K//2] uint8 (transposed, two MXFP4 per byte).
        a_scales: A scales [M, K//32] float8_e8m0fnu.
        b_scales: B scales [N, K//32] float8_e8m0fnu.
        ctx: Device context for kernel launch.
    """
    comptime assert c.dtype == DType.float32, "output must be float32"
    comptime assert a.dtype == DType.uint8, "A must be uint8 (packed MXFP4)"
    comptime assert b.dtype == DType.uint8, "B must be uint8 (packed MXFP4)"
    comptime assert (
        a_scales.dtype == DType.float8_e8m0fnu
    ), "A scales must be float8_e8m0fnu"
    comptime assert (
        b_scales.dtype == DType.float8_e8m0fnu
    ), "B scales must be float8_e8m0fnu"

    var M = Int(c.dim[0]())
    comptime N = type_of(c).static_shape[1]

    comptime BM = MXFP4MatmulAMD.BM
    comptime BN = MXFP4MatmulAMD.BN

    comptime kernel = MXFP4MatmulAMD.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scales).LayoutType,
        type_of(b_scales).LayoutType,
    ]

    ctx.enqueue_function[kernel, kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=MXFP4MatmulAMD.num_threads,
    )
