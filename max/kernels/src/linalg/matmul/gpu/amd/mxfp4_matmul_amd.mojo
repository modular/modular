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
from structured_kernels.amd_tile_io import RegTileLoader, RegTileWriter
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

    # NOTE this is hardcoded to work with 16x16x128 matmul where
    # where 16 threads operate on a 1x32 tile (total 16x32). Each
    # thread owns one row, that row is 16 bytes.
    comptime mma_frag_width_bytes: Int = 16

    # Scales: 4 E8M0 bytes per MFMA call (128 MXFP4 / 32 per scale = 4).
    comptime scales_per_mma = Self.MMA_K // MX_BLOCK_SIZE  # 4

    comptime _a_reg_layout = row_major[
        Self.num_m_mmas * Self.num_k_tiles,
        Self.mma_frag_width_bytes,
    ]()
    comptime _b_reg_layout = row_major[
        Self.num_n_mmas * Self.num_k_tiles,
        Self.mma_frag_width_bytes,
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

    # Packed scale VGPRs: one Int32 per k_tile for A and B. Byte i of
    # _a_scale_packed[k] holds the scale for spatial A tile m_mma=i of
    # k-tile k. Separate slots per k_tile so that schedules which
    # interleave LOAD_FRAG / COMPUTE across k_tiles don't clobber each
    # other.
    comptime _scale_layout = row_major[1, Self.num_k_tiles]()
    var _a_scale_packed: TileTensor[
        DType.int32,
        type_of(Self._scale_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _b_scale_packed: TileTensor[
        DType.int32,
        type_of(Self._scale_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

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
        _ = self._c_reg.fill(Scalar[DType.float32](0))
        self._a_scale_packed = stack_allocation[
            DType.int32, AddressSpace.LOCAL
        ](Self._scale_layout)
        self._b_scale_packed = stack_allocation[
            DType.int32, AddressSpace.LOCAL
        ](Self._scale_layout)
        _ = self._a_scale_packed.fill(Scalar[DType.int32](0))
        _ = self._b_scale_packed.fill(Scalar[DType.int32](0))

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
        comptime frag_w = Self.mma_frag_width_bytes  # 16
        comptime mma_k_bytes = Self.packed_k_per_mma  # 64

        # groups of 16 threads are responsible for 16 rows, i.e threads 0, 16, 32, 48 handle row 0 ...
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
    def load_scales_from_smem[
        k_tile_idx: Int
    ](
        mut self,
        a_scale_smem_warp: TileTensor[
            DType.uint8, address_space=AddressSpace.SHARED, ...
        ],
        b_scale_smem_warp: TileTensor[
            DType.uint8, address_space=AddressSpace.SHARED, ...
        ],
    ):
        """Load packed scale VGPRs for k-tile k_tile_idx from SMEM.

        Packs num_m_mmas (A) or num_n_mmas (B) scale bytes into one
        Int32 each using the same col_major[MMA_M, WARP_SIZE/MMA_M]
        distribute pattern as load_frag_from_smem. Each lane picks one
        scale byte via (lane_row, lane_k_group). TileTensor's stride
        handling means this works for any parent SMEM layout.

        The MFMA byte-index selector (a_scale_byte_index=m_mma,
        b_scale_byte_index=n_mma) picks the correct byte — no shifts
        or masks at consumption time.
        """
        var a_packed = Int32(0)
        comptime for m_mma in range(Self.num_m_mmas):
            var byte_val = UInt32(
                a_scale_smem_warp.tile[Self.MMA_M, Self.scales_per_mma](
                    m_mma, 0
                ).distribute[
                    col_major[Self.MMA_M, WARP_SIZE // Self.MMA_M](),
                ](
                    lane_id()
                )[
                    0, 0
                ][
                    0
                ]
            )
            a_packed = a_packed | rebind[Scalar[DType.int32]](
                byte_val << UInt32(m_mma * 8)
            )
        self._a_scale_packed.raw_store(k_tile_idx, a_packed)

        var b_packed = Int32(0)
        comptime for n_mma in range(Self.num_n_mmas):
            var byte_val = UInt32(
                b_scale_smem_warp.tile[Self.MMA_N, Self.scales_per_mma](
                    n_mma, 0
                ).distribute[
                    col_major[Self.MMA_N, WARP_SIZE // Self.MMA_N](),
                ](
                    lane_id()
                )[
                    0, 0
                ][
                    0
                ]
            )
            b_packed = b_packed | rebind[Scalar[DType.int32]](
                byte_val << UInt32(n_mma * 8)
            )
        self._b_scale_packed.raw_store(k_tile_idx, b_packed)

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
                comptime a_row = k_tile_idx * Self.num_m_mmas + m
                comptime b_row = k_tile_idx * Self.num_n_mmas + n
                comptime a_off = a_row * Self.mma_frag_width_bytes
                comptime b_off = b_row * Self.mma_frag_width_bytes
                comptime c_off = (
                    m * Self.num_n_mmas * Self.c_frag_size
                    + n * Self.c_frag_size
                )

                var a_data = self._a_reg.raw_load[
                    width=Self.mma_frag_width_bytes
                ](a_off)
                var b_data = self._b_reg.raw_load[
                    width=Self.mma_frag_width_bytes
                ](b_off)

                var a_frag = SIMD[DType.uint8, 32](0)
                var b_frag = SIMD[DType.uint8, 32](0)
                a_frag = a_frag.insert[offset=0](a_data)
                b_frag = b_frag.insert[offset=0](b_data)

                var c_frag = self._c_reg.raw_load[width=Self.c_frag_size](c_off)

                var a_scale = rebind[Int32](
                    self._a_scale_packed.raw_load(k_tile_idx)
                )
                var b_scale = rebind[Int32](
                    self._b_scale_packed.raw_load(k_tile_idx)
                )
                cdna4_block_scaled_mfma[
                    Int32(n),
                    Int32(m),
                    CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                    CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                ](
                    c_frag,
                    b_frag,
                    a_frag,
                    b_scale,
                    a_scale,
                )

                self._c_reg.raw_store[width=Self.c_frag_size](c_off, c_frag)


# ===----------------------------------------------------------------------=== #
# MXFP4MatmulAMD — kernel struct
# ===----------------------------------------------------------------------=== #

# TODO hardcoded support multiple MMAs in the future
comptime MXFP4_MMA_M = 16
comptime MXFP4_MMA_N = 16
comptime MXFP4_MMA_K = 128


struct MXFP4MatmulAMD[
    BM: Int = 128,
    BN: Int = 128,
    BK_ELEMS: Int = 128,
    WM: Int = 64,
    WN: Int = 64,
]:
    """Native MXFP4 block-scaled matmul for AMD CDNA4.

    Uses cdna4_block_scaled_mfma with FLOAT4_E2M1 format directly.
    Single-buffer pipeline with schedule-driven prologue/kernel/epilogue.
    SMEM is plain row-major (no blocked-product, no swizzle).

    Parameters:
        BM: Block tile rows (output M per block). Default 128.
        BN: Block tile cols (output N per block). Default 128.
        BK_ELEMS: Block tile K in logical FP4 elements. Default 128.
        WM: Warp tile rows. BM must be divisible by WM. Default 64.
        WN: Warp tile cols. BN must be divisible by WN. Default 64.
    """

    comptime BK_BYTES = Self.BK_ELEMS // 2

    comptime MMA_M = MXFP4_MMA_M
    comptime MMA_N = MXFP4_MMA_N
    comptime MMA_K = MXFP4_MMA_K

    comptime num_warps_m = Self.BM // Self.WM
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_warps = Self.num_warps_m * Self.num_warps_n
    comptime num_threads = Self.num_warps * WARP_SIZE

    comptime num_m_mmas = Self.WM // Self.MMA_M
    comptime num_n_mmas = Self.WN // Self.MMA_N

    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE  # 4
    comptime packed_k_per_mma = Self.MMA_K // 2  # 64 bytes per MFMA
    comptime num_k_tiles = Self.BK_BYTES // Self.packed_k_per_mma

    # DRAM→regs loading constants.
    comptime simd_width = simd_width_of[DType.uint8]()  # 16
    comptime k_tile_size = Self.BK_BYTES

    # Scale tile: [BM, scales_per_mma] uint8 per BK iteration.
    # scales_per_mma = MMA_K / 32 = 4 bytes per row.
    comptime scales_per_mma = Self.MMA_K // MX_BLOCK_SIZE  # 4

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
        b_layout: TensorLayout,
        sfa_layout: TensorLayout,
        sfb_layout: TensorLayout,
    ](
        c: TileTensor[out_dtype, c_layout, MutAnyOrigin],
        a: TileTensor[DType.uint8, a_layout, ImmutAnyOrigin],
        b: TileTensor[DType.uint8, b_layout, ImmutAnyOrigin],
        sfa: TileTensor[DType.float8_e8m0fnu, sfa_layout, ImmutAnyOrigin],
        sfb: TileTensor[DType.float8_e8m0fnu, sfb_layout, ImmutAnyOrigin],
    ):
        """MXFP4 block-scaled GEMM kernel with SMEM pipeline."""
        comptime BK_BYTES = Self.BK_BYTES
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
        # K must divide evenly into BK_BYTES — the main K-loop uses
        # integer division `K_BYTES // BK_BYTES`, so any remainder is
        # silently dropped (trailing chunk of K never computed).
        comptime assert K_BYTES % BK_BYTES == 0, (
            "K (packed bytes) must be a multiple of BK_BYTES; otherwise"
            " the trailing K chunk is silently skipped. Either pick a BK_ELEMS"
            " that divides K, or pad K."
        )

        comptime K_SCALES = type_of(sfa).static_shape[1]  # K//32

        # Dynamic M for OOB bounds handling when M is not a multiple of Self.BM.
        var M = Int(a.dim[0]())

        var _warp_id = warp_id()
        var warp_m, warp_n = divmod(_warp_id, Self.num_warps_n)

        # === GMEM views ===
        var a_gmem = TileTensor(a.ptr.bitcast[Scalar[DType.uint8]](), a.layout)
        var b_gmem = TileTensor(b.ptr.bitcast[Scalar[DType.uint8]](), b.layout)

        # === SMEM tiles (row-major, no swizzle) ===
        var a_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.BM, BK_BYTES]()
        )
        var b_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.BN, BK_BYTES]()
        )

        comptime scales_per_mma = Self.scales_per_mma
        var sfa_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.BM, scales_per_mma * num_k_tiles]()
        )
        var sfb_smem = stack_allocation[DType.uint8, AddressSpace.SHARED](
            row_major[Self.BN, scales_per_mma * num_k_tiles]()
        )

        # === DRAM→regs→SMEM loading ===
        # Two-phase: LOAD_DRAM loads to register buffers, STORE_SMEM
        # copies registers to SMEM. This keeps the schedule's barrier
        # placement correct (barrier between STORE_SMEM and LOAD_FRAG).
        #
        # Thread distribution: row_major[load_rows, load_cols] maps each
        # thread to a (row, col) position. Each thread loads loads_per_tile
        # vector-width chunks, covering the full [Self.BM, BK_BYTES] tile.
        comptime load_thread_cols = BK_BYTES // simd_width
        comptime load_thread_rows = num_threads // load_thread_cols
        comptime load_layout = row_major[load_thread_rows, load_thread_cols]()
        # A and B can have different row counts (BM vs BN). Each matrix
        # needs its own loads_per_tile and register buffer size.
        comptime a_loads_per_tile = Self.BM // load_thread_rows
        comptime b_loads_per_tile = Self.BN // load_thread_rows
        comptime a_reg_elems = Self.BM * BK_BYTES // num_threads
        comptime b_reg_elems = Self.BN * BK_BYTES // num_threads

        # Block-row tiles spanning the full K dimension for tile-based indexing.
        var a_blockrow = a_gmem.tile[Self.BM, K_BYTES](block_idx.y, 0)
        var b_blockrow = b_gmem.tile[Self.BN, K_BYTES](block_idx.x, 0)

        # Register buffers for DRAM loads (one per matrix). Sized per
        # matrix so BM != BN works.
        var a_load_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            row_major[1, a_reg_elems]()
        )
        var b_load_reg = stack_allocation[DType.uint8, AddressSpace.LOCAL](
            row_major[1, b_reg_elems]()
        )

        # RegTileLoader wraps each block-row in an AMD buffer resource
        # descriptor. `bounds_from=a_gmem` clamps OOB buffer_load_dwordx4
        # reads to zero at the hardware level (no fault, no garbage).
        var a_loader = RegTileLoader[DType.uint8, load_layout](
            a_blockrow,
            bounds_from=a_gmem,
        )
        var b_loader = RegTileLoader[DType.uint8, load_layout](
            b_blockrow,
            bounds_from=b_gmem,
        )

        # === MMA operator ===
        var mma_op = BlockScaledMmaOp[
            mma_shape=IndexList[3](MMA_M, MMA_N, Self.MMA_K),
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            num_k_tiles=num_k_tiles,
        ]()

        # === Output writer ===
        # RegTileWriter casts from float32 accumulators to out_dtype.
        var c_writer = RegTileWriter[out_dtype, MMA_M, WARP_SIZE // MMA_M](c)

        # === Pipeline helpers ===
        var k_counter = 0
        var k_scale_counter = 0

        @always_inline
        @parameter
        def load_tiles_from_dram():
            """Load one BK-wide tile from DRAM to register buffers."""
            var a_block = a_blockrow.tile[Self.BM, BK_BYTES](0, k_counter)
            var b_block = b_blockrow.tile[Self.BN, BK_BYTES](0, k_counter)
            a_loader.load(a_load_reg, a_block.vectorize[1, simd_width]())
            b_loader.load(b_load_reg, b_block.vectorize[1, simd_width]())
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
            comptime for v in range(a_loads_per_tile):
                a_smem_dist[v, 0] = a_load_reg.raw_load[width=simd_width](
                    v * simd_width
                )
            comptime for v in range(b_loads_per_tile):
                b_smem_dist[v, 0] = b_load_reg.raw_load[width=simd_width](
                    v * simd_width
                )

        @always_inline
        @parameter
        def load_scales_to_smem():
            """Cooperatively load scale tiles from GMEM to SMEM.

            Scale tile per BK iteration: [Self.BM, scales_per_mma] for A and
            [Self.BN, scales_per_mma] for B, both uint8. Each row is 4 bytes.
            Threads 0..BM-1 load A scales, threads Self.BM..BM+Self.BN-1 load B.
            Each active thread loads one 4-byte row as an Int32, giving
            coalesced 4-byte aligned GMEM reads.
            """
            # Per BK iteration we load num_k_tiles MFMAs' worth of scales
            # per row. Each thread writes num_k_tiles contiguous Int32
            # dwords at row `tid` in the interleaved SMEM layout.
            var tid = Int(thread_idx.x)
            var base_scale_k = k_scale_counter * scales_per_mma * num_k_tiles
            var a_base_row = Int(block_idx.y) * Self.BM
            var b_base_row = Int(block_idx.x) * Self.BN

            # A scales: guard M-OOB rows.
            if tid < Self.BM:
                var row = a_base_row + tid
                if row < M:
                    comptime for k in range(num_k_tiles):
                        var src_off = (
                            row * K_SCALES + base_scale_k + k * scales_per_mma
                        )
                        var src_word = sfa.ptr.bitcast[Scalar[DType.int32]]()[
                            src_off // scales_per_mma
                        ]
                        sfa_smem.ptr.bitcast[Scalar[DType.int32]]()[
                            tid * num_k_tiles + k
                        ] = src_word
                else:
                    comptime for k in range(num_k_tiles):
                        sfa_smem.ptr.bitcast[Scalar[DType.int32]]()[
                            tid * num_k_tiles + k
                        ] = Int32(0)
            # B scales: guard N-OOB rows (B is transposed).
            if tid < Self.BN:
                var row = b_base_row + tid
                if row < N:
                    comptime for k in range(num_k_tiles):
                        var src_off = (
                            row * K_SCALES + base_scale_k + k * scales_per_mma
                        )
                        var src_word = sfb.ptr.bitcast[Scalar[DType.int32]]()[
                            src_off // scales_per_mma
                        ]
                        sfb_smem.ptr.bitcast[Scalar[DType.int32]]()[
                            tid * num_k_tiles + k
                        ] = src_word
                else:
                    comptime for k in range(num_k_tiles):
                        sfb_smem.ptr.bitcast[Scalar[DType.int32]]()[
                            tid * num_k_tiles + k
                        ] = Int32(0)

            k_scale_counter += 1

        # === Schedule-driven pipeline ===
        # The schedule prologue pre-loads 2 tiles, so we need at least 2
        # K-iterations. For K with only 1 tile, fall back to a simple loop.
        comptime a_loads_per_thread = Self.BM // load_thread_rows
        comptime b_loads_per_thread = Self.BN // load_thread_rows

        @always_inline
        @parameter
        def simple_k_loop():
            """Fallback for small K where schedule prologue doesn't fit."""
            for k_iter in range(K_BYTES // BK_BYTES):
                load_tiles_from_dram()
                load_scales_to_smem()
                copy_tiles_to_smem()
                barrier()

                var a_warp = a_smem.tile[Self.WM, BK_BYTES](warp_m, 0)
                var b_warp = b_smem.tile[Self.WN, BK_BYTES](warp_n, 0)

                comptime for k in range(num_k_tiles):
                    mma_op.load_frag_from_smem[k](a_warp, b_warp)

                    # k_tiles are interleaved along the column axis, so
                    # slice (warp, k_tile) → [WM/WN, scales_per_mma].
                    var sfa_k = sfa_smem.tile[Self.WM, scales_per_mma](
                        warp_m, k
                    )
                    var sfb_k = sfb_smem.tile[Self.WN, scales_per_mma](
                        warp_n, k
                    )
                    mma_op.load_scales_from_smem[k](sfa_k, sfb_k)

                    mma_op.mma[k]()
                barrier()

        @always_inline
        @parameter
        def scheduled_k_loop():
            """Pipelined K-loop via build_default_matmul_schedule."""
            comptime schedule = build_default_matmul_schedule[
                num_k_tiles=num_k_tiles,
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                num_k_mmas=num_k_tiles,
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
                    comptime k = entry.op.subtile
                    var a_warp = a_smem.tile[Self.WM, BK_BYTES](warp_m, 0)
                    var b_warp = b_smem.tile[Self.WN, BK_BYTES](warp_n, 0)
                    mma_op.load_frag_from_smem[k](a_warp, b_warp)
                    # k_tiles interleaved along the column axis.
                    var sfa_k = sfa_smem.tile[Self.WM, scales_per_mma](
                        warp_m, k
                    )
                    var sfb_k = sfb_smem.tile[Self.WN, scales_per_mma](
                        warp_n, k
                    )
                    mma_op.load_scales_from_smem[k](sfa_k, sfb_k)
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
        # RegTileWriter uses buffer_store_dwordx4 with an AMD buffer
        # resource descriptor built from the full [M, N] output tensor.
        # The V#'s bounds field is derived from the tensor's runtime M,
        # so OOB stores (rows >= M) are hardware-clamped and silently
        # dropped. No per-element guards needed.
        var c_reg = mma_op.accum_tile()
        var c_block = c.tile[Self.BM, Self.BN](block_idx.y, block_idx.x)
        var c_warp = c_block.tile[Self.WM, Self.WN](warp_m, warp_n)

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


def _launch_mxfp4[
    BM: Int, BN: Int, BK_ELEMS: Int, WM: Int, WN: Int
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    M: Int,
    ctx: DeviceContext,
) raises:
    """Instantiate MXFP4MatmulAMD with the given tile shape and launch."""
    comptime Kernel = MXFP4MatmulAMD[
        BM=BM, BN=BN, BK_ELEMS=BK_ELEMS, WM=WM, WN=WN
    ]
    comptime N = type_of(c).static_shape[1]

    comptime out_dtype = type_of(c).dtype

    comptime kernel = Kernel.run[
        out_dtype,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scales).LayoutType,
        type_of(b_scales).LayoutType,
    ]

    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=Kernel.num_threads,
    )


def mxfp4_block_scaled_matmul_amd(
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, DType.uint8, ...],
    b: TileTensor[mut=False, DType.uint8, ...],
    a_scales: TileTensor[mut=False, DType.float8_e8m0fnu, ...],
    b_scales: TileTensor[mut=False, DType.float8_e8m0fnu, ...],
    ctx: DeviceContext,
) raises:
    """Launch native MXFP4 block-scaled matmul on AMD CDNA4.

    Uses cdna4_block_scaled_mfma with FLOAT4_E2M1 directly — no
    dequantization to FP8. Both A and B must be packed uint8 with
    E8M0 scaling factors. Accumulates in float32, casts to c.dtype
    during the store epilogue.

    Tile shape is selected at runtime based on M to match the expected
    arithmetic intensity regime (decode vs. prefill). A proper
    (N, K)-keyed dispatch table is planned for a follow-up PR.

    Args:
        c: Output [M, N] (any float dtype, e.g. float32 or bfloat16).
        a: Packed A [M, K//2] uint8 (two MXFP4 elements per byte).
        b: Packed B [N, K//2] uint8 (transposed, two MXFP4 per byte).
        a_scales: A scales [M, K//32] float8_e8m0fnu.
        b_scales: B scales [N, K//32] float8_e8m0fnu.
        ctx: Device context for kernel launch.
    """

    # MMA tile c_frag_size is 4 regardless of block tile shape.
    comptime assert type_of(c).static_shape[1] % 4 == 0, (
        "N must be a multiple of c_frag_size=4 for the MXFP4 block-scaled"
        " matmul; non-aligned N is not yet supported"
    )

    # Aggressive BK values require K_BYTES to be divisible by BK_BYTES,
    # else MXFP4MatmulAMD's comptime assert fires. Gate each bucket so
    # a small-K caller (e.g. tests with K=128) falls back to the safe
    # default instead of hitting a build error.
    comptime K_BYTES = type_of(a).static_shape[1]
    comptime can_use_bk_256 = K_BYTES >= 128 and K_BYTES % 128 == 0
    comptime can_use_bk_512 = K_BYTES >= 256 and K_BYTES % 256 == 0

    var M = Int(c.dim[0]())
    comptime N = type_of(c).static_shape[1]

    if M == 0 or N == 0:
        return

    # Runtime M-bucket dispatch. Tile shapes tuned for Kimi K2.5 on MI355.
    #   M <=  16  → single-row decode regime
    #   M <=  64  → short prefill
    #   else      → general prefill / training
    if M <= 16:
        comptime if can_use_bk_512:
            _launch_mxfp4[BM=64, BN=32, BK_ELEMS=512, WM=64, WN=32](
                c, a, b, a_scales, b_scales, M, ctx
            )
        else:
            _launch_mxfp4[BM=128, BN=128, BK_ELEMS=128, WM=64, WN=64](
                c, a, b, a_scales, b_scales, M, ctx
            )
    elif M <= 64:
        comptime if can_use_bk_256:
            _launch_mxfp4[BM=128, BN=64, BK_ELEMS=256, WM=64, WN=64](
                c, a, b, a_scales, b_scales, M, ctx
            )
        else:
            _launch_mxfp4[BM=128, BN=128, BK_ELEMS=128, WM=64, WN=64](
                c, a, b, a_scales, b_scales, M, ctx
            )
    else:
        _launch_mxfp4[BM=128, BN=128, BK_ELEMS=128, WM=64, WN=64](
            c, a, b, a_scales, b_scales, M, ctx
        )
