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
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from std.gpu.host import DeviceContext
from layout import TensorLayout, TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation

from std.utils import IndexList, StaticTuple
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)
from structured_kernels.amd_tile_io import RegTileWriter

# MXFP4: 32 MXFP4 elements per E8M0 scale.
comptime MX_BLOCK_SIZE = 32


# ===----------------------------------------------------------------------=== #
# BlockScaledMmaOp — MFMA with inline scale application
# ===----------------------------------------------------------------------=== #


struct BlockScaledMmaOp[
    mma_shape: IndexList[3],  # (16, 16, 128) for MXFP4
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_k_tiles: Int,
]:
    """Register ownership + GMEM loading + block-scaled MFMA execution.

    Loads packed uint8 A/B fragments from GMEM and executes
    cdna4_block_scaled_mfma with per-lane E8M0 scale values.

    Scale operand model:
      Each lane holds 32 FP4 elements and one E8M0 scale byte,
      matching the MX format's per-32-element granularity exactly.
      For 16x16x128: 64 lanes cover 16 rows x 4 K-groups.
        lane_row = lane_id % 16   (matrix row)
        lane_k_group = lane_id / 16  (K-group 0..3)
      Each lane loads its own scale: scale[row, base_k + k_group].
      The scale byte is placed in byte 0 of the Int32 word.
    """

    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]

    comptime packed_k_per_mma = Self.MMA_K // 2  # bytes consumed per MFMA
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE  # 4

    # Per-lane data width from the SMEM tile.
    # 16x16x128 FP4: 16 rows x 64 packed bytes / 64 lanes = 16 bytes
    # per lane of actual FP4 data.
    # The MFMA operand VGPR is always SIMD[uint8, 32] (256 bits), but
    # since 32 FP4 elements occupy only 128 bits, the remaining 128 bits are
    # zeroed. We load 16 bytes from SMEM and zero-extend to 32 at the
    # MFMA call site.
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

    # Scale registers: per-lane packed Int32 words. Each lane holds 4
    # E8M0 scale bytes (one per K-group) for that lane's matrix row.
    # Stored per (m_mma, k_tile) or (n_mma, k_tile) combination.
    # Uses int32 TileTensors in LOCAL for mutability.
    comptime _a_scales_count = Self.num_m_mmas * Self.num_k_tiles
    comptime _b_scales_count = Self.num_n_mmas * Self.num_k_tiles
    comptime _a_scales_layout = row_major[1, Self._a_scales_count]()
    comptime _b_scales_layout = row_major[1, Self._b_scales_count]()
    var _a_scales: TileTensor[
        DType.int32,
        type_of(Self._a_scales_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]
    var _b_scales: TileTensor[
        DType.int32,
        type_of(Self._b_scales_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    @always_inline
    def __init__(out self):
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
            self._c_reg.ptr[i] = Scalar[DType.float32](0)
        self._a_scales = stack_allocation[DType.int32, AddressSpace.LOCAL](
            Self._a_scales_layout
        )
        self._b_scales = stack_allocation[DType.int32, AddressSpace.LOCAL](
            Self._b_scales_layout
        )
        comptime for i in range(Self._a_scales_count):
            self._a_scales.ptr[i] = Scalar[DType.int32](0)
        comptime for i in range(Self._b_scales_count):
            self._b_scales.ptr[i] = Scalar[DType.int32](0)

    @always_inline
    def accum_tile(self) -> ref[self._c_reg] type_of(self._c_reg):
        return self._c_reg

    @always_inline
    def load_frag_from_gmem[
        k_tile_idx: Int,
        a_stride: Int,
        b_stride: Int,
    ](
        self,
        a_gmem_ptr: UnsafePointer[Scalar[DType.uint8], _],
        b_gmem_ptr: UnsafePointer[Scalar[DType.uint8], _],
        a_m_offset: Int,
        b_n_offset: Int,
        k_byte_offset: Int,
    ):
        """Load MXFP4 A/B fragments directly from GMEM (bypasses SMEM).

        Uses the row-major MFMA lane mapping from the CDNA4 ISA:
          lane_row = lane_id % MMA_M
          lane_chunk = lane_id / MMA_M
          offset = lane_row * stride + lane_chunk * 16 + k_byte_offset
        """
        comptime frag_w = Self.mma_frag_width  # 16

        var lane = Int(lane_id())
        var lane_row = lane % Self.MMA_M
        var lane_chunk = lane // Self.MMA_M

        comptime for i in range(Self.num_m_mmas):
            var a_idx = k_tile_idx * Self.num_m_mmas + i
            var a_row = a_m_offset + i * Self.MMA_M + lane_row
            var a_off = a_row * a_stride + k_byte_offset + lane_chunk * frag_w
            self._a_reg.ptr.store[width=frag_w](
                a_idx * frag_w,
                a_gmem_ptr.load[width=frag_w](a_off),
            )

        comptime for i in range(Self.num_n_mmas):
            var b_idx = k_tile_idx * Self.num_n_mmas + i
            var b_row = b_n_offset + i * Self.MMA_N + lane_row
            var b_off = b_row * b_stride + k_byte_offset + lane_chunk * frag_w
            self._b_reg.ptr.store[width=frag_w](
                b_idx * frag_w,
                b_gmem_ptr.load[width=frag_w](b_off),
            )

    @always_inline
    def load_scales(
        self,
        a_scale_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], _],
        b_scale_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], _],
        a_m_base: Int,
        b_n_base: Int,
        k_tile_idx: Int,
        scale_stride_a: Int,
        scale_stride_b: Int,
    ):
        """Load per-lane E8M0 scale bytes for the MFMA.

        Each lane holds 32 FP4 elements and needs the matching scale.
        Lane mapping for 16x16x128:
          lane_row = lane % MMA_M     (matrix row, 0..15)
          lane_k_group = lane / MMA_M (K-group, 0..3)
        Scale index: scale_ptr[row * stride + base_k + lane_k_group]

        This gives full per-32-element MX compliance: 64 lanes x 1
        scale each = 16 rows x 4 K-groups = 64 distinct scale values.

        Args:
            a_scale_ptr: Base pointer to A scales [M, K//32].
            b_scale_ptr: Base pointer to B scales [N, K//32].
            a_m_base: Global M-row offset for this warp's first MMA tile.
            b_n_base: Global N-row offset for this warp's first MMA tile.
            k_tile_idx: Which k-tile we're processing.
            scale_stride_a: Row stride of A scales tensor.
            scale_stride_b: Row stride of B scales tensor.
        """
        var base_scale_k = k_tile_idx * Self.scales_per_mma
        var base_m = a_m_base
        var base_n = b_n_base

        var lane = Int(lane_id())
        var lane_row = lane % Self.MMA_M
        var lane_k_group = lane // Self.MMA_M

        # A scales: each lane loads scale for its own row + K-group.
        comptime for m_mma in range(Self.num_m_mmas):
            var row = base_m + m_mma * Self.MMA_M + lane_row
            var sb = a_scale_ptr[
                row * scale_stride_a + base_scale_k + lane_k_group
            ]
            var byte_val = bitcast[DType.uint8](sb)
            self._a_scales.ptr[m_mma] = rebind[Scalar[DType.int32]](
                Int32(UInt32(byte_val))
            )

        # B scales: same per-lane K-group mapping.
        comptime for n_mma in range(Self.num_n_mmas):
            var col = base_n + n_mma * Self.MMA_N + lane_row
            var sb = b_scale_ptr[
                col * scale_stride_b + base_scale_k + lane_k_group
            ]
            var byte_val = bitcast[DType.uint8](sb)
            self._b_scales.ptr[n_mma] = rebind[Scalar[DType.int32]](
                Int32(UInt32(byte_val))
            )

    @always_inline
    def mma[k_tile_idx: Int](self):
        """Execute block-scaled MFMA for k-tile k_tile_idx.

        The MFMA call swaps B→src_a and A→src_b to match the AMD
        convention (gpu_mma(c, b, a, c)). The accumulator is stored
        in row-major order [num_m_mmas, num_n_mmas * c_frag_size]
        matching the output store's (m_mma, n_mma) indexing.

        Loop order: outer=m (output M-tiles), inner=n (output N-tiles).
        For each (m, n) pair, we read the A fragment for M-tile m and
        B fragment for N-tile n, then pass B as src_a and A as src_b.
        """
        comptime for m in range(Self.num_m_mmas):
            comptime for n in range(Self.num_n_mmas):
                var a_row = k_tile_idx * Self.num_m_mmas + m
                var b_row = k_tile_idx * Self.num_n_mmas + n
                var c_off = (
                    m * Self.num_n_mmas * Self.c_frag_size
                    + n * Self.c_frag_size
                )

                var a_data = self._a_reg.ptr.load[width=Self.mma_frag_width](
                    a_row * Self.mma_frag_width
                )
                var b_data = self._b_reg.ptr.load[width=Self.mma_frag_width](
                    b_row * Self.mma_frag_width
                )

                # Zero-extend to SIMD[uint8, 32] for the MFMA operand.
                var a_frag = SIMD[DType.uint8, 32](0)
                var b_frag = SIMD[DType.uint8, 32](0)
                a_frag = a_frag.insert[offset=0](a_data)
                b_frag = b_frag.insert[offset=0](b_data)

                var c_frag = self._c_reg.ptr.load[width=Self.c_frag_size](c_off)

                var a_scale = rebind[Int32](self._a_scales.ptr[m])
                var b_scale = rebind[Int32](self._b_scales.ptr[n])

                # B→src_a (drives MFMA rows → N coords in output),
                # A→src_b (drives MFMA cols → M coords in output).
                cdna4_block_scaled_mfma[
                    0,
                    0,
                    CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                    CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                ](c_frag, b_frag, a_frag, b_scale, a_scale)

                self._c_reg.ptr.store[width=Self.c_frag_size](c_off, c_frag)


# ===----------------------------------------------------------------------=== #
# MXFP4MatmulAMD — kernel struct
# ===----------------------------------------------------------------------=== #

# Fixed config for MXFP4: 16x16x128 MFMA is the only shape that
# supports FLOAT4_E2M1 on CDNA4.
comptime MXFP4_MMA_M = 16
comptime MXFP4_MMA_N = 16
comptime MXFP4_MMA_K = 128

# Block tile shape: 128x128 output tile, BK=128 MXFP4 elements = 64 packed bytes.
comptime MXFP4_BM = 128
comptime MXFP4_BN = 128
comptime MXFP4_BK_ELEMS = 128
comptime MXFP4_BK_BYTES = MXFP4_BK_ELEMS // 2  # 64 packed bytes

# Warp tile shape (4 warps: 2x2 grid).
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
        """MXFP4 block-scaled GEMM kernel.

        All inputs/outputs are TileTensor. A and B are packed uint8
        (K//2 columns). Scales are [rows, K//32] float8_e8m0fnu.

        Args:
            c: Output [M, N] float32.
            a: Packed A [M, K//2] uint8.
            b: Packed B [N, K//2] uint8 (transposed).
            sfa: A scales [M, K//32] float8_e8m0fnu.
            sfb: B scales [N, K//32] float8_e8m0fnu.
        """
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

        comptime K_BYTES = type_of(a).static_shape[1]  # K//2
        comptime N = type_of(b).static_shape[0]
        comptime assert N > 0, "N must be known at compile time"
        comptime assert K_BYTES > 0, "K (packed) must be known at compile time"

        comptime K_SCALES = type_of(sfa).static_shape[1]  # K//32
        comptime scale_stride_a = K_SCALES
        comptime scale_stride_b = K_SCALES

        var _warp_id = warp_id()
        var warp_m, warp_n = divmod(_warp_id, Self.num_warps_n)

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

        # === Direct GMEM loop (no SMEM, correctness-first) ===
        # Load A/B tiles directly from global memory into registers
        # using the MFMA's native row-major lane mapping, then execute
        # the block-scaled MFMA. SMEM pipeline will be added back once
        # this path produces correct results.
        var a_m_base = Int(block_idx.y) * BM + warp_m * WM
        var b_n_base = Int(block_idx.x) * BN + warp_n * WN

        for k_iter in range(K_BYTES // BK_BYTES):
            var k_byte_off = k_iter * BK_BYTES

            mma_op.load_frag_from_gmem[0, K_BYTES, K_BYTES](
                a.ptr,
                b.ptr,
                a_m_base,
                b_n_base,
                k_byte_off,
            )

            mma_op.load_scales(
                sfa.ptr,
                sfb.ptr,
                a_m_base,
                b_n_base,
                k_iter,
                scale_stride_a,
                scale_stride_b,
            )

            mma_op.mma[0]()

        # === Output store ===
        # The MFMA uses (b_frag, a_frag) so its accumulator is laid
        # out with the N-tile index in the outer (m) loop and M-tile
        # index in the inner (n) loop. When reading the accumulator,
        # accum row `m` corresponds to N-tile `m` and accum column
        # group `n` to M-tile `n`. We swap the output tile indices
        # so that accum(m, n) writes to output C(n, m).
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
