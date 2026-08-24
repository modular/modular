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
consumes MXFP4 operands with E8M0 scale words: no dequantization needed.

Structure mirrors AMDMatmul: TileTensor throughout, RegTileLoader for
DRAM→regs, row-major SMEM (no blocked-product layout: the FP4/FP8 MFMA
expects a simple row-major lane-to-data mapping unlike BF16), schedule-
driven pipeline. A conditional XOR swizzle (see `use_smem_swizzle`)
removes LDS bank conflicts on the A/B fragment SMEM read/write when the
tile config qualifies.

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

Entry point: block_scaled_matmul_amd()
"""

from std.math import ceildiv
from std.memory import bitcast
from std.sys import simd_width_of, size_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
)
from layout import Coord, Idx, TensorLayout, TileTensor
from layout.tile_layout import row_major, col_major
from layout.tile_tensor import stack_allocation
from layout.swizzle import Swizzle

from std.utils import IndexList, StaticTuple
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)
from structured_kernels.amd_tile_io import (
    RegTileEpilogue,
    RegTileLoader,
    RegTileWriter,
)
from ....utils import elementwise_epilogue_type
from .amd_matmul_schedule import build_default_matmul_schedule
from .amd_matmul_schedule import (
    DefaultMatmulOps,
    COMPUTE,
    LOAD_DRAM,
    LOAD_FRAG,
    STORE_SMEM,
)
from pipeline.pipeline_dsl import ScheduleEntry

from .block_scaled_preshuffle_loaders import PreshuffledBLoader
from .amd_4wave_split_k_matmul import (
    SplitKWorkspace,
    _split_k_reduce_kernel,
)

# MXFP4: 32 MXFP4 elements per E8M0 scale.
comptime MX_BLOCK_SIZE = 32

comptime SCHED_MASK_DS_READ = 0
comptime SCHED_MASK_DS_WRITE = 1
comptime SCHED_MASK_VMEM_READ = 2
comptime SCHED_MASK_MFMA = 3


# `_full_row_lds_swizzle`: XOR LDS swizzle on a row-major `[rows, BK_BYTES]`
# uint8 A/B tile, applied to the full flat `row*BK_BYTES + col_byte` in-tile
# offset (not a sub-tile-local offset) at both SMEM write and read; a
# mismatch produces wrong logits. `use_smem_swizzle`'s gate admits exactly
# two BK_BYTES values, 64 (MXFP4) and 128 (MXFP8), and both use bits=3:
#
#   * BK_BYTES == 64 (MXFP4): bit-for-bit the pattern this kernel shipped
#     with before MXFP8 support existed (`distribute(...,
#     swizzle=Swizzle(3, 0, 3))`, byte-space equivalent `Swizzle(3, 4, 3)`,
#     applied over a per-MFMA sub-tile that happened to span the whole
#     row). That formulation depends only on `lane_row` (0..15); since
#     `row = i*MMA_M + lane_row` folds `i` into bits outside
#     `Swizzle(3, 4, 3)`'s [4, 10) span, applying the same swizzle to the
#     explicit flat offset reproduces it exactly for every `i`.
#   * BK_BYTES == 128 (MXFP8): a new pattern. The old kernel disabled the
#     swizzle here entirely (see `use_smem_swizzle` below). Validated on
#     MI355 via rocprofv3: it eliminates the measured LDS bank conflicts
#     on the A/B fragment reads.
@always_inline
def _full_row_lds_swizzle[BK_BYTES: Int]() -> Swizzle:
    """Builds the LDS swizzle for a row-major `[rows, BK_BYTES]` uint8 tile.

    Parameters:
        BK_BYTES: Width of the SMEM tile in bytes. Only 64 (MXFP4) and
            128 (MXFP8) are derived/validated (see comment above); any
            other value is a comptime error until re-derived.

    Returns:
        The `Swizzle` on the flat in-tile byte offset.
    """
    comptime assert BK_BYTES == 64 or BK_BYTES == 128, (
        "_full_row_lds_swizzle is only derived and measured for BK_BYTES"
        " in {64, 128} (the only values use_smem_swizzle's gate admits);"
        " a relaxed gate must re-derive and re-measure `bits` here rather"
        " than inherit this constant."
    )
    comptime bits = 3
    return Swizzle(bits, 4, bits)


@always_inline
def _swizzled_smem_off[
    BK_BYTES: Int, swizzle: Optional[Swizzle]
](row: Int, col_byte: Int) -> Int:
    """Flat `row*BK_BYTES + col_byte` in-tile byte offset, XOR-swizzled if enabled.
    """
    var off = row * BK_BYTES + col_byte
    comptime if swizzle:
        off = swizzle.value()(off)
    return off


# ===----------------------------------------------------------------------=== #
# BlockScaledMmaOp — MFMA with inline scale application
# ===----------------------------------------------------------------------=== #


struct BlockScaledMmaOp[
    mma_shape: IndexList[3],  # (16, 16, 128) for MXFP4 and MXFP8 alike
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_k_tiles: Int,
    num_b_slots: Int = 1,
    matrix_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
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
      one Int32 VGPR: byte i holds the scale for m_mma=i (A) or
      n_mma=i (B). The MFMA byte-index selector (OP_SEL) picks the
      correct byte for each MMA tile, so one scale load covers all
      4 m_mma or n_mma positions with zero overhead.

    Parameters:
        mma_shape: MFMA tile shape as `(M, N, K)` in logical FP4
            elements, `(16, 16, 128)`.
        num_m_mmas: Number of spatial M MMA tiles per warp tile
            (`WM // MMA_M`). Must be <= 4.
        num_n_mmas: Number of spatial N MMA tiles per warp tile
            (`WN // MMA_N`). Must be <= 4.
        num_k_tiles: Number of K sub-tiles within one BK iteration
            (`BK_BYTES // packed_k_per_mma`).
        num_b_slots: Number of B register slots for depth-2 prefetch
            (defaults to 1).
        matrix_format: `f8f6f4` operand encoding for A and B. A lane covers
            32 K-elements in every format; what changes is the bytes that
            occupies -- 16 (FP4), 24 (FP6), 32 (FP8) -- which `lane_bytes`
            below derives.
    """

    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]

    comptime a_bits = Self.matrix_format.bits_per_element()
    comptime b_bits = Self.matrix_format.bits_per_element()
    comptime bits_per_element = Self.a_bits
    comptime a_packed_k_per_mma = (Self.MMA_K * Self.a_bits) // 8
    comptime b_packed_k_per_mma = (Self.MMA_K * Self.b_bits) // 8
    comptime packed_k_per_mma = Self.a_packed_k_per_mma
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE  # 4

    comptime lanes_per_row = WARP_SIZE // Self.MMA_M

    comptime a_frag_width_bytes: Int = (
        Self.a_packed_k_per_mma // Self.lanes_per_row
    )
    comptime b_frag_width_bytes: Int = (
        Self.b_packed_k_per_mma // Self.lanes_per_row
    )
    comptime mma_frag_width_bytes: Int = Self.a_frag_width_bytes
    comptime lane_bytes: Int = Self.mma_frag_width_bytes
    comptime a_reg_frag_bytes: Int = Self.matrix_format.simd_width()
    comptime b_reg_frag_bytes: Int = Self.matrix_format.simd_width()
    comptime reg_frag_bytes: Int = Self.a_reg_frag_bytes
    # At MXFP8 a lane's two 16-byte halves sit K_HALF_STRIDE apart in K, not
    # contiguous; treating them as contiguous mis-pairs the scale blocks.
    comptime FRAG_HALF_BYTES = 16
    comptime a_num_frag_halves = (
        Self.a_frag_width_bytes // Self.FRAG_HALF_BYTES
    )
    comptime b_num_frag_halves = (
        Self.b_frag_width_bytes // Self.FRAG_HALF_BYTES
    )
    comptime num_frag_halves = Self.a_num_frag_halves
    comptime A_K_HALF_STRIDE = (
        Self.a_packed_k_per_mma // Self.a_num_frag_halves
    )
    comptime B_K_HALF_STRIDE = (
        Self.b_packed_k_per_mma // Self.b_num_frag_halves
    )
    comptime K_HALF_STRIDE = Self.A_K_HALF_STRIDE

    comptime A_BK_BYTES = Self.num_k_tiles * Self.a_packed_k_per_mma
    comptime B_BK_BYTES = Self.num_k_tiles * Self.b_packed_k_per_mma
    comptime BK_BYTES = Self.A_BK_BYTES

    # XOR swizzle removing LDS bank conflicts on the A/B fragment SMEM
    # read/write, computed on the full flat offset (see
    # `_full_row_lds_swizzle`). Gated to the 16x16x128/64B-`K_HALF_STRIDE`
    # config this formula was derived and tested against.
    comptime use_smem_swizzle = (
        Self.num_k_tiles == 1
        and Self.A_K_HALF_STRIDE == 64
        and Self.B_K_HALF_STRIDE == 64
        and Self.FRAG_HALF_BYTES == 16
        and Self.a_frag_width_bytes == 16
        and Self.b_frag_width_bytes == 16
        and Self.MMA_M == 16
    )
    comptime smem_swizzle = Optional[Swizzle](
        _full_row_lds_swizzle[Self.BK_BYTES]()
    ) if Self.use_smem_swizzle else Optional[Swizzle]()

    # Scales: 4 E8M0 bytes per MFMA call (128 MXFP4 / 32 per scale = 4).
    comptime scales_per_mma = Self.MMA_K // MX_BLOCK_SIZE  # 4

    comptime _a_reg_layout = row_major[
        Self.num_m_mmas * Self.num_k_tiles,
        Self.a_reg_frag_bytes,
    ]()
    comptime _b_reg_layout = row_major[
        Self.num_b_slots * Self.num_n_mmas * Self.num_k_tiles,
        Self.b_reg_frag_bytes,
    ]()
    comptime _b_slot_stride = Self.num_n_mmas * Self.num_k_tiles
    comptime _c_reg_layout = row_major[
        Self.num_m_mmas,
        Self.num_n_mmas * Self.c_frag_size,
    ]()

    var _a_reg: TileTensor[
        .uint8,
        type_of(Self._a_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _b_reg: TileTensor[
        .uint8,
        type_of(Self._b_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _c_reg: TileTensor[
        .float32,
        type_of(Self._c_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]

    # Packed scale VGPRs: one Int32 per k_tile for A and B. Byte i of
    # _a_scale_packed[k] holds the scale for spatial A tile m_mma=i of
    # k-tile k. Separate slots per k_tile so that schedules which
    # interleave LOAD_FRAG / COMPUTE across k_tiles don't clobber each
    # other.
    comptime _scale_layout = row_major[1, Self.num_k_tiles]()
    var _a_scale_packed: TileTensor[
        .int32,
        type_of(Self._scale_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _b_scale_packed: TileTensor[
        .int32,
        type_of(Self._scale_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]

    @always_inline
    def __init__(out self):
        comptime assert (
            Self.num_m_mmas <= 4
        ), "num_m_mmas must be <= 4 for packed scales"
        comptime assert (
            Self.num_n_mmas <= 4
        ), "num_n_mmas must be <= 4 for packed scales"
        self._a_reg = stack_allocation[DType.uint8, address_space=.LOCAL](
            Self._a_reg_layout
        )
        self._b_reg = stack_allocation[DType.uint8, address_space=.LOCAL](
            Self._b_reg_layout
        )
        self._c_reg = stack_allocation[DType.float32, address_space=.LOCAL](
            Self._c_reg_layout
        )
        _ = self._c_reg.fill(Float32(0))
        self._a_scale_packed = stack_allocation[
            DType.int32, address_space=.LOCAL
        ](Self._scale_layout)
        self._b_scale_packed = stack_allocation[
            DType.int32, address_space=.LOCAL
        ](Self._scale_layout)
        _ = self._a_scale_packed.fill(Int32(0))
        _ = self._b_scale_packed.fill(Int32(0))

    @always_inline
    def accum_tile(self) -> ref[self._c_reg] type_of(self._c_reg):
        return self._c_reg

    @always_inline
    def load_frag_from_smem[
        k_tile_idx: Int
    ](
        mut self,
        a_smem_warp: TileTensor[.uint8, _, _, address_space=.SHARED, ...],
        b_smem_warp: TileTensor[.uint8, _, _, address_space=.SHARED, ...],
    ):
        """Loads A/B fragments from row-major SMEM for k-tile `k_tile_idx`.

        Dispatches on the operand format at compile time. The dispatch lives
        here rather than at the call sites so a new call site cannot forget it
        and silently instantiate the power-of-two loader for FP6, whose 24-byte
        fragment `vectorize`/`distribute` cannot express.

        Parameters:
            k_tile_idx: K-tile index within the current BK iteration.

        Args:
            a_smem_warp: SMEM view of the A tile for this warp.
            b_smem_warp: SMEM view of the B tile for this warp.
        """
        comptime if Self.b_bits == 6:
            comptime for i in range(Self.num_n_mmas):
                var b_idx = k_tile_idx * Self.num_n_mmas + i
                self._b_reg.vectorize[1, Self.b_reg_frag_bytes]()[
                    b_idx, 0
                ] = self._load_fp6_lane_fragment[
                    Self.b_packed_k_per_mma,
                    Self.b_frag_width_bytes,
                    Self.b_reg_frag_bytes,
                ](
                    b_smem_warp, i, k_tile_idx
                )
        else:
            self._load_b_frag_vectorized[k_tile_idx](b_smem_warp)

        comptime if Self.a_bits == 6:
            comptime for i in range(Self.num_m_mmas):
                var a_idx = k_tile_idx * Self.num_m_mmas + i
                self._a_reg.vectorize[1, Self.a_reg_frag_bytes]()[
                    a_idx, 0
                ] = self._load_fp6_lane_fragment[
                    Self.a_packed_k_per_mma,
                    Self.a_frag_width_bytes,
                    Self.a_reg_frag_bytes,
                ](
                    a_smem_warp, i, k_tile_idx
                )
        else:
            self._load_a_frag_vectorized[k_tile_idx](a_smem_warp)

    @always_inline
    def _load_b_frag_vectorized[
        k_tile_idx: Int
    ](
        mut self,
        b_smem_warp: TileTensor[.uint8, _, _, address_space=.SHARED, ...],
    ):
        """Loads B fragments for a power-of-two encoding (FP4 or FP8).

        One lane's operand is `b_num_frag_halves` 16-byte halves, each in its
        own k-window (`B_K_HALF_STRIDE` apart). At FP4 that is a single half.

        Computes each lane's flat `row*B_BK_BYTES + col_byte` in-tile byte
        offset explicitly (XOR-swizzled when `use_smem_swizzle`) and reads it
        directly, matching the write side's identical formula in
        `copy_tiles_to_smem`. col_major[MMA_M, WARP_SIZE/MMA_M] gives each
        lane's (row, k_vec) position, matching the MFMA native lane mapping.

        Parameters:
            k_tile_idx: K-tile index within the current BK iteration.

        Args:
            b_smem_warp: SMEM view of the B tile for this warp.
        """
        comptime half_w = Self.FRAG_HALF_BYTES

        # groups of 16 threads are responsible for 16 rows, i.e threads 0, 16, 32, 48 handle row 0 ...
        comptime lane_layout = col_major[Self.MMA_M, WARP_SIZE // Self.MMA_M]()
        var lane_crd = lane_layout.idx2crd(Int(lane_id()))
        var lane_row = Int(lane_crd[0])
        var lane_k_vec = Int(lane_crd[1])

        var b_reg_v = self._b_reg.vectorize[1, half_w]()
        comptime for i in range(Self.num_n_mmas):
            var b_idx = k_tile_idx * Self.num_n_mmas + i
            var row = i * Self.MMA_N + lane_row
            comptime for h in range(Self.b_num_frag_halves):
                var col_byte = (
                    k_tile_idx * Self.b_packed_k_per_mma
                    + h * Self.B_K_HALF_STRIDE
                    + lane_k_vec * half_w
                )
                var off = _swizzled_smem_off[
                    Self.B_BK_BYTES, Self.smem_swizzle
                ](row, col_byte)
                b_reg_v[b_idx, h] = b_smem_warp.raw_load[width=half_w](off)

    @always_inline
    def _load_a_frag_vectorized[
        k_tile_idx: Int
    ](
        mut self,
        a_smem_warp: TileTensor[.uint8, _, _, address_space=.SHARED, ...],
    ):
        """A-side counterpart of `_load_b_frag_vectorized`.

        Parameters:
            k_tile_idx: K-tile index within the current BK iteration.

        Args:
            a_smem_warp: SMEM view of the A tile for this warp.
        """
        comptime half_w = Self.FRAG_HALF_BYTES
        comptime lane_layout = col_major[Self.MMA_M, WARP_SIZE // Self.MMA_M]()
        var lane_crd = lane_layout.idx2crd(Int(lane_id()))
        var lane_row = Int(lane_crd[0])
        var lane_k_vec = Int(lane_crd[1])

        var a_reg_v = self._a_reg.vectorize[1, half_w]()
        comptime for i in range(Self.num_m_mmas):
            var a_idx = k_tile_idx * Self.num_m_mmas + i
            var row = i * Self.MMA_M + lane_row
            comptime for h in range(Self.a_num_frag_halves):
                var col_byte = (
                    k_tile_idx * Self.a_packed_k_per_mma
                    + h * Self.A_K_HALF_STRIDE
                    + lane_k_vec * half_w
                )
                var off = _swizzled_smem_off[
                    Self.A_BK_BYTES, Self.smem_swizzle
                ](row, col_byte)
                a_reg_v[a_idx, h] = a_smem_warp.raw_load[width=half_w](off)

    @always_inline
    def _load_fp6_lane_fragment[
        packed_k: Int, frag_w: Int, reg_w: Int
    ](
        self,
        smem_warp: TileTensor[.uint8, _, _, address_space=.SHARED, ...],
        mma_idx: Int,
        k_tile_idx: Int,
    ) -> SIMD[.uint8, reg_w]:
        """Reads one lane's 24 FP6 payload bytes out of row-major SMEM.

        `vectorize`/`distribute` cannot express a 24-byte element, so this
        indexes SMEM directly. The lane mapping is deliberately identical to
        the power-of-two path's `col_major[MMA_M, lanes_per_row]`: lane `l`
        owns row `l % MMA_M` and K-group `l // MMA_M`.

        Three 8-byte reads rather than 16+8: a lane's byte offset is a multiple
        of 24, which is 8-byte but not 16-byte aligned, so a 16-byte read would
        be misaligned for odd K-groups.

        Returns:
            The lane's fragment, payload in bytes 0..23 and zeros above.
        """
        comptime assert frag_w % 8 == 0, "FP6 fragment must be 8-byte aligned"

        var lane = Int(lane_id())
        var row = mma_idx * Self.MMA_M + (lane % Self.MMA_M)
        var byte_base = k_tile_idx * packed_k + (lane // Self.MMA_M) * frag_w

        var fragment = SIMD[.uint8, reg_w](0)
        comptime for chunk in range(frag_w // 8):
            fragment = fragment.insert[offset=chunk * 8](
                rebind[SIMD[.uint8, 8]](
                    smem_warp.load[8](Coord(row, byte_base + chunk * 8))
                )
            )
        return fragment

    @always_inline
    def load_a_frag_from_smem[
        k_tile_idx: Int
    ](self, a_smem_warp: TileTensor[.uint8, _, _, address_space=.SHARED, ...],):
        """A-only variant of `load_frag_from_smem` for callers that source B
        elsewhere (e.g. preshuffled DRAM via PreshuffledBLoader).

        Parameters:
            k_tile_idx: K-tile index within the current BK iteration.

        Args:
            a_smem_warp: SMEM view of the A tile for this warp, shape
                `[WM, BK_BYTES]` uint8.
        """
        comptime half_w = Self.FRAG_HALF_BYTES
        comptime lane_layout = col_major[Self.MMA_M, WARP_SIZE // Self.MMA_M]()
        var lane_crd = lane_layout.idx2crd(Int(lane_id()))
        var lane_row = Int(lane_crd[0])
        var lane_k_vec = Int(lane_crd[1])
        var a_reg_v = self._a_reg.vectorize[1, half_w]()
        comptime for i in range(Self.num_m_mmas):
            var a_idx = k_tile_idx * Self.num_m_mmas + i
            var row = i * Self.MMA_M + lane_row
            comptime for h in range(Self.a_num_frag_halves):
                var col_byte = (
                    k_tile_idx * Self.a_packed_k_per_mma
                    + h * Self.A_K_HALF_STRIDE
                    + lane_k_vec * half_w
                )
                var off = _swizzled_smem_off[
                    Self.A_BK_BYTES, Self.smem_swizzle
                ](row, col_byte)
                a_reg_v[a_idx, h] = a_smem_warp.raw_load[width=half_w](off)

    @always_inline
    def load_b_frag_preshuffled[
        k_tile_idx: Int, N: Int, K_BYTES: Int, slot: Int = 0
    ](
        self,
        b_loader: PreshuffledBLoader[N, K_BYTES],
        warp_n_off: Int,
        k_byte_base: Int,
    ):
        """Load B fragments directly from preshuffled DRAM into b_reg slot `slot`.

        Each lane issues one `buffer_load_dwordx4` per (k_tile, n_mma) at the
        per-lane MFMA mapping `(lane%16 → n-row, lane//16 → k-group)`. The
        `slot` parameter selects which b_reg half to write into when
        `num_b_slots > 1` (depth-2 prefetch).

        Parameters:
            k_tile_idx: K-tile index within the current BK iteration.
            N: Total N dimension of the B matrix in output columns.
            K_BYTES: Total K dimension in packed bytes (`K // 2`).
            slot: B register slot to write into (defaults to 0).

        Args:
            b_loader: Preshuffled B DRAM loader issuing per-lane
                `buffer_load_dwordx4` reads.
            warp_n_off: Starting N-row offset of this warp's B tile
                within the block.
            k_byte_base: Base byte offset in K for the current BK
                iteration.
        """
        comptime assert slot < Self.num_b_slots, "slot out of range"
        comptime assert Self.lane_bytes == 16, (
            "the preshuffled-B DRAM path is MXFP4-only; MXFP8 dense reads"
            " row-major B through the SMEM loaders"
        )
        # MXFP4-only path (asserted above), so the operand is exactly one
        # 16-byte half; use the concrete width the loader returns.
        comptime frag_w = Self.FRAG_HALF_BYTES
        comptime mma_k_bytes = Self.packed_k_per_mma
        var lane_nlane = lane_id() % Self.MMA_N
        var lane_klane = lane_id() // Self.MMA_N
        comptime for i in range(Self.num_n_mmas):
            var b_idx = (
                slot * Self._b_slot_stride + k_tile_idx * Self.num_n_mmas + i
            )
            var n_log = warp_n_off + i * Self.MMA_N + Int(lane_nlane)
            var k_byte_log = (
                k_byte_base
                + k_tile_idx * mma_k_bytes
                + Int(lane_klane) * frag_w
            )
            self._b_reg.vectorize[1, frag_w]()[
                b_idx, 0
            ] = b_loader.load_fragment(n_log, k_byte_log)

    @always_inline
    def load_scales_from_smem[
        k_tile_idx: Int
    ](
        mut self,
        a_scale_smem_warp: TileTensor[.uint8, address_space=.SHARED, ...],
        b_scale_smem_warp: TileTensor[.uint8, address_space=.SHARED, ...],
    ):
        """Load packed scale VGPRs for k-tile k_tile_idx from SMEM.

        Packs num_m_mmas (A) or num_n_mmas (B) scale bytes into one
        Int32 each using the same col_major[MMA_M, WARP_SIZE/MMA_M]
        distribute pattern as load_frag_from_smem. Each lane picks one
        scale byte via (lane_row, lane_k_group). TileTensor's stride
        handling means this works for any parent SMEM layout.

        The MFMA byte-index selector (a_scale_byte_index=m_mma,
        b_scale_byte_index=n_mma) picks the correct byte: no shifts
        or masks at consumption time.

        Parameters:
            k_tile_idx: K-tile index within the current BK iteration.

        Args:
            a_scale_smem_warp: SMEM view of A scale bytes for this
                warp, shape `[WM, scales_per_mma]` uint8.
            b_scale_smem_warp: SMEM view of B scale bytes for this
                warp, shape `[WN, scales_per_mma]` uint8.
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
            a_packed = a_packed | bitcast[.int32](byte_val << UInt32(m_mma * 8))
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
            b_packed = b_packed | bitcast[.int32](byte_val << UInt32(n_mma * 8))
        self._b_scale_packed.raw_store(k_tile_idx, b_packed)

    @always_inline
    def mma[k_tile_idx: Int, slot: Int = 0](self):
        """Execute block-scaled MFMA for k-tile k_tile_idx using B from `slot`.

        B→src_a, A→src_b (AMD MFMA convention).
        The packed scale VGPRs hold one byte per spatial MMA tile.
        a_scale_byte_index=m selects byte m from _a_scale_packed,
        b_scale_byte_index=n selects byte n from _b_scale_packed.

        `slot` selects which b_reg half to read when `num_b_slots > 1`.

        Parameters:
            k_tile_idx: K-tile index within the current BK iteration.
            slot: B register slot to read from when `num_b_slots > 1`
                (defaults to 0).
        """
        comptime assert slot < Self.num_b_slots, "slot out of range"
        comptime for m in range(Self.num_m_mmas):
            comptime for n in range(Self.num_n_mmas):
                comptime a_row = k_tile_idx * Self.num_m_mmas + m
                comptime b_row = (
                    slot * Self._b_slot_stride
                    + k_tile_idx * Self.num_n_mmas
                    + n
                )
                comptime a_off = a_row * Self.a_reg_frag_bytes
                comptime b_off = b_row * Self.b_reg_frag_bytes
                comptime c_off = (
                    m * Self.num_n_mmas * Self.c_frag_size
                    + n * Self.c_frag_size
                )

                var a_frag = self._a_reg.raw_load[width=Self.a_reg_frag_bytes](
                    a_off
                )
                var b_frag = self._b_reg.raw_load[width=Self.b_reg_frag_bytes](
                    b_off
                )

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
                    Self.matrix_format,
                    Self.matrix_format,
                ](
                    c_frag,
                    b_frag,
                    a_frag,
                    b_scale,
                    a_scale,
                )

                self._c_reg.raw_store[width=Self.c_frag_size](c_off, c_frag)


# ===----------------------------------------------------------------------=== #
# BlockScaledMatmulAMD — kernel struct
# ===----------------------------------------------------------------------=== #


struct BlockScaledMatmulAMD[
    BM: Int = 128,
    BN: Int = 128,
    BK_ELEMS: Int = 128,
    WM: Int = 64,
    WN: Int = 64,
    MMA_M: Int = 16,
    MMA_N: Int = 16,
    MMA_K: Int = 128,
    matrix_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """Native MXFP4 block-scaled matmul for AMD CDNA4.

    Uses cdna4_block_scaled_mfma with FLOAT4_E2M1 format directly.
    Single-buffer pipeline with schedule-driven prologue/kernel/epilogue.
    SMEM is plain row-major (no blocked-product), with a conditional XOR
    swizzle (`BlockScaledMmaOp.use_smem_swizzle`) removing LDS bank
    conflicts on the A/B fragment read/write when the tile config qualifies.

    Parameters:
        BM: Block tile rows (output M per block). Default 128.
        BN: Block tile cols (output N per block). Default 128.
        BK_ELEMS: Block tile K in logical FP4 elements. Default 128.
        WM: Warp tile rows. BM must be divisible by WM. Default 64.
        WN: Warp tile cols. BN must be divisible by WN. Default 64.
        MMA_M: MFMA tile rows. WM must be divisible by MMA_M. Default 16.
        MMA_N: MFMA tile cols. WN must be divisible by MMA_N. Default 16.
        MMA_K: MFMA K-depth in logical FP4 elements. Default 128.
        matrix_format: `f8f6f4` operand encoding for A and B (FP4 E2M1 by
            default). `BK_ELEMS` counts ELEMENTS, so a given `BK_ELEMS` costs
            1.5x the registers and LDS at MXFP6 and 2x at MXFP8.
        elementwise_lambda_fn: Optional fused epilogue. When set, each
            output fragment is handed to the lambda with its global
            `(m, n)` instead of being stored to `c`, which lets a caller
            route output bands elsewhere (e.g. scattering K/V into a
            paged KV cache). Requires `MMA_M == 16` and `num_splits == 1`.
    """

    comptime a_bits = Self.matrix_format.bits_per_element()
    comptime b_bits = Self.matrix_format.bits_per_element()
    comptime bits_per_element = Self.a_bits
    comptime A_BK_BYTES = (Self.BK_ELEMS * Self.a_bits) // 8
    comptime B_BK_BYTES = (Self.BK_ELEMS * Self.b_bits) // 8
    comptime BK_BYTES = Self.A_BK_BYTES
    comptime lane_bytes = (32 * Self.a_bits) // 8

    comptime num_warps_m = Self.BM // Self.WM
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_warps = Self.num_warps_m * Self.num_warps_n
    comptime num_threads = Self.num_warps * WARP_SIZE

    comptime num_m_mmas = Self.WM // Self.MMA_M
    comptime num_n_mmas = Self.WN // Self.MMA_N

    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE  # 4
    comptime a_packed_k_per_mma = (Self.MMA_K * Self.a_bits) // 8
    comptime b_packed_k_per_mma = (Self.MMA_K * Self.b_bits) // 8
    comptime packed_k_per_mma = Self.a_packed_k_per_mma
    comptime num_k_tiles = Self.A_BK_BYTES // Self.a_packed_k_per_mma

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
    @__name(
        t"mx_dense_lb{Self.lane_bytes}_BM{Self.BM}_BN{Self.BN}_WM{Self.WM}_WN{Self.WN}_BK{Self.BK_ELEMS}_N{b_layout.static_shape[0]}_KB{a_layout.static_shape[1]}_SK{num_splits}"
    )
    @staticmethod
    def run[
        out_dtype: DType,
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        sfa_layout: TensorLayout,
        sfb_layout: TensorLayout,
        num_splits: Int = 1,
    ](
        c: TileTensor[out_dtype, c_layout, MutAnyOrigin],
        a: TileTensor[.uint8, a_layout, ImmutAnyOrigin],
        b: TileTensor[.uint8, b_layout, ImmutAnyOrigin],
        sfa: TileTensor[.float8_e8m0fnu, sfa_layout, ImmutAnyOrigin],
        sfb: TileTensor[.float8_e8m0fnu, sfb_layout, ImmutAnyOrigin],
    ):
        """MXFP4 block-scaled GEMM kernel with SMEM pipeline.

        With `num_splits > 1` this is the inter-block split-K body: each
        `block_idx.z` slice accumulates one disjoint K-band into its own
        `[M, N]` region of a stacked `(num_splits * M, N)` float32
        workspace (`out_dtype` is float32 in that mode). A separate
        reduce kernel sums the `num_splits` partials and casts to the
        real output dtype. `num_splits == 1` is byte-identical to the
        no-split path (`split_id == 0`, full K range, zero offset).

        Parameters:
            out_dtype: Element type of the output tensor `c`; must be
                `float32` when `num_splits > 1`.
            c_layout: Compile-time layout of the output tensor `c`.
            a_layout: Compile-time layout of the A operand.
            b_layout: Compile-time layout of the B operand.
            sfa_layout: Compile-time layout of the A scales tensor `sfa`.
            sfb_layout: Compile-time layout of the B scales tensor `sfb`.
            num_splits: Number of disjoint K-bands the K dimension is
                partitioned into (defaults to 1, no split).

        Args:
            c: Output matrix `[M, N]` of dtype `out_dtype`; in split-K
                mode a stacked `(num_splits * M, N)` float32 workspace.
            a: Packed A operand `[M, K//2]` uint8, two MXFP4 nibbles
                per byte.
            b: Packed B operand `[N, K//2]` uint8, transposed with two
                MXFP4 nibbles per byte.
            sfa: A block scales `[M, K//32]` as `float8_e8m0fnu`, one
                scale per 32 MXFP4 elements.
            sfb: B block scales `[N, K//32]` as `float8_e8m0fnu`, one
                scale per 32 MXFP4 elements.
        """
        comptime A_BK_BYTES = Self.A_BK_BYTES
        comptime B_BK_BYTES = Self.B_BK_BYTES
        comptime BK_BYTES = A_BK_BYTES
        comptime num_m_mmas = Self.num_m_mmas
        comptime num_n_mmas = Self.num_n_mmas
        comptime c_frag_size = Self.c_frag_size
        comptime num_k_tiles = Self.num_k_tiles
        comptime simd_width = Self.simd_width
        comptime num_threads = Self.num_threads

        comptime A_K_BYTES = type_of(a).static_shape[1]
        comptime B_K_BYTES = type_of(b).static_shape[1]
        comptime K_BYTES = A_K_BYTES
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

        # === Split-K K-banding ===
        # The K dimension is partitioned into `num_splits` disjoint bands.
        # Split `split_id` covers BK-tiles [split_id*tiles_per_split,
        # (split_id+1)*tiles_per_split). For num_splits=1 this is the full
        # K range (split_id=0, tiles_per_split = K_BYTES // BK_BYTES).
        comptime K_per_split_bytes = K_BYTES // num_splits
        comptime assert (
            K_per_split_bytes * num_splits == K_BYTES
        ), "num_splits must evenly divide K (packed bytes)"
        comptime assert K_per_split_bytes % BK_BYTES == 0, (
            "K_BYTES // num_splits must be a multiple of BK_BYTES; otherwise"
            " the trailing K chunk of a split is silently skipped."
        )
        comptime tiles_per_split = K_per_split_bytes // BK_BYTES
        # split_id is the K-band index, read from grid_dim.z *only* in split-K
        # mode. With num_splits == 1 the split dimension does not exist, so we
        # must NOT read block_idx.z: callers that reuse this kernel as a device
        # function (e.g. the grouped/persistent matmuls) launch with
        # grid_dim.z = num_experts, where block_idx.z is the expert index, not
        # a split. Forcing split_id = 0 there keeps the K range full and the
        # output offset zero — byte-identical to the no-split path.
        var split_id = Int(block_idx.z) if num_splits > 1 else 0

        # Dynamic M for OOB bounds handling when M is not a multiple of Self.BM.
        var M = Int(a.dim[0]())

        var _warp_id = warp_id()
        var warp_m, warp_n = divmod(_warp_id, Self.num_warps_n)

        # === GMEM views ===
        var a_gmem = TileTensor(a.ptr.bitcast[UInt8](), a.layout)
        var b_gmem = TileTensor(b.ptr.bitcast[UInt8](), b.layout)

        # === SMEM tiles (row-major allocation; access is conditionally
        # XOR-swizzled per BlockScaledMmaOp.use_smem_swizzle) ===
        var a_smem = stack_allocation[DType.uint8, address_space=.SHARED](
            row_major[Self.BM, A_BK_BYTES]()
        )
        var b_smem = stack_allocation[DType.uint8, address_space=.SHARED](
            row_major[Self.BN, B_BK_BYTES]()
        )

        comptime scales_per_mma = Self.scales_per_mma
        var sfa_smem = stack_allocation[DType.uint8, address_space=.SHARED](
            row_major[Self.BM, scales_per_mma * num_k_tiles]()
        )
        var sfb_smem = stack_allocation[DType.uint8, address_space=.SHARED](
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
        comptime a_load_cols = A_BK_BYTES // simd_width
        comptime b_load_cols = B_BK_BYTES // simd_width
        comptime a_load_rows = _largest_divisor_at_most[Self.BM](
            num_threads // a_load_cols
        )
        comptime b_load_rows = _largest_divisor_at_most[Self.BN](
            num_threads // b_load_cols
        )
        comptime a_active_threads = a_load_rows * a_load_cols
        comptime b_active_threads = b_load_rows * b_load_cols
        comptime a_load_layout = row_major[a_load_rows, a_load_cols]()
        comptime b_load_layout = row_major[b_load_rows, b_load_cols]()
        comptime load_thread_rows = a_load_rows
        comptime load_layout = a_load_layout
        comptime a_loads_per_tile = Self.BM // a_load_rows
        comptime b_loads_per_tile = Self.BN // b_load_rows
        comptime a_reg_elems = Self.BM * A_BK_BYTES // a_active_threads
        comptime b_reg_elems = Self.BN * B_BK_BYTES // b_active_threads

        # Block-row tiles spanning the full K dimension for tile-based indexing.
        var a_blockrow = a_gmem.tile[Self.BM, A_K_BYTES](block_idx.y, 0)
        var b_blockrow = b_gmem.tile[Self.BN, B_K_BYTES](block_idx.x, 0)

        # Register buffers for DRAM loads (one per matrix). Sized per
        # matrix so BM != BN works.
        var a_load_reg = stack_allocation[DType.uint8, address_space=.LOCAL](
            row_major[1, a_reg_elems]()
        )
        var b_load_reg = stack_allocation[DType.uint8, address_space=.LOCAL](
            row_major[1, b_reg_elems]()
        )

        # RegTileLoader wraps each block-row in an AMD buffer resource
        # descriptor. `bounds_from=a_gmem` clamps OOB buffer_load_dwordx4
        # reads to zero at the hardware level (no fault, no garbage).
        var a_loader = RegTileLoader[.uint8, load_layout](
            a_blockrow,
            bounds_from=a_gmem,
        )
        var b_loader = RegTileLoader[.uint8, b_load_layout](
            b_blockrow,
            bounds_from=b_gmem,
        )

        # === MMA operator ===
        var mma_op = BlockScaledMmaOp[
            mma_shape=IndexList[3](Self.MMA_M, Self.MMA_N, Self.MMA_K),
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            num_k_tiles=num_k_tiles,
            matrix_format=Self.matrix_format,
        ]()

        # === Output writer ===
        # RegTileWriter casts from float32 accumulators to out_dtype.
        #
        # Split-K: `c` is a stacked `(num_splits * M, N)` workspace; this
        # split writes its partial into the `[M, N]` region starting at
        # element offset `split_id * M * N`. Building the writer over that
        # per-split view means make_amd_buffer_resource bounds the V# at
        # exactly one split's M*N extent, so OOB warp rows (rows >= M when
        # M is not BM-aligned) are hardware-clamped to this split's region
        # and never bleed into the next split. For num_splits==1 the offset
        # is 0 and `c_split` is byte-identical to `c`.
        var c_split = TileTensor(
            c.ptr + split_id * M * N, row_major((Int(M), Idx[N]))
        )
        var c_writer = RegTileWriter[
            out_dtype, Self.MMA_M, WARP_SIZE // Self.MMA_M
        ](c_split)

        # A fused epilogue has to see the FINAL value, so it cannot ride a
        # per-split partial; `RegTileEpilogue`'s contract sends that case to
        # the reduce kernel instead. The 32x32 MFMA's per-lane fragment is
        # spread over non-contiguous columns, which the chunked writer below
        # cannot address as one span.
        comptime if Bool(Self.elementwise_lambda_fn):
            comptime assert num_splits == 1, (
                "fused epilogue is unsupported with split-K: it would fire on"
                " each K partial rather than the reduced output"
            )
            comptime assert Self.MMA_M == 16, (
                "fused epilogue requires the 16x16 MFMA; the 32x32 fragment is"
                " column-strided per lane"
            )
        var c_epilogue = RegTileEpilogue[
            out_dtype,
            Self.c_frag_size,
            elementwise_lambda_fn=Self.elementwise_lambda_fn,
        ](c_split)

        # === Pipeline helpers ===
        # Both counters start at this split's first BK-tile. The DRAM
        # loaders index `a_blockrow.tile[BM, BK_BYTES](0, k_counter)`, so
        # this offset selects the split's K-slice; `load_scales_to_smem`
        # mirrors it via `k_scale_counter * scales_per_mma * num_k_tiles`.
        var k_counter = split_id * tiles_per_split
        var k_scale_counter = split_id * tiles_per_split

        @always_inline
        @__parameter
        def load_tiles_from_dram():
            """Load one BK-wide tile from DRAM to register buffers."""
            var a_block = a_blockrow.tile[Self.BM, A_BK_BYTES](0, k_counter)
            var b_block = b_blockrow.tile[Self.BN, B_BK_BYTES](0, k_counter)
            a_loader.load(a_load_reg, a_block.vectorize[1, simd_width]())
            b_loader.load(b_load_reg, b_block.vectorize[1, simd_width]())
            k_counter += 1

        @always_inline
        @__parameter
        def copy_tiles_to_smem():
            """Copy register buffers to SMEM in row-major order.

            Computes each thread's flat `row*BK_BYTES + col_byte` in-tile
            byte offset explicitly (XOR-swizzled when `use_smem_swizzle`),
            matching the fragment read side's identical formula byte-for-byte
            — see `_swizzled_smem_off`. A and B derive theirs from their own
            grid and row width, which diverge once the two operands carry
            different encodings.

            A thread outside an operand's grid holds undefined register
            contents, since its `RegTileLoader` never wrote them; the grids
            only fall short of `num_threads` when a row count had to be
            snapped down to divide the tile height.
            """
            comptime MmaOpT = type_of(mma_op)
            var tid = Int(thread_idx.x)

            @always_inline
            @__parameter
            def store_a():
                var a_row = tid // a_load_cols
                var a_col_byte = (tid % a_load_cols) * simd_width
                comptime for v in range(a_loads_per_tile):
                    var off = _swizzled_smem_off[
                        A_BK_BYTES, MmaOpT.smem_swizzle
                    ](a_row + v * a_load_rows, a_col_byte)
                    a_smem.raw_store[width=simd_width](
                        off,
                        a_load_reg.raw_load[width=simd_width](v * simd_width),
                    )

            @always_inline
            @__parameter
            def store_b():
                var b_row = tid // b_load_cols
                var b_col_byte = (tid % b_load_cols) * simd_width
                comptime for v in range(b_loads_per_tile):
                    var off = _swizzled_smem_off[
                        B_BK_BYTES, MmaOpT.smem_swizzle
                    ](b_row + v * b_load_rows, b_col_byte)
                    b_smem.raw_store[width=simd_width](
                        off,
                        b_load_reg.raw_load[width=simd_width](v * simd_width),
                    )

            comptime if a_active_threads == num_threads:
                store_a()
            else:
                if tid < a_active_threads:
                    store_a()

            comptime if b_active_threads == num_threads:
                store_b()
            else:
                if tid < b_active_threads:
                    store_b()

        @always_inline
        @__parameter
        def load_scales_to_smem():
            """Cooperatively load scale tiles from GMEM to SMEM.

            Scale tile per BK iteration: [Self.BM, scales_per_mma] for A and
            [Self.BN, scales_per_mma] for B, both uint8. Each row is
            scales_per_mma * num_k_tiles bytes.
            Threads 0..BM-1 load A scales, threads Self.BM..BM+Self.BN-1 load B.
            Each active thread loads SCALE_WORDS_PER_ROW Int32 dwords per BK
            iteration, giving coalesced 4-byte aligned GMEM reads.
            """
            comptime SCALE_WORDS_PER_ROW = (scales_per_mma * num_k_tiles) // 4

            var tid = Int(thread_idx.x)
            var base_scale_k = k_scale_counter * scales_per_mma * num_k_tiles
            var a_base_row = Int(block_idx.y) * Self.BM
            var b_base_row = Int(block_idx.x) * Self.BN

            # A scales: guard M-OOB rows.
            if tid < Self.BM:
                var row = a_base_row + tid
                if row < M:
                    var src_word_base = (row * K_SCALES + base_scale_k) // 4
                    comptime for w in range(SCALE_WORDS_PER_ROW):
                        sfa_smem.ptr.bitcast[Int32]()[
                            tid * SCALE_WORDS_PER_ROW + w
                        ] = sfa.ptr.bitcast[Int32]()[src_word_base + w]
                else:
                    comptime for w in range(SCALE_WORDS_PER_ROW):
                        sfa_smem.ptr.bitcast[Int32]()[
                            tid * SCALE_WORDS_PER_ROW + w
                        ] = Int32(0)
            # B scales: guard N-OOB rows (B is transposed).
            if tid < Self.BN:
                var row = b_base_row + tid
                if row < N:
                    var src_word_base = (row * K_SCALES + base_scale_k) // 4
                    comptime for w in range(SCALE_WORDS_PER_ROW):
                        sfb_smem.ptr.bitcast[Int32]()[
                            tid * SCALE_WORDS_PER_ROW + w
                        ] = sfb.ptr.bitcast[Int32]()[src_word_base + w]
                else:
                    comptime for w in range(SCALE_WORDS_PER_ROW):
                        sfb_smem.ptr.bitcast[Int32]()[
                            tid * SCALE_WORDS_PER_ROW + w
                        ] = Int32(0)

            k_scale_counter += 1

        # === Schedule-driven pipeline ===
        # The schedule prologue pre-loads 2 tiles, so we need at least 2
        # K-iterations. For K with only 1 tile, fall back to a simple loop.
        comptime a_loads_per_thread = Self.BM // a_load_rows
        comptime b_loads_per_thread = Self.BN // b_load_rows

        @always_inline
        @__parameter
        def simple_k_loop():
            """Fallback for small K where schedule prologue doesn't fit."""
            for k_iter in range(tiles_per_split):
                load_tiles_from_dram()
                load_scales_to_smem()
                copy_tiles_to_smem()
                barrier()

                var a_warp = a_smem.tile[Self.WM, A_BK_BYTES](warp_m, 0)
                var b_warp = b_smem.tile[Self.WN, B_BK_BYTES](warp_n, 0)

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
        @__parameter
        def scheduled_k_loop():
            """Pipelined K-loop via build_default_matmul_schedule."""
            comptime schedule = build_default_matmul_schedule[
                num_k_tiles=num_k_tiles,
                num_m_mmas=num_m_mmas,
                num_n_mmas=num_n_mmas,
                num_k_mmas=num_k_tiles,
                MMA_M=Self.MMA_M,
                MMA_N=Self.MMA_N,
                a_loads_per_thread=a_loads_per_thread,
                b_loads_per_thread=b_loads_per_thread,
            ]()

            @__parameter
            @always_inline
            def _bind[entry: ScheduleEntry]():
                comptime if entry.op.tag == LOAD_DRAM:
                    load_tiles_from_dram()
                elif entry.op.tag == STORE_SMEM:
                    copy_tiles_to_smem()
                    load_scales_to_smem()
                elif entry.op.tag == LOAD_FRAG:
                    comptime k = entry.op.subtile
                    var a_warp = a_smem.tile[Self.WM, A_BK_BYTES](warp_m, 0)
                    var b_warp = b_smem.tile[Self.WN, B_BK_BYTES](warp_n, 0)
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

            # Main K-loop (bounded to this split's tile count).
            for _ in range(2, tiles_per_split):
                comptime for i in range(len(schedule.kernel)):
                    _bind[schedule.kernel[i]]()

            # Epilogue.
            comptime for i in range(len(schedule.epilogue)):
                _bind[schedule.epilogue[i]]()

        if tiles_per_split < 2:
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
        var c_block = c_split.tile[Self.BM, Self.BN](block_idx.y, block_idx.x)
        var c_warp = c_block.tile[Self.WM, Self.WN](warp_m, warp_n)

        # AMD buffer_store dispatches at most 16 bytes per lane. For 16x16
        # MFMA `c_frag_size == 4` and vectorize[1, 4] hits 16 bytes exactly.
        # For 32x32 `c_frag_size == 16` (16 FP32 per lane per MMA), so we
        # keep the vectorize at literal 4 and use `store[mfma32=True]`,
        # which iterates the source as 4 register groups of 4 floats and
        # reorders them via the CDNA 32x32 register permutation
        # (`src[4*n + 16*m]` → fragment position `4*m + n`).
        comptime if Bool(Self.elementwise_lambda_fn):
            # The lambda mode addresses the output by global (m, n) rather than
            # through the buffer resource, so it loses the V#'s row clamp and
            # has to gate `m` itself. For the 16x16 MFMA each lane owns
            # `c_frag_size` contiguous columns of one row: row = lane % MMA_M,
            # first column = (lane // MMA_M) * c_frag_size.
            var lane_group, thread_m = divmod(Int(lane_id()), Self.MMA_M)
            var m_warp_base = Int(block_idx.y) * Self.BM + Int(warp_m) * Self.WM
            var n_warp_base = Int(block_idx.x) * Self.BN + Int(warp_n) * Self.WN

            comptime for m_mma in range(num_m_mmas):
                var m_global = m_warp_base + m_mma * Self.MMA_M + Int(thread_m)
                if m_global < M:
                    comptime for n_mma in range(num_n_mmas):
                        var v = (
                            c_reg.tile[1, c_frag_size](m_mma, n_mma)
                            .raw_load[width=c_frag_size](0)
                            .cast[out_dtype]()
                        )
                        c_epilogue.store(
                            v,
                            m=m_global,
                            n=n_warp_base
                            + n_mma * Self.MMA_N
                            + Int(lane_group) * c_frag_size,
                        )
        else:
            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    c_writer.store[mfma32=Self.MMA_M == 32](
                        c_warp.tile[Self.MMA_M, Self.MMA_N](
                            m_mma, n_mma
                        ).vectorize[1, 4](),
                        c_reg.tile[1, c_frag_size](m_mma, n_mma),
                    )


# ===----------------------------------------------------------------------=== #
# Public entry point
# ===----------------------------------------------------------------------=== #


def _launch_block_scaled[
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int = 16,
    MMA_N: Int = 16,
    MMA_K: Int = 128,
    matrix_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    M: Int,
    ctx: DeviceContext,
) raises:
    """Instantiate BlockScaledMatmulAMD with the given tile shape and launch."""
    comptime Kernel = BlockScaledMatmulAMD[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        MMA_M=MMA_M,
        MMA_N=MMA_N,
        MMA_K=MMA_K,
        matrix_format=matrix_format,
        elementwise_lambda_fn=elementwise_lambda_fn,
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


def _launch_block_scaled_split_k[
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    WM: Int,
    WN: Int,
    num_splits: Int,
    MMA_M: Int = 16,
    MMA_N: Int = 16,
    MMA_K: Int = 128,
    lane_bytes: Int = 16,
    matrix_format: CDNA4F8F6F4MatrixFormat = (
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1 if lane_bytes
        == 16 else CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3
    ),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    M: Int,
    ctx: DeviceContext,
) raises:
    """Inter-block split-K launch of BlockScaledMatmulAMD + reduce.

    Mirrors `amd_4wave_split_k_matmul`: allocate a `(num_splits * M, N)`
    float32 workspace, launch the matmul over a `grid_dim.z = num_splits`
    grid (each z-slice accumulates one K-band's partial into its `[M, N]`
    region), then run `_split_k_reduce_kernel` on the same stream to sum
    the partials and cast to `c`'s dtype.

    `elementwise_lambda_fn` rides the REDUCE so it fires once on the summed
    value: per-split firing would be wrong arithmetic and would repeat the
    fused QKV+index consumer's KV-cache scatter. The reduce then routes every
    element through the lambda and never stores to `c`.
    """
    # The kernel below is instantiated at the default MFMA shape; refuse any
    # other request rather than silently downgrading it to 16x16x128.
    comptime assert MMA_M == 16 and MMA_N == 16 and MMA_K == 128, (
        "split-K only instantiates the 16x16x128 MFMA; route a different MFMA"
        " shape to the non-split launch instead"
    )
    comptime Kernel = BlockScaledMatmulAMD[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        matrix_format=matrix_format,
    ]
    comptime N = type_of(c).static_shape[1]
    comptime c_dtype = type_of(c).dtype

    # Without the lambda the matmul stores the f32 workspace through
    # `RegTileWriter`, which spills an unaligned last column block into the next
    # row (see `_sk_n_aligned`); this backstops a direct caller.
    comptime assert (
        N % BN == 0
    ), "split-K needs N % BN == 0; `RegTileWriter` spills across rows otherwise"

    var elems_per_split = M * N
    var workspace = SplitKWorkspace[num_splits](ctx, elems_per_split)

    # Stacked (num_splits * M, N) row-major float32 workspace. The kernel
    # offsets into split `split_id`'s [M, N] region at element
    # `split_id * M * N`, which is byte-identical to a (num_splits, M, N)
    # buffer — exactly the layout `_split_k_reduce_kernel` expects.
    var ws_tile = TileTensor(
        workspace.scratch.unsafe_ptr(),
        row_major((Int(num_splits * M), Idx[N])),
    )

    comptime kernel = Kernel.run[
        .float32,
        type_of(ws_tile).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scales).LayoutType,
        type_of(b_scales).LayoutType,
        num_splits=num_splits,
    ]

    ctx.enqueue_function[kernel](
        ws_tile,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), num_splits),
        block_dim=Kernel.num_threads,
    )

    # Reduce + cast on the same stream — naturally serialized after the
    # matmul launch. Sums the `num_splits` f32 partials at flat index
    # `tid` and casts to c's dtype.
    comptime block_dim_x: Int = 256
    var total_elems = M * N
    var num_blocks = ceildiv(total_elems, block_dim_x)
    comptime reduce_kernel = _split_k_reduce_kernel[
        num_splits, c_dtype, elementwise_lambda_fn
    ]
    ctx.enqueue_function[reduce_kernel](
        workspace.scratch.unsafe_ptr(),
        c.ptr,
        Int32(total_elems),
        Int32(elems_per_split),
        Int32(N),
        grid_dim=num_blocks,
        block_dim=block_dim_x,
    )

    # Keep the workspace alive until both kernels are enqueued.
    _ = workspace^


def _pick_num_splits[
    K_BYTES: Int, N: Int, BN: Int, BK_BYTES: Int, cta_cap: Int
]() -> Int:
    """Comptime split-K factor for the small-M decode regime.

    Picks the largest `num_splits` such that the split is legal AND the
    resulting CTA count `ceildiv(N, BN) * num_splits` stays under
    `cta_cap`. Legality (mirrors `BlockScaledMatmulAMD.run`'s split-K asserts):
      * `K_BYTES % num_splits == 0`, and
      * `(K_BYTES // num_splits) % BK_BYTES == 0`
    i.e. `num_splits` divides `K_BYTES // BK_BYTES`. Additionally each
    split must own at least 2 BK-tiles (`K_BYTES // num_splits >=
    2*BK_BYTES`): with only 1 tile per split the kernel falls back to the
    non-pipelined `simple_k_loop`, and the separate reduce launch over
    the full `[M, N]` output then dominates the (now tiny) per-split
    matmul, which regresses small-K shapes (e.g. down-proj K=2048). The
    2-tile floor confines split-K to the regime where it actually wins.
    Returns 1 if no split qualifies (caller takes the plain single-launch
    path).

    With M fitting in a single M-tile, total CTAs ≈ ceildiv(N, BN) *
    num_splits, so this targets enough WGs to saturate ~256 CUs.
    """
    comptime total_tiles = K_BYTES // BK_BYTES
    comptime n_blocks = ceildiv(N, BN)
    var best = 1
    comptime for s in range(2, total_tiles + 1):
        comptime if (
            K_BYTES % s == 0
            and (K_BYTES // s) % BK_BYTES == 0
            and (K_BYTES // s) >= 2 * BK_BYTES
            and n_blocks * s <= cta_cap
        ):
            best = s
    return best


def block_scaled_matmul_amd[
    MMA_M: Int = 16,
    MMA_N: Int = 16,
    MMA_K: Int = 128,
    lane_bytes: Int = 16,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, .uint8, ...],
    b: TileTensor[mut=False, .uint8, ...],
    a_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    b_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    ctx: DeviceContext,
) raises:
    """Launch native MXFP4 block-scaled matmul on AMD CDNA4.

    Uses cdna4_block_scaled_mfma with FLOAT4_E2M1 directly: no
    dequantization to FP8. Both A and B must be packed uint8 with
    E8M0 scaling factors. Accumulates in float32, casts to c.dtype
    during the store epilogue.

    Tile shape is selected at runtime based on M to match the expected
    arithmetic intensity regime (decode vs. prefill). A proper
    (N, K)-keyed dispatch table is planned for a follow-up PR.

    Parameters:
        MMA_M: MFMA tile rows. Default 16. Pass 32 to opt into the
            32x32x64 MFMA shape (must be paired with MMA_N=32, MMA_K=64).
            Non-split branches only -- split-K and the narrow wide-N tile
            instantiate 16x16x128, so a 32x32x64 request routes past them.
        MMA_N: MFMA tile cols. Default 16.
        MMA_K: MFMA K-depth in logical FP4 elements. Default 128.
        lane_bytes: Operand bytes per lane per MFMA: 16 (MXFP4, default) or
            32 (MXFP8), which selects the operand format and byte extents.
        elementwise_lambda_fn: Optional fused epilogue receiving each output
            fragment with its global `(m, n)`; `c` is left untouched. Split-K
            carries it too, on the reduce — one scalar per cell rather than a
            fragment, still exactly once per cell.

    Args:
        c: Output [M, N] (any float dtype, e.g. float32 or bfloat16).
        a: Packed A [M, K//2] uint8 (MXFP4) or [M, K] uint8 (MXFP8).
        b: Packed B [N, K//2] uint8 (transposed, MXFP4) or [N, K] (MXFP8).
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
    # else BlockScaledMatmulAMD's comptime assert fires. Gate each bucket so
    # a small-K caller (e.g. tests with K=128) falls back to the safe
    # default instead of hitting a build error.
    comptime K_BYTES = type_of(a).static_shape[1]
    # The buckets are named in ELEMENTS but the divisibility they need is in
    # BYTES, and the two only coincide at MXFP4. Convert, or the gate is off by
    # 2x at MXFP8 and admits a K it cannot tile.
    comptime assert (
        lane_bytes == 16 or lane_bytes == 32
    ), "lane_bytes must be 16 (MXFP4) or 32 (MXFP8); FP6 has its own entry"
    comptime _fmt = (
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1 if lane_bytes
        == 16 else CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3
    )
    comptime _elems_per_byte = 32 // lane_bytes
    comptime _bk_256_bytes = 256 // _elems_per_byte
    comptime _bk_512_bytes = 512 // _elems_per_byte
    comptime can_use_bk_256 = (
        K_BYTES >= _bk_256_bytes and K_BYTES % _bk_256_bytes == 0
    )
    comptime can_use_bk_512 = (
        K_BYTES >= _bk_512_bytes and K_BYTES % _bk_512_bytes == 0
    )

    var M = Int(c.dim[0]())
    comptime N = type_of(c).static_shape[1]

    if M == 0 or N == 0:
        return

    # Split-K small-M config. At small M the whole problem is one M-tile, so the
    # plain kernel launches only ceildiv(N, BN) CTAs and starves the GPU;
    # splitting K into `_sk_splits` disjoint bands multiplies the CTA count up
    # toward `_sk_cta_cap`. `_sk_splits == 1` means no split qualified.
    # The cap is sm_count * 2 — every CU plus a second wave for latency hiding —
    # i.e. 512 on MI355X, the value the split factors were tuned at.
    comptime _gpu = ctx.default_device_info
    comptime SK_CTA_WAVES = 2
    comptime _sk_cta_cap = _gpu.sm_count * SK_CTA_WAVES
    comptime SK_BM = 64
    comptime SK_BN = 128
    comptime SK_BK_ELEMS = 256
    # Bytes, not elements -- see _bk_256_bytes above.
    comptime SK_BK_BYTES = SK_BK_ELEMS // _elems_per_byte
    comptime SK_WM = 64
    comptime SK_WN = 32
    comptime _sk_splits = _pick_num_splits[
        K_BYTES, N, SK_BN, SK_BK_BYTES, cta_cap=_sk_cta_cap
    ]()

    # The f32 workspace grows with M while the benefit does not, so a byte
    # budget on it is an M ceiling (everything but M is comptime). 128 MiB stays
    # under the 205 MB a `mojo_test` gets from the MAX memory manager and still
    # admits M <= 1092 on M3's QKV+index GEMM (N=2560, 12 splits).
    comptime SK_MAX_WORKSPACE_BYTES = 128 * 1024 * 1024
    comptime _sk_max_m = SK_MAX_WORKSPACE_BYTES // (
        _sk_splits * N * size_of[DType.float32]()
    )

    # Narrow-M split-K tile (M <= 16): BM=16 wastes no M rows, where BM=64 would
    # load and run MFMA on 48-63 OOB-zero rows. The DRAM→SMEM loader requires
    # load_thread_rows = num_threads / (BK_BYTES/simd_width) <= BM, else
    # a_loads_per_tile == 0 and the A tile is never loaded — at BK_BYTES=128,
    # simd_width=16 that caps num_threads at 8*BM = 128, so this has to be the
    # 2-warp WN=64 shape, not the 4-warp WN=32 one the BM=64 tile uses.
    comptime SK16_BM = 16
    comptime SK16_BN = 128
    comptime SK16_WM = 16
    comptime SK16_WN = 64

    # `RegTileWriter`'s buffer-resource OOB clamp (`amd_tile_io.mojo`) bounds a
    # store by the destination's total byte extent, not per row: at
    # `N % BN != 0` the workspace's last column block spills into the next row
    # instead of being clipped, and the reduce sums the corruption. The
    # non-split branches use the column-gated `RegTileEpilogue` instead.
    comptime _sk_n_aligned = N % SK_BN == 0 and N % SK16_BN == 0

    # BM=WM=16 leaves `num_m_mmas = 16//32 = 0` at MMA_M=32, so the split tiles
    # and the narrow wide-N tile below can only be the default MFMA; route a
    # 32x32x64 request past them rather than downgrading it silently.
    comptime _mma_is_default = MMA_M == 16 and MMA_N == 16 and MMA_K == 128

    # Wide-N short-K decode gate (e.g. down-proj N=16384, K<=3072). For wide
    # N the plain launch already yields ceildiv(N, BN) CTAs that fill the GPU,
    # and split-K's reduce kernel cost scales with the large output M*N —
    # measured ~41% of total for N=16384,K=2048 (matmul 6.6us + reduce 4.7us),
    # while at num_splits=4 the steady-state K-loop is empty. A single small-BN
    # kernel keeps the CTA count high WITHOUT the reduce tax and matches/beats
    # aiter (whose autotuned config sets NUM_KSPLIT=1 for this exact shape).
    # Two conditions:
    #   * wide N: ceildiv(N, 32) >= sm_count, so BN=32 alone gives >=1 CTA/CU
    #     (down-proj N=16384 -> 512 CTAs; up-proj N=2304 -> 72 and Kimi N=4096
    #     -> 128 fall through to split-K, which is correct for their narrow N /
    #     long K, matching aiter NUM_KSPLIT>1).
    #   * short K: K_BYTES <= 1536 (K <= 3072 FP4 elems). At larger K each CTA's
    #     full-K loop becomes latency-bound under the single-buffer pipeline and
    #     split-K (shorter per-CTA loop) wins instead — measured crossover sits
    #     between K=3072 (single ~= split) and K=4096 (split wins).
    # Measured: single BN=32 closes the down-proj gap vs aiter from +25..36%
    # (split-K) to +2..8% at K=2048, and is faster than aiter at K=2560.
    # Compare in ELEMENTS: the crossover above was measured in FP4 elements, so
    # a byte comparison would halve the threshold at MXFP8. The crossover
    # itself has not been re-measured for MXFP8; it is reused as-is.
    comptime _wide_n_short_k_decode = (
        ceildiv(N, 32) >= _gpu.sm_count
        and K_BYTES * _elems_per_byte <= 3072
        and can_use_bk_512
    )

    # Runtime M-bucket dispatch. Tile shapes tuned for Kimi K2.5 on MI355.
    #   M <=  16  → decode → single small-BN kernel for the wide-N short-K
    #               regime, else narrow split-K (BM=16, no wasted M rows)
    #   M >   16  → BM=64 split-K when `_sk_splits` found a legal factor and
    #               the workspace fits (`M <= _sk_max_m`). `_sk_splits` is
    #               keyed on N/K/cta_cap, not M, so a shape that's still
    #               CTA-starved at large M keeps benefiting there too. The
    #               BK512 fallback stays capped at M<=64 — see that branch.
    if M <= 16:
        comptime if _wide_n_short_k_decode and _mma_is_default:
            # Single kernel, no split-K, no reduce. BN=32 → ceildiv(N,32) CTAs
            # fill the GPU; BM=16 wastes no M rows. Mirrors aiter NUM_KSPLIT=1.
            _launch_block_scaled[
                BM=16,
                BN=32,
                BK_ELEMS=512,
                WM=16,
                WN=16,
                matrix_format=_fmt,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b, a_scales, b_scales, M, ctx)
        elif (
            can_use_bk_256
            and _sk_splits > 1
            and _sk_n_aligned
            and _mma_is_default
        ):
            _launch_block_scaled_split_k[
                BM=SK16_BM,
                BN=SK16_BN,
                BK_ELEMS=SK_BK_ELEMS,
                WM=SK16_WM,
                WN=SK16_WN,
                num_splits=_sk_splits,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                MMA_K=MMA_K,
                matrix_format=_fmt,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b, a_scales, b_scales, M, ctx)
        elif can_use_bk_512:
            _launch_block_scaled[
                BM=64,
                BN=32,
                BK_ELEMS=512,
                WM=64,
                WN=32,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                MMA_K=MMA_K,
                matrix_format=_fmt,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b, a_scales, b_scales, M, ctx)
        else:
            _launch_block_scaled[
                BM=128,
                BN=128,
                BK_ELEMS=128,
                WM=64,
                WN=64,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                MMA_K=MMA_K,
                matrix_format=_fmt,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b, a_scales, b_scales, M, ctx)
    else:
        comptime if (
            can_use_bk_256
            and _sk_splits > 1
            and _sk_n_aligned
            and _mma_is_default
        ):
            # Past `_sk_max_m` the scratch outweighs the split, and the plain
            # BM=128 tile already has the parallelism it was buying.
            if M <= _sk_max_m:
                _launch_block_scaled_split_k[
                    BM=SK_BM,
                    BN=SK_BN,
                    BK_ELEMS=SK_BK_ELEMS,
                    WM=SK_WM,
                    WN=SK_WN,
                    num_splits=_sk_splits,
                    MMA_M=MMA_M,
                    MMA_N=MMA_N,
                    MMA_K=MMA_K,
                    matrix_format=_fmt,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b, a_scales, b_scales, M, ctx)
            else:
                _launch_block_scaled[
                    BM=128,
                    BN=128,
                    BK_ELEMS=128,
                    WM=64,
                    WN=64,
                    MMA_M=MMA_M,
                    MMA_N=MMA_N,
                    MMA_K=MMA_K,
                    matrix_format=_fmt,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b, a_scales, b_scales, M, ctx)
        elif can_use_bk_512:
            # Non-split BK_ELEMS=512 fallback, M<=64 only (checked below at
            # runtime — `M` isn't comptime-known here). Reached when
            # can_use_bk_256 holds, can_use_bk_512 holds, and _sk_splits == 1
            # — i.e. split-K found no legal factor. Since the split requires
            # >=2 BK256-tiles per split (BK_BYTES=128), the only K that lands
            # here at small M is K_BYTES=256 (K=512 FP4 elems): 2 total
            # K-tiles, so s=2 gives 1 tile/split (below the floor) and s>2
            # doesn't divide. A BK512 split is no better — at K_BYTES=256
            # there is exactly 1 BK512-tile, so no split is legal there
            # either. There is simply nothing to split at K=512, so the
            # non-split BK512 tile is correct for this tiny-K corner (rare in
            # production: Kimi up=7168, down=2048).
            #
            # Not extended past M=64 like the split-K branch above: split-K's
            # legality is self-limiting by cta_cap, but this fallback isn't —
            # at large M with a wide-N shape where split-K legitimately found
            # no factor, this small decode-tuned tile is a worse choice than
            # the general BM=128 kernel below, not a substitute for it.
            if M <= 64:
                _launch_block_scaled[
                    BM=64,
                    BN=32,
                    BK_ELEMS=512,
                    WM=64,
                    WN=32,
                    MMA_M=MMA_M,
                    MMA_N=MMA_N,
                    MMA_K=MMA_K,
                    matrix_format=_fmt,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b, a_scales, b_scales, M, ctx)
            else:
                _launch_block_scaled[
                    BM=128,
                    BN=128,
                    BK_ELEMS=128,
                    WM=64,
                    WN=64,
                    MMA_M=MMA_M,
                    MMA_N=MMA_N,
                    MMA_K=MMA_K,
                    matrix_format=_fmt,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b, a_scales, b_scales, M, ctx)
        else:
            _launch_block_scaled[
                BM=128,
                BN=128,
                BK_ELEMS=128,
                WM=64,
                WN=64,
                MMA_M=MMA_M,
                MMA_N=MMA_N,
                MMA_K=MMA_K,
                matrix_format=_fmt,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b, a_scales, b_scales, M, ctx)


@always_inline
def _largest_divisor_at_most[n: Int](cap: Int) -> Int:
    """Returns the largest divisor of `n` that is at most `cap`.

    Parameters:
        n: The value to divide, positive.

    Args:
        cap: Upper bound on the returned divisor, positive.

    Returns:
        The largest `d` with `n % d == 0` and `d <= cap`.
    """
    var d = min(cap, n)
    while n % d != 0:
        d -= 1
    return d


def mxfp6_block_scaled_matmul_amd[
    fp6_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3,
    num_splits: Int = 1,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, .uint8, ...],
    b: TileTensor[mut=False, .uint8, ...],
    a_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    b_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    ctx: DeviceContext,
) raises:
    """Launch native MXFP6 block-scaled matmul on AMD CDNA4.

    Feeds FP6 operands straight to `cdna4_block_scaled_mfma` with no
    dequantization, reusing `BlockScaledMatmulAMD`'s pipeline via its
    `matrix_format` parameter. Accumulates in float32 and casts to `c.dtype`
    in the store epilogue.

    Single tile shape for now, chosen by arithmetic rather than tuning. FP6's
    `BK_BYTES` is a multiple of 3 (96 bytes at `BK_ELEMS=128`), so the
    DRAM->SMEM thread grid needs a thread count divisible by 3 -- hence the
    3-warp `BM=96, WM=32` shape rather than the usual power-of-two tile. A
    tuned (N, K)-keyed dispatch, the preshuffle path, and split-K are all
    follow-up work.

    Parameters:
        fp6_format: FP6 encoding of both operands (E2M3 or E3M2).
        num_splits: Inter-block split-K factor. `1` (default) is the plain
            launch; `> 1` partitions K into that many disjoint bands, each
            accumulating into its own float32 workspace slice, then reduces.
            Requires `K_BYTES % num_splits == 0` and
            `(K_BYTES / num_splits) % BK_BYTES == 0`; `BlockScaledMatmulAMD.run`
            asserts both.

    Args:
        c: Output `[M, N]` (any float dtype).
        a: Packed A `[M, K * 6 // 8]` uint8, four FP6 codes per three bytes.
        b: Packed B `[N, K * 6 // 8]` uint8, transposed, same packing.
        a_scales: A scales `[M, K // 32]` float8_e8m0fnu.
        b_scales: B scales `[N, K // 32]` float8_e8m0fnu.
        ctx: Device context for kernel launch.
    """
    comptime assert (
        fp6_format.bits_per_element() == 6
    ), "mxfp6_block_scaled_matmul_amd requires an FP6 matrix format"
    comptime BM = 96
    comptime BN = 64
    comptime BK_ELEMS = 128
    comptime WM = 32
    comptime WN = 64

    comptime assert type_of(c).static_shape[1] % BN == 0, (
        "N must be a multiple of BN=64 for the MXFP6 block-scaled matmul; a"
        " column overhang wraps into the next row and corrupts it"
    )

    comptime Kernel = BlockScaledMatmulAMD[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        matrix_format=fp6_format,
    ]

    var M = Int(c.dim[0]())
    comptime N = type_of(c).static_shape[1]

    if M == 0 or N == 0:
        return

    comptime if num_splits > 1:
        _launch_block_scaled_split_k[
            BM=BM,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WM=WM,
            WN=WN,
            num_splits=num_splits,
            matrix_format=fp6_format,
        ](c, a, b, a_scales, b_scales, M, ctx)
        return

    comptime kernel = Kernel.run[
        type_of(c).dtype,
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
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=Kernel.num_threads,
    )
