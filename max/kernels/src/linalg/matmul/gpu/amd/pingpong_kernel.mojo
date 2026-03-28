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
from std.math import ceildiv
from std.bit import log2_floor
from std.sys import simd_width_of, size_of

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx_int as block_idx,
    lane_id_int as lane_id,
    thread_idx_int as thread_idx,
    warp_id_uint as warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.intrinsics import AMDBufferResource
from std.gpu.memory import AddressSpace
from std.gpu.compute.mma import mma
from std.sys import llvm_intrinsic
from std.gpu.sync import schedule_barrier, s_waitcnt
from std.memory.unsafe import bitcast

from std.utils import Index, IndexList, StaticTuple
from std.utils.numerics import get_accum_type

from std.sys.intrinsics import readfirstlane, llvm_intrinsic

from std.gpu._utils import to_i64


from ....structuring import SMemTile, RegTile
from ....utils import elementwise_epilogue_type
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout
from layout.swizzle import Swizzle
from layout._utils import make_amd_buffer_resource
from .matmul import write_output_fragments
from pipeline.config import ScheduleConfig, SchedulingStrategy
from pipeline.pipeline_dsl import ScheduleEntry
from pipeline.program_builder import derive_safe_max_globals
from .amd_target import mi355x_target
from .pingpong_schedule import (
    build_schedule,
    PingPongOps,
    LOAD_A,
    LOAD_B,
    MMA_LOAD_A,
    MMA_LOAD_B,
    MMA,
)


# =============================================================================
# AMD Ping-Pong Matmul Kernel for MI355X (CDNA4)
# =============================================================================
#
# High-performance matrix multiplication using 8-warp double-buffered scheduling.
#
# Performance (8K×8K×8K on MI355X):
#   - BF16: ~1.44 TFLOPS (16×16×32 MMA, BK=64)
#   - FP8:  ~2.85 TFLOPS (32×32×64 MMA, BK=128, split-K)
#
# Supported MMA Configurations:
#   - BF16: 16×16×32  (BK=64,  interleaved LDS layout)
#   - FP8:  32×32×64  (BK=128, row-major LDS, 2 MMAs per tile) - default
#   - FP8:  16×16×128 (BK=128, row-major LDS) - fallback for M % 32 != 0
#
# Key Features:
#   - Supports arbitrary M dimensions (partial blocks handled correctly)
#   - Row-major LDS for FP8 enables correct OOB handling
#   - AMDBufferResource provides automatic OOB → zero clamping
#
# Components:
#   - TileLoaderLDS: Global→LDS cooperative loading with swizzle
#   - MmaOp: LDS→register loading and MMA execution
#   - TileBuffers: Double-buffered LDS tile management
#   - AMDPingPongMatmul: Kernel orchestration and scheduling
#
# =============================================================================


def make_mma_swizzle[dtype: DType, MMA_M: Int, MMA_K: Int]() -> Swizzle:
    """Create swizzle pattern for MMA LDS access.

    AMD MI355X have 64 LDS banks × 4 bytes each. Without swizzling,
    the MMA thread access pattern causes 4-way bank conflicts. The swizzle
    XORs high-order address bits into the bank selection bits to distribute
    accesses across banks.

    Swizzle parameters:
    - log_tile: Number of bits to XOR, scales with MMA_K
    - base: Log2 of read granularity in bytes (lds_frag_width * elem_size)
    - shift: Fixed at 4 for AMD LDS bank geometry

    Configuration examples:
        BF16 16×16×32:  lds_frag=8  bytes=16  → Swizzle(1, 4, 4)
        FP8  16×16×128: lds_frag=16 bytes=16  → Swizzle(3, 4, 4)
        FP8  32×32×64:  lds_frag=32 bytes=32  → Swizzle(2, 5, 4)

    Parameters:
        dtype: Element data type (affects byte size).
        MMA_M: M dimension of MMA instruction.
        MMA_K: K dimension of MMA instruction.

    Returns:
        Swizzle pattern for bank-conflict-free LDS access.
    """
    # Compute lds_frag_width (elements loaded per LDS read)
    comptime mma_frag_width = (MMA_M * MMA_K) // WARP_SIZE
    # FP8 16×16×128: each lane group holds two non-contiguous 16-element
    # K-chunks (see CDNA4 ISA §7.1.5.1 FP8 layout), so load 16 at a time.
    comptime use_split_k = (dtype.is_float8() and MMA_M == 16 and MMA_K == 128)
    comptime lds_frag_width = 16 if use_split_k else mma_frag_width

    # Compute swizzle parameters
    comptime log_tile = log2_floor(MMA_K // 32) + 1
    comptime frag_bytes = lds_frag_width * size_of[dtype]()
    comptime base = log2_floor(frag_bytes)

    return Swizzle(log_tile, base, 4)


# =============================================================================
# TileLoaderLDS: Cooperative Global→LDS Tile Loader
# =============================================================================


struct TileLoaderLDS[
    dtype: DType,
    src_layout: Layout,  # Full tensor layout (stride = shape[1])
    src_tile_layout: Layout,  # Tile shape for loading geometry
    num_loading_warps: Int,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
    load_width: Int = simd_width_of[dtype](),
    use_full_tile_width: Bool = False,  # FP8 row-major mode
](TrivialRegisterPassable):
    """Cooperative global→LDS tile loader with swizzle support.

    Loads tiles from global memory to LDS using AMDBufferResource which provides
    automatic out-of-bounds clamping to zero - critical for partial block support.

    Loading Modes (controlled by use_full_tile_width):
    - False (default): Interleaved layout. Each warp handles 32-col subtile.
      Used for BF16 where MMA_K (32) < BK (64).
    - True: Row-major layout. Each source row maps 1:1 to LDS row.
      Used for FP8 where MMA_K == BK, enabling correct partial block handling.
    """

    # =========================================================================
    # Tile Geometry (derived from src_tile_layout at compile time)
    # =========================================================================
    comptime tile_rows = Self.src_tile_layout.shape[0].value()
    comptime tile_cols = Self.src_tile_layout.shape[1].value()

    # =========================================================================
    # Thread Layout Geometry
    # =========================================================================
    # subtile_cols controls loading pattern:
    # - 32 (default): Interleaved for BF16 (MMA_K=32 < BK=64)
    # - tile_cols: Row-major for FP8 (MMA_K == BK), enables partial block OOB handling
    comptime subtile_cols = Self.tile_cols if Self.use_full_tile_width else 32
    comptime threads_per_row = Self.subtile_cols // Self.load_width
    comptime thread_rows = WARP_SIZE // Self.threads_per_row
    comptime threads_per_warp = WARP_SIZE  # 64

    # =========================================================================
    # Warp Grid Configuration (parameterized on tile_cols)
    # =========================================================================
    # Arrange warps in a 2D grid to cover the full tile width.
    # num_warp_cols = how many 32-col subtiles fit in tile_cols
    # num_warp_rows = remaining warps for row coverage
    #
    # Examples with 8 loading warps:
    #   tile_cols=64:  num_warp_cols=2, num_warp_rows=4 → 4×2 grid
    #   tile_cols=128: num_warp_cols=4, num_warp_rows=2 → 2×4 grid
    comptime num_warp_cols = Self.tile_cols // Self.subtile_cols
    comptime num_warp_rows = Self.num_loading_warps // Self.num_warp_cols

    # =========================================================================
    # Per-Warp Coverage
    # =========================================================================
    comptime elements_per_warp = Self.threads_per_warp * Self.load_width
    comptime rows_per_warp = Self.elements_per_warp // Self.tile_cols

    # Multi-warp cooperative loading geometry
    comptime loading_threads = Self.num_loading_warps * Self.threads_per_warp
    comptime loads_per_row = Self.tile_cols // Self.load_width
    comptime rows_per_iteration = Self.loading_threads // Self.loads_per_row
    comptime num_iterations = Self.tile_rows // Self.rows_per_iteration

    # Instance state
    var buffer: AMDBufferResource
    var thread_row: Int  # With per-warp swizzle applied if enabled
    var thread_col: Int
    var warp_id: Int
    var lane_id: Int  # Stored for per-iteration swizzle computation (FP8)

    # Stride derived from src_layout at compile time
    comptime stride = Self.src_layout.shape[1].value()

    # Byte-level constants for per-iteration swizzle (FP8 row-major path)
    comptime warp_subtile_bytes = Self.rows_per_warp * Self.tile_cols * size_of[
        Self.dtype
    ]()
    comptime lane_load_bytes = Self.load_width * size_of[Self.dtype]()
    comptime row_bytes = Self.tile_cols * size_of[Self.dtype]()

    # Whether swizzle needs per-iteration computation (vs pre-computed per-warp).
    #
    # XOR swizzle Swizzle(bits, base, shift) is translation-invariant over
    # warp subtiles of S bytes iff (base + shift + bits - 1) < log2(S).
    # BF16 Swizzle(1,5,4): top bit 9 < log2(1024)=10 -> invariant, pre-compute OK.
    # FP8  Swizzle(3,4,4): top bit 10 = log2(1024)=10 -> NOT invariant, per-iter needed.
    # FP8  Swizzle(2,5,4): top bit 10 = log2(1024)=10 -> NOT invariant, per-iter needed.
    #
    # use_full_tile_width is True exactly for FP8 row-major, which is the non-invariant case.
    comptime _needs_per_iter_swizzle = Bool(
        Self.swizzle
    ) and Self.use_full_tile_width

    @always_inline
    def __init__(
        out self,
        src: LayoutTensor,
        warp_id: Int,
        lane_id: Int,
    ):
        """Pre-compute thread position with swizzle inversion.

        For BF16 (interleaved layout), the per-warp swizzle inversion computed
        here is exact because the swizzle pattern is translation-invariant over
        warp subtile boundaries. For FP8 (row-major layout), per-iteration
        computation is used instead (see load_tile).
        """
        self.buffer = make_amd_buffer_resource(src)
        self.warp_id = warp_id
        self.lane_id = lane_id

        # Per-warp swizzle inversion: correct for translation-invariant swizzles (BF16).
        # For non-invariant swizzles (FP8), this is overridden per-iteration in load_tile.
        var effective_lane = lane_id

        comptime if Self.swizzle and not Self._needs_per_iter_swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        var warp_row, warp_col = divmod(warp_id, Self.num_warp_cols)
        var subtile_row, subtile_col_idx = divmod(
            effective_lane, Self.threads_per_row
        )
        var subtile_col = subtile_col_idx * Self.load_width

        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

    @always_inline
    def load_tile[
        dst_layout: Layout,
        //,
    ](
        self,
        dst: SMemTile[Self.dtype, dst_layout, ...],
        src_row: Int,
        src_col: Int,
    ):
        """Load a tile from source coordinates to LDS.

        Two paths depending on swizzle translation-invariance:

        1. Pre-computed path (BF16 / no swizzle): Uses thread_row/thread_col
           computed once in __init__. The BF16 Swizzle(1,5,4) is invariant
           over 1024-byte warp subtiles, so the per-warp approximation is exact.

        2. Per-iteration path (FP8 with swizzle): Computes the full byte offset
           within the half-tile each iteration. Required because FP8
           Swizzle(3,4,4) and Swizzle(2,5,4) have their top source bit at
           the subtile boundary (bit 10 = log2(1024)), breaking invariance.
        """
        comptime if Self._needs_per_iter_swizzle:
            var lane_byte = self.lane_id * Self.lane_load_bytes

            comptime for i in range(Self.num_iterations):
                var tile_idx = i * Self.num_loading_warps + self.warp_id
                var warp_subtile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                    tile_idx, 0
                )
                var smem_ptr = readfirstlane(warp_subtile.ptr)

                var full_byte = tile_idx * Self.warp_subtile_bytes + lane_byte
                var swizzled_byte = Self.swizzle.value()(full_byte)

                var swizzled_row = swizzled_byte // Self.row_bytes
                var swizzled_col = (swizzled_byte % Self.row_bytes) // size_of[
                    Self.dtype
                ]()

                var lane_offset = swizzled_col + swizzled_row * self.stride
                var uniform_offset = src_col + src_row * self.stride

                self.buffer.load_to_lds[width=Self.load_width](
                    Int32(lane_offset),
                    smem_ptr,
                    scalar_offset=Int32(uniform_offset),
                )
        else:
            var lane_offset = self.thread_col + self.thread_row * self.stride

            comptime for i in range(Self.num_iterations):
                var tile_idx = i * Self.num_loading_warps + self.warp_id
                var warp_subtile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                    tile_idx, 0
                )
                var smem_ptr = readfirstlane(warp_subtile.ptr)

                var tile_row = src_row + i * Self.rows_per_iteration
                var uniform_offset = src_col + tile_row * self.stride

                self.buffer.load_to_lds[width=Self.load_width](
                    Int32(lane_offset),
                    smem_ptr,
                    scalar_offset=Int32(uniform_offset),
                )


@always_inline
def _load_from_lds[
    dtype: DType,
    //,
    width: Int = 1,
](
    shared_ptr: UnsafePointer[
        Scalar[dtype],
        _,
        address_space=AddressSpace.SHARED,
    ],
) -> SIMD[dtype, width]:
    """Load a SIMD vector from LDS with LLVM alias scopes.

    Uses alias scopes to help LLVM distinguish LDS reads from async global→LDS
    copies, enabling better instruction scheduling and avoiding false dependencies.
    """
    comptime alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
    comptime no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`

    # Convert to LLVM address space 3 (shared memory)
    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](shared_ptr)

    # Compute alignment based on total load size
    comptime load_bytes = width * size_of[dtype]()
    comptime alignment = min(load_bytes, 16)  # Cap at 16-byte alignment

    # Generate the appropriate LLVM vector type and load
    # Using compile-time dispatch based on dtype and width
    comptime if dtype == DType.bfloat16 and width == 4:
        # BF16 x4 = 64 bits = ds_read_b64
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<4 x bf16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[DType.bfloat16, 4]._mlir_type
            ](llvm_res)
        )
    elif dtype == DType.bfloat16 and width == 8:
        # BF16 x8 = 128 bits = ds_read_b128
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x bf16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[DType.bfloat16, 8]._mlir_type
            ](llvm_res)
        )
    elif dtype.is_float8() and width == 8:
        # FP8 x8 = 64 bits = ds_read_b64
        # Load as i8 vector, then bitcast to fp8 (same bit pattern)
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 8]._mlir_type
        ](llvm_res)
        # Bitcast uint8 → float8_e4m3fn (same 8-bit representation)
        return bitcast[dtype, width](
            rebind[SIMD[DType.uint8, width]](uint8_vec)
        )
    elif dtype.is_float8() and width == 16:
        # FP8 x16 = 128 bits = ds_read_b128
        # Used for 16×16×128 MMA (HipKittens-style FP8 schedule)
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res)
        # Bitcast uint8 → float8_e4m3fn (same 8-bit representation)
        return bitcast[dtype, width](
            rebind[SIMD[DType.uint8, width]](uint8_vec)
        )
    elif dtype.is_float8() and width == 32:
        # FP8 x32 = 256 bits = 2 x ds_read_b128
        # Used for 32×32×64 MMA (mfma_scale_f32_32x32x64)
        # Load as two 128-bit chunks using pointer arithmetic
        var llvm_res0 = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        # Offset pointer by 16 bytes for second load
        var shared_ptr_offset = shared_ptr + 16
        var shared_ptr3_hi = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_type.`!llvm.ptr<3>`
        ](shared_ptr_offset)
        var llvm_res1 = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3_hi)
        var uint8_vec0 = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res0)
        var uint8_vec1 = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res1)
        var uint8_vec = rebind[SIMD[DType.uint8, 16]](uint8_vec0).join(
            rebind[SIMD[DType.uint8, 16]](uint8_vec1)
        )
        return bitcast[dtype, width](uint8_vec)
    else:
        comptime assert (
            False
        ), "Unsupported dtype/width combination for _load_from_lds"


@always_inline
def load_lds_fragment[
    dtype: DType,
    smem_layout: Layout,
    smem_element_layout: Layout,
    frag_layout: Layout,
    frag_element_layout: Layout,
    //,
    mma_access_layout: Layout,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](
    smem_tile: SMemTile[
        dtype, smem_layout, element_layout=smem_element_layout, ...
    ],
    reg_frag: RegTile[
        dtype, frag_layout, element_layout=frag_element_layout, ...
    ],
):
    """Load LDS → registers with MMA access pattern."""
    comptime num_iterations = frag_layout.size()
    comptime frag_width = frag_element_layout.size()
    comptime FragElement = SIMD[dtype, frag_width]

    # =========================================================================
    # Compile-time layout compatibility validation
    # =========================================================================

    # mma_access_layout maps lane_id (0..63) to LDS offsets within one iteration
    comptime assert mma_access_layout.size() == WARP_SIZE, String(
        "mma_access_layout must map exactly ",
        WARP_SIZE,
        " threads, got ",
        mma_access_layout.size(),
    )

    # smem must have enough elements for all iterations
    # Each iteration: WARP_SIZE threads × frag_width elements per thread
    comptime smem_elements = smem_layout.size() * smem_element_layout.size()
    comptime required_smem = num_iterations * WARP_SIZE * frag_width
    comptime assert smem_elements >= required_smem, String(
        "smem has ",
        smem_elements,
        " elements but loading requires ",
        required_smem,
        " (iterations=",
        num_iterations,
        " × WARP_SIZE=",
        WARP_SIZE,
        " × width=",
        frag_width,
        ")",
    )

    # frag must hold all loaded elements
    comptime frag_elements = frag_layout.size() * frag_element_layout.size()
    comptime required_frag = num_iterations * frag_width
    comptime assert frag_elements == required_frag, String(
        "frag has ",
        frag_elements,
        " elements but expects ",
        required_frag,
        " (iterations=",
        num_iterations,
        " × width=",
        frag_width,
        ")",
    )

    # =========================================================================

    var lane = lane_id()
    # RuntimeLayout wraps compile-time layout for efficient evaluation
    var lane_offset = Int(RuntimeLayout[mma_access_layout]()(lane))
    var frag_ptr = reg_frag.ptr.bitcast[FragElement]()

    comptime mma_m = Int(mma_access_layout.shape[0])
    comptime col_groups = Int(mma_access_layout.shape[1])
    comptime mma_k = Int(mma_access_layout.stride[0])
    comptime elements_per_iter = col_groups * frag_width
    comptime use_split_k = mma_k > elements_per_iter

    comptime if use_split_k:
        # Split-K pattern: multiple K-iterations per M-position
        comptime k_splits = mma_k // elements_per_iter  # 2 for 128/64
        comptime m_positions = num_iterations // k_splits
        comptime k_stride = elements_per_iter  # 64 elements between K halves
        comptime m_stride = mma_k * mma_m  # 128*16=2048 between M positions

        comptime for m_idx in range(m_positions):
            comptime for k_idx in range(k_splits):
                var iter_base = m_idx * m_stride + k_idx * k_stride
                var full_offset = iter_base + lane_offset

                comptime if swizzle:
                    full_offset = swizzle.value()(full_offset)

                comptime frag_idx = m_idx * k_splits + k_idx
                frag_ptr[frag_idx] = rebind[FragElement](
                    _load_from_lds[width=frag_width](
                        smem_tile.ptr + full_offset
                    )
                )
    else:
        # Standard iteration: iter_base = i * WARP_SIZE * frag_width
        comptime for i in range(num_iterations):
            var iter_base = i * WARP_SIZE * frag_width
            var full_offset = iter_base + lane_offset

            comptime if swizzle:
                full_offset = swizzle.value()(full_offset)

            frag_ptr[i] = rebind[FragElement](
                _load_from_lds[width=frag_width](smem_tile.ptr + full_offset)
            )


struct KernelConfig(ImplicitlyCopyable, Movable, Writable):
    var block_shape: IndexList[3]
    var warp_shape: IndexList[3]
    var mma_shape: IndexList[3]

    def __init__(
        out self,
        *,
        block_shape: IndexList[3],
        warp_shape: IndexList[3],
        mma_shape: IndexList[3],
    ):
        self.block_shape = block_shape
        self.warp_shape = warp_shape
        self.mma_shape = mma_shape

    @staticmethod
    def _write_index_list(
        mut writer: Some[Writer], list: IndexList, sep: StaticString
    ):
        comptime for i in range(list.size):
            if i != 0:
                writer.write(sep)
            writer.write(list[i])

    @always_inline
    def num_threads(self) -> Int:
        var num_warps = self.block_shape // self.warp_shape
        return num_warps.flattened_length() * WARP_SIZE

    def write_to(self, mut writer: Some[Writer]):
        writer.write("config_")
        Self._write_index_list(writer, self.block_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.warp_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.mma_shape, "x")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct MmaOp[
    in_type: DType,
    accum_type: DType,
    WM: Int,
    WN: Int,
    BK: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    alignment: Int,
    swizzle: Optional[Swizzle],  # Swizzle pattern (None if disabled)
]:
    """Encapsulates MMA register tiles and operations for matrix multiplication.

    This struct manages register tiles and MMA operations for a single warp.
    It processes warp-sized tiles (WM × BK for A, WN × BK for B) without
    knowledge of the broader kernel architecture.

    MmaOp accepts generic SMemTile and validates compatibility at
    compile-time via load_lds_fragment constraints.

    Note: Several values are derived from other parameters:
    - num_m_mmas = WM // MMA_M
    - num_n_mmas = WN // MMA_N
    - num_k_mmas = BK // MMA_K
    - load_width = simd_width_of[in_type]() (SIMD width for input type)
    - accum_width = (MMA_M * MMA_N) // WARP_SIZE (elements per thread)

    Quadrant Processing:
    The warp tile is divided into 4 quadrants for MMA scheduling:
    - quadrant_m_mmas = num_m_mmas // 2 (M-dimension quadrant size)
    - quadrant_n_mmas = num_n_mmas // 2 (N-dimension quadrant size)
    This enables efficient interleaving of loads and computes.

    Thread Layout for MMA:
    AMD's expected pattern: 64 threads → 4 rows × 16 cols (row-major)
    Lane offset computed on-the-fly via lane_id()

    Swizzle Configuration:
    MmaOp receives the swizzle pattern from the kernel/TileBuffers, since it's
    determined by how data is loaded into LDS. MmaOp must read using
    the same swizzle pattern that was used for writing.
    - BF16: Swizzle(1, 5, 4) - 1 bit XOR
    - FP8 16×128: Swizzle(3, 4, 4) - 3 bit XOR (HipKittens st_16x128)
    """

    # Derived values (computed from other parameters)
    comptime num_m_mmas = Self.WM // Self.MMA_M
    comptime num_n_mmas = Self.WN // Self.MMA_N
    comptime num_k_mmas = Self.BK // Self.MMA_K

    # =========================================================================
    # FP8 MMA Mode Detection
    # =========================================================================
    # FP8 supports two MMA configurations:
    # - 16×16×128: Highest throughput (4× K), 32-element LDS reads (2×ds_read_b128)
    # - 32×32×64:  Medium throughput (2× K), 32-element LDS reads (2×ds_read_b128)
    # Both use row-major LDS layout (MMA_K == BK) for correct partial blocks.
    # =========================================================================
    comptime use_fp8_16x16x128_mma = (
        Self.in_type.is_float8() and Self.MMA_M == 16 and Self.MMA_K == 128
    )
    comptime use_fp8_32x32x64_mma = (
        Self.in_type.is_float8() and Self.MMA_M == 32 and Self.MMA_K == 64
    )

    # Fragment widths:
    # - load_width: SIMD width for global→LDS (8 bf16, 16 fp8)
    # - lds_frag_width: Elements per LDS read (16 for FP8 16×16×128, else mma_frag_width)
    # - mma_frag_width: MMA hardware expectation = (MMA_M * MMA_K) // WARP_SIZE
    #   - BF16 16×16×32:  8 elements
    #   - FP8  16×16×128: 32 elements (loaded as 2×16 interleaved K-chunks)
    #   - FP8  32×32×64:  32 elements (single load)
    comptime load_width = simd_width_of[Self.in_type]()
    comptime mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime lds_frag_width = (
        16 if Self.use_fp8_16x16x128_mma else Self.mma_frag_width
    )
    comptime k_loads_per_mma = Self.mma_frag_width // Self.lds_frag_width
    comptime accum_width = (Self.MMA_M * Self.MMA_N) // WARP_SIZE

    # Quadrant dimensions (warp tile divided into 4 quadrants for MMA scheduling)
    comptime quadrant_m_mmas = Self.num_m_mmas // 2
    comptime quadrant_n_mmas = Self.num_n_mmas // 2

    # Swizzle pattern for LDS reads (must match TileBuffers write pattern)
    comptime elem_swizzle = Self.swizzle

    # LDS load counts for lgkmcnt tracking (ds_read ops per load_a/load_b call)
    # Each ds_read_b128 loads 16 bytes. For larger fragment sizes, multiple
    # ds_read operations are needed:
    #   - FP8 x8 (64 bits):  1 ds_read_b64  → 1 lgkm
    #   - FP8 x16 (128 bits): 1 ds_read_b128 → 1 lgkm
    #   - FP8 x32 (256 bits): 2 ds_read_b128 → 2 lgkm
    comptime bytes_per_frag = Self.lds_frag_width * size_of[Self.in_type]()
    comptime ds_reads_per_frag = ceildiv(
        Self.bytes_per_frag, 16
    )  # Max 16 bytes per ds_read
    comptime lgkm_per_load_a = Self.quadrant_m_mmas * Self.num_k_mmas * Self.k_loads_per_mma * Self.ds_reads_per_frag
    comptime lgkm_per_load_b = Self.quadrant_n_mmas * Self.num_k_mmas * Self.k_loads_per_mma * Self.ds_reads_per_frag
    comptime lgkm_per_load_ab = Self.lgkm_per_load_a + Self.lgkm_per_load_b  # Combined A+B load

    # Register tile type aliases - sized for MMA fragment width
    comptime RegTile[num_mmas: Int] = RegTile[
        Self.in_type,
        Layout.row_major(num_mmas, Self.num_k_mmas * Self.mma_frag_width),
        alignment=Self.alignment,
    ]
    comptime ARegTile = Self.RegTile[Self.num_m_mmas]
    comptime BRegTile = Self.RegTile[Self.num_n_mmas]

    # Output layout: Single contiguous register tile for all accumulators
    # Layout: (num_m_mmas, num_n_mmas * accum_width) = (8, 4*4) = (8, 16)
    # Quadrants are accessed via .tile[] views during MMA execution.
    # This enables reuse of write_output_fragments from matmul.mojo.
    comptime out_reg_layout = Layout.row_major(
        Self.num_m_mmas, Self.num_n_mmas * Self.accum_width
    )
    comptime OutRegTile = RegTile[
        Self.accum_type,
        Self.out_reg_layout,
        alignment=Self.alignment,
    ]

    # Quadrant dimensions for tile view access
    comptime quadrant_m_size = Self.quadrant_m_mmas
    comptime quadrant_n_size = Self.quadrant_n_mmas * Self.accum_width

    # MMA LDS access layout: maps 64 lanes to (MMA_M × MMA_K) subtile
    # Shape (MMA_M, col_groups) with strides (lds_row_stride, lds_frag_width)
    # RuntimeLayout enables compile-time offset computation
    #
    # Row stride depends on LDS layout mode:
    # - FP8 row-major: BK (full tile width, since MMA_K may be < BK)
    # - BF16 interleaved: MMA_K (each 32-col subtile is contiguous)
    # When lds_row_stride > MMA_K, load_lds_fragment's split-K path
    # automatically handles the K-dimension iteration.
    comptime col_groups = WARP_SIZE // Self.MMA_M
    comptime lds_row_stride = Self.BK if Self.in_type.is_float8() else Self.MMA_K
    comptime mma_access_layout = Layout(
        IntTuple(Self.MMA_M, Self.col_groups),
        IntTuple(Self.lds_row_stride, Self.lds_frag_width),
    )

    # Register tiles for A and B inputs
    var a_reg_tile: Self.ARegTile
    var b_reg_tile: Self.BRegTile

    # Single contiguous output register tile
    # Quadrants accessed via .tile[] views: out_reg_tile.tile[quadrant_m_size, quadrant_n_size](qa, qb)
    var out_reg_tile: Self.OutRegTile

    @always_inline
    def __init__(out self):
        """Initialize MMA operation with register tiles."""
        comptime assert Self.WM % Self.MMA_M == 0
        comptime assert Self.WN % Self.MMA_N == 0
        comptime assert Self.BK % Self.MMA_K == 0
        comptime assert (Self.MMA_M * Self.MMA_N) % WARP_SIZE == 0

        self.a_reg_tile = Self.ARegTile.stack_allocation()
        self.b_reg_tile = Self.BRegTile.stack_allocation()

        # Initialize contiguous output tile to zero
        self.out_reg_tile = Self.OutRegTile.stack_allocation().fill(0)

    @always_inline
    def reset_accumulator(self):
        """Reset output register tile to zero."""
        _ = self.out_reg_tile.fill(0)

    @always_inline
    def load_a[which: Int](self, smem_tile: SMemTile[Self.in_type, ...]):
        """Load A[which] from LDS → registers."""
        var smem_frag = smem_tile.vectorize[1, Self.lds_frag_width]()
        var reg_frag = self.a_reg_tile.tile[
            Self.quadrant_m_mmas, Self.num_k_mmas * Self.mma_frag_width
        ](which, 0).vectorize[1, Self.lds_frag_width]()
        load_lds_fragment[
            mma_access_layout=Self.mma_access_layout,
            swizzle=Self.elem_swizzle,
        ](smem_frag, reg_frag)

    @always_inline
    def load_b[which: Int](self, smem_tile: SMemTile[Self.in_type, ...]):
        """Load B[which] from LDS → registers."""
        var smem_frag = smem_tile.vectorize[1, Self.lds_frag_width]()
        var reg_frag = self.b_reg_tile.tile[
            Self.quadrant_n_mmas, Self.num_k_mmas * Self.mma_frag_width
        ](which, 0).vectorize[1, Self.lds_frag_width]()
        load_lds_fragment[
            mma_access_layout=Self.mma_access_layout,
            swizzle=Self.elem_swizzle,
        ](smem_frag, reg_frag)

    @always_inline
    def mma[which_a: Int, which_b: Int](self):
        """Execute MMA operations for a quadrant of the output tile.

        Accesses quadrant via .tile[] view into the contiguous out_reg_tile.
        Uses mma_frag_width for fragment sizing (4 for BF16, 8 for FP8).

        Works for both BF16 and FP8 via stdlib mma() dispatch.

        Parameters:
            which_a: A quadrant index (0 or 1).
            which_b: B quadrant index (0 or 1).
        """
        # Create vectorized views with mma_frag_width (dtype-aware)
        var a_mma_frag = self.a_reg_tile.tile[
            Self.quadrant_m_mmas, Self.num_k_mmas * Self.mma_frag_width
        ](which_a, 0).vectorize[1, Self.mma_frag_width]()

        var b_mma_frag = self.b_reg_tile.tile[
            Self.quadrant_n_mmas, Self.num_k_mmas * Self.mma_frag_width
        ](which_b, 0).vectorize[1, Self.mma_frag_width]()

        # Access quadrant via tile view into contiguous out_reg_tile
        var c_accum_frag = self.out_reg_tile.tile[
            Self.quadrant_m_size, Self.quadrant_n_size
        ](which_a, which_b).vectorize[1, Self.accum_width]()

        comptime for k in range(Self.num_k_mmas):
            comptime for m in range(Self.quadrant_m_mmas):
                comptime for n in range(Self.quadrant_n_mmas):
                    mma(
                        c_accum_frag[m, n],
                        b_mma_frag[n, k],
                        a_mma_frag[m, k],
                        c_accum_frag[m, n],
                    )


struct TileBuffers[
    in_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    //,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,  # MMA instruction M dimension - needed for swizzle matching
    MMA_K: Int,  # MMA instruction K dimension - needed for swizzle matching
    num_threads: Int,
    alignment: Int,
    enable_swizzle: Bool,
    load_width: Int,  # From MmaOp - how consumer wants data vectorized
    loading_warps: Int = 8,  # Number of warps participating in each load (4 or 8)
]:
    """Double-buffered LDS tiles and TileLoaders for ping-pong matmul.

    a_layout and b_layout are infer-only parameters (note `//`), automatically
    extracted from the input tensors passed to __init__. K is derived as an
    comptime from a_layout.shape[1].
    """

    # =========================================================================
    # Swizzle Configuration
    # =========================================================================
    # Write and read swizzle patterns must produce identical physical LDS
    # addresses. The write swizzle (TileLoaderLDS) operates in BYTE space,
    # while the read swizzle (MmaOp/load_lds_fragment) operates in ELEMENT
    # space. For N-byte elements, write base = read base + log2(N):
    #
    #   BF16 (2B): write Swizzle(1,5,4) in bytes == read Swizzle(1,4,4) in elems
    #   FP8  (1B): write Swizzle(3,4,4) in bytes == read Swizzle(3,4,4) in elems
    #   FP8  (1B): write Swizzle(2,5,4) in bytes == read Swizzle(2,5,4) in elems
    # =========================================================================

    # Compute lds_frag_width (same logic as make_mma_swizzle)
    comptime mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime use_split_k = (Self.in_type.is_float8() and Self.MMA_K == 128)
    comptime lds_frag_width = 16 if Self.use_split_k else Self.mma_frag_width

    # Swizzle parameters matching make_mma_swizzle
    comptime swizzle_log_tile = log2_floor(Self.MMA_K // 32) + 1
    comptime frag_bytes = Self.lds_frag_width * size_of[Self.in_type]()

    # BF16: use original (1, 5, 4) which works despite mismatch
    # FP8: use computed base to match make_mma_swizzle exactly
    comptime swizzle_subtile_cols = 4 * Self.load_width
    comptime elem_size = size_of[Self.in_type]()
    comptime swizzle_base = (
        log2_floor(Self.frag_bytes) if Self.in_type.is_float8() else (
            log2_floor(Self.swizzle_subtile_cols // 2)
            + log2_floor(Self.elem_size)
        )
    )
    comptime swizzle_shift = 4

    comptime byte_swizzle = Optional(
        Swizzle(Self.swizzle_log_tile, Self.swizzle_base, Self.swizzle_shift)
    ) if Self.enable_swizzle else Optional[Swizzle]()

    # FP8 uses row-major LDS layout for correct partial block OOB handling
    comptime use_fp8_row_major = Self.in_type.is_float8()

    # Half-tile dimensions (each warp group loads/uses one half independently)
    # half_BM == WM so A half-tile matches the warp's A region exactly.
    # This holds for both 2xN grids (BM//2 == WM) and 1xN grids (BM == WM).
    comptime half_BM = Self.WM
    comptime half_BN = Self.BN // 2

    # MMA tile dimensions (2 tiles per warp dimension for quadrant processing)
    comptime mma_tile_m = Self.WM // 2
    comptime mma_tile_n = Self.WN // 2

    # Type aliases for shared memory tiles
    comptime SMemTile[rows: Int, cols: Int] = SMemTile[
        Self.in_type,
        Layout.row_major(rows, cols),
        alignment=Self.alignment,
    ]

    # Half-tile types - allocated directly (groups are staggered, independent)
    comptime AHalfTile = Self.SMemTile[Self.half_BM, Self.BK]
    comptime BHalfTile = Self.SMemTile[Self.half_BN, Self.BK]
    comptime HalfTile[rows: Int] = Self.SMemTile[rows, Self.BK]

    # MMA tile types - views into half-tiles for quadrant processing
    # These are what MmaOp.load_a/load_b receive (validated via compile-time constraints)
    comptime AMmaTile = Self.AHalfTile.TileType[Self.mma_tile_m, Self.BK]
    comptime BMmaTile = Self.BHalfTile.TileType[Self.WN, Self.BK].TileType[
        Self.mma_tile_n, Self.BK
    ]

    # Nested tuple type for [stage][idx] indexing (2 stages × 2 tiles)
    comptime AMmaTilePair = Tuple[Self.AMmaTile, Self.AMmaTile]
    comptime BMmaTilePair = Tuple[Self.BMmaTile, Self.BMmaTile]

    # Loading configuration constants
    # Configurable number of warps for cooperative loading (4 or 8)
    comptime total_warps = 8  # Total warps in the block (always 8 for ping-pong)
    comptime loading_threads = Self.loading_warps * WARP_SIZE  # 256 or 512 threads
    comptime elements_per_warp = WARP_SIZE * Self.load_width  # 64 * 8 = 512
    comptime rows_per_warp = Self.elements_per_warp // Self.BK  # 8 rows per warp
    comptime loads_per_row = Self.BK // Self.load_width  # 8

    # Derived loading geometry (depends on loading_warps)
    comptime rows_per_load_iteration = Self.loading_threads // Self.loads_per_row
    # With 8 warps: 512/8 = 64 rows per iteration
    # With 4 warps: 256/8 = 32 rows per iteration

    # Global load counts for vmcnt tracking (load_to_lds ops per half-tile)
    # half_BM = WM (A half-tile matches warp M region)
    comptime vmcnt_per_load_a = Self.half_BM // Self.rows_per_load_iteration  # 8-warp A half
    comptime vmcnt_per_load_b = (
        Self.BN // 2
    ) // Self.rows_per_load_iteration  # 8-warp B half
    comptime vmcnt_per_load_ab = Self.vmcnt_per_load_a + Self.vmcnt_per_load_b  # Combined A+B

    # =========================================================================
    # TileLoader Configuration
    # =========================================================================
    # A and B use separate tile layouts since half_BM and half_BN may differ
    # (e.g. skinny 128x256: half_BM=64, half_BN=128)
    comptime a_half_tile_layout = Layout.row_major(Self.half_BM, Self.BK)
    comptime b_half_tile_layout = Layout.row_major(Self.half_BN, Self.BK)

    comptime ATileLoader = TileLoaderLDS[
        Self.in_type,
        Self.a_layout,
        Self.a_half_tile_layout,
        Self.loading_warps,
        Self.byte_swizzle,
        Self.load_width,
        Self.use_fp8_row_major,
    ]
    comptime BTileLoader = TileLoaderLDS[
        Self.in_type,
        Self.b_layout,
        Self.b_half_tile_layout,
        Self.loading_warps,
        Self.byte_swizzle,
        Self.load_width,
        Self.use_fp8_row_major,
    ]

    # MMA tiles: [stage][half] for double-buffered compute
    var a_mma_tiles: Tuple[Self.AMmaTilePair, Self.AMmaTilePair]
    var b_mma_tiles: Tuple[Self.BMmaTilePair, Self.BMmaTilePair]

    # Load tiles: [stage][which] - LDS destinations
    comptime AHalfTilePair = Tuple[Self.AHalfTile, Self.AHalfTile]
    comptime BHalfTilePair = Tuple[Self.BHalfTile, Self.BHalfTile]
    var a_load_tiles: Tuple[Self.AHalfTilePair, Self.AHalfTilePair]
    var b_load_tiles: Tuple[Self.BHalfTilePair, Self.BHalfTilePair]

    var loader_a: Self.ATileLoader
    var loader_b: Self.BTileLoader
    var warp_id_m: Int

    # K derived from a_layout at compile time
    comptime K = Self.a_layout.shape[1].value()

    @always_inline
    def __init__(
        out self,
        a: LayoutTensor[Self.in_type, Self.a_layout, ...],
        b: LayoutTensor[_, Self.b_layout, ...],
        block_row: Int,
        block_col: Int,
        warp_id: Int,
        warp_id_m: Int,
        warp_id_n: Int,
        lane_id: Int,
    ):
        """Initialize LDS tiles and loaders. Layouts inferred from a and b tensors.
        """
        # Validate configuration
        comptime assert (
            Self.loading_warps == 4 or Self.loading_warps == 8
        ), "loading_warps must be 4 or 8"
        comptime assert (
            Self.load_width == simd_width_of[Self.in_type]()
        ), "load_width must match simd_width_of[in_type]()"
        # Allocate half-tiles: [stage][warp_group]
        # A: 2 stages × 2 warp_m groups (half_BM == WM, so 1:1 mapping)
        var a_s0_g0 = Self.AHalfTile.stack_allocation()  # stage 0, warp_id_m=0
        var a_s0_g1 = Self.AHalfTile.stack_allocation()  # stage 0, warp_id_m=1
        var a_s1_g0 = Self.AHalfTile.stack_allocation()  # stage 1, warp_id_m=0
        var a_s1_g1 = Self.AHalfTile.stack_allocation()  # stage 1, warp_id_m=1

        # B: 2 stages × 2 halves (each half covers 2 warp_n groups)
        var b_s0_h0 = Self.BHalfTile.stack_allocation()  # stage 0, warp_n 0-1
        var b_s0_h1 = Self.BHalfTile.stack_allocation()  # stage 0, warp_n 2-3
        var b_s1_h0 = Self.BHalfTile.stack_allocation()  # stage 1, warp_n 0-1
        var b_s1_h1 = Self.BHalfTile.stack_allocation()  # stage 1, warp_n 2-3

        # Select this warp's half-tiles based on warp_id_m and warp_id_n
        var a_s0 = a_s0_g0 if warp_id_m == 0 else a_s0_g1
        var a_s1 = a_s1_g0 if warp_id_m == 0 else a_s1_g1

        # warps_per_b_half: number of N-warps sharing each B half-tile
        # 256x256 (2x4 grid): 4/2 = 2 warps per B half
        # 128x512 (1x8 grid): 8/2 = 4 warps per B half
        comptime num_warps_n = Self.BN // Self.WN
        comptime warps_per_b_half = num_warps_n // 2
        var b_half_idx, b_local_n = divmod(
            warp_id_n, warps_per_b_half
        )  # which B half, position within
        var b_s0 = b_s0_h0 if b_half_idx == 0 else b_s0_h1
        var b_s1 = b_s1_h0 if b_half_idx == 0 else b_s1_h1

        # Create MMA tiles directly from half-tiles
        self.a_mma_tiles = (
            (
                a_s0.tile[Self.mma_tile_m, Self.BK](0, 0),
                a_s0.tile[Self.mma_tile_m, Self.BK](1, 0),
            ),
            (
                a_s1.tile[Self.mma_tile_m, Self.BK](0, 0),
                a_s1.tile[Self.mma_tile_m, Self.BK](1, 0),
            ),
        )
        self.b_mma_tiles = (
            (
                b_s0.tile[Self.WN, Self.BK](b_local_n, 0).tile[
                    Self.mma_tile_n, Self.BK
                ](0, 0),
                b_s0.tile[Self.WN, Self.BK](b_local_n, 0).tile[
                    Self.mma_tile_n, Self.BK
                ](1, 0),
            ),
            (
                b_s1.tile[Self.WN, Self.BK](b_local_n, 0).tile[
                    Self.mma_tile_n, Self.BK
                ](0, 0),
                b_s1.tile[Self.WN, Self.BK](b_local_n, 0).tile[
                    Self.mma_tile_n, Self.BK
                ](1, 0),
            ),
        )

        # LDS destination tiles for loading: [stage][which]
        self.a_load_tiles = (
            (a_s0_g0, a_s0_g1),  # stage 0: (group 0, group 1)
            (a_s1_g0, a_s1_g1),  # stage 1: (group 0, group 1)
        )
        self.b_load_tiles = (
            (b_s0_h0, b_s0_h1),  # stage 0: (half 0, half 1)
            (b_s1_h0, b_s1_h1),  # stage 1: (half 0, half 1)
        )

        # Initialize loaders with tile views (block offset embedded in ptr)
        var a_block = a.tile[Self.BM, Self.K](block_row // Self.BM, 0)
        var b_block = b.tile[Self.BN, Self.K](block_col // Self.BN, 0)
        self.loader_a = Self.ATileLoader(a_block, warp_id, lane_id)
        self.loader_b = Self.BTileLoader(b_block, warp_id, lane_id)
        self.warp_id_m = warp_id_m

    # =========================================================================
    # 8-Warp Loading
    # =========================================================================

    @always_inline
    def load_a[stage: Int, which: Int](self, *, k: Int):
        """Load A[stage][which] from global to LDS using all 8 warps."""
        self.loader_a.load_tile(
            self.a_load_tiles[stage][which],
            src_row=which * Self.half_BM,
            src_col=k,
        )

    @always_inline
    def load_b[stage: Int, which: Int](self, *, k: Int):
        """Load B[stage][which] from global to LDS using all 8 warps."""
        self.loader_b.load_tile(
            self.b_load_tiles[stage][which],
            src_row=which * Self.half_BN,
            src_col=k,
        )


struct AMDPingPongMatmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    config: KernelConfig,
    /,
    # Enable 16×32 swizzle pattern for bank conflict avoidance
    enable_swizzle: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """8-warp ping-pong matmul for AMD MI355X.

    Warps are split into 2 groups of 4, alternating between load and compute
    phases for overlapped execution. Uses double-buffered LDS with swizzled
    access patterns to avoid bank conflicts.

    Key features:
    - load_to_lds for direct DRAM→LDS transfer (bypasses L1/L2)
    - Swizzle pattern for bank-conflict-free LDS access
    - Fine-grained lgkmcnt/vmcnt waits for maximum overlap
    """

    # Extract configuration
    comptime BM = Self.config.block_shape[0]
    comptime BN = Self.config.block_shape[1]
    comptime BK = Self.config.block_shape[2]

    comptime WM = Self.config.warp_shape[0]
    comptime WN = Self.config.warp_shape[1]
    comptime WK = Self.config.warp_shape[2]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    # Derived configuration
    comptime num_warps_m = Self.BM // Self.WM
    comptime num_warps_n = Self.BN // Self.WN
    comptime total_warps = Self.num_warps_m * Self.num_warps_n

    comptime num_m_mmas = Self.WM // Self.MMA_M
    comptime num_n_mmas = Self.WN // Self.MMA_N
    comptime num_k_mmas = Self.WK // Self.MMA_K

    # Memory configuration
    comptime load_width = simd_width_of[Self.a_type]()
    comptime ping_pong_stages = 2
    comptime total_smem_a = Self.ping_pong_stages * Self.BM * Self.BK
    comptime total_smem_b = Self.ping_pong_stages * Self.BN * Self.BK

    # Accumulator configuration (fp32 for bf16/fp8 output)
    comptime accum_dtype = get_accum_type[Self.c_type]()
    comptime accum_width = (Self.MMA_M * Self.MMA_N) // WARP_SIZE
    comptime num_accums = Self.num_m_mmas * Self.num_n_mmas

    # =========================================================================
    # Async Load Counts for s_waitcnt
    # =========================================================================
    # These aliases enable precise tracking of outstanding async operations.
    # Operations complete in FIFO order, so we can count how many must complete
    # before specific data is ready.
    #
    # Usage: s_waitcnt[lgkmcnt=N]() waits until ≤N lgkm ops remain in flight.
    #        s_waitcnt[vmcnt=N]() waits until ≤N global loads remain in flight.
    #
    # Example: After issuing load_b + load_a (4+8=12 lgkm ops), to wait for
    #          only load_b to complete, use lgkmcnt=8 (8 load_a ops still ok).
    # =========================================================================

    # Quadrant MMA counts (used in load_a/load_b loops)
    comptime quadrant_m_mmas = Self.num_m_mmas // 2  # 4
    comptime quadrant_n_mmas = Self.num_n_mmas // 2  # 2

    # LDS → Registers (lgkmcnt): ds_read ops per mma_op.load_* call
    # Must match MmaOp's lgkm_per_load_{a,b} exactly. Each LDS fragment
    # load issues ds_reads_per_frag ds_read ops (2 for 32-byte FP8 frags,
    # 1 for 16-byte), and k_loads_per_mma handles split K-loading.
    comptime _mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime _use_split_lds = (
        Self.a_type.is_float8() and Self.MMA_M == 16 and Self.MMA_K == 128
    )
    comptime _lds_frag_width = 16 if Self._use_split_lds else Self._mma_frag_width
    comptime _k_loads_per_mma = Self._mma_frag_width // Self._lds_frag_width
    comptime _ds_reads_per_frag = ceildiv(
        Self._lds_frag_width * size_of[Self.a_type](), 16
    )
    comptime LGKM_PER_LOAD_A = (
        Self.quadrant_m_mmas
        * Self.num_k_mmas
        * Self._k_loads_per_mma
        * Self._ds_reads_per_frag
    )
    comptime LGKM_PER_LOAD_B = (
        Self.quadrant_n_mmas
        * Self.num_k_mmas
        * Self._k_loads_per_mma
        * Self._ds_reads_per_frag
    )
    comptime LGKM_PER_LOAD_AB = Self.LGKM_PER_LOAD_A + Self.LGKM_PER_LOAD_B

    # Global → LDS (vmcnt): load_to_lds ops per buffers.load_* call (8-warp)
    # half_BM = WM (matches TileBuffers), half_BN = BN // 2
    comptime loads_per_row = Self.BK // Self.load_width  # 8
    comptime loading_threads_8warp = 8 * WARP_SIZE  # 512
    comptime rows_per_iter_8warp = Self.loading_threads_8warp // Self.loads_per_row  # 64
    comptime half_BM = Self.config.warp_shape[0]  # WM
    comptime VMCNT_PER_LOAD_A = Self.half_BM // Self.rows_per_iter_8warp  # 2
    comptime VMCNT_PER_LOAD_B = (Self.BN // 2) // Self.rows_per_iter_8warp  # 2

    @staticmethod
    def validate_config():
        """Validate the kernel configuration."""
        comptime assert (
            Self.BM % Self.WM == 0
        ), "Block M must be divisible by Warp M"
        comptime assert (
            Self.BN % Self.WN == 0
        ), "Block N must be divisible by Warp N"
        comptime assert (
            Self.BK % Self.WK == 0
        ), "Block K must be divisible by Warp K"
        comptime assert (
            Self.WM % Self.MMA_M == 0
        ), "Warp M must be divisible by MMA M"
        comptime assert (
            Self.WN % Self.MMA_N == 0
        ), "Warp N must be divisible by MMA N"
        comptime assert (
            Self.WK % Self.MMA_K == 0
        ), "Warp K must be divisible by MMA K"
        comptime assert (
            Self.total_warps == 8
        ), "Ping-pong kernel requires exactly 8 warps"
        comptime assert (
            Self.num_warps_m == 2
        ), "Ping-pong kernel requires 2 warps in M dimension"

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads())
        )
    )
    @staticmethod
    def matmul_ping_pong(
        a: LayoutTensor[
            Self.a_type,
            Self.a_layout,
            ImmutAnyOrigin,
            address_space=AddressSpace.GENERIC,
        ],
        b: LayoutTensor[
            Self.b_type,
            Self.b_layout,
            ImmutAnyOrigin,
            address_space=AddressSpace.GENERIC,
        ],
        c: LayoutTensor[
            Self.c_type,
            Self.c_layout,
            MutAnyOrigin,
            address_space=AddressSpace.GENERIC,
        ],
    ):
        var M = a.dim(0)
        # Makes enable_l2_cache_optimization useful
        # comptime M = a.layout.shape[0].value()
        comptime N = b.layout.shape[0].value()
        comptime K = a.layout.shape[1].value()

        comptime assert (
            Self.a_type == Self.b_type
        ), "A and B must have the same type"

        comptime in_type = Self.a_type

        # Validate configuration
        Self.validate_config()

        # Use struct's configuration directly
        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK

        comptime WM = Self.WM
        comptime WN = Self.WN
        comptime WK = Self.WK

        comptime MMA_M = Self.MMA_M
        comptime MMA_N = Self.MMA_N
        comptime MMA_K = Self.MMA_K

        comptime num_warps_m = Self.num_warps_m
        comptime num_warps_n = Self.num_warps_n
        comptime total_warps = Self.total_warps

        comptime num_m_mmas = Self.num_m_mmas
        comptime num_n_mmas = Self.num_n_mmas
        comptime num_k_mmas = Self.num_k_mmas

        comptime accum_dtype = Self.accum_dtype
        comptime accum_width = Self.accum_width
        comptime num_accums = Self.num_accums

        comptime load_width = Self.load_width
        comptime total_smem_a = Self.total_smem_a
        comptime total_smem_b = Self.total_smem_b

        comptime alignment = 128  # align_of[in_type]()

        # AMD MI355: 64 LDS banks × 4 bytes each. With enable_swizzle=True (default),
        # the 16×32 subtile swizzle (XOR bit 9→5) breaks 4-way bank conflicts during MMA.
        # See module header and TileBuffers struct for detailed swizzle documentation.

        # Thread and warp identification
        var thread_id = thread_idx.x
        var lane_id = lane_id()
        var warp_id = readfirstlane(Int(warp_id()))

        # Block coordinates from block indices
        var n = block_idx.x * BN
        var m = block_idx.y * BM

        # Swizzle for LDS bank conflict avoidance (see make_mma_swizzle docs)
        comptime mma_swizzle = Optional(
            make_mma_swizzle[in_type, MMA_M, MMA_K]()
        ) if Self.enable_swizzle else Optional[Swizzle]()

        # MmaOp handles both BF16 and FP8 via dtype-aware mma_frag_width
        comptime MmaOpType = MmaOp[
            in_type,
            accum_dtype,
            WM,
            WN,
            BK,
            MMA_M,
            MMA_N,
            MMA_K,
            alignment,
            mma_swizzle,
        ]
        comptime BuffersType = TileBuffers[
            BM,
            BN,
            BK,
            WM,
            WN,
            MMA_M,
            MMA_K,  # For swizzle matching with MmaOp
            Self.config.num_threads(),
            alignment,
            Self.enable_swizzle,
            load_width,
        ]

        # MMA operation encapsulates register tiles and compute operations
        var mma_op = MmaOpType()

        # Warp position in warp grid
        var warp_id_m, warp_id_n = divmod(warp_id, num_warps_n)

        # Warp group for ping-pong scheduling (always 2 groups of 4 warps)
        # Decoupled from warp_id_m to support both 2xN and 1xN warp grids
        var warp_group_id = warp_id // 4

        # Tile Buffers struct
        var buffers = BuffersType(
            a, b, m, n, warp_id, warp_id_m, warp_id_n, lane_id
        )

        @always_inline
        def s_barrier():
            llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

        @always_inline
        def s_setprio[priority: Int16]():
            llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)

        # ================================================================
        # PHASE SYNCHRONIZATION
        # ================================================================
        # Single-counter barrier model with local phase tracking.
        # - All 8 warps share a single atomic counter
        # - Each warp tracks its own local barrier_count
        # - G1 has barrier_count = G0's barrier_count + 1 (from stagger)
        # - This creates the 1-phase offset between groups
        #
        # Key insight: s_barrier releases when all 8 waves hit SOME barrier.
        # By giving G1 a higher barrier_count, G1 waits for more increments.
        # ================================================================

        comptime lgkm_a = Self.LGKM_PER_LOAD_A
        comptime lgkm_b = Self.LGKM_PER_LOAD_B

        # ================================================================
        # FRAMEWORK-DRIVEN SCHEDULE
        # ================================================================
        # Build the schedule at compile time. The framework derives edges,
        # prologue, kernel body, and epilogue automatically from the
        # DeclarativeSchedule specification in pingpong_schedule.mojo.
        # ================================================================

        comptime is_fp8 = in_type.is_float8()
        comptime sched_config = ScheduleConfig(
            scheduling=SchedulingStrategy.CSP, auto_waits=True
        )

        # Derive safe max_globals from num_k_mmas. With warp stagger,
        # uniform distribution is only safe when enough MMA latency
        # covers async LDS writes (num_k_mmas >= 2 for BF16, not for FP8).
        comptime target = mi355x_target(
            vm_per_load_a=Self.VMCNT_PER_LOAD_A,
            vm_per_load_b=Self.VMCNT_PER_LOAD_B,
            max_globals=derive_safe_max_globals(Self.num_k_mmas),
        )
        comptime schedule = build_schedule[
            is_fp8,
            lgkm_a,
            lgkm_b,
        ](sched_config, target)

        # ================================================================
        # TILE ORGANIZATION AND PING-PONG SCHEDULE
        # ================================================================
        #
        # BLOCK TILE (BM×BN, 8 warps in num_warps_m × num_warps_n grid):
        #
        # Standard (256×256, 2×4):            Skinny (128×256, 2×4):
        # ┌──────┬──────┬──────┬──────┐       ┌──────┬──────┬──────┬──────┐
        # │  W0  │  W1  │  W2  │  W3  │ g0    │  W0  │  W1  │  W2  │  W3  │ g0
        # │128×64│128×64│128×64│128×64│       │ 64×64│ 64×64│ 64×64│ 64×64│
        # ├──────┼──────┼──────┼──────┤       ├──────┼──────┼──────┼──────┤
        # │  W4  │  W5  │  W6  │  W7  │ g1    │  W4  │  W5  │  W6  │  W7  │ g1
        # │128×64│128×64│128×64│128×64│       │ 64×64│ 64×64│ 64×64│ 64×64│
        # └──────┴──────┴──────┴──────┘       └──────┴──────┴──────┴──────┘
        #
        # warp_group_id = warp_id // 4 (group selection, decoupled from grid)
        # warp_id_m, warp_id_n = divmod(warp_id, num_warps_n) (tile coords)
        #
        # DOUBLE BUFFERING (ping-pong):
        #   Stage 0: A[0][0,1], B[0][0,1]  ←→  Stage 1: A[1][0,1], B[1][0,1]
        #   K loop: compute[0] while load[1], then compute[1] while load[0]
        #
        # WARP OUTPUT QUADRANTS (WM×WN → 4 quadrants):
        # ┌─────────────────┬─────────────────┐
        # │ mma[0,0]        │ mma[0,1]        │
        # ├─────────────────┼─────────────────┤
        # │ mma[1,0]        │ mma[1,1]        │
        # └─────────────────┴─────────────────┘
        #
        # ================================================================

        # Dispatch: map schedule entries to hardware calls.
        @parameter
        @always_inline
        def _bind[entry: ScheduleEntry](k_base: Int):
            comptime k_off = entry.op.k_offset.signed_bk_multiple()
            var k = k_base + k_off * BK
            comptime if entry.op.tag == LOAD_A:
                buffers.load_a[entry.op.stage, entry.op.subtile](k=k)
            elif entry.op.tag == LOAD_B:
                buffers.load_b[entry.op.stage, entry.op.subtile](k=k)
            elif entry.op.tag == MMA_LOAD_A:
                mma_op.load_a[entry.op.subtile](
                    buffers.a_mma_tiles[entry.op.stage][entry.op.subtile]
                )
            elif entry.op.tag == MMA_LOAD_B:
                mma_op.load_b[entry.op.subtile](
                    buffers.b_mma_tiles[entry.op.stage][entry.op.subtile]
                )
            elif entry.op.tag == MMA:
                mma_op.mma[entry.op.stage, entry.op.subtile]()
            elif entry.op.tag == PingPongOps.BARRIER.value:
                s_barrier()
            elif entry.op.tag == PingPongOps.WAIT_VM.value:
                s_waitcnt[vmcnt=UInt32(entry.op.wait_value)]()
            elif entry.op.tag == PingPongOps.WAIT_LGKM.value:
                s_waitcnt[lgkmcnt=UInt32(entry.op.wait_value)]()
            elif entry.op.tag == PingPongOps.SET_PRIO.value:
                s_setprio[Int16(entry.op.wait_value)]()
            elif entry.op.tag == PingPongOps.SCHEDULE_BARRIER.value:
                schedule_barrier()

        # Prologue: stage 0 loads.
        comptime for i in range(schedule.warp_stagger_index):
            _bind[schedule.prologue[i]](0)

        # Warp stagger: G1 starts one phase ahead of G0.
        if warp_group_id == 1:
            s_barrier()

        # Prologue: waits, stage 1 loads, final barrier.
        comptime for i in range(
            schedule.warp_stagger_index, len(schedule.prologue)
        ):
            _bind[schedule.prologue[i]](0)
        s_barrier()

        # Main loop: steady-state double-buffered execution.
        for k in range(BK * 2, K, BK * 2):
            comptime for i in range(len(schedule.kernel)):
                _bind[schedule.kernel[i]](k)

        # Epilogue: drain remaining compute.
        # k_base = K so completion loads at k_offset=-1 resolve to K-BK.
        comptime for i in range(len(schedule.epilogue)):
            _bind[schedule.epilogue[i]](K)

        # Re-balance warp staggering.
        if warp_group_id == 0:
            s_barrier()
        # ================================================================
        # Output Store
        # ================================================================
        # MMA call: mma(d, b_frag, a_frag, d) — B is MFMA's A-operand
        # (drives MFMA rows → N-coords), A is MFMA's B-operand (drives
        # MFMA cols → M-coords). The store transposes to compensate:
        #     stored_row = M-coord = m_mma*MMA_M + MFMA_col
        #     stored_col = N-coord = n_mma*MMA_N + MFMA_row

        var warp_tile_m = m + WM * warp_id_m
        var warp_tile_n = n + WN * warp_id_n

        comptime out_frag_layout = Layout(
            IntTuple(num_m_mmas, num_n_mmas, accum_width),
            IntTuple(num_n_mmas * accum_width, accum_width, 1),
        )

        var c_reg_fragment = mma_op.out_reg_tile.reshape[
            out_frag_layout
        ]().vectorize[1, 1, accum_width]()

        if warp_tile_m < M and warp_tile_n < N:
            comptime if MMA_M == 32:
                # 32×32 MMA: Interleaved accumulator store.
                # d[j] maps to N = (j//4)*8 + (lane//32)*4 + j%4
                # The 16 accumulators form 4 groups of 4 contiguous N values
                # at stride 8. Write each group as SIMD[4].
                var lane_m = Int(lane_id) % MMA_M
                var lane_group = Int(lane_id) // MMA_M  # 0 or 1

                comptime for frag_m in range(num_m_mmas):
                    comptime for frag_n in range(num_n_mmas):
                        var accum_vec = c_reg_fragment[frag_m, frag_n, 0]
                        var row = warp_tile_m + frag_m * MMA_M + lane_m

                        if row < M:
                            comptime for g in range(4):
                                var col = (
                                    warp_tile_n
                                    + frag_n * MMA_N
                                    + g * 8
                                    + lane_group * 4
                                )
                                if col < N:
                                    var group_vals = accum_vec.slice[
                                        4, offset=g * 4
                                    ]().cast[Self.c_type]()
                                    c.store[width=4](
                                        Index(row, col), group_vals
                                    )
            else:
                # 16×16 MMA: 4 contiguous accumulators per lane → SIMD[4] store.
                comptime output_thread_layout = Layout(
                    IntTuple(MMA_M, WARP_SIZE // MMA_M),
                    IntTuple(1, MMA_M),
                )
                var c_warp_tile = c.tile[WM, WN](
                    warp_tile_m // WM, warp_tile_n // WN
                )
                var c_gmem_fragment = c_warp_tile.vectorize[
                    1, accum_width
                ]().distribute[output_thread_layout](Int(lane_id))

                comptime if Self.elementwise_lambda_fn:
                    write_output_fragments[
                        Self.c_type,
                        accum_width,
                        MMA_M,
                        MMA_N,
                        output_thread_layout,
                        Self.elementwise_lambda_fn,
                    ](
                        c_reg_fragment,
                        c_gmem_fragment,
                        warp_tile_m,
                        warp_tile_n,
                        M,
                        N,
                    )
                else:
                    write_output_fragments[
                        Self.c_type,
                        accum_width,
                        MMA_M,
                        MMA_N,
                        output_thread_layout,
                    ](
                        c_reg_fragment,
                        c_gmem_fragment,
                        warp_tile_m,
                        warp_tile_n,
                        M,
                        N,
                    )


@always_inline
def ping_pong_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    //,
    enable_swizzle: Bool = True,
](
    a_device_tensor: LayoutTensor[a_type, a_layout, ...],
    b_device_tensor: LayoutTensor[b_type, b_layout, ...],
    c_device_tensor: LayoutTensor[c_type, c_layout, ...],
    ctx: DeviceContext,
) raises:
    comptime assert a_type == b_type, "A and B must have the same type"
    comptime assert (
        a_type == DType.bfloat16 or a_type.is_float8()
    ), "A must be bfloat16 or float8_e4m3fn"

    # Select kernel based on input dtype
    comptime is_fp8 = a_type.is_float8()

    var N = c_device_tensor.dim(1)
    var M = c_device_tensor.dim(0)

    comptime use_swizzle = enable_swizzle

    @always_inline
    @parameter
    def run_kernel[config: KernelConfig]() raises:
        comptime kernel = AMDPingPongMatmul[
            a_type,
            b_type,
            c_type,
            a_layout,
            b_layout,
            c_layout,
            config,
            use_swizzle,
        ].matmul_ping_pong

        ctx.enqueue_function[kernel, kernel](
            a_device_tensor,
            b_device_tensor,
            c_device_tensor,
            grid_dim=(
                ceildiv(N, config.block_shape[1]),
                ceildiv(M, config.block_shape[0]),
            ),
            block_dim=config.num_threads(),
        )

    # Dispatch: FP8 uses 32×32×64 MMA with BK=128 (best perf: ~2.9 TFLOPS)
    # Fallback to 16×16×128 for M % 32 != 0 (still high perf, supports any M)
    # BF16 uses 16×16×32 with BK=64
    #
    # Skinny 128x256 wins in a mid-M band whose boundaries depend on N:
    #   Large N (>=4096): skinny wins for 150 < M <= 512
    #   Small N (<4096):  skinny wins for 512 < M <= 2048
    # Outside those bands, baseline 256x256 is faster.
    comptime if is_fp8:
        comptime BM = 256
        comptime BN = 256
        comptime BK = 128

        # Standard 256x256 configs
        comptime config_32x32 = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(32, 32, 64),
        )
        comptime config_16x16 = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(16, 16, 128),
        )

        # Skinny 128x256 config (170 FLOP/B, 96 KB LDS)
        comptime skinny_config = KernelConfig(
            block_shape=Index(128, BN, BK),
            warp_shape=Index(64, BN // 4, BK),
            mma_shape=Index(16, 16, 128),
        )

        # TODO: skinny_config (BM=128) has a pre-existing race condition
        # in the pipeline scheduling that causes intermittent wrong results.
        # Disabled until the pipeline framework can generate correct
        # synchronization for non-square block shapes.
        # TODO: config_32x32 (32x32x64 MMA) has a pre-existing
        # correctness bug with random FP8 data (also broken on main).
        # Use config_16x16 for all FP8 until the root cause is fixed.
        run_kernel[config_16x16]()
    else:
        # BF16: 16×16×32 MMA with BK=64 (~1.55 TFLOPS)
        comptime BM = 256
        comptime BN = 256
        comptime BK = 64
        comptime config = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(16, 16, 32),
        )
        run_kernel[config]()
