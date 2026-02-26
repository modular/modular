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
from math import ceildiv
from bit import log2_floor
from sys import simd_width_of, size_of

from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    grid_dim,
)
from gpu import warp_id
from gpu.host import DeviceContext
from gpu.intrinsics import AMDBufferResource
from gpu.memory import AddressSpace
from gpu.compute.mma import mma
from sys import llvm_intrinsic
from gpu.sync import barrier, schedule_barrier, s_waitcnt
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from memory.unsafe import bitcast

from utils import Index, IndexList, StaticTuple
from utils.numerics import get_accum_type

from sys.intrinsics import readfirstlane, llvm_intrinsic
from sys._assembly import inlined_assembly
from os.atomic import Atomic

from gpu._utils import to_i64

from collections import OptionalReg

from ....structuring import SMemTile, RegTile, eval
from ....utils import elementwise_epilogue_type
from layout import Layout, LayoutTensor, IntTuple, RuntimeLayout
from layout.swizzle import Swizzle
from layout._utils import make_amd_buffer_resource, idx2crd
from .matmul import write_output_fragments


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


fn make_mma_swizzle[dtype: DType, MMA_M: Int, MMA_K: Int]() -> Swizzle:
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
    # FP8 16×16×128 uses split-K with 16-element fragments
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

    # Instance state (pre-computed for efficient load_tile calls)
    var buffer: AMDBufferResource
    var thread_row: Int  # With swizzle applied if enabled
    var thread_col: Int
    var warp_id: Int

    # Stride derived from src_layout at compile time
    comptime stride = Self.src_layout.shape[1].value()

    @always_inline
    fn __init__(
        out self,
        src: LayoutTensor,
        warp_id: Int,
        lane_id: Int,
    ):
        """Pre-compute thread position with swizzle inversion for bank-conflict-free reads.

        Args:
            src: Source tensor (tile view for BF16, full tensor for FP8).
            warp_id: Warp ID within the block.
            lane_id: Lane ID within the warp.
        """
        self.buffer = make_amd_buffer_resource(src)
        self.warp_id = warp_id

        # Swizzle inversion: load from swizzle(T) so writing to T is conflict-free
        var effective_lane = lane_id

        comptime if Self.swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        # Map warp_id to 2D grid position and lane to subtile position.
        # Warp grid adapts to tile width: 4×2 for BK=64, 2×4 for BK=128.
        var warp_row, warp_col = divmod(warp_id, Self.num_warp_cols)
        var subtile_row, subtile_col_idx = divmod(
            effective_lane, Self.threads_per_row
        )
        var subtile_col = subtile_col_idx * Self.load_width

        # Final thread position in tile: combine warp grid position + lane subtile position
        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

    @always_inline
    fn load_tile[
        dst_layout: Layout,
        //,
    ](
        self,
        dst: SMemTile[Self.dtype, dst_layout, ...],
        src_row: Int,
        src_col: Int,
    ):
        """Load a tile from source coordinates to LDS.

        Combines pre-computed thread position with source coordinates.
        Uses the buffer resource stored at init time.
        Only warps 0 to num_loading_warps-1 participate; others return immediately.

        Args:
            dst: Destination LDS tile.
            src_row: Starting row in source tensor.
            src_col: Starting column in source tensor (typically k_offset).
        """
        # Per-lane offset (constant across iterations) - uses vector registers
        # This is the thread-varying part that differs per lane
        var lane_offset = self.thread_col + self.thread_row * self.stride

        comptime for i in range(Self.num_iterations):
            var tile_idx = i * Self.num_loading_warps + self.warp_id
            var warp_subtile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                tile_idx, 0
            )
            var smem_ptr = readfirstlane(warp_subtile.ptr)

            # Uniform offset (same for all lanes) - uses scalar registers
            var tile_row = src_row + i * Self.rows_per_iteration
            var uniform_offset = src_col + tile_row * self.stride

            self.buffer.load_to_lds[width = Self.load_width](
                Int32(lane_offset),
                smem_ptr,
                scalar_offset=Int32(uniform_offset),
            )


@always_inline
fn _load_from_lds[
    dtype: DType,
    //,
    width: Int = 1,
](
    shared_ptr: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
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
        _type = __mlir_type.`!llvm.ptr<3>`
    ](shared_ptr)

    # Compute alignment based on total load size
    comptime load_bytes = width * size_of[dtype]()
    comptime alignment = min(load_bytes, 16)  # Cap at 16-byte alignment

    # Generate the appropriate LLVM vector type and load
    # Using compile-time dispatch based on dtype and width
    comptime if dtype == DType.bfloat16 and width == 4:
        # BF16 x4 = 64 bits = ds_read_b64
        var llvm_res = __mlir_op.`llvm.load`[
            _type = __mlir_type.`vector<4 x bf16>`,
            alignment = to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type = SIMD[DType.bfloat16, 4]._mlir_type
            ](llvm_res)
        )
    elif dtype == DType.bfloat16 and width == 8:
        # BF16 x8 = 128 bits = ds_read_b128
        var llvm_res = __mlir_op.`llvm.load`[
            _type = __mlir_type.`vector<8 x bf16>`,
            alignment = to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type = SIMD[DType.bfloat16, 8]._mlir_type
            ](llvm_res)
        )
    elif dtype.is_float8() and width == 8:
        # FP8 x8 = 64 bits = ds_read_b64
        # Load as i8 vector, then bitcast to fp8 (same bit pattern)
        var llvm_res = __mlir_op.`llvm.load`[
            _type = __mlir_type.`vector<8 x i8>`,
            alignment = to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type = SIMD[DType.uint8, 8]._mlir_type
        ](llvm_res)
        # Bitcast uint8 → float8_e4m3fn (same 8-bit representation)
        return bitcast[dtype, width](
            rebind[SIMD[DType.uint8, width]](uint8_vec)
        )
    elif dtype.is_float8() and width == 16:
        # FP8 x16 = 128 bits = ds_read_b128
        # Used for 16×16×128 MMA (HipKittens-style FP8 schedule)
        var llvm_res = __mlir_op.`llvm.load`[
            _type = __mlir_type.`vector<16 x i8>`,
            alignment = to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type = SIMD[DType.uint8, 16]._mlir_type
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
            _type = __mlir_type.`vector<16 x i8>`,
            alignment = to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        # Offset pointer by 16 bytes for second load
        var shared_ptr_offset = shared_ptr + 16
        var shared_ptr3_hi = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = __mlir_type.`!llvm.ptr<3>`
        ](shared_ptr_offset)
        var llvm_res1 = __mlir_op.`llvm.load`[
            _type = __mlir_type.`vector<16 x i8>`,
            alignment = to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3_hi)
        var uint8_vec0 = __mlir_op.`pop.cast_from_builtin`[
            _type = SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res0)
        var uint8_vec1 = __mlir_op.`pop.cast_from_builtin`[
            _type = SIMD[DType.uint8, 16]._mlir_type
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
fn load_lds_fragment[
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
    """Load LDS → registers with MMA access pattern.

    Why mma_access_layout differs from the global→LDS thread layout:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Layout          │ Purpose              │ Constraint                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │ load_thread     │ Global → LDS write   │ Coalesced global reads     │
    │ mma_access      │ LDS → Registers read │ AMD WMMA hardware pattern  │
    └─────────────────────────────────────────────────────────────────────┘

    mma_access_layout encodes how AMD's WMMA instruction expects data:
    - Lane decomposition: (lane % 16, lane // 16) = (col_group, row_group)
    - Offset computation: col_group * 32 + row_group * 8

    Using RuntimeLayout ensures compile-time evaluation (no GPU heap alloc).

    Layout compatibility requirements:
    - mma_access_layout must map exactly WARP_SIZE (64) threads
    - smem must have enough elements for: num_iterations * WARP_SIZE * frag_width
    - frag must store: num_iterations * frag_width elements
    """
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

    var lane = Int(lane_id())
    # RuntimeLayout wraps compile-time layout for efficient evaluation
    var lane_offset = Int(RuntimeLayout[mma_access_layout]()(lane))
    var frag_ptr = reg_frag.ptr.bitcast[FragElement]()

    # Iteration pattern:
    # - Standard (BF16/FP8 16×16×32): iter_base = i * WARP_SIZE * frag_width
    # - FP8 16×16×128: Split-K pattern with 2 iterations per MMA position
    #
    # Detect split-K mode: when MMA_K (stride[0]) requires more elements than
    # one iteration can provide (col_groups * frag_width).
    comptime mma_m = Int(mma_access_layout.shape[0])
    comptime col_groups = Int(mma_access_layout.shape[1])
    comptime mma_k = Int(mma_access_layout.stride[0])
    comptime elements_per_iter = col_groups * frag_width
    comptime use_split_k = mma_k > elements_per_iter  # True for FP8 16×16×128

    comptime if use_split_k:
        # FP8 16×16×128 split-K pattern: 2 iterations per MMA position
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


struct KernelConfig(ImplicitlyCopyable, Movable, Stringable, Writable):
    var block_shape: IndexList[3]
    var warp_shape: IndexList[3]
    var mma_shape: IndexList[3]

    fn __init__(
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
    fn _write_index_list(
        mut writer: Some[Writer], list: IndexList, sep: StaticString
    ):
        comptime for i in range(list.size):
            if i != 0:
                writer.write(sep)
            writer.write(list[i])

    @always_inline
    fn num_threads(self) -> Int:
        var num_warps = self.block_shape // self.warp_shape
        return num_warps.flattened_length() * WARP_SIZE

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("config_")
        Self._write_index_list(writer, self.block_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.warp_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.mma_shape, "x")

    fn __str__(self) -> String:
        return String.write(self)

    fn __repr__(self) -> String:
        return String.write(self)


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
    # - 16×16×128: Highest throughput (4× K), split-K LDS reads (2×16 elements)
    # - 32×32×64:  Medium throughput (2× K), single 32-element LDS read
    # Both use row-major LDS layout (MMA_K == BK) for correct partial blocks.
    # =========================================================================
    comptime use_fp8_16x16x128_mma = (
        Self.in_type == DType.float8_e4m3fn
        and Self.MMA_M == 16
        and Self.MMA_K == 128
    )
    comptime use_fp8_32x32x64_mma = (
        Self.in_type == DType.float8_e4m3fn
        and Self.MMA_M == 32
        and Self.MMA_K == 64
    )

    # Fragment widths:
    # - load_width: SIMD width for global→LDS (8 bf16, 16 fp8)
    # - lds_frag_width: Elements per LDS read (16 for FP8 16×16×128, else mma_frag_width)
    # - mma_frag_width: MMA hardware expectation = (MMA_M * MMA_K) // WARP_SIZE
    #   - BF16 16×16×32:  8 elements
    #   - FP8  16×16×128: 32 elements (loaded as 2×16 split-K)
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
        alignment = Self.alignment,
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
        alignment = Self.alignment,
    ]

    # Quadrant dimensions for tile view access
    comptime quadrant_m_size = Self.quadrant_m_mmas
    comptime quadrant_n_size = Self.quadrant_n_mmas * Self.accum_width

    # MMA LDS access layout: maps 64 lanes to (MMA_M × MMA_K) subtile
    # Shape (MMA_M, col_groups) with strides (MMA_K, lds_frag_width)
    # RuntimeLayout enables compile-time offset computation
    comptime col_groups = WARP_SIZE // Self.MMA_M
    comptime mma_access_layout = Layout(
        IntTuple(Self.MMA_M, Self.col_groups),
        IntTuple(Self.MMA_K, Self.lds_frag_width),
    )

    # Register tiles for A and B inputs
    var a_reg_tile: Self.ARegTile
    var b_reg_tile: Self.BRegTile

    # Single contiguous output register tile
    # Quadrants accessed via .tile[] views: out_reg_tile.tile[quadrant_m_size, quadrant_n_size](qa, qb)
    var out_reg_tile: Self.OutRegTile

    @always_inline
    fn __init__(out self):
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
    fn reset_accumulator(self):
        """Reset output register tile to zero."""
        _ = self.out_reg_tile.fill(0)

    @always_inline
    fn load_a[which: Int](self, smem_tile: SMemTile[Self.in_type, ...]):
        """Load A[which] from LDS → registers.

        Accepts SMemTile with matching dtype - layout compatibility validated
        at compile-time via load_lds_fragment constraints.

        For FP8 16×16×128: Uses lds_frag_width=16 with 2 K-iterations per MMA.
        For FP8 32×32×64:  Uses lds_frag_width=32 with single load.
        For BF16 16×16×32: Uses lds_frag_width=8.
        """
        # Vectorize with the LDS load width (16 for FP8 128, else mma_frag_width)
        var smem_frag = smem_tile.vectorize[1, Self.lds_frag_width]()
        var reg_frag = self.a_reg_tile.tile[
            Self.quadrant_m_mmas, Self.num_k_mmas * Self.mma_frag_width
        ](which, 0).vectorize[1, Self.lds_frag_width]()
        load_lds_fragment[
            mma_access_layout = Self.mma_access_layout,
            swizzle = Self.elem_swizzle,
        ](smem_frag, reg_frag)

    @always_inline
    fn load_b[which: Int](self, smem_tile: SMemTile[Self.in_type, ...]):
        """Load B[which] from LDS → registers.

        Accepts SMemTile with matching dtype - layout compatibility validated
        at compile-time via load_lds_fragment constraints.

        For FP8 16×16×128: Uses lds_frag_width=16 with 2 K-iterations per MMA.
        For FP8 32×32×64:  Uses lds_frag_width=32 with single load.
        For BF16 16×16×32: Uses lds_frag_width=8.
        """
        # Vectorize with the LDS load width (16 for FP8 128, else mma_frag_width)
        var smem_frag = smem_tile.vectorize[1, Self.lds_frag_width]()
        var reg_frag = self.b_reg_tile.tile[
            Self.quadrant_n_mmas, Self.num_k_mmas * Self.mma_frag_width
        ](which, 0).vectorize[1, Self.lds_frag_width]()
        load_lds_fragment[
            mma_access_layout = Self.mma_access_layout,
            swizzle = Self.elem_swizzle,
        ](smem_frag, reg_frag)

    @always_inline
    fn mma[which_a: Int, which_b: Int](self):
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
                    # stdlib mma() handles both BF16 and FP8 dispatch
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
    # MUST MATCH make_mma_swizzle exactly for read/write consistency!
    #
    # Configurations (matching make_mma_swizzle):
    #   BF16 16×16×32:  lds_frag=8  bytes=16  → Swizzle(1, 4, 4)
    #   FP8  16×16×128: lds_frag=16 bytes=16  → Swizzle(3, 4, 4)
    #   FP8  32×32×64:  lds_frag=32 bytes=32  → Swizzle(2, 5, 4)
    #
    # Note: BF16 read path uses (1,4,4) but write uses (1,5,4) - mismatch
    # tolerated due to interleaved layout. FP8 must match exactly.
    # =========================================================================

    # Compute lds_frag_width (same logic as make_mma_swizzle)
    comptime mma_frag_width = (
        16 * Self.MMA_K
    ) // WARP_SIZE  # Assume MMA_M=16 or 32
    comptime use_split_k = (
        Self.in_type == DType.float8_e4m3fn and Self.MMA_K == 128
    )
    comptime lds_frag_width = 16 if Self.use_split_k else Self.mma_frag_width

    # Swizzle parameters matching make_mma_swizzle
    comptime swizzle_log_tile = log2_floor(Self.MMA_K // 32) + 1
    comptime frag_bytes = Self.lds_frag_width * size_of[Self.in_type]()

    # BF16: use original (1, 5, 4) which works despite mismatch
    # FP8: use computed base to match make_mma_swizzle exactly
    comptime swizzle_subtile_cols = 4 * Self.load_width
    comptime elem_size = size_of[Self.in_type]()
    comptime swizzle_base = (
        log2_floor(Self.frag_bytes) if Self.in_type
        == DType.float8_e4m3fn else (
            log2_floor(Self.swizzle_subtile_cols // 2)
            + log2_floor(Self.elem_size)
        )
    )
    comptime swizzle_shift = 4

    comptime byte_swizzle = Optional(
        Swizzle(Self.swizzle_log_tile, Self.swizzle_base, Self.swizzle_shift)
    ) if Self.enable_swizzle else Optional[Swizzle]()

    # FP8 uses row-major LDS layout for correct partial block OOB handling
    comptime use_fp8_row_major = Self.in_type == DType.float8_e4m3fn

    # Half-tile dimensions (each warp group loads/uses one half independently)
    # Note: half_BM == WM, so A half-tile == warp's A region
    comptime half_BM = Self.BM // 2
    comptime half_BN = Self.BN // 2

    # MMA tile dimensions (2 tiles per warp dimension for quadrant processing)
    comptime mma_tile_m = Self.WM // 2
    comptime mma_tile_n = Self.WN // 2

    # Type aliases for shared memory tiles
    comptime SMemTile[rows: Int, cols: Int] = SMemTile[
        Self.in_type,
        Layout.row_major(rows, cols),
        alignment = Self.alignment,
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
    comptime vmcnt_per_load_a = (
        Self.BM // 2
    ) // Self.rows_per_load_iteration  # 8-warp A half
    comptime vmcnt_per_load_b = (
        Self.BN // 2
    ) // Self.rows_per_load_iteration  # 8-warp B half
    comptime vmcnt_per_load_ab = Self.vmcnt_per_load_a + Self.vmcnt_per_load_b  # Combined A+B

    # 4-warp loading counts (for load_a_as_group, load_b_as_group)
    comptime rows_per_iter_4warp = (
        4 * WARP_SIZE
    ) // Self.loads_per_row  # 32 rows
    comptime vmcnt_per_load_a_4warp = (
        Self.BM // 2
    ) // Self.rows_per_iter_4warp  # 4 ops
    comptime vmcnt_per_load_b_4warp = (
        Self.BN // 2
    ) // Self.rows_per_iter_4warp  # 4 ops

    # LDS pointer type aliases
    comptime smem_ptr = UnsafePointer[
        Scalar[Self.in_type], address_space = AddressSpace.SHARED
    ]

    # =========================================================================
    # TileLoader Configuration
    # =========================================================================
    comptime half_tile_layout = Layout.row_major(Self.half_BM, Self.BK)

    # TileLoader parameterized on source layout, with use_full_tile_width for FP8
    comptime TileLoader[src_layout: Layout] = TileLoaderLDS[
        Self.in_type,
        src_layout,
        Self.half_tile_layout,
        Self.loading_warps,
        Self.byte_swizzle,
        Self.load_width,
        Self.use_fp8_row_major,  # FP8: row-major for partial block support
    ]
    comptime ATileLoader = Self.TileLoader[Self.a_layout]
    comptime BTileLoader = Self.TileLoader[Self.b_layout]

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

    # 4-warp loading: row shift to remap warps 4-7 → 0-3
    var warp_shift_rows: Int

    @always_inline
    fn __init__(
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

        var b_half_idx, b_local_n = divmod(
            warp_id_n, 2
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

        # 4-warp loading: remap warps 4-7 to match row positions of warps 0-3
        # For FP8 row-major (BK=128, 2×4 warp grid): warps 0-3 all have warp_row=0
        # For BF16 interleaved (BK=64, 4×2 warp grid): warps 0-1 row=0, warps 2-3 row=1
        comptime num_warp_cols = Self.ATileLoader.num_warp_cols
        comptime thread_rows = Self.ATileLoader.thread_rows
        var group_warp_id = warp_id % 4
        var target_warp_row = group_warp_id // num_warp_cols
        var actual_warp_row = warp_id // num_warp_cols
        self.warp_shift_rows = (target_warp_row - actual_warp_row) * thread_rows

    # =========================================================================
    # 8-Warp Loading
    # =========================================================================

    @always_inline
    fn load_a[stage: Int, which: Int](self, *, k: Int):
        """Load A[stage][which] from global to LDS using all 8 warps."""
        self.loader_a.load_tile(
            self.a_load_tiles[stage][which],
            src_row=which * Self.half_BM,
            src_col=k,
        )

    @always_inline
    fn load_b[stage: Int, which: Int](self, *, k: Int):
        """Load B[stage][which] from global to LDS using all 8 warps."""
        self.loader_b.load_tile(
            self.b_load_tiles[stage][which],
            src_row=which * Self.half_BN,
            src_col=k,
        )

    # =========================================================================
    # 4-Warp Loading (for true ping-pong overlap)
    # =========================================================================

    @always_inline
    fn _load_tile_4warp[
        half_data_rows: Int,
        which: Int,
    ](
        self,
        loader: Self.TileLoader[_],
        dst_tile: Self.HalfTile[half_data_rows],
        k_offset: Int,
    ):
        """4-warp cooperative load: global → LDS.

        Uses warp_shift_rows to remap warps 4-7 → 0-3 so both warp groups
        (0-3 and 4-7) can independently load their tiles.
        """
        # Apply warp shift to thread position for group-local row
        var effective_thread_row = loader.thread_row + self.warp_shift_rows
        var group_warp_id = loader.warp_id % 4

        # Per-lane offset (constant across iterations) - uses vector registers
        # This is the thread-varying part that differs per lane
        var lane_offset = loader.thread_col + effective_thread_row * self.K

        # Correct row coverage for 4-warp loading:
        # - BF16 (4×2 grid): warps 0-1 row=0, warps 2-3 row=1 → 32 rows/iter
        # - FP8 (2×4 grid): warps 0-3 all row=0 → 32 rows/iter
        comptime num_warp_cols = Self.ATileLoader.num_warp_cols
        comptime thread_rows = Self.ATileLoader.thread_rows
        comptime num_warp_rows_in_4 = (4 + num_warp_cols - 1) // num_warp_cols
        comptime rows_per_iter_4warp = thread_rows * num_warp_rows_in_4
        comptime num_iterations = half_data_rows // rows_per_iter_4warp

        comptime for i in range(num_iterations):
            var tile_idx = i * 4 + group_warp_id
            var warp_subtile = dst_tile.tile[Self.rows_per_warp, Self.BK](
                tile_idx, 0
            )
            var smem_ptr = readfirstlane(warp_subtile.ptr)

            # Uniform offset (same for all lanes) - uses scalar registers
            var tile_row = which * half_data_rows + i * rows_per_iter_4warp
            var uniform_offset = k_offset + tile_row * self.K

            loader.buffer.load_to_lds[width = Self.load_width](
                Int32(lane_offset),
                smem_ptr,
                scalar_offset=Int32(uniform_offset),
            )

    # =========================================================================
    # 4-Warp Loading (for ping-pong overlap)
    # =========================================================================

    @always_inline
    fn load_a_as_group[
        stage: Int,
        target_group: Int,
    ](self, caller_group: Int, *, k: Int):
        """Load A[stage][target_group] from global to LDS using 4 warps."""
        if caller_group != target_group:
            return
        self._load_tile_4warp[which=target_group](
            self.loader_a, self.a_load_tiles[stage][target_group], k
        )

    @always_inline
    fn load_b_as_group[
        stage: Int,
        which: Int,
    ](self, caller_group: Int, loading_group: Int, *, k: Int):
        """Load B[stage][which] from global to LDS using 4 warps."""
        if caller_group != loading_group:
            return
        self._load_tile_4warp[which=which](
            self.loader_b, self.b_load_tiles[stage][which], k
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
    # Optional elementwise epilogue function (e.g., for FP8 per-tensor scaling)
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
    comptime LGKM_PER_LOAD_A = Self.quadrant_m_mmas * Self.num_k_mmas  # 4*2 = 8
    comptime LGKM_PER_LOAD_B = Self.quadrant_n_mmas * Self.num_k_mmas  # 2*2 = 4
    comptime LGKM_PER_LOAD_AB = Self.LGKM_PER_LOAD_A + Self.LGKM_PER_LOAD_B  # 12

    # Global → LDS (vmcnt): load_to_lds ops per buffers.load_* call (8-warp)
    comptime loads_per_row = Self.BK // Self.load_width  # 8
    comptime loading_threads_8warp = 8 * WARP_SIZE  # 512
    comptime rows_per_iter_8warp = Self.loading_threads_8warp // Self.loads_per_row  # 64
    comptime VMCNT_PER_LOAD_A = (Self.BM // 2) // Self.rows_per_iter_8warp  # 2
    comptime VMCNT_PER_LOAD_B = (Self.BN // 2) // Self.rows_per_iter_8warp  # 2

    # 4-warp loading (vmcnt): for load_a_as_group / load_b_as_group
    comptime loading_threads_4warp = 4 * WARP_SIZE  # 256
    comptime rows_per_iter_4warp = Self.loading_threads_4warp // Self.loads_per_row  # 32
    comptime VMCNT_PER_LOAD_A_4WARP = (
        Self.BM // 2
    ) // Self.rows_per_iter_4warp  # 4
    comptime VMCNT_PER_LOAD_B_4WARP = (
        Self.BN // 2
    ) // Self.rows_per_iter_4warp  # 4

    @staticmethod
    fn validate_config():
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
    fn matmul_ping_pong(
        a: LayoutTensor[
            Self.a_type,
            Self.a_layout,
            ImmutAnyOrigin,
            address_space = AddressSpace.GENERIC,
        ],
        b: LayoutTensor[
            Self.b_type,
            Self.b_layout,
            ImmutAnyOrigin,
            address_space = AddressSpace.GENERIC,
        ],
        c: LayoutTensor[
            Self.c_type,
            Self.c_layout,
            MutAnyOrigin,
            address_space = AddressSpace.GENERIC,
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
        var thread_id = Int(thread_idx.x)
        var lane_id = Int(lane_id())
        var warp_id = readfirstlane(Int(warp_id()))

        # Block coordinates from block indices
        var n = Int(block_idx.x) * BN
        var m = Int(block_idx.y) * BM

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
            MMA_K,  # For swizzle matching with MmaOp
            Self.config.num_threads(),
            alignment,
            Self.enable_swizzle,
            load_width,
        ]

        # MMA operation encapsulates register tiles and compute operations
        var mma_op = MmaOpType()

        # Warp position in 2×4 grid
        var warp_id_m, warp_id_n = divmod(warp_id, num_warps_n)

        # Tile Buffers struct
        var buffers = BuffersType(
            a, b, m, n, warp_id, warp_id_m, warp_id_n, lane_id
        )

        @always_inline
        fn s_barrier():
            llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

        @always_inline
        fn s_setprio[priority: Int16]():
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

        # Parameterized lgkmcnt values for compute_stage (BK-dependent)
        # lgkm_per_load_a = quadrant_m_mmas * num_k_mmas (8 for BK=64, 16 for BK=128)
        # lgkm_per_load_b = quadrant_n_mmas * num_k_mmas (4 for BK=64, 8 for BK=128)
        comptime lgkm_a = Self.LGKM_PER_LOAD_A
        comptime lgkm_b = Self.LGKM_PER_LOAD_B
        # Wait values: lgkmcnt=N means "wait until ≤N ops remain"
        # After issuing b[0], a[0], b[1], a[1]: total = 2*lgkm_a + 2*lgkm_b
        # To wait for b[0]+a[0]: lgkmcnt = lgkm_b + lgkm_a (ops remaining = b[1]+a[1])
        comptime lgkm_wait_a0_b0 = lgkm_b + lgkm_a  # 12 for BK=64
        comptime lgkm_wait_b1 = lgkm_a  # 8 for BK=64

        # Helper to extract compute logic (used by both groups)
        @parameter
        @always_inline
        fn compute_stage[stage: Int]():
            """Execute MMA operations for a given stage with fine-grained lgkmcnt.

            Dependencies:
              mma[0,0] ← load_a[0], load_b[0]
              mma[0,1] ← load_a[0], load_b[1]
              mma[1,0] ← load_a[1], load_b[0]
              mma[1,1] ← load_a[1], load_b[1]

            lgkmcnt tracking (FIFO order):
              load_b[0]: 4 ops  → cumulative 4
              load_a[0]: 8 ops  → cumulative 12
              load_b[1]: 4 ops  → cumulative 16
              load_a[1]: 8 ops  → cumulative 24

            Fine-grained waits:
              lgkmcnt=12: 12 complete → b[0], a[0] done
              lgkmcnt=8:  16 complete → b[1] done (a[1] has 8 in flight)
              lgkmcnt=0:  24 complete → a[1] done
            """

            # Issue all loads asynchronously (24 lgkm ops total)
            mma_op.load_b[0](buffers.b_mma_tiles[stage][0])  # +4 = 4
            mma_op.load_a[0](buffers.a_mma_tiles[stage][0])  # +8 = 12
            mma_op.load_b[1](buffers.b_mma_tiles[stage][1])  # +4 = 16
            mma_op.load_a[1](buffers.a_mma_tiles[stage][1])  # +8 = 24

            # Wait for b[0], a[0] complete
            s_waitcnt[lgkmcnt = UInt32(lgkm_wait_a0_b0)]()
            mma_op.mma[0, 0]()  # Uses a[0], b[0] ✓

            # Wait for b[1] complete (a[1] still in flight)
            s_waitcnt[lgkmcnt = UInt32(lgkm_wait_b1)]()
            mma_op.mma[0, 1]()  # Uses a[0] (done), b[1] ✓

            # Wait for a[1] (drain remaining)
            s_waitcnt[lgkmcnt=0]()
            mma_op.mma[1, 0]()  # Uses a[1] ✓, b[0] (done)
            mma_op.mma[1, 1]()  # Uses a[1] ✓, b[1] (done)

        # ================================================================
        # TILE ORGANIZATION AND PING-PONG SCHEDULE
        # ================================================================
        #
        # BLOCK TILE (256×256, 8 warps in 2×4 grid):
        # ┌─────────────┬─────────────┬─────────────┬─────────────┐
        # │   Warp 0    │   Warp 1    │   Warp 2    │   Warp 3    │ group 0
        # │   128×64    │   128×64    │   128×64    │   128×64    │
        # ├─────────────┼─────────────┼─────────────┼─────────────┤
        # │   Warp 4    │   Warp 5    │   Warp 6    │   Warp 7    │ group 1
        # │   128×64    │   128×64    │   128×64    │   128×64    │
        # └─────────────┴─────────────┴─────────────┴─────────────┘
        #   warp_id_m = warp_id // 4, warp_id_n = warp_id % 4
        #
        # DOUBLE BUFFERING (ping-pong):
        #   Stage 0: A[0][0,1], B[0][0,1]  ←→  Stage 1: A[1][0,1], B[1][0,1]
        #   K loop: compute[0] while load[1], then compute[1] while load[0]
        #
        # WARP OUTPUT QUADRANTS (128×64 → 4 quadrants):
        # ┌─────────────────┬─────────────────┐
        # │ mma[0,0] 64×32  │ mma[0,1] 64×32  │
        # ├─────────────────┼─────────────────┤
        # │ mma[1,0] 64×32  │ mma[1,1] 64×32  │
        # └─────────────────┴─────────────────┘
        #
        # ================================================================

        # Schedule selection:
        # - USE_SIMPLIFIED_SCHEDULE: Simple schedule with clear load/compute phases
        # - USE_FP8_HIPKITTENS_SCHEDULE: Optimized FP8 16×16×128 (from HipKittens)
        # - else: Optimized BF16 schedule with aggressive interleaving
        comptime USE_SIMPLIFIED_SCHEDULE = False
        comptime USE_FP8_HIPKITTENS_SCHEDULE = (
            in_type == DType.float8_e4m3fn and MMA_K == 128
        )

        comptime if USE_SIMPLIFIED_SCHEDULE:
            # ================================================================
            # SIMPLIFIED SCHEDULE
            # ================================================================
            # Clear load/compute phases with 4-warp group loading.
            # Warp group 0 and 1 alternate loading while the other computes.
            # ================================================================

            for k in range(0, K, BK * 2):
                # K offsets for this iteration
                var k0 = k  # Stage 0 K offset
                var k1 = k + BK  # Stage 1 K offset
                var k_next = k + 2 * BK  # Next iteration's stage 0

                if k == 0:
                    # Prologue: load stage 0
                    buffers.load_a_as_group[0, 0](warp_id_m, k=k0)
                    buffers.load_a_as_group[0, 1](warp_id_m, k=k0)
                    buffers.load_b_as_group[0, 0](warp_id_m, 0, k=k0)
                    buffers.load_b_as_group[0, 1](warp_id_m, 1, k=k0)
                    s_waitcnt[vmcnt=0]()
                    s_barrier()
                    if warp_id_m == 1:
                        s_barrier()  # Warp stagger

                # Load stage 1
                s_barrier()
                buffers.load_a_as_group[1, 0](warp_id_m, k=k1)
                buffers.load_a_as_group[1, 1](warp_id_m, k=k1)
                buffers.load_b_as_group[1, 0](warp_id_m, 0, k=k1)
                buffers.load_b_as_group[1, 1](warp_id_m, 1, k=k1)
                s_waitcnt[vmcnt=0]()

                # Compute stage 0
                s_barrier()
                compute_stage[0]()

                # Prefetch A for next iteration (overlaps with compute[1])
                if k < K - BK * 2:
                    buffers.load_a_as_group[0, 0](warp_id_m, k=k_next)
                    buffers.load_a_as_group[0, 1](warp_id_m, k=k_next)

                # Compute stage 1
                s_barrier()
                compute_stage[1]()

                # Prefetch B for next iteration
                if k < K - BK * 2:
                    s_barrier()
                    buffers.load_b_as_group[0, 0](warp_id_m, 0, k=k_next)
                    buffers.load_b_as_group[0, 1](warp_id_m, 1, k=k_next)
                    s_waitcnt[vmcnt=0]()

            # Epilogue: rebalance warp stagger
            if warp_id_m == 0:
                s_barrier()

        elif USE_FP8_HIPKITTENS_SCHEDULE:
            # ================================================================
            # FP8 PING-PONG SCHEDULE (HipKittens style)
            # ================================================================
            # Double-buffered schedule with 8-warp cooperative loading.
            # Based on HipKittens 8_wave.cu schedule.
            # ================================================================

            # Wait counts from HipKittens
            comptime lgkm_partial = 8  # lgkmcnt(8)

            # === PROLOGUE ===
            # Stage 0: all 4 tiles at k=0
            buffers.load_b[0, 0](k=0)
            buffers.load_a[0, 0](k=0)
            buffers.load_b[0, 1](k=0)
            buffers.load_a[0, 1](k=0)

            # Warp staggering
            if warp_id_m == 1:
                s_barrier()

            s_waitcnt[vmcnt=4]()
            s_barrier()

            # Stage 1: 3 tiles at k=0 (A[1,1] loaded in main loop)
            buffers.load_a[1, 0](k=0)
            buffers.load_b[1, 0](k=0)
            buffers.load_b[1, 1](k=0)

            s_waitcnt[vmcnt=6]()
            s_barrier()

            # === MAIN LOOP ===
            # Each iteration processes 2*BK elements (one full ping-pong cycle).
            # Loop variable k is the K offset for stage 0 prefetch.
            for k in range(BK, K - BK, 2 * BK):
                # K offsets: stage 1 completion matches prologue/previous prefetch,
                # stage 0 prefetch+completion at same K, stage 1 prefetch for next iter
                var k_complete_1 = k - BK  # Matches stage 1's other tiles
                var k_prefetch_0 = k  # All 4 stage 0 tiles at same K
                var k_complete_0 = k  # Matches stage 0's prefetch
                var k_prefetch_1 = k + BK  # All 4 stage 1 tiles at same K

                # --- First half: compute stage 0, prefetch stage 0 ---
                mma_op.load_b[0](buffers.b_mma_tiles[0][0])
                mma_op.load_a[0](buffers.a_mma_tiles[0][0])
                buffers.load_a[1, 1](
                    k=k_complete_1
                )  # Complete stage 1 (4th tile)
                s_waitcnt[lgkmcnt=lgkm_partial]()
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                mma_op.load_b[1](buffers.b_mma_tiles[0][1])
                buffers.load_a[0, 0](k=k_prefetch_0)
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 1]()
                s_setprio[0]()
                s_barrier()

                mma_op.load_a[1](buffers.a_mma_tiles[0][1])
                buffers.load_b[0, 0](k=k_prefetch_0)
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[1, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                buffers.load_b[0, 1](k=k_prefetch_0)
                s_waitcnt[vmcnt=6]()
                s_barrier()

                s_setprio[1]()
                mma_op.mma[1, 1]()
                s_setprio[0]()
                s_barrier()

                # --- Second half: compute stage 1, prefetch stage 1 ---
                mma_op.load_b[0](buffers.b_mma_tiles[1][0])
                mma_op.load_a[0](buffers.a_mma_tiles[1][0])
                buffers.load_a[0, 1](
                    k=k_complete_0
                )  # Complete stage 0 (4th tile)
                s_waitcnt[lgkmcnt=lgkm_partial]()
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                mma_op.load_b[1](buffers.b_mma_tiles[1][1])
                buffers.load_a[1, 0](k=k_prefetch_1)
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 1]()
                s_setprio[0]()
                s_barrier()

                mma_op.load_a[1](buffers.a_mma_tiles[1][1])
                buffers.load_b[1, 0](k=k_prefetch_1)
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[1, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                buffers.load_b[1, 1](k=k_prefetch_1)
                s_waitcnt[vmcnt=6]()
                s_barrier()

                s_setprio[1]()
                mma_op.mma[1, 1]()
                s_setprio[0]()
                s_barrier()

            # === EPILOGUE ===
            # Process final stage 0, then final stage 1
            mma_op.load_b[0](buffers.b_mma_tiles[0][0])
            mma_op.load_a[0](buffers.a_mma_tiles[0][0])
            buffers.load_a[1, 1](k=K - 2 * BK)  # Complete stage 1
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[0, 0]()
            s_setprio[0]()
            s_barrier()
            schedule_barrier()

            mma_op.load_b[1](buffers.b_mma_tiles[0][1])
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[0, 1]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_a[1](buffers.a_mma_tiles[0][1])
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[1, 0]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_b[0](buffers.b_mma_tiles[1][0])
            s_waitcnt[vmcnt=4]()
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[1, 1]()
            s_setprio[0]()
            s_barrier()

            # Last iteration (stage 1 at k = K - BK)
            mma_op.load_a[0](buffers.a_mma_tiles[1][0])
            s_waitcnt[vmcnt=2]()
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[0, 0]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_b[1](buffers.b_mma_tiles[1][1])
            s_waitcnt[vmcnt=0]()
            s_barrier()
            schedule_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[0, 1]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_a[1](buffers.a_mma_tiles[1][1])
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[1, 0]()
            mma_op.mma[1, 1]()
            s_setprio[0]()
            s_barrier()

            # Re-balance warp staggering: if (warp_m == 0) barrier
            if warp_id_m == 0:
                s_barrier()

        else:
            # ================================================================
            # BF16 PING-PONG SCHEDULE (HipKittens style)
            # ================================================================
            # Double-buffered schedule with 8-warp cooperative loading.
            # Each load specifies its K offset for clear data flow.
            # ================================================================

            # === PROLOGUE ===
            # Stage 0: all 4 tiles at k=0
            buffers.load_b[0, 0](k=0)
            buffers.load_a[0, 0](k=0)
            buffers.load_b[0, 1](k=0)
            buffers.load_a[0, 1](k=0)

            # Warp staggering
            if warp_id_m == 1:
                s_barrier()

            # HipKittens-style: vmcnt=4 after 4 loads (more conservative for BF16)
            s_waitcnt[vmcnt=4]()
            s_barrier()

            # Stage 1: 3 tiles at k=BK (A[1,1] loaded in main loop)
            buffers.load_b[1, 0](k=BK)
            buffers.load_a[1, 0](k=BK)
            buffers.load_b[1, 1](k=BK)

            # HipKittens-style: vmcnt=6 after 7 loads
            s_waitcnt[vmcnt=6]()
            s_barrier()

            # === MAIN LOOP ===
            # Each iteration processes 2*BK elements (one full ping-pong cycle).
            # Loop variable k is the K offset for stage 0 prefetch.
            for k in range(BK * 2, K, BK * 2):
                # K offsets: stage 1 completion matches prologue/previous prefetch,
                # stage 0 prefetch for current iter, stage 1 prefetch for next iter
                var k_complete_1 = k - BK  # Matches stage 1's other tiles
                var k_prefetch_0 = k  # All 4 stage 0 tiles at same K
                var k_prefetch_1 = k + BK  # All 4 stage 1 tiles at same K

                # --- First half: compute stage 0, prefetch stage 0 ---
                mma_op.load_b[0](buffers.b_mma_tiles[0][0])
                mma_op.load_a[0](buffers.a_mma_tiles[0][0])
                buffers.load_a[1, 1](
                    k=k_complete_1
                )  # Complete stage 1 (4th tile)
                s_waitcnt[lgkmcnt=8]()
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                mma_op.load_b[1](buffers.b_mma_tiles[0][1])
                buffers.load_b[0, 0](k=k_prefetch_0)
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 1]()
                s_setprio[0]()
                s_barrier()

                mma_op.load_a[1](buffers.a_mma_tiles[0][1])
                buffers.load_a[0, 0](k=k_prefetch_0)
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[1, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                mma_op.load_b[0](buffers.b_mma_tiles[1][0])
                buffers.load_b[0, 1](k=k_prefetch_0)
                s_waitcnt[vmcnt=6]()
                s_barrier()

                s_setprio[1]()
                mma_op.mma[1, 1]()
                s_setprio[0]()
                s_barrier()

                # --- Second half: compute stage 1, prefetch stage 1 ---
                mma_op.load_a[0](buffers.a_mma_tiles[1][0])
                # Complete stage 0 (4th tile)
                buffers.load_a[0, 1](k=k_prefetch_0)
                s_waitcnt[lgkmcnt=8]()
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                mma_op.load_b[1](buffers.b_mma_tiles[1][1])
                buffers.load_b[1, 0](k=k_prefetch_1)
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 1]()
                s_setprio[0]()
                s_barrier()

                mma_op.load_a[1](buffers.a_mma_tiles[1][1])
                buffers.load_a[1, 0](k=k_prefetch_1)
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[1, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                buffers.load_b[1, 1](k=k_prefetch_1)
                s_waitcnt[vmcnt=6]()
                s_barrier()

                s_setprio[1]()
                mma_op.mma[1, 1]()
                s_setprio[0]()
                s_barrier()

            # === EPILOGUE ===
            # Process final stage 0, then final stage 1
            mma_op.load_b[0](buffers.b_mma_tiles[0][0])
            mma_op.load_a[0](buffers.a_mma_tiles[0][0])
            # Complete stage 1
            buffers.load_a[1, 1](k=K - BK)
            s_barrier()
            s_waitcnt[lgkmcnt=0]()

            s_setprio[1]()
            mma_op.mma[0, 0]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_b[1](buffers.b_mma_tiles[0][1])
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[0, 1]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_a[1](buffers.a_mma_tiles[0][1])
            s_waitcnt[vmcnt=0]()  # 4
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[1, 0]()
            mma_op.mma[1, 1]()
            s_setprio[0]()
            s_barrier()

            # Final stage 1
            mma_op.load_b[0](buffers.b_mma_tiles[1][0])
            mma_op.load_a[0](buffers.a_mma_tiles[1][0])
            s_waitcnt[vmcnt=0]()  # 2
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[0, 0]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_b[1](buffers.b_mma_tiles[1][1])
            s_waitcnt[vmcnt=0]()
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[0, 1]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_a[1](buffers.a_mma_tiles[1][1])
            s_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[1, 0]()
            mma_op.mma[1, 1]()
            s_setprio[0]()
            s_barrier()

            # Re-balance warp staggering
            if warp_id_m == 0:
                s_barrier()

        # ================================================================
        # Output Store using write_output_fragments (reused from matmul.mojo)
        # ================================================================
        # Thread layout for output: maps lane_id to (m_offset, n_offset)
        # For 16×16 MMA with accum_width=4:
        #   lane_crd[0] = lane_id % 16 (m offset within MMA tile)
        #   lane_crd[1] = lane_id // 16 (n group, multiplied by accum_width)
        comptime output_thread_layout = Layout(
            IntTuple(MMA_M, WARP_SIZE // MMA_M),
            IntTuple(1, MMA_M),
        )

        # Fragment layout: (num_m_mmas, num_n_mmas, accum_width)
        # Strides must match out_reg_layout = row_major(num_m_mmas, num_n_mmas * accum_width)
        # Element at (m, n, f) is at: m * (num_n_mmas * accum_width) + n * accum_width + f
        comptime out_frag_layout = Layout(
            IntTuple(num_m_mmas, num_n_mmas, accum_width),
            IntTuple(num_n_mmas * accum_width, accum_width, 1),
        )

        # Reshape MMA output to fragment view and vectorize
        var c_reg_fragment = mma_op.out_reg_tile.reshape[
            out_frag_layout
        ]().vectorize[1, 1, accum_width]()

        # Create warp tile in output matrix
        var warp_tile_m = m + WM * warp_id_m
        var warp_tile_n = n + WN * warp_id_n

        var c_warp_tile = c.tile[WM, WN](warp_tile_m // WM, warp_tile_n // WN)

        # Create global memory fragment distributed across threads
        var c_gmem_fragment = c_warp_tile.vectorize[
            1, accum_width
        ]().distribute[output_thread_layout](UInt(lane_id))

        # Use shared write_output_fragments function
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


@always_inline
fn ping_pong_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    //,
    enable_swizzle: Bool = True,
](
    a_device_tensor: LayoutTensor[a_type, a_layout],
    b_device_tensor: LayoutTensor[b_type, b_layout],
    c_device_tensor: LayoutTensor[c_type, c_layout],
    ctx: DeviceContext,
) raises:
    comptime assert a_type == b_type, "A and B must have the same type"
    comptime assert (
        a_type == DType.bfloat16 or a_type == DType.float8_e4m3fn
    ), "A must be bfloat16 or float8_e4m3fn"

    # Select kernel based on input dtype
    comptime is_fp8 = a_type == DType.float8_e4m3fn

    # Block tile dimensions (used by all configurations)
    comptime BM = 256
    comptime BN = 256

    var N = c_device_tensor.dim(1)
    var M = c_device_tensor.dim(0)

    @always_inline
    @parameter
    fn run_kernel[config: KernelConfig]() raises:
        comptime kernel = AMDPingPongMatmul[
            a_type,
            b_type,
            c_type,
            a_layout,
            b_layout,
            c_layout,
            config,
            enable_swizzle,
        ].matmul_ping_pong

        ctx.enqueue_function[kernel, kernel](
            a_device_tensor,
            b_device_tensor,
            c_device_tensor,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=config.num_threads(),
        )

    # Dispatch: FP8 uses 32×32×64 MMA with BK=128 (best perf: ~2.9 TFLOPS)
    # Fallback to 16×16×128 for M % 32 != 0 (still high perf, supports any M)
    # BF16 uses 16×16×32 with BK=64
    #
    # Swizzle enabled for all configs (each has matched write/read swizzle patterns)
    comptime if is_fp8:
        # FP8 32×32×64 MMA with BK=128 (split-K: 2 MMAs per tile)
        # Requires M % 32 == 0: transpose reads mix data across 32 rows,
        # and split-K offsets corrupt partial groups with OOB zeros
        comptime BK = 128
        comptime config_32x32 = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(32, 32, 64),
        )
        # FP8 16×16×128 fallback for unaligned M
        comptime config_16x16 = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(16, 16, 128),
        )

        if M % 32 == 0:
            run_kernel[config_32x32]()
        else:
            run_kernel[config_16x16]()
    else:
        # BF16: 16×16×32 MMA with BK=64 (~1.55 TFLOPS)
        comptime BK = 64
        comptime config = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(16, 16, 32),
        )
        run_kernel[config]()
