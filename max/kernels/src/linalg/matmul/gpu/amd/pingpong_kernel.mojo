# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from stdlib.bit import log2_floor
from sys import simd_width_of, size_of

from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    grid_dim,
)
from gpu import warp_id as get_warp_id
from gpu.host import DeviceContext
from gpu.intrinsics import AMDBufferResource
from gpu.memory import AddressSpace
from gpu.mma import mma
from gpu.sync import barrier, schedule_barrier, s_waitcnt
from memory import LegacyUnsafePointer as UnsafePointer

from utils import Index, IndexList, StaticTuple
from utils.numerics import get_accum_type

from sys.intrinsics import readfirstlane, llvm_intrinsic
from sys._assembly import inlined_assembly
from os.atomic import Atomic

from gpu._utils import to_i64

from collections import OptionalReg

from ....structuring import SMemTileType, RegTileType, eval
from layout import Layout, LayoutTensor, IntTuple, RuntimeLayout
from layout.swizzle import Swizzle
from layout._utils import make_amd_buffer_resource
from layout.tensor_core import load_b_nt


# =============================================================================
# Layout-Based Memory Access Patterns
# =============================================================================
#
# This kernel uses Layout and RuntimeLayout for structured memory access:
#
# 1. GLOBAL → LDS LOADING (TileBuffers.__init__, _load_tile_to_lds_*)
#    - Thread layout: Layout.row_major(16, 4) maps 64 threads to 16×32 subtile
#    - Swizzle: Swizzle(1, 5, 4) XORs bit 9→5 in byte offsets
#    - LayoutTensor.distribute() computes per-thread global positions
#    - 8 warps fill 64 rows × 64 cols per iteration
#
# 2. LDS → REGISTER LOADING (MmaOp.load_a/b)
#    - MMA access layout: Layout((16, 4), (32, 8))
#    - RuntimeLayout converts lane_id to LDS offset at compile time
#    - Element swizzle: Swizzle(1, 4, 4) applied to computed offsets
#
# 3. TILE-BASED ADDRESSING (_load_tile_to_lds_*, _load_tile_4warp_*)
#    - Pass LayoutTensor tiles directly to loading functions
#    - Warp position computed via tile[rows_per_warp, BK](tile_idx, 0).ptr
#    - readfirstlane() ensures scalar (SGPR) pointer for load_to_lds
#
# Swizzle Pattern (Swizzle(1, 5, 4) for bytes, Swizzle(1, 4, 4) for elements):
#    - AMD MI300X/MI355X: 64 LDS banks × 4 bytes = 256 bytes per cycle
#    - Without swizzle: MMA 4×16 read causes 4-way bank conflicts
#    - Swizzle XORs bit 9→5: breaks conflicts within 16×32 bf16 subtiles
#    - Result: ~6% performance improvement
#
# =============================================================================


from gpu.memory import CacheOperation
from gpu._utils import to_i32, to_i64


@always_inline
fn _load_to_lds[
    dtype: DType,
    *,
    width: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](
    resource: AMDBufferResource,
    vector_offset: Int32,
    shared_ptr: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ],
    *,
    scalar_offset: Int32 = 0,
):
    comptime bytes = size_of[dtype]() * width
    comptime aux = 0  # _cache_operation_to_amd_aux[cache_policy]()
    var vector_offset_bytes = vector_offset * size_of[dtype]()
    var scalar_offset_bytes = scalar_offset * size_of[dtype]()

    comptime alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
    comptime no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`

    # Create a null pointer in address space 8 (BUFFER_RESOURCE)
    var desc_ptr_ = UnsafePointer[
        Scalar[DType.bfloat16], address_space = AddressSpace.BUFFER_RESOURCE
    ]()

    var ptr_to_ptr = UnsafePointer(to=desc_ptr_)
    var ptr_to_simd = UnsafePointer(to=resource.desc)
    ptr_to_ptr[0] = ptr_to_simd.bitcast[
        UnsafePointer[
            Scalar[DType.bfloat16],
            address_space = AddressSpace.BUFFER_RESOURCE,
        ]
    ]()[0]

    # Cast the null pointer to LLVM pointer type
    var desc_ptr_llvm = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr<8>`
    ](desc_ptr_)

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr<3>`
    ](shared_ptr)

    __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
        # no_alias_scope=no_alias_scope_attr,
        alias_scopes=alias_scope_attr,
        _type=None,
    ](
        desc_ptr_llvm,
        shared_ptr3,
        to_i32(bytes),
        to_i32(vector_offset_bytes),
        to_i32(scalar_offset_bytes),
        to_i32(0),
        to_i32(aux),
    )


# =============================================================================
# TileLoaderLDS: Cooperative Global→LDS Tile Loader
# =============================================================================


@register_passable("trivial")
struct TileLoaderLDS[
    dtype: DType,
    src_layout: Layout,  # Full tensor layout (stride = shape[1])
    src_tile_layout: Layout,  # Tile shape for loading geometry
    num_loading_warps: Int,
    swizzle: OptionalReg[Swizzle] = OptionalReg[Swizzle](),
    load_width: Int = simd_width_of[dtype](),
]:
    """Encapsulates load_to_lds with pre-computed thread positions and swizzle.
    """

    # Tile geometry (derived from src_tile_layout at compile time)
    comptime tile_rows = Self.src_tile_layout.shape[0].value()
    comptime tile_cols = Self.src_tile_layout.shape[1].value()

    # Thread layout geometry (derived from subtile organization)
    # Each warp covers a 16×32 subtile: 4 threads per row × 8 elements = 32 cols
    comptime subtile_cols = 32
    comptime threads_per_row = Self.subtile_cols // Self.load_width  # 4
    comptime thread_rows = WARP_SIZE // Self.threads_per_row  # 16
    comptime threads_per_warp = WARP_SIZE  # 64

    # Per-warp coverage
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
    fn __init__(out self, src: LayoutTensor, warp_id: Int, lane_id: Int):
        """Pre-compute thread position with swizzle inversion for bank-conflict-free reads.
        """
        self.buffer = make_amd_buffer_resource(src)
        self.warp_id = warp_id

        # Swizzle inversion: load from swizzle(T) so writing to T is conflict-free
        var effective_lane = lane_id

        @parameter
        if Self.swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        # 8 warps as 4×2 grid, each covers 16×32 subtile
        var warp_row = warp_id // 2
        var warp_col = warp_id % 2
        var subtile_row = effective_lane // Self.threads_per_row
        var subtile_col = (
            effective_lane % Self.threads_per_row
        ) * Self.load_width

        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

    @always_inline
    fn load_tile[
        dst_layout: Layout, //,
    ](
        self,
        dst: SMemTileType[Self.dtype, dst_layout, **_],
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
        # Only participating warps execute
        if self.warp_id >= Self.num_loading_warps:
            return

        # Base offset: column position (thread_col relative to src_col)
        var col_offset = src_col + self.thread_col

        @parameter
        for i in range(Self.num_iterations):
            var tile_idx = i * Self.num_loading_warps + self.warp_id
            var warp_subtile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                tile_idx, 0
            )
            var smem_ptr = readfirstlane(warp_subtile.ptr)

            # Row position: src_row + thread_row + iteration offset
            var row = src_row + self.thread_row + i * Self.rows_per_iteration
            var offset = col_offset + row * self.stride

            self.buffer.load_to_lds[width = Self.load_width](
                Int32(offset), smem_ptr, scalar_offset=0
            )


@always_inline
fn _load_from_lds[
    dtype: DType, //,
    width: Int = 1,
](
    shared_ptr: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ],
) -> SIMD[dtype, width]:
    """Load a SIMD vector from LDS (shared memory) with alias annotations.

    This function generates an LLVM load with proper alias scopes to help
    the compiler distinguish between async copy operations and local loads,
    enabling better instruction scheduling.

    Parameters:
        dtype: The data type of elements to load.
        width: Number of elements to load (SIMD width).

    Args:
        shared_ptr: Pointer to shared memory location.

    Returns:
        SIMD vector of `width` elements of type `dtype`.
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
    @parameter
    if dtype == DType.bfloat16 and width == 8:
        comptime use_asm = False

        @parameter
        if use_asm:
            return inlined_assembly[
                "ds_read_b128 $0, $1",
                SIMD[dtype, width],
                constraints="=v,v,~{memory}",
                has_side_effect=True,
            ](shared_ptr3)
        else:
            var llvm_res = __mlir_op.`llvm.load`[
                _type = __mlir_type.`vector<8 x bf16>`,
                alignment = to_i64(alignment),
                noalias_scopes=alias_scope_attr,
                alias_scopes=no_alias_scope_attr,
            ](shared_ptr3)
            return rebind[SIMD[dtype, width]](
                __mlir_op.`pop.cast_from_builtin`[
                    _type = SIMD[DType.bfloat16, 8]._mlir_type
                ](llvm_res)
            )
    else:
        constrained[
            False, "Unsupported dtype/width combination for _load_from_lds"
        ]()
        return SIMD[dtype, width]()


@always_inline
fn load_lds_fragment[
    dtype: DType,
    smem_layout: Layout,
    smem_element_layout: Layout,
    frag_layout: Layout,
    frag_element_layout: Layout, //,
    mma_access_layout: Layout,
    swizzle: OptionalReg[Swizzle] = OptionalReg[Swizzle](),
](
    smem_tile: SMemTileType[
        dtype, smem_layout, element_layout=smem_element_layout, **_
    ],
    reg_frag: RegTileType[
        dtype, frag_layout, element_layout=frag_element_layout, **_
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
    constrained[
        mma_access_layout.size() == WARP_SIZE,
        String(
            "mma_access_layout must map exactly ",
            WARP_SIZE,
            " threads, got ",
            mma_access_layout.size(),
        ),
    ]()

    # smem must have enough elements for all iterations
    # Each iteration: WARP_SIZE threads × frag_width elements per thread
    comptime smem_elements = smem_layout.size() * smem_element_layout.size()
    comptime required_smem = num_iterations * WARP_SIZE * frag_width
    constrained[
        smem_elements >= required_smem,
        String(
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
        ),
    ]()

    # frag must hold all loaded elements
    comptime frag_elements = frag_layout.size() * frag_element_layout.size()
    comptime required_frag = num_iterations * frag_width
    constrained[
        frag_elements == required_frag,
        String(
            "frag has ",
            frag_elements,
            " elements but expects ",
            required_frag,
            " (iterations=",
            num_iterations,
            " × width=",
            frag_width,
            ")",
        ),
    ]()

    # =========================================================================

    var lane = Int(lane_id())
    # RuntimeLayout wraps compile-time layout for efficient evaluation
    var lane_offset = Int(RuntimeLayout[mma_access_layout]()(lane))
    var frag_ptr = reg_frag.ptr.bitcast[FragElement]()

    @parameter
    for i in range(num_iterations):
        var iter_base = i * WARP_SIZE * frag_width
        var full_offset = iter_base + lane_offset

        @parameter
        if swizzle:
            full_offset = swizzle.value()(full_offset)

        frag_ptr[i] = rebind[FragElement](
            _load_from_lds[width=frag_width](smem_tile.ptr.offset(full_offset))
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
        @parameter
        for i in range(list.size):
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
    enable_swizzle: Bool,
    swizzle_elem_base: Int,  # From loading pattern - log2(subtile_cols // 2)
    swizzle_shift: Int,  # From loading pattern - log2(subtile_rows)
]:
    """Encapsulates MMA register tiles and operations for matrix multiplication.

    This struct manages register tiles and MMA operations for a single warp.
    It processes warp-sized tiles (WM × BK for A, WN × BK for B) without
    knowledge of the broader kernel architecture.

    MmaOp accepts generic SMemTileType and validates compatibility at
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

    Swizzle Configuration (enable_swizzle=True):
    MmaOp receives swizzle parameters from the kernel/TileBuffers, since they
    are determined by how data is loaded into LDS. MmaOp must read using
    the same swizzle pattern that was used for writing.
    - swizzle_elem_base: bit position for XOR (from loading subtile width)
    - swizzle_shift: XOR source distance (from loading subtile rows)
    """

    # Derived values (computed from other parameters)
    comptime num_m_mmas = Self.WM // Self.MMA_M
    comptime num_n_mmas = Self.WN // Self.MMA_N
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime load_width = simd_width_of[Self.in_type]()
    comptime accum_width = (Self.MMA_M * Self.MMA_N) // WARP_SIZE

    # Quadrant dimensions (warp tile divided into 4 quadrants for MMA scheduling)
    comptime quadrant_m_mmas = Self.num_m_mmas // 2
    comptime quadrant_n_mmas = Self.num_n_mmas // 2

    # =========================================================================
    # Swizzle Configuration (received from kernel/TileBuffers)
    # =========================================================================
    # MmaOp receives swizzle parameters that match how TileBuffers loads data.
    # The element swizzle is used when reading from LDS to registers.
    # =========================================================================
    comptime elem_swizzle = OptionalReg(
        Swizzle(1, Self.swizzle_elem_base, Self.swizzle_shift)
    ) if Self.enable_swizzle else OptionalReg[Swizzle]()

    # =========================================================================
    # Async LDS Load Counts (for s_waitcnt[lgkmcnt=N])
    # =========================================================================
    # Each _load_from_lds call issues one ds_read instruction.
    # load_a/load_b iterate `half_*_mmas * num_k_mmas` times.
    # These aliases enable precise lgkmcnt tracking in the optimized schedule.
    # =========================================================================
    comptime lgkm_per_load_a = Self.quadrant_m_mmas * Self.num_k_mmas  # ds_read ops per load_a[which]
    comptime lgkm_per_load_b = Self.quadrant_n_mmas * Self.num_k_mmas  # ds_read ops per load_b[which]
    comptime lgkm_per_load_ab = Self.lgkm_per_load_a + Self.lgkm_per_load_b  # Combined A+B load

    # Register tile type aliases
    comptime RegTileType[num_mmas: Int] = RegTileType[
        Self.in_type,
        Layout.row_major(num_mmas, Self.num_k_mmas * Self.load_width),
        alignment = Self.alignment,
    ]
    comptime ARegTileType = Self.RegTileType[Self.num_m_mmas]
    comptime BRegTileType = Self.RegTileType[Self.num_n_mmas]

    # Output layout: Four separate contiguous quadrants
    # Each quadrant is (quadrant_m_mmas, quadrant_n_mmas * accum_width) for MMA scheduling
    # This gives truly contiguous memory for each mma[which_a, which_b] operation.
    comptime OutQuadrantType = RegTileType[
        Self.accum_type,
        Layout.row_major(
            Self.quadrant_m_mmas, Self.quadrant_n_mmas * Self.accum_width
        ),
        alignment = Self.alignment,
    ]

    # MMA LDS Read Access Layout (using RuntimeLayout for compile-time evaluation)
    #
    # AMD WMMA/MMA lane assignment for bf16 splits 64 lanes into 4 groups of 16:
    #   - lane // 16 gives group index (0-3)
    #   - lane % 16 gives position within group (0-15)
    #
    # Layout((16, 4), (32, 8)) maps lane_id to LDS offset:
    #   - Shape (16, 4): decompose lane as col = lane % 16, row = lane // 16
    #   - Stride (32, 8): compute offset as col * 32 + row * 8
    #   - Equivalent to: (lane // 16) * 8 + (lane % 16) * 32
    #
    # RuntimeLayout[layout]() enables compile-time offset computation when
    # the layout has all known dimensions, avoiding GPU heap allocation.
    comptime mma_access_layout = Layout(
        IntTuple(16, 4), IntTuple(4 * Self.load_width, Self.load_width)
    )

    # Register tiles for A and B inputs
    var a_reg_tile: Self.ARegTileType
    var b_reg_tile: Self.BRegTileType

    # Four separate output quadrants - each is contiguous in memory
    # Indexed by [which_a][which_b]: out_quadrants[0][0], out_quadrants[0][1], etc.
    var out_quadrants: StaticTuple[StaticTuple[Self.OutQuadrantType, 2], 2]

    @always_inline
    fn __init__(out self):
        """Initialize MMA operation with register tiles."""
        constrained[Self.WM % Self.MMA_M == 0]()
        constrained[Self.WN % Self.MMA_N == 0]()
        constrained[Self.BK % Self.MMA_K == 0]()
        constrained[(Self.MMA_M * Self.MMA_N) % WARP_SIZE == 0]()

        self.a_reg_tile = Self.ARegTileType.stack_allocation()
        self.b_reg_tile = Self.BRegTileType.stack_allocation()

        # Initialize all four output quadrants to zero
        self.out_quadrants = StaticTuple[
            StaticTuple[Self.OutQuadrantType, 2], 2
        ](
            StaticTuple[Self.OutQuadrantType, 2](
                Self.OutQuadrantType.stack_allocation().fill(0),
                Self.OutQuadrantType.stack_allocation().fill(0),
            ),
            StaticTuple[Self.OutQuadrantType, 2](
                Self.OutQuadrantType.stack_allocation().fill(0),
                Self.OutQuadrantType.stack_allocation().fill(0),
            ),
        )

    @always_inline
    fn reset_accumulator(self):
        """Reset all output quadrants to zero."""
        _ = self.out_quadrants[0][0].fill(0)
        _ = self.out_quadrants[0][1].fill(0)
        _ = self.out_quadrants[1][0].fill(0)
        _ = self.out_quadrants[1][1].fill(0)

    @always_inline
    fn load_a[which: Int](self, smem_tile: SMemTileType[Self.in_type, **_]):
        """Load A[which] from LDS → registers.

        Accepts SMemTileType with matching dtype - layout compatibility validated
        at compile-time via load_lds_fragment constraints.
        """
        var smem_frag = smem_tile.vectorize[1, Self.load_width]()
        var reg_frag = self.a_reg_tile.tile[
            Self.quadrant_m_mmas, Self.num_k_mmas * Self.load_width
        ](which, 0).vectorize[1, Self.load_width]()
        load_lds_fragment[
            mma_access_layout = Self.mma_access_layout,
            swizzle = Self.elem_swizzle,
        ](smem_frag, reg_frag)

    @always_inline
    fn load_b[which: Int](self, smem_tile: SMemTileType[Self.in_type, **_]):
        """Load B[which] from LDS → registers.

        Accepts SMemTileType with matching dtype - layout compatibility validated
        at compile-time via load_lds_fragment constraints.
        """
        var smem_frag = smem_tile.vectorize[1, Self.load_width]()
        var reg_frag = self.b_reg_tile.tile[
            Self.quadrant_n_mmas, Self.num_k_mmas * Self.load_width
        ](which, 0).vectorize[1, Self.load_width]()
        load_lds_fragment[
            mma_access_layout = Self.mma_access_layout,
            swizzle = Self.elem_swizzle,
        ](smem_frag, reg_frag)

    @always_inline
    fn load_b_with_transpose[
        which: Int
    ](self, smem_tile: SMemTileType[Self.in_type, **_]):
        """Load B[which] from LDS → registers using hardware transpose.

        Uses ds_read_tr16_b64 instruction for efficient transposed LDS read.
        This function expects B tiles in (N, K) storage order and produces
        data in the format expected by AMD MFMA instructions.

        Supports swizzle: When enable_swizzle is True, applies the byte swizzle
        pattern to LDS read offsets for bank-conflict-free access.

        Requires: MMA shape must be 16x16x32 or 32x32x16 (double-rate MFMA).

        Args:
            smem_tile: B tile in LDS with shape (mma_tile_n, BK) = (N, K) order.
        """
        # Validate MMA shape supports load_b_nt
        comptime mma_shape = IndexList[3](Self.MMA_M, Self.MMA_N, Self.MMA_K)
        constrained[
            mma_shape in (IndexList[3](32, 32, 16), IndexList[3](16, 16, 32)),
            "load_b_with_transpose requires 16x16x32 or 32x32x16 MMA shape",
        ]()

        # Get register tile for this quadrant
        # Shape: (quadrant_n_mmas, num_k_mmas * load_width) = (2, 16) vectorized
        var reg_frag = self.b_reg_tile.tile[
            Self.quadrant_n_mmas, Self.num_k_mmas * Self.load_width
        ](which, 0)

        # smem_tile shape: (mma_tile_n, BK) = (32, 64) = (2*MMA_N, 2*MMA_K)
        # Load each (MMA_N, MMA_K) = (16, 32) subtile using hardware transpose
        @parameter
        for n_idx in range(Self.quadrant_n_mmas):

            @parameter
            for k_idx in range(Self.num_k_mmas):
                # Extract (MMA_N, MMA_K) subtile from smem_tile
                var subtile = smem_tile.tile[Self.MMA_N, Self.MMA_K](
                    n_idx, k_idx
                )

                # Load with hardware transpose: converts (N, K) → (K, N) for MMA
                var b_values = load_b_nt[mma_shape, Self.elem_swizzle](subtile)

                # Store to register tile at position (n_idx, k_idx * load_width)
                # reg_frag[n_idx, k_idx] corresponds to load_width elements
                reg_frag.ptr.offset(
                    n_idx * Self.num_k_mmas * Self.load_width
                    + k_idx * Self.load_width
                ).store(b_values)

    @always_inline
    fn mma[which_a: Int, which_b: Int](self):
        """Execute MMA operations for a quadrant of the output tile.

        Each quadrant is stored in a separate contiguous register tile.

        Parameters:
            which_a: A quadrant index (0 or 1).
            which_b: B quadrant index (0 or 1).
        """
        # Create vectorized views of the A and B quadrants
        var a_mma_frag = self.a_reg_tile.tile[
            Self.quadrant_m_mmas, Self.num_k_mmas * Self.load_width
        ](which_a, 0).vectorize[1, Self.load_width]()

        var b_mma_frag = self.b_reg_tile.tile[
            Self.quadrant_n_mmas, Self.num_k_mmas * Self.load_width
        ](which_b, 0).vectorize[1, Self.load_width]()

        # Access the contiguous quadrant directly
        var c_accum_frag = self.out_quadrants[which_a][which_b].vectorize[
            1, Self.accum_width
        ]()

        @parameter
        for k in range(Self.num_k_mmas):

            @parameter
            for m in range(Self.quadrant_m_mmas):

                @parameter
                for n in range(Self.quadrant_n_mmas):
                    mma(
                        c_accum_frag[m, n],
                        b_mma_frag[n, k],
                        a_mma_frag[m, k],
                        c_accum_frag[m, n],
                    )


struct TileBuffers[
    in_type: DType,
    a_layout: Layout,
    b_layout: Layout, //,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_threads: Int,
    alignment: Int,
    enable_swizzle: Bool,
    load_width: Int,  # From MmaOp - how consumer wants data vectorized
    loading_warps: Int = 8,  # Number of warps participating in each load (4 or 8)
]:
    """Double-buffered LDS tiles and TileLoaders for ping-pong matmul.

    a_layout and b_layout are infer-only parameters (note `//`), automatically
    extracted from the input tensors passed to __init__. K is derived as a
    comptime from a_layout.shape[1].
    """

    # Swizzle derived from 16×4 thread layout
    comptime swizzle_subtile_rows = 16
    comptime swizzle_subtile_cols = 4 * Self.load_width
    comptime elem_size = size_of[Self.in_type]()

    comptime swizzle_elem_base = log2_floor(Self.swizzle_subtile_cols // 2)
    comptime swizzle_byte_base = Self.swizzle_elem_base + log2_floor(
        Self.elem_size
    )
    comptime swizzle_shift = log2_floor(Self.swizzle_subtile_rows)
    comptime byte_swizzle = OptionalReg(
        Swizzle(1, Self.swizzle_byte_base, Self.swizzle_shift)
    ) if Self.enable_swizzle else OptionalReg[Swizzle]()

    # Half-tile dimensions (each warp group loads/uses one half independently)
    # Note: half_BM == WM, so A half-tile == warp's A region
    comptime half_BM = Self.BM // 2
    comptime half_BN = Self.BN // 2

    # MMA tile dimensions (2 tiles per warp dimension for quadrant processing)
    comptime mma_tile_m = Self.WM // 2
    comptime mma_tile_n = Self.WN // 2

    # Type aliases for shared memory tiles
    comptime SMemTile[rows: Int, cols: Int] = SMemTileType[
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

    # =========================================================================
    # Async Global Load Counts (for s_waitcnt[vmcnt=N])
    # =========================================================================
    # Each _load_to_lds call issues one buffer load instruction (vmcnt).
    # These aliases enable precise vmcnt tracking in the optimized schedule.
    #
    # 8-warp loading (buffers.load_a, buffers.load_b):
    #   - half_rows = BM/2 = 128 rows
    #   - rows_per_iteration = loading_threads / loads_per_row = 512/8 = 64
    #   - iterations = 128 / 64 = 2 load_to_lds per half-tile
    #
    # 4-warp loading (buffers.load_a_as_group, buffers.load_b_as_group):
    #   - half_rows = BM/2 = 128 rows
    #   - rows_per_iter_4warp = (4*64) / 8 = 32
    #   - iterations = 128 / 32 = 4 load_to_lds per half-tile
    # =========================================================================
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
    # TileLoader parameterized on source layout (inferred from a_layout/b_layout)
    comptime half_tile_layout = Layout.row_major(Self.half_BM, Self.BK)
    comptime TileLoader[src_layout: Layout] = TileLoaderLDS[
        Self.in_type,
        src_layout,
        Self.half_tile_layout,
        Self.loading_warps,
        Self.byte_swizzle,
        Self.load_width,
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
    var k_offset: Int

    # 4-warp loading: row shift to remap warps 4-7 → 0-3
    var warp_shift_rows: Int

    @always_inline
    fn __init__(
        out self,
        a: LayoutTensor[Self.in_type, Self.a_layout, *_, **_],
        b: LayoutTensor[_, Self.b_layout, *_, **_],
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
        constrained[
            Self.loading_warps == 4 or Self.loading_warps == 8,
            "loading_warps must be 4 or 8",
        ]()
        constrained[
            Self.load_width == simd_width_of[Self.in_type](),
            "load_width must match simd_width_of[in_type]()",
        ]()
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

        var b_half_idx = warp_id_n // 2  # which B half (0 or 1)
        var b_local_n = warp_id_n % 2  # position within half (0 or 1)
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

        # Create block tiles and initialize loaders (layouts inferred → stride derived)
        var a_block = a.tile[Self.BM, Self.K](block_row // Self.BM, 0)
        var b_block = b.tile[Self.BN, Self.K](block_col // Self.BN, 0)
        self.loader_a = Self.ATileLoader(a_block, warp_id, lane_id)
        self.loader_b = Self.BTileLoader(b_block, warp_id, lane_id)
        self.warp_id_m = warp_id_m
        self.k_offset = 0

        # 4-warp loading: remap warps 4-7 to positions 0-3
        var group_warp_id = warp_id % 4
        self.warp_shift_rows = (group_warp_id - warp_id) * Self.rows_per_warp

    @always_inline
    fn advance_k(mut self):
        """Advance k_offset by BK for the next K iteration."""
        self.k_offset += Self.BK

    @always_inline
    fn load_a[stage: Int, which: Int](self):
        """Load A[stage][which] using 8 warps."""
        self.loader_a.load_tile(
            self.a_load_tiles[stage][which],
            src_row=which * Self.half_BM,
            src_col=self.k_offset,
        )

    @always_inline
    fn load_b[stage: Int, which: Int](self):
        """Load B[stage][which] using 8 warps."""
        self.loader_b.load_tile(
            self.b_load_tiles[stage][which],
            src_row=which * Self.half_BN,
            src_col=self.k_offset,
        )

    # =========================================================================
    # 4-Warp Loading (for true ping-pong overlap)
    # =========================================================================

    @always_inline
    fn _load_tile_4warp[
        half_data_rows: Int, //,
        which: Int,
    ](
        self,
        loader: Self.TileLoader[_],
        dst_tile: Self.HalfTile[half_data_rows],
    ):
        """4-warp load: global → LDS. Uses warp_shift_rows to remap warps 4-7 → 0-3.

        Uses source coordinates with warp_shift adjustment so both warp groups
        (0-3 and 4-7) can independently load their tiles.
        """
        # Apply warp shift to thread position for group-local row
        var effective_thread_row = loader.thread_row + self.warp_shift_rows
        var group_warp_id = loader.warp_id % 4

        # Base column offset
        var col_offset = self.k_offset + loader.thread_col

        comptime rows_per_iter_4warp = 4 * Self.rows_per_warp  # 32 rows
        comptime num_iterations = half_data_rows // rows_per_iter_4warp

        @parameter
        for i in range(num_iterations):
            var tile_idx = i * 4 + group_warp_id
            var warp_subtile = dst_tile.tile[Self.rows_per_warp, Self.BK](
                tile_idx, 0
            )
            var smem_ptr = readfirstlane(warp_subtile.ptr)

            # Row: which * half_data_rows + thread_row + iteration offset
            var row = (
                which * half_data_rows
                + effective_thread_row
                + i * rows_per_iter_4warp
            )
            var offset = col_offset + row * self.K

            loader.buffer.load_to_lds[width = Self.load_width](
                Int32(offset),
                smem_ptr,
                scalar_offset=0,
            )

    @always_inline
    fn load_a_as_group[stage: Int, target_group: Int](self, caller_group: Int):
        """Load A[stage][target_group] using 4 warps. Only executes if caller_group == target_group.
        """
        if caller_group != target_group:
            return
        self._load_tile_4warp[which=target_group](
            self.loader_a, self.a_load_tiles[stage][target_group]
        )

    @always_inline
    fn load_b_as_group[
        stage: Int, which: Int
    ](self, caller_group: Int, loading_group: Int,):
        """Load B[stage][which] using 4 warps. Only executes if caller_group == loading_group.
        """
        if caller_group != loading_group:
            return
        self._load_tile_4warp[which=which](
            self.loader_b, self.b_load_tiles[stage][which]
        )


@always_inline
fn chiplet_transform_chunked[
    num_xcds: Int, chunk_size: Int
](workgroup_id: Int, num_workgroups: Int) -> Int:
    """Transform work group ID for better chiplet locality.

    AMD MI300X/MI355X have 8 XCDs (chiplets), each with its own L2 cache.
    This function reorganizes blocks from round-robin distribution to chunked
    allocation, improving cache locality.

    Original pattern: WG0→XCD0, WG1→XCD1, ..., WG8→XCD0
    Transformed: WG0-63→XCD0, WG64-127→XCD1, etc.

    Args:
        workgroup_id: Original block ID.
        num_workgroups: Total number of blocks.

    Parameters:
        num_xcds: Number of XCDs (8 for MI300X/MI355X).
        chunk_size: Number of blocks per XCD chunk.

    Returns:
        Transformed block ID for better XCD locality.
    """
    # Current XCD under round-robin
    var xcd = workgroup_id % num_xcds

    # Largest full (NUM_XCDS * CHUNK_SIZE)-aligned block
    comptime block = num_xcds * chunk_size
    var limit = (num_workgroups // block) * block

    # If beyond last full block, leave unchanged
    if workgroup_id > limit:
        return workgroup_id

    # Local PID (within round-robin assignment)
    var local_pid = workgroup_id // num_xcds
    var chunk_idx, pos_in_chunk = divmod(local_pid, chunk_size)

    # New PID with chunked assignment
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk


struct AMDPingPongMatmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    config: KernelConfig,
    /,
    # Only helps when M is known at compile time and matrix dims are not regular
    enable_l2_cache_optimization: Bool,
    # Enable 16×32 swizzle pattern for bank conflict avoidance
    enable_swizzle: Bool,
    # Use ds_read_tr16_b64 for B loads (experimental, requires enable_swizzle=False)
    use_transpose_load: Bool,
]:
    """High-level ping-pong matmul implementation for AMD GPUs.

    This implements the 8-warp ping-pong pattern where warps alternate
    between loading data and computing, achieving overlapped execution.

    Memory Layout Strategy for Bank Conflict Avoidance:

    1. Shared Memory Organization (AMD MI355 has 64 banks, 4 bytes each):
       - Uses double-buffered shared memory (ping-pong buffers)
       - Each buffer holds BM×BK elements for A, BN×BK for B

    2. Bank Conflict Avoidance Pattern:
       - Bank index = (address / 4) % 64
       - Swizzled access pattern distributes consecutive thread accesses across banks
       - Column swizzle: (lane_id % 4) * load_width spreads within 32 bytes
       - Row stride: (lane_id // 4) * K ensures different rows map to different banks
       - Warp-level offsets further distribute accesses

    3. Load Pattern (Global → Shared Memory):
       - Uses AMD's load_to_lds instruction for direct DRAM→LDS transfer
       - Bypasses L1/L2 caches for lower latency
       - Coalesced global memory access (consecutive threads → consecutive addresses)
       - Bank-conflict-free shared memory writes via swizzled offsets

    4. MMA Access Pattern (Shared Memory → Registers):
       - Optimized for AMD's matrix cores (4 per CU on MI355)
       - 16×4 thread layout within each warp for MMA fragments
       - Ensures all 4 matrix cores stay busy throughout execution
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

    # Accumulator configuration
    comptime accum_dtype = get_accum_type[
        Self.c_type
    ]()  # FIXME: which one should this be?
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
        constrained[
            Self.BM % Self.WM == 0, "Block M must be divisible by Warp M"
        ]()
        constrained[
            Self.BN % Self.WN == 0, "Block N must be divisible by Warp N"
        ]()
        constrained[
            Self.BK % Self.WK == 0, "Block K must be divisible by Warp K"
        ]()
        constrained[
            Self.WM % Self.MMA_M == 0, "Warp M must be divisible by MMA M"
        ]()
        constrained[
            Self.WN % Self.MMA_N == 0, "Warp N must be divisible by MMA N"
        ]()
        constrained[
            Self.WK % Self.MMA_K == 0, "Warp K must be divisible by MMA K"
        ]()
        constrained[
            Self.total_warps == 8, "Ping-pong kernel requires exactly 8 warps"
        ]()
        constrained[
            Self.num_warps_m == 2,
            "Ping-pong kernel requires 2 warps in M dimension",
        ]()

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Self.config.num_threads()
        )
    )
    @staticmethod
    fn matmul_demo_ping_pong(
        a: LayoutTensor[
            Self.a_type,
            Self.a_layout,
            MutAnyOrigin,
            address_space = AddressSpace.GENERIC,
        ],
        b: LayoutTensor[
            Self.b_type,
            Self.b_layout,
            MutAnyOrigin,
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
        # alias M = a.layout.shape[0].value()
        comptime N = b.layout.shape[0].value()
        comptime K = a.layout.shape[1].value()

        constrained[
            Self.a_type == Self.b_type, "A and B must have the same type"
        ]()

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
        var lane_id = thread_id % WARP_SIZE
        var warp_id = readfirstlane(Int(get_warp_id()))

        @parameter
        @always_inline
        fn compute_block_coords() -> Tuple[Int, Int]:
            @parameter
            if Self.enable_l2_cache_optimization:
                # Apply chiplet transform for better XCD locality
                comptime NUM_XCDS = 8  # MI355X has 8 XCDs
                comptime CHUNK_SIZE = 64  # Blocks per XCD chunk

                # If M is a compile-time constant
                # alias grid_dim_x = ceildiv(N, Self.config.block_shape[1])
                # alias grid_dim_y = ceildiv(M, Self.config.block_shape[0])

                var num_wgs = Int(grid_dim.x) * Int(grid_dim.y)
                var wgid = Int(block_idx.y) * Int(grid_dim.x) + Int(block_idx.x)
                wgid = chiplet_transform_chunked[NUM_XCDS, CHUNK_SIZE](
                    wgid, num_wgs
                )

                # Further transform within XCD for L2 cache locality
                comptime WGM = 8  # 4, 8, 12, 16, 32  # Super-block size
                var num_pid_m = ceildiv(M, BM)
                comptime num_pid_n = ceildiv(N, BN)
                comptime num_wgid_in_group = WGM * num_pid_n
                var group_id = wgid // num_wgid_in_group
                var first_pid_m = group_id * WGM
                var group_size_m = min(num_pid_m - first_pid_m, WGM)
                var local_wgid = wgid % num_wgid_in_group
                var pid_m = first_pid_m + (local_wgid % group_size_m)
                var pid_n = local_wgid // group_size_m

                # Set block coordinates based on transformed IDs
                return (pid_n * BN, pid_m * BM)
            else:
                # Direct mapping from block indices to coordinates
                return (Int(block_idx.x) * BN, Int(block_idx.y) * BM)

        var n, m = compute_block_coords()

        var count_n = min(N - n, BN)
        var count_m = min(M - m, BM)

        # ================================================================
        # Swizzle Configuration (determined by loading thread layout)
        # ================================================================
        # The loading pattern uses 16×4 threads per warp, each loading
        # load_width elements. This determines the swizzle parameters.
        # ================================================================
        comptime swizzle_subtile_rows = 16  # Loading thread layout rows
        comptime swizzle_subtile_cols = 4 * load_width  # Cols × SIMD width
        comptime swizzle_elem_base = log2_floor(swizzle_subtile_cols // 2)
        comptime swizzle_shift = log2_floor(swizzle_subtile_rows)

        # Type aliases - TileBuffers owns loading, MmaOp receives swizzle params
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
            Self.enable_swizzle,
            swizzle_elem_base,  # From loading pattern
            swizzle_shift,  # From loading pattern
        ]
        comptime BuffersType = TileBuffers[
            BM,
            BN,
            BK,
            WM,
            WN,
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
            llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](Int16(priority))

        # ================================================================
        # EXPLICIT PHASE SYNCHRONIZATION (replaces s_barrier abuse)
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
            @parameter
            if Self.use_transpose_load:
                # Use hardware transpose for B loads (ds_read_tr16_b64)
                mma_op.load_b_with_transpose[0](buffers.b_mma_tiles[stage][0])
                mma_op.load_a[0](buffers.a_mma_tiles[stage][0])
                mma_op.load_b_with_transpose[1](buffers.b_mma_tiles[stage][1])
                mma_op.load_a[1](buffers.a_mma_tiles[stage][1])
            else:
                mma_op.load_b[0](buffers.b_mma_tiles[stage][0])  # +4 = 4
                mma_op.load_a[0](buffers.a_mma_tiles[stage][0])  # +8 = 12
                mma_op.load_b[1](buffers.b_mma_tiles[stage][1])  # +4 = 16
                mma_op.load_a[1](buffers.a_mma_tiles[stage][1])  # +8 = 24

            # Wait for b[0], a[0] (first 12 ops)
            s_waitcnt[lgkmcnt=12]()
            mma_op.mma[0, 0]()  # Uses a[0], b[0] ✓

            # Wait for b[1] (next 4 ops, 8 remaining from a[1])
            s_waitcnt[lgkmcnt=8]()
            mma_op.mma[0, 1]()  # Uses a[0] (done), b[1] ✓

            # Wait for a[1] (final 8 ops)
            s_waitcnt[lgkmcnt=0]()
            mma_op.mma[1, 0]()  # Uses a[1] ✓, b[0] (done)
            mma_op.mma[1, 1]()  # Uses a[1] ✓, b[1] (done)

        # ================================================================
        # TILE ORGANIZATION AND PING-PONG SCHEDULE
        # ================================================================
        #
        # BLOCK TILE (BM=256 × BN=256):
        # ┌─────────────────────────────────────────────────────────────┐
        # │                    BN = 256 columns                         │
        # │  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
        # │  │   Warp 0    │   Warp 1    │   Warp 2    │   Warp 3    │  │
        # │  │  WM×WN      │  WM×WN      │  WM×WN      │  WM×WN      │  │
        # │  │  128×64     │  128×64     │  128×64     │  128×64     │  │ half_BM
        # │  │ (group 0)   │ (group 0)   │ (group 0)   │ (group 0)   │  │ = 128
        # │  ├─────────────┼─────────────┼─────────────┼─────────────┤  │ rows
        # │  │   Warp 4    │   Warp 5    │   Warp 6    │   Warp 7    │  │
        # │  │  WM×WN      │  WM×WN      │  WM×WN      │  WM×WN      │  │
        # │  │  128×64     │  128×64     │  128×64     │  128×64     │  │ half_BM
        # │  │ (group 1)   │ (group 1)   │ (group 1)   │ (group 1)   │  │ = 128
        # │  └─────────────┴─────────────┴─────────────┴─────────────┘  │ rows
        # └─────────────────────────────────────────────────────────────┘
        #    warp_id_m = warp_id // 4    (0 or 1, selects row group)
        #    warp_id_n = warp_id % 4     (0-3, selects column)
        #
        # DOUBLE BUFFERING:
        # ┌─────────────────────────────────────────────────────────────┐
        # │  Stage 0 LDS Buffers        │  Stage 1 LDS Buffers          │
        # │  ┌─────────┐ ┌─────────┐    │  ┌─────────┐ ┌─────────┐      │
        # │  │ A_s0[0] │ │ B_s0[0] │    │  │ A_s1[0] │ │ B_s1[0] │      │
        # │  │ 128×64  │ │ 128×64  │    │  │ 128×64  │ │ 128×64  │      │
        # │  ├─────────┤ ├─────────┤    │  ├─────────┤ ├─────────┤      │
        # │  │ A_s0[1] │ │ B_s0[1] │    │  │ A_s1[1] │ │ B_s1[1] │      │
        # │  │ 128×64  │ │ 128×64  │    │  │ 128×64  │ │ 128×64  │      │
        # │  └─────────┘ └─────────┘    │  └─────────┘ └─────────┘      │
        # │  [0] = group 0's region     │  [1] = group 1's region       │
        # └─────────────────────────────────────────────────────────────┘
        #
        # PING-PONG COMPUTE SCHEDULE (each k-iteration processes BK×2):
        # ┌─────────────────────────────────────────────────────────────┐
        # │  K dimension: ──k=0────k=BK────k=2BK────k=3BK────...        │
        # │                  │      │       │       │                   │
        # │  compute_stage:  │  [0] │  [1]  │  [0]  │  [1]  │ ...       │
        # │                  │      │       │       │                   │
        # │  Stage 0 bufs:   │ COMP │ LOAD  │ COMP  │ LOAD  │ ...       │
        # │  Stage 1 bufs:   │ LOAD │ COMP  │ LOAD  │ COMP  │ ...       │
        # └─────────────────────────────────────────────────────────────┘
        #
        # compute_stage[stage] QUADRANT PROCESSING:
        # Each warp's output tile (WM×WN = 128×64) is divided into 4 quadrants:
        # ┌───────────────────────────────────────┐
        # │  Warp Output (128×64)                 │
        # │  ┌─────────────────┬─────────────────┐│
        # │  │ mma[0,0]        │ mma[0,1]        ││ quadrant_m=0
        # │  │ (64×32 output)  │ (64×32 output)  ││
        # │  ├─────────────────┼─────────────────┤│
        # │  │ mma[1,0]        │ mma[1,1]        ││ quadrant_m=1
        # │  │ (64×32 output)  │ (64×32 output)  ││
        # │  └─────────────────┴─────────────────┘│
        # │     quadrant_n=0      quadrant_n=1    │
        # └───────────────────────────────────────┘
        #
        # Each mma[qa,qb] call executes quadrant_m_mmas × quadrant_n_mmas MMA ops
        # using A quadrant [qa] and B quadrant [qb] from the current stage's LDS.
        #
        # ================================================================

        # Toggle to switch between simplified and optimized schedules
        comptime USE_SIMPLIFIED_SCHEDULE = False

        @parameter
        if USE_SIMPLIFIED_SCHEDULE:
            # ================================================================
            # SIMPLIFIED SCHEDULE WITH 4-WARP GROUP LOADING
            # ================================================================
            # Uses 4-warp group loading. More robust to race conditions.
            # ================================================================

            @parameter
            for _k in range(0, K, BK * 2):

                @parameter
                if _k == 0:
                    # PROLOGUE: Load stage 0
                    buffers.load_a_as_group[0, 0](warp_id_m)
                    buffers.load_a_as_group[0, 1](warp_id_m)
                    buffers.load_b_as_group[0, 0](warp_id_m, 0)
                    buffers.load_b_as_group[0, 1](warp_id_m, 1)
                    s_waitcnt[vmcnt=0]()
                    buffers.advance_k()
                    s_barrier()
                    if warp_id_m == 1:
                        s_barrier()

                # STEP 1: Load stage 1
                s_barrier()
                buffers.load_a_as_group[1, 0](warp_id_m)
                buffers.load_a_as_group[1, 1](warp_id_m)
                buffers.load_b_as_group[1, 0](warp_id_m, 0)
                buffers.load_b_as_group[1, 1](warp_id_m, 1)
                s_waitcnt[vmcnt=0]()
                buffers.advance_k()

                # STEP 2: compute[0]
                s_barrier()
                compute_stage[0]()

                # STEP 3: Issue A stage 0 load (overlaps with compute[1])
                @parameter
                if _k < K - BK * 2:
                    buffers.load_a_as_group[0, 0](warp_id_m)
                    buffers.load_a_as_group[0, 1](warp_id_m)

                # STEP 4: compute[1]
                s_barrier()
                compute_stage[1]()

                # STEP 5: Load B stage 0
                @parameter
                if _k < K - BK * 2:
                    s_barrier()
                    buffers.load_b_as_group[0, 0](warp_id_m, 0)
                    buffers.load_b_as_group[0, 1](warp_id_m, 1)
                    s_waitcnt[vmcnt=0]()
                    buffers.advance_k()

            # EPILOGUE
            if warp_id_m == 0:
                s_barrier()

        else:
            # ================================================================
            # OPTIMIZED PING-PONG SCHEDULE (HipKittens-style)
            # ================================================================
            # Aggressively interleaved schedule for maximum throughput.
            # Overlaps global loads, LDS loads, and MMA compute operations.
            # Uses original 8-warp loading functions.
            # ================================================================

            # === PROLOGUE ===
            buffers.load_b[0, 0]()
            buffers.load_a[0, 0]()
            buffers.load_b[0, 1]()
            buffers.load_a[0, 1]()

            buffers.advance_k()

            # Warp staggering
            if warp_id_m == 1:
                s_barrier()

            s_waitcnt[vmcnt=0]()  # down from 4
            schedule_barrier()
            s_barrier()
            schedule_barrier()

            # Issue 3 loads to stage 1
            buffers.load_b[1, 0]()
            buffers.load_a[1, 0]()
            buffers.load_b[1, 1]()

            s_waitcnt[vmcnt=0]()  # down from 6
            schedule_barrier()
            s_barrier()
            schedule_barrier()

            # === MAIN LOOP ===
            @parameter
            for _k in range(BK * 2, K, BK * 2):
                mma_op.load_b[0](buffers.b_mma_tiles[0][0])
                mma_op.load_a[0](buffers.a_mma_tiles[0][0])
                buffers.load_a[1, 1]()
                s_waitcnt[lgkmcnt=8]()
                s_barrier()

                buffers.advance_k()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                mma_op.load_b[1](buffers.b_mma_tiles[0][1])
                buffers.load_b[0, 0]()
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 1]()
                s_setprio[0]()
                s_barrier()

                mma_op.load_a[1](buffers.a_mma_tiles[0][1])
                buffers.load_a[0, 0]()
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[1, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                mma_op.load_b[0](buffers.b_mma_tiles[1][0])
                buffers.load_b[0, 1]()
                s_waitcnt[vmcnt=6]()
                schedule_barrier()
                s_barrier()
                schedule_barrier()

                s_setprio[1]()
                mma_op.mma[1, 1]()
                s_setprio[0]()
                s_barrier()

                mma_op.load_a[0](buffers.a_mma_tiles[1][0])
                buffers.load_a[0, 1]()
                s_waitcnt[lgkmcnt=8]()
                s_barrier()

                buffers.advance_k()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                mma_op.load_b[1](buffers.b_mma_tiles[1][1])
                buffers.load_b[1, 0]()
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[0, 1]()
                s_setprio[0]()
                s_barrier()

                mma_op.load_a[1](buffers.a_mma_tiles[1][1])
                buffers.load_a[1, 0]()
                s_barrier()

                s_waitcnt[lgkmcnt=0]()
                s_setprio[1]()
                mma_op.mma[1, 0]()
                s_setprio[0]()
                s_barrier()
                schedule_barrier()

                buffers.load_b[1, 1]()
                s_waitcnt[vmcnt=6]()
                schedule_barrier()
                s_barrier()
                schedule_barrier()

                s_setprio[1]()
                mma_op.mma[1, 1]()
                s_setprio[0]()
                s_barrier()

            # === EPILOGUE BLOCK 1 ===
            mma_op.load_b[0](buffers.b_mma_tiles[0][0])
            mma_op.load_a[0](buffers.a_mma_tiles[0][0])
            buffers.load_a[1, 1]()
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
            s_waitcnt[vmcnt=0]()  # Down from 2
            schedule_barrier()
            s_barrier()
            schedule_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[1, 0]()
            mma_op.mma[1, 1]()
            s_setprio[0]()
            s_barrier()

            # === EPILOGUE BLOCK 2 ===
            mma_op.load_b[0](buffers.b_mma_tiles[1][0])
            mma_op.load_a[0](buffers.a_mma_tiles[1][0])
            s_waitcnt[vmcnt=0]()  # down from 1
            schedule_barrier()
            s_barrier()
            schedule_barrier()

            s_waitcnt[lgkmcnt=0]()
            s_setprio[1]()
            mma_op.mma[0, 0]()
            s_setprio[0]()
            s_barrier()

            mma_op.load_b[1](buffers.b_mma_tiles[1][1])
            s_waitcnt[vmcnt=0]()
            schedule_barrier()
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

            # Re-balance warp staggering
            if warp_id_m == 0:
                s_barrier()

        barrier()

        lane_id_n, lane_id_m = divmod(lane_id, MMA_N)

        # Create output block tile
        var c_block = c.tile[BM, BN](
            m // BM, n // BN
        )  # Block at (m//BM, n//BN)

        c_resource = AMDBufferResource(c_block.ptr, count_m * N)

        # Quadrant dimensions for output storage
        comptime quadrant_m_mmas = num_m_mmas // 2
        comptime quadrant_n_mmas = num_n_mmas // 2

        # Store each quadrant separately
        # Quadrant (qa, qb) contains elements for:
        #   om in [qa*quadrant_m_mmas, (qa+1)*quadrant_m_mmas)
        #   on in [qb*quadrant_n_mmas, (qb+1)*quadrant_n_mmas)
        @parameter
        for qa in range(2):

            @parameter
            for qb in range(2):
                # Get vectorized view of this quadrant
                var c_quad = mma_op.out_quadrants[qa][qb].vectorize[
                    1, accum_width
                ]()

                # Base output position for this quadrant
                comptime quad_m_offset = qa * quadrant_m_mmas * MMA_M
                comptime quad_n_offset = qb * quadrant_n_mmas * MMA_N

                var quad_base_offset = (
                    (WM * warp_id_m + lane_id_m + quad_m_offset) * N
                    + WN * warp_id_n
                    + lane_id_n * 4
                    + quad_n_offset
                )

                @parameter
                for j in range(0, accum_width, 4):

                    @parameter
                    for local_m in range(quadrant_m_mmas):

                        @parameter
                        for local_n in range(quadrant_n_mmas):
                            output_reg = (
                                c_quad[local_m, local_n]
                                .slice[4, offset=j]()
                                .cast[Self.c_type]()
                            )

                            c_resource.store(
                                quad_base_offset
                                + (local_m * MMA_M) * N
                                + local_n * MMA_N,
                                output_reg,
                            )

                    quad_base_offset += 8


@always_inline
fn ping_pong_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout, //,
    enable_l2_cache_optimization: Bool = False,
    enable_swizzle: Bool = True,
    use_transpose_load: Bool = False,
](
    a_device_tensor: LayoutTensor[a_type, a_layout],
    b_device_tensor: LayoutTensor[b_type, b_layout],
    c_device_tensor: LayoutTensor[c_type, c_layout],
    ctx: DeviceContext,
) raises:
    comptime config = KernelConfig(
        block_shape=Index(256, 256, 64),
        warp_shape=Index(128, 64, 64),
        mma_shape=Index(16, 16, 32),
    )

    var N = c_device_tensor.dim(1)
    var M = c_device_tensor.dim(0)

    comptime kernel = AMDPingPongMatmul[
        a_type,
        b_type,
        c_type,
        a_layout,
        b_layout,
        c_layout,
        config,
        enable_l2_cache_optimization,
        enable_swizzle,
        use_transpose_load,
    ].matmul_demo_ping_pong

    ctx.enqueue_function_checked[kernel, kernel](
        a_device_tensor,
        b_device_tensor,
        c_device_tensor,
        grid_dim=(
            ceildiv(N, config.block_shape[1]),
            ceildiv(M, config.block_shape[0]),
        ),
        block_dim=config.num_threads(),
    )
