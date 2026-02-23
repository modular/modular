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
"""Tensor Memory (TMEM) abstractions for SM100 Blackwell GPUs.

TMEM is dedicated memory for MMA accumulators, separate from registers and
shared memory. This module provides type-safe abstractions:

- TmemAllocation: Manages TMEM lifecycle (alloc/dealloc)
- TmemTensor: Layout-parameterized typed view over TMEM accumulators
- TmemStage: Represents a pipeline stage for accumulator buffering
- TmemAddress: Simple address wrapper for TMEM load operations
"""

from layout import Layout

from gpu import syncwarp
from gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)

from linalg.structuring import SMemArray


struct TmemAllocation[
    cta_group: Int,
    max_cols: Int = 512,
](TrivialRegisterPassable):
    """Handle to allocated Tensor Memory.

    Lifecycle: allocate() → use → release_lock() → wait → deallocate()

    Parameters:
        cta_group: Cooperating CTAs (1 or 2).
        max_cols: TMEM columns (512 for SM100).
    """

    comptime SmemAddrStorage = SMemArray[UInt32, 1]

    var addr: UInt32

    @always_inline
    fn __init__(out self, addr: UInt32):
        self.addr = addr

    @staticmethod
    fn allocate(smem_addr: Self.SmemAddrStorage) -> Self:
        """Allocate TMEM (MMA warp). Address stored in smem for epilogue."""
        tcgen05_alloc[Int32(Self.cta_group)](
            smem_addr.ptr, UInt32(Self.max_cols)
        )
        syncwarp()
        return Self(smem_addr.ptr[0])

    @staticmethod
    fn from_shared(smem_addr: Self.SmemAddrStorage) -> Self:
        """Get handle from existing allocation (epilogue warp)."""
        return Self(smem_addr.ptr[0])

    fn release_lock(self):
        """Release allocation lock before waiting for epilogue."""
        tcgen05_release_allocation_lock[Int32(Self.cta_group)]()

    fn deallocate(self):
        """Free TMEM after epilogue completion."""
        tcgen05_dealloc[Int32(Self.cta_group)](self.addr, UInt32(Self.max_cols))


# TMEM Address Encoding (SM100 Blackwell)
# =========================================
# TMEM addresses encode row and column offsets in a packed format:
#
#   Address = [row_offset : 16 bits] [column_offset : 16 bits]
#
# SM100 MMA accumulators span 32 rows × N columns per tile:
#   - Upper fragment: rows 0-15  (accessed at base address)
#   - Lower fragment: rows 16-31 (accessed at base + TMEM_LOWER_ROW_OFFSET)
#
# The value 16 << 16 encodes "row 16, column 0" as the starting offset
# for the lower half of the accumulator.
comptime TMEM_LOWER_ROW_OFFSET: UInt32 = 16 << 16


struct TmemAddress(TrivialRegisterPassable):
    """Simple TMEM address wrapper for load/store operations.

    Encapsulates TMEM address encoding for accumulator fragment access.
    SM100 MMA accumulators are organized as 32 rows, split into:
      - Upper fragment (rows 0-15): accessed via upper_addr()
      - Lower fragment (rows 16-31): accessed via lower_addr()

    The lower fragment address adds TMEM_LOWER_ROW_OFFSET (16 << 16) to
    encode the row offset in the upper 16 bits of the address.

    Usage:
        var tmem = TmemAddress(base_offset)

        # Load operations
        var upper = tmem.load_upper[dtype, size]()
        var lower = tmem.load_lower[dtype, size]()
        TmemAddress.wait_load()

        # Store operations
        tmem.store_upper[dtype, size](upper_frag)
        tmem.store_lower[dtype, size](lower_frag)
        TmemAddress.wait_store()

        # Low-level address access for custom operations
        raw_upper = tmem.upper_addr()
        raw_lower = tmem.lower_addr()
    """

    var addr: UInt32

    @always_inline
    fn __init__(out self, addr: Int):
        """Create TmemAddress from integer column address."""
        self.addr = UInt32(addr)

    @always_inline
    fn __init__(out self, addr: UInt32):
        """Create TmemAddress from hardware address (UInt32)."""
        self.addr = addr

    @always_inline
    fn __add__(self, offset: Int) -> Self:
        """Create new TmemAddress with column offset added."""
        return Self(Int(self.addr) + offset)

    @always_inline
    fn upper_addr(self) -> UInt32:
        """Raw address for upper fragment (rows 0-15)."""
        return self.addr

    @always_inline
    fn lower_addr(self) -> UInt32:
        """Raw address for lower fragment (rows 16-31)."""
        return self.addr + TMEM_LOWER_ROW_OFFSET

    @always_inline
    fn load_upper[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self) -> SIMD[dtype, width]:
        """Load upper accumulator fragment (rows 0-15)."""
        return tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=dtype,
            pack=False,
            width=width,
        ](self.upper_addr())

    @always_inline
    fn load_lower[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self) -> SIMD[dtype, width]:
        """Load lower accumulator fragment (rows 16-31)."""
        return tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=dtype,
            pack=False,
            width=width,
        ](self.lower_addr())

    @always_inline
    fn store_upper[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self, data: SIMD[dtype, width]):
        """Store upper accumulator fragment (rows 0-15)."""
        tcgen05_st[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            pack=False,
        ](self.upper_addr(), data)

    @always_inline
    fn store_lower[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self, data: SIMD[dtype, width]):
        """Store lower accumulator fragment (rows 16-31)."""
        tcgen05_st[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            pack=False,
        ](self.lower_addr(), data)

    @staticmethod
    @always_inline
    fn wait_store():
        """Wait for TMEM store operations to complete."""
        tcgen05_store_wait()

    @staticmethod
    @always_inline
    fn wait_load():
        """Wait for TMEM load operations to complete."""
        tcgen05_load_wait()


# =============================================================================
# TmemTensor - Layout-parameterized typed view over TMEM
# =============================================================================


struct TmemTensor[
    dtype: DType,
    layout: Layout,
    *,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """Typed tensor view over Tensor Memory (TMEM) for MMA accumulators.

    Provides a LayoutTensor-like abstraction for TMEM with:
    - Type safety: dtype and layout known at compile time
    - Fragment access: upper (rows 0-15) and lower (rows 16-31)
    - MMA integration: offset() returns raw address for MMA operations

    The layout parameter captures the logical accumulator shape (M × N),
    enabling future extensions like custom tiling patterns or multi-tile
    accumulator management.

    Parameters:
        dtype: Accumulator data type (typically float32).
        layout: Logical layout of the accumulator tile (M × N).
        cta_group: CTA cooperation level (1 or 2).

    Example:
        # Create typed TMEM view with (64, 128) accumulator layout
        comptime layout = Layout.row_major(64, 128)
        var tmem = TmemTensor[DType.float32, layout](col_offset)

        # Use with MMA operations (returns raw UInt32 offset)
        mma_op.mma(a_tile, b_tile, tmem.offset(), init_c=True)

        # Load fragments for epilogue
        var upper = tmem.load_upper[repeat=4]()
        var lower = tmem.load_lower[repeat=4]()
        TmemTensor.wait_load()
    """

    # SM100 tcgen05 fragment parameters
    comptime data_paths = 16
    comptime bits = 256
    comptime frag_size = (Self.data_paths * (Self.bits // 32)) // 32

    # Lower fragment required unless cta_group=1 and M=64
    comptime tile_m = Self.layout.shape[0].value()
    comptime is_lower_required = not (Self.cta_group == 1 and Self.tile_m == 64)

    var col_addr: Int

    @always_inline
    fn __init__(out self, col_addr: Int):
        """Create TMEM tensor view at the given column address."""
        self.col_addr = col_addr

    @always_inline
    fn __init__(out self, addr: TmemAddress):
        """Create TMEM tensor view from a TmemAddress."""
        self.col_addr = Int(addr.addr)

    @always_inline
    fn offset(self) -> Int:
        """TMEM column address for this tensor."""
        return self.col_addr

    @always_inline
    fn address(self) -> TmemAddress:
        """Get TmemAddress for low-level fragment operations."""
        return TmemAddress(self.col_addr)

    @always_inline
    fn load_upper[
        repeat: Int = 1,
    ](self) -> SIMD[Self.dtype, Self.frag_size * repeat]:
        """Load upper accumulator fragment (rows 0-15).

        Parameters:
            repeat: Number of times to repeat the load pattern.

        Returns:
            SIMD vector containing the upper fragment data.
        """
        return self.address().load_upper[
            Self.dtype,
            Self.frag_size * repeat,
            Self.data_paths,
            Self.bits,
            repeat,
        ]()

    @always_inline
    fn load_lower[
        repeat: Int = 1,
    ](self) -> SIMD[Self.dtype, Self.frag_size * repeat]:
        """Load lower accumulator fragment (rows 16-31).

        Parameters:
            repeat: Number of times to repeat the load pattern.

        Returns:
            SIMD vector containing the lower fragment data.
        """
        return self.address().load_lower[
            Self.dtype,
            Self.frag_size * repeat,
            Self.data_paths,
            Self.bits,
            repeat,
        ]()

    @always_inline
    fn store_upper[
        repeat: Int = 1,
    ](self, data: SIMD[Self.dtype, Self.frag_size * repeat]):
        """Store upper accumulator fragment (rows 0-15).

        Parameters:
            repeat: Number of times to repeat the store pattern.

        Args:
            data: SIMD vector containing the data to store.
        """
        self.address().store_upper[
            Self.dtype,
            Self.frag_size * repeat,
            Self.data_paths,
            Self.bits,
            repeat,
        ](data)

    @always_inline
    fn store_lower[
        repeat: Int = 1,
    ](self, data: SIMD[Self.dtype, Self.frag_size * repeat]):
        """Store lower accumulator fragment (rows 16-31).

        Parameters:
            repeat: Number of times to repeat the store pattern.

        Args:
            data: SIMD vector containing the data to store.
        """
        self.address().store_lower[
            Self.dtype,
            Self.frag_size * repeat,
            Self.data_paths,
            Self.bits,
            repeat,
        ](data)

    # ========== Unified Fragment Access ==========

    comptime Fragments = TmemFragments[
        Self.dtype,
        Self.frag_size,
        is_lower_required = Self.is_lower_required,
        data_paths = Self.data_paths,
        bits = Self.bits,
    ]

    @always_inline
    fn load_fragments[
        repeat: Int = 1,
    ](self) -> TmemFragments[
        Self.dtype,
        Self.frag_size * repeat,
        is_lower_required = Self.is_lower_required,
    ]:
        """Load both upper and lower fragments in one call.

        Handles is_lower_required automatically based on layout.

        Parameters:
            repeat: Number of times to repeat the load pattern.

        Returns:
            TmemFragments containing upper and (conditionally) lower data.
        """
        return TmemFragments[
            Self.dtype,
            Self.frag_size,
            is_lower_required = Self.is_lower_required,
        ].load[repeat](self.address())

    @always_inline
    fn store_fragments[
        repeat: Int = 1,
    ](
        self,
        frags: TmemFragments[
            Self.dtype,
            Self.frag_size * repeat,
            is_lower_required = Self.is_lower_required,
        ],
    ):
        """Store both upper and lower fragments in one call.

        Handles is_lower_required automatically based on layout.

        Parameters:
            repeat: Number of times to repeat the store pattern.

        Args:
            frags: TmemFragments containing upper and (conditionally) lower data.
        """
        frags.store[repeat](self.address())

    @staticmethod
    @always_inline
    fn wait_load():
        """Wait for TMEM load operations to complete."""
        TmemAddress.wait_load()

    @staticmethod
    @always_inline
    fn wait_store():
        """Wait for TMEM store operations to complete."""
        TmemAddress.wait_store()


# =============================================================================
# TmemFragments - Paired upper/lower accumulator fragments
# =============================================================================


struct TmemFragments[
    dtype: DType,
    frag_size: Int,
    *,
    is_lower_required: Bool = True,
    data_paths: Int = 16,
    bits: Int = 256,
](TrivialRegisterPassable):
    """Paired upper/lower accumulator fragments from TMEM.

    Encapsulates the SM100 TMEM row-split hardware detail:
    - Upper fragment: rows 0-15 (always present)
    - Lower fragment: rows 16-31 (only when is_lower_required=True)

    The is_lower_required flag is determined by:
    - False when cta_group=1 and MMA_M=64 (fits in 16 rows)
    - True otherwise (needs both halves)

    Parameters:
        dtype: Fragment data type (typically float32).
        frag_size: Elements per fragment (derived from data_paths and bits).
        is_lower_required: Whether lower fragment is needed.
        data_paths: SM100 data paths (typically 16).
        bits: Bits per fragment load (typically 256).

    Example:
        # Load both fragments in one call
        var frags = TmemFragments[DType.float32, 16].load(tmem_addr)

        # Work with fragments
        frags.upper = process(frags.upper)
        frags.lower = process(frags.lower)

        # Store both fragments
        frags.store(tmem_addr)
        TmemFragments.wait_store()
    """

    var upper: SIMD[Self.dtype, Self.frag_size]
    var lower: SIMD[Self.dtype, Self.frag_size]

    @always_inline
    fn __init__(out self):
        """Initialize with zero fragments."""
        self.upper = SIMD[Self.dtype, Self.frag_size]()
        self.lower = SIMD[Self.dtype, Self.frag_size]()

    @always_inline
    fn __init__(
        out self,
        upper: SIMD[Self.dtype, Self.frag_size],
        lower: SIMD[Self.dtype, Self.frag_size],
    ):
        """Initialize with provided fragments."""
        self.upper = upper
        self.lower = lower

    @staticmethod
    @always_inline
    fn load[
        repeat: Int = 1
    ](tmem: TmemAddress) -> TmemFragments[
        Self.dtype,
        Self.frag_size * repeat,
        is_lower_required = Self.is_lower_required,
    ]:
        """Load fragments from TMEM address.

        Loads upper fragment always; loads lower only if required.

        Parameters:
            repeat: Number of times to repeat the load pattern.

        Args:
            tmem: TMEM address to load from.

        Returns:
            TmemFragments containing upper and (optionally) lower data.
        """
        comptime width = Self.frag_size * repeat
        var result = TmemFragments[
            Self.dtype, width, is_lower_required = Self.is_lower_required
        ]()
        result.upper = tmem.load_upper[
            Self.dtype, width, Self.data_paths, Self.bits, repeat
        ]()

        comptime if Self.is_lower_required:
            result.lower = tmem.load_lower[
                Self.dtype, width, Self.data_paths, Self.bits, repeat
            ]()

        return result

    @always_inline
    fn store[repeat: Int = 1](self, tmem: TmemAddress):
        """Store fragments to TMEM address.

        Stores upper fragment always; stores lower only if required.

        Parameters:
            repeat: Number of times to repeat the store pattern.

        Args:
            tmem: TMEM address to store to.
        """
        tmem.store_upper[
            Self.dtype, Self.frag_size, Self.data_paths, Self.bits, repeat
        ](self.upper)

        comptime if Self.is_lower_required:
            tmem.store_lower[
                Self.dtype, Self.frag_size, Self.data_paths, Self.bits, repeat
            ](self.lower)

    @always_inline
    fn cast[
        target_dtype: DType
    ](self) -> TmemFragments[
        target_dtype, Self.frag_size, is_lower_required = Self.is_lower_required
    ]:
        """Cast fragments to a different dtype."""
        var result = TmemFragments[
            target_dtype,
            Self.frag_size,
            is_lower_required = Self.is_lower_required,
        ]()
        result.upper = self.upper.cast[target_dtype]()

        comptime if Self.is_lower_required:
            result.lower = self.lower.cast[target_dtype]()

        return result

    @staticmethod
    @always_inline
    fn wait_load():
        """Wait for TMEM load operations to complete."""
        TmemAddress.wait_load()

    @staticmethod
    @always_inline
    fn wait_store():
        """Wait for TMEM store operations to complete."""
        TmemAddress.wait_store()


# =============================================================================
# TmemArrayType - Array of tiles in Tensor Memory
# =============================================================================


struct TmemArrayType[
    dtype: DType,
    layout: Layout,
    num_tiles: Int,
    *,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """Array of tiles in Tensor Memory (TMEM).

    Similar to SMemArray but for TMEM-resident tiles. Provides indexed
    access to a contiguous array of TmemTensor tiles.

    Parameters:
        dtype: Element dtype for tiles.
        layout: Layout of each tile.
        num_tiles: Number of tiles in the array.
        cta_group: CTA group size (1 or 2).

    Compile-time constants:
        Tile: TmemTensor type for each tile.
        tile_stride: Columns per tile (derived from layout.size()).
        num_cols: Total TMEM columns used (num_tiles × tile_stride).
    """

    comptime Tile = TmemTensor[
        Self.dtype, Self.layout, cta_group = Self.cta_group
    ]
    # TMEM addresses are in column units, so stride is N dimension (shape[1])
    comptime tile_stride = Self.layout.shape[1].value()
    comptime num_cols = Self.num_tiles * Self.tile_stride

    var base_addr: Int

    @always_inline
    fn __init__(out self, base_addr: Int):
        """Initialize array at the given TMEM base address."""
        self.base_addr = base_addr

    @always_inline
    fn __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at the given index."""
        return Self.Tile(self.base_addr + Int(index) * Self.tile_stride)


# =============================================================================
# BlockScaledTmem - TMEM region for block-scaled MMA operations
# =============================================================================


struct BlockScaledTmem[
    # Accumulator configuration
    accum_dtype: DType,
    MMA_M: Int,
    MMA_N: Int,
    num_accum_stages: Int,
    # Scaling factor configuration
    sf_dtype: DType,
    BM: Int,  # Block M dimension (for SFA)
    num_pipeline_stages: Int,
    *,
    cta_group: Int = 1,
    total_cols: Int = 512,
    num_sf_k_tiles: Int = 1,
](TrivialRegisterPassable):
    """TMEM region for block-scaled matmul with typed tile accessors.

    Manages the TMEM address space for block-scaled MMA operations,
    providing typed TmemTensor access to:
    - Accumulator tiles (one per output pipeline stage)
    - SFA scaling factor tiles (one per k-iteration)
    - SFB scaling factor tiles (one per k-iteration)

    Memory layout (512 columns total):
    ┌────────────────────────────────────────────────────────────┐
    │ Accumulators     │ SFA Scales        │ SFB Scales         │
    │ (stages × MMA_N) │ (iters × cols)    │ (iters × cols)     │
    └────────────────────────────────────────────────────────────┘

    Parameters:
        accum_dtype: Accumulator data type (typically float32).
        MMA_M: MMA M dimension.
        MMA_N: MMA N dimension (also stage stride for accumulators).
        num_accum_stages: Number of accumulator pipeline stages.
        sf_dtype: Scaling factor data type.
        BM: Block M dimension (for SFA sizing).
        num_pipeline_stages: Number of k-iteration pipeline stages.
        cta_group: CTA group size (1 or 2).
        total_cols: Total TMEM columns (512 for SM100).
        num_sf_k_tiles: Scaling factor tiles per K-iteration.
            MXFP8 uses 1 (one SF vector per K-tile).
            NVFP4 uses 4 (multiple SF vectors per K-tile).
    """

    # Tile layouts (stride derived automatically from layout.size())
    # Each SFA/SFB tile in TMEM covers num_sf_k_tiles SF vectors,
    # so the column width is num_sf_k_tiles * (dim // 32).
    comptime accum_layout = Layout.row_major(Self.MMA_M, Self.MMA_N)
    comptime sfa_layout = Layout.row_major(
        1, Self.num_sf_k_tiles * (Self.BM // 32)
    )
    comptime sfb_layout = Layout.row_major(
        1, Self.num_sf_k_tiles * (Self.MMA_N // 32)
    )

    # Array types for each TMEM region
    comptime AccumArray = TmemArrayType[
        Self.accum_dtype,
        Self.accum_layout,
        num_tiles = Self.num_accum_stages,
        cta_group = Self.cta_group,
    ]
    comptime SFAArray = TmemArrayType[
        Self.sf_dtype,
        Self.sfa_layout,
        num_tiles = Self.num_pipeline_stages,
        cta_group = Self.cta_group,
    ]
    comptime SFBArray = TmemArrayType[
        Self.sf_dtype,
        Self.sfb_layout,
        num_tiles = Self.num_pipeline_stages,
        cta_group = Self.cta_group,
    ]

    # Tile types (for convenience)
    comptime AccumTile = Self.AccumArray.Tile
    comptime SFATile = Self.SFAArray.Tile
    comptime SFBTile = Self.SFBArray.Tile

    # Region base offsets (compile-time constants)
    comptime accum_offset = 0
    comptime sfa_offset = Self.AccumArray.num_cols
    comptime sfb_offset = Self.sfa_offset + Self.SFAArray.num_cols
    comptime used_cols = Self.sfb_offset + Self.SFBArray.num_cols

    var base_addr: Int

    @always_inline
    fn __init__(out self, base_addr: Int):
        """Create TMEM region view at the given base address."""
        comptime assert (
            Self.used_cols <= Self.total_cols
        ), "Block-scaled TMEM region exceeds capacity"
        self.base_addr = base_addr

    @always_inline
    fn __init__(out self, addr: TmemAddress):
        """Create TMEM region view from a TmemAddress."""
        comptime assert (
            Self.used_cols <= Self.total_cols
        ), "Block-scaled TMEM region exceeds capacity"
        self.base_addr = Int(addr.addr)

    @always_inline
    fn __init__[
        cta: Int, max_cols: Int
    ](out self, alloc: TmemAllocation[cta, max_cols]):
        """Create TMEM region view from a TmemAllocation."""
        comptime assert (
            Self.used_cols <= Self.total_cols
        ), "Block-scaled TMEM region exceeds capacity"
        self.base_addr = Int(alloc.addr)

    @always_inline
    fn accum_tiles(self) -> Self.AccumArray:
        """Get array of accumulator tiles."""
        return Self.AccumArray(self.base_addr + Self.accum_offset)

    @always_inline
    fn sfa_tiles(self) -> Self.SFAArray:
        """Get array of SFA scaling factor tiles."""
        return Self.SFAArray(self.base_addr + Self.sfa_offset)

    @always_inline
    fn sfb_tiles(self) -> Self.SFBArray:
        """Get array of SFB scaling factor tiles."""
        return Self.SFBArray(self.base_addr + Self.sfb_offset)

    # Convenience accessors (delegate to arrays)
    @always_inline
    fn accum[T: Intable](self, stage: T) -> Self.AccumTile:
        """Get accumulator tile for the given pipeline stage."""
        return self.accum_tiles()[stage]

    @always_inline
    fn sfa[T: Intable](self, index: T) -> Self.SFATile:
        """Get SFA scaling factor tile for the given k-iteration index."""
        return self.sfa_tiles()[index]

    @always_inline
    fn sfb[T: Intable](self, index: T) -> Self.SFBTile:
        """Get SFB scaling factor tile for the given k-iteration index."""
        return self.sfb_tiles()[index]


# =============================================================================
# TmemStage - Pipeline stage wrapper for accumulator buffering
# =============================================================================


struct TmemStage[
    num_stages: Int,
    stage_stride: Int,
    cta_group: Int,
](TrivialRegisterPassable):
    """A pipeline stage within TMEM for accumulator buffering.

    Used by OutputTilePipeline to manage MMA→Epilogue synchronization.
    MMA writes to one stage while epilogue reads from another.

    Wraps TmemAddress with stage-specific offset calculation:
      - offset(): Column address for this stage (base + index * stride)
      - address(): TmemAddress for this stage (for load/store ops)
      - tensor[layout](): Get typed TmemTensor view

    Parameters:
        num_stages: Pipeline stages (typically 2-4).
        stage_stride: Columns per stage (512 / num_stages).
        cta_group: Cooperating CTAs (1 or 2).
    """

    var base_addr: Int
    var index: Int

    @always_inline
    fn __init__(out self, base_addr: Int, index: Int):
        self.base_addr = base_addr
        self.index = index

    @always_inline
    fn __init__(out self, addr: TmemAddress, index: Int):
        """Create stage from TmemAddress and stage index."""
        self.base_addr = Int(addr.addr)
        self.index = index

    @always_inline
    fn __init__[
        cta: Int, max_cols: Int
    ](out self, alloc: TmemAllocation[cta, max_cols], index: Int):
        """Create stage from TmemAllocation and stage index."""
        self.base_addr = Int(alloc.addr)
        self.index = index

    @staticmethod
    @always_inline
    fn from_offset(offset: Int, index: Int) -> Self:
        """Create stage from pre-computed offset (for legacy pipeline compatibility).

        Use this when the caller has already computed the TMEM offset
        (e.g., `base + stage * stride`) and just needs to wrap it.

        The index is preserved for barrier signaling, and we back-calculate
        the base_addr such that offset() = base + index * stride = offset.

        Args:
            offset: Pre-computed TMEM column offset for this stage.
            index: Pipeline stage index (for barrier signaling).

        Returns:
            TmemStage with offset() returning the given value.
        """
        # Back-calculate base_addr so that base_addr + index * stride = offset
        # base_addr = offset - index * stride
        var base_addr = offset - index * Self.stage_stride
        return Self(base_addr, index)

    @always_inline
    fn offset(self) -> Int:
        """TMEM column address for this stage."""
        return self.base_addr + self.index * Self.stage_stride

    @always_inline
    fn address(self) -> TmemAddress:
        """Get TmemAddress for this stage's offset."""
        return TmemAddress(self.offset())

    @always_inline
    fn tensor[
        accum_dtype: DType,
        accum_layout: Layout,
    ](self) -> TmemTensor[
        accum_dtype, accum_layout, cta_group = Self.cta_group
    ]:
        """Get typed TmemTensor view of this stage's accumulator.

        Parameters:
            accum_dtype: Accumulator data type.
            accum_layout: Logical accumulator layout (M × N).

        Returns:
            TmemTensor providing typed access to the accumulator.
        """
        return TmemTensor[
            accum_dtype, accum_layout, cta_group = Self.cta_group
        ](self.base_addr + self.index * Self.stage_stride)

    @always_inline
    fn load_upper[
        dtype: DType,
        frag_size: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 4,
    ](self) -> SIMD[dtype, frag_size]:
        """Load upper accumulator fragment (rows 0-15)."""
        return self.address().load_upper[
            dtype, frag_size, data_paths, bits, repeat
        ]()

    @always_inline
    fn load_lower[
        dtype: DType,
        frag_size: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 4,
    ](self) -> SIMD[dtype, frag_size]:
        """Load lower accumulator fragment (rows 16-31)."""
        return self.address().load_lower[
            dtype, frag_size, data_paths, bits, repeat
        ]()

    @staticmethod
    @always_inline
    fn wait_load():
        """Wait for TMEM load operations to complete."""
        TmemAddress.wait_load()
