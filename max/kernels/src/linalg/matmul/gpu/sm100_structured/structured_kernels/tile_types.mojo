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
"""Native TileTensor types for SM100 structured kernels.

This module provides TileTensor-based tile types for SM100 structured kernels.
All SMEM storage uses TileTensor natively. Conversion to LayoutTensor only
happens at external API boundaries (TMA, MMA) using explicit LayoutTensor
construction from the tile pointer.

Usage:
    from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
        SMemTile, SMemTileArray2D, SMemTileArrayWithLayout
    )

    # Create tile with a layout
    comptime my_layout = row_major[64, 32]()
    comptime MyTile = SMemTile[DType.float16, my_layout]

    # At TMA/MMA boundaries, construct LayoutTensor from pointer
    comptime lt_type = LayoutTensor[dtype, layout, ...]
    tma_op.async_load(lt_type(tile.ptr), barrier, coords)
"""

from sys import size_of

from gpu.memory import AddressSpace
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout import LayoutTensor
from layout._coord import Coord, Idx
from layout._layout import Layout, row_major
from layout._tile_tensor import TileTensor
from memory import LegacyUnsafePointer, stack_allocation

# Alias for mutable UnsafePointer (same pattern as structuring.mojo)
comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

# Core matrix constant from tensor_core_async.mojo
comptime _CM_NUM_ROWS = 8


# ============================================================================
# Internal Swizzled Layout Aliases
# ============================================================================
# These create internal Layout types that match the swizzled structures from
# tile_layout_k_major. Using internal Layout allows passing through struct
# parameters while preserving type information for .to_layout_tensor().
#
# The key insight: internal Layout (from _layout.mojo) has compile-time type
# parameters (shape_types, stride_types) that are preserved through struct
# chains, unlike public Layout (from layout.mojo) which uses runtime IntTuple.

# Internal swizzled layout for K-major access with configurable swizzle
# Matches tile_layout_k_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_*]()
comptime internal_k_major[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_bytes: Int,
] = Layout(
    Coord(
        Coord(Idx[BM // _CM_NUM_ROWS](), Idx[_CM_NUM_ROWS]()),
        Coord(
            Idx[swizzle_bytes // size_of[dtype]()](),
            Idx[BK * size_of[dtype]() // swizzle_bytes](),
        ),
    ),
    Coord(
        Coord(
            Idx[swizzle_bytes // size_of[dtype]()](),
            Idx[(BM // _CM_NUM_ROWS) * (swizzle_bytes // size_of[dtype]())](),
        ),
        Coord(Idx[1](), Idx[0]()),
    ),
)

# Convenience aliases for common swizzle sizes
comptime internal_k_major_128B[
    dtype: DType, BM: Int, BK: Int
] = internal_k_major[dtype, BM, BK, 128]

comptime internal_k_major_64B[
    dtype: DType, BM: Int, BK: Int
] = internal_k_major[dtype, BM, BK, 64]

comptime internal_k_major_32B[
    dtype: DType, BM: Int, BK: Int
] = internal_k_major[dtype, BM, BK, 32]

# Internal row-major layout (no swizzle) - just delegates to row_major
comptime internal_k_major_none[
    dtype: DType,
    BM: Int,
    BK: Int,
] = row_major[BM, BK]()


# ============================================================================
# Core TileTensor type for shared memory
# ============================================================================

comptime SMemTile[
    dtype: DType,
    layout: Layout,
    *,
    alignment: Int = 128,
] = TileTensor[
    shape_types = layout.shape_types,
    stride_types = layout.stride_types,
    dtype,
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
]
"""Shared memory tile using TileTensor with a Layout.

The Layout parameter preserves swizzle information, enabling
.to_layout_tensor() to produce correctly swizzled LayoutTensors.

Parameters:
    dtype: The data type of tile elements.
    layout: The full layout including swizzle information.
    alignment: Memory alignment (default 128 for shared memory).
"""

# ============================================================================
# Compile-time Accessors for SMemTile
# ============================================================================
# These comptimes allow accessing shape and stride values at compile time.
# Usage:
#   comptime shape0 = SMemTileShape[0, type_of(tile)]
#   comptime stride1 = SMemTileStride[1, type_of(tile)]

comptime SMemTileShape[
    idx: Int,
    Tile: TileTensor,
] = Tile.shape_types[idx].static_value
"""Get compile-time shape value at index from a TileTensor type.

Parameters:
    idx: The dimension index.
    Tile: The TileTensor type (use type_of(tile)).

Returns:
    The static shape value, or -1 if runtime-determined.
"""

comptime SMemTileStride[
    idx: Int,
    Tile: TileTensor,
] = Tile.stride_types[idx].static_value
"""Get compile-time stride value at index from a TileTensor type.

Parameters:
    idx: The dimension index.
    Tile: The TileTensor type (use type_of(tile)).

Returns:
    The static stride value, or -1 if runtime-determined.
"""

# Backward-compatible alias: SMemTile2D[dtype, dim0, dim1] -> SMemTile[dtype, row_major[dim0, dim1]()]
comptime SMemTile2D[
    dtype: DType,
    dim0: Int,
    dim1: Int,
    *,
    alignment: Int = 128,
] = SMemTile[dtype, row_major[dim0, dim1](), alignment=alignment]
"""Backward-compatible alias for SMemTile with explicit 2D dimensions."""


# ============================================================================
# SMemTileArrayWithLayout - Array of tiles with explicit Layout
# ============================================================================


struct SMemTileArrayWithLayout[
    dtype: DType,
    tile_layout: Layout,
    num_tiles: Int,
    alignment: Int = 128,
](TrivialRegisterType):
    """Array of TileTensor tiles with explicit Layout (preserves swizzle info).

    Unlike SMemTileArray2D which uses row_major internally, this type preserves
    the full layout information from TMA swizzling, enabling .to_layout_tensor()
    to produce correctly swizzled LayoutTensors.

    Parameters:
        dtype: Tile element data type.
        tile_layout: The full layout including swizzle information.
        num_tiles: Number of tiles in the array.
        alignment: Memory alignment (default 128 for shared memory).

    Example:
        comptime swizzled_layout = tile_layout_k_major[dtype, BM, BK, ...]()
        comptime MyArray = SMemTileArrayWithLayout[dtype, swizzled_layout, 4]

        var array = MyArray.stack_allocation()
        var tile = array[0]  # Returns TileTensor with swizzled layout
        var lt = tile.to_layout_tensor()  # Correctly swizzled!
    """

    # The TileTensor-based tile type with correct layout
    comptime Tile = SMemTile[Self.dtype, Self.tile_layout]

    # Size calculations - use layout.size() for element count
    comptime tile_size: Int = Self.tile_layout.size()
    comptime num_elements: Int = Self.tile_size * Self.num_tiles
    comptime storage_size: Int = Self.num_elements * size_of[Self.dtype]()

    # Storage type for stack allocation
    comptime Storage = InlineArray[Scalar[Self.dtype], Self.num_elements]

    # Pointer to the array data
    var ptr: UnsafePointer[
        Scalar[Self.dtype], address_space = AddressSpace.SHARED
    ]

    fn __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from inline storage.

        Args:
            storage: The inline storage array.

        Returns:
            A new SMemTileArrayWithLayout pointing to the storage.
        """
        return Self(storage.unsafe_ptr())

    fn __init__[
        mut: Bool, //, origin: Origin[mut=mut]
    ](
        out self,
        unsafe_ptr: LegacyUnsafePointer[
            Scalar[Self.dtype],
            address_space = AddressSpace.SHARED,
            origin=origin,
        ],
    ):
        """Initialize with a shared memory pointer.

        Args:
            unsafe_ptr: Pointer to shared memory storage.
        """
        self.ptr = unsafe_ptr

    @always_inline
    fn __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at the given index.

        Args:
            index: The tile index.

        Returns:
            A TileTensor with correct swizzled layout at the given index.
        """
        var tile_ptr = self.ptr + Self.tile_size * Int(index)
        return Self.Tile(tile_ptr, Self.tile_layout)

    fn slice[
        length: Int
    ](
        self,
        start: Int,
        out result: SMemTileArrayWithLayout[
            Self.dtype, Self.tile_layout, length, Self.alignment
        ],
    ):
        """Get a slice of the array.

        Parameters:
            length: The length of the slice.

        Args:
            start: The starting index.

        Returns:
            A new SMemTileArrayWithLayout representing the slice.
        """
        return type_of(result)(self.ptr + Self.tile_size * start)

    @always_inline
    @staticmethod
    fn stack_allocation() -> Self:
        """Allocate the array on the stack (in shared memory).

        Returns:
            A new SMemTileArrayWithLayout backed by stack-allocated shared memory.
        """
        var ptr = stack_allocation[
            Self.storage_size,
            Self.dtype,
            alignment = Self.alignment,
            address_space = AddressSpace.SHARED,
        ]()
        return Self(ptr)


# ============================================================================
# Compatibility Patterns - Converting TileTensor at API boundaries
# ============================================================================
#
# TileTensor is used natively throughout the kernel. At external API boundaries
# (TMA, MMA), convert to LayoutTensor using one of these patterns:
#
# 1. {ptr} SYNTAX (preferred when type inference works)
#
#    The {ptr} syntax constructs LayoutTensor from just the pointer, inferring
#    the type from the function parameter signature:
#
#        tma_op.async_multicast_load[...](
#            {tile.ptr},  # LayoutTensor inferred from parameter type
#            barrier, coords, mask
#        )
#
#        mma_op.mma(
#            {a_tile.ptr},
#            {b_tile.ptr},
#            ...
#        )
#
# 2. EXPLICIT CONSTRUCTION (when {ptr} doesn't work)
#
#    For TMA functions with complex type parameters (e.g., async_multicast_load_3d),
#    construct LayoutTensor explicitly:
#
#        comptime ATileLT = LayoutTensor[
#            Self.a_type,
#            Self.a_smem_layout,
#            address_space = AddressSpace.SHARED,
#        ]
#        tma_op.async_multicast_load_3d[...](
#            ATileLT(tile.ptr),
#            barrier, coords, mask
#        )
#
# 3. PEER TILE CREATION (offset pointer with same layout)
#
#    TileTensor requires both ptr and layout for construction:
#
#        var peer_tile = type_of(tile)(
#            tile.ptr + offset,
#            tile.layout,
#        )
#


# ============================================================================
# SMemTileArray - Array of tiles in shared memory (TileTensor-based)
# ============================================================================


struct SMemTileArray2D[
    dtype: DType,
    dim0: Int,
    dim1: Int,
    num_tiles: Int,
    alignment: Int = 128,
](TrivialRegisterType):
    """Array of TileTensor tiles in shared memory with explicit dimensions.

    Parameters:
        dtype: Tile element data type.
        dim0: First dimension (rows).
        dim1: Second dimension (columns).
        num_tiles: Number of tiles in the array.
        alignment: Memory alignment (default 128 for shared memory).

    Example:
        comptime MyArray = SMemTileArray2D[DType.float16, 64, 32, 4, 128]

        var array = MyArray.stack_allocation()
        var tile = array[0]  # Returns TileTensor
    """

    # The TileTensor-based tile type
    comptime Tile = SMemTile[
        Self.dtype,
        row_major[Self.dim0, Self.dim1](),
        alignment = Self.alignment,
    ]

    # Size calculations
    comptime tile_size: Int = Self.dim0 * Self.dim1
    comptime num_elements: Int = Self.tile_size * Self.num_tiles
    comptime storage_size: Int = Self.num_elements * size_of[Self.dtype]()

    # Storage type for stack allocation
    comptime Storage = InlineArray[Scalar[Self.dtype], Self.num_elements]

    # Pointer to the array data
    var ptr: UnsafePointer[
        Scalar[Self.dtype], address_space = AddressSpace.SHARED
    ]

    fn __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from inline storage.

        Args:
            storage: The inline storage array.

        Returns:
            A new SMemTileArray2D pointing to the storage.
        """
        return Self(storage.unsafe_ptr())

    fn __init__[
        mut: Bool, //, origin: Origin[mut=mut]
    ](
        out self,
        unsafe_ptr: LegacyUnsafePointer[
            Scalar[Self.dtype],
            address_space = AddressSpace.SHARED,
            origin=origin,
        ],
    ):
        """Initialize with a shared memory pointer.

        Args:
            unsafe_ptr: Pointer to shared memory storage.
        """
        self.ptr = unsafe_ptr

    @always_inline
    fn __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at the given index.

        Args:
            index: The tile index.

        Returns:
            A TileTensor-based tile at the given index.
        """
        var tile_ptr = self.ptr + Self.tile_size * Int(index)
        return Self.Tile(
            tile_ptr,
            row_major[Self.dim0, Self.dim1](),
        )

    fn slice[
        length: Int
    ](
        self,
        start: Int,
        out result: SMemTileArray2D[
            Self.dtype, Self.dim0, Self.dim1, length, Self.alignment
        ],
    ):
        """Get a slice of the array.

        Parameters:
            length: The length of the slice.

        Args:
            start: The starting index.

        Returns:
            A new SMemTileArray2D representing the slice.
        """
        return type_of(result)(self.ptr + Self.tile_size * start)

    @always_inline
    @staticmethod
    fn stack_allocation() -> Self:
        """Allocate the array on the stack (in shared memory).

        Returns:
            A new SMemTileArray2D backed by stack-allocated shared memory.
        """
        var ptr = stack_allocation[
            Self.storage_size,
            Self.dtype,
            alignment = Self.alignment,
            address_space = AddressSpace.SHARED,
        ]()
        return Self(ptr)


# ============================================================================
# BlockwiseFP8TileStorage - TileTensor-based tile storage for blockwise FP8
# ============================================================================


struct BlockwiseFP8TileStorage[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # C tile dimensions
    c_dim0: Int,
    c_dim1: Int,
    # A-scales tile dimensions
    a_scales_dim0: Int,
    a_scales_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
    num_output_stages: Int,
]:
    """TileTensor-based storage for blockwise FP8 matmul tiles.

    IMPORTANT: Field order preserves SMEM layout compatibility: a, b, c, a_scales.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        c_type: Data type for C matrix tiles.
        a_scales_type: Data type for A scale tiles.
        a_dim0: First dimension for A tiles.
        a_dim1: Second dimension for A tiles.
        b_dim0: First dimension for B tiles.
        b_dim1: Second dimension for B tiles.
        c_dim0: First dimension for C tiles.
        c_dim1: Second dimension for C tiles.
        a_scales_dim0: First dimension for A scale tiles.
        a_scales_dim1: Second dimension for A scale tiles.
        num_pipeline_stages: Number of input pipeline stages.
        num_output_stages: Number of output pipeline stages.
    """

    comptime ATileArray = SMemTileArray2D[
        Self.a_type, Self.a_dim0, Self.a_dim1, Self.num_pipeline_stages, 128
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type, Self.b_dim0, Self.b_dim1, Self.num_pipeline_stages, 128
    ]
    comptime CTileArray = SMemTileArray2D[
        Self.c_type, Self.c_dim0, Self.c_dim1, Self.num_output_stages, 128
    ]
    comptime AScalesTileArray = SMemTileArray2D[
        Self.a_scales_type,
        Self.a_scales_dim0,
        Self.a_scales_dim1,
        Self.num_pipeline_stages,
        128,
    ]

    # Field order preserves SMEM layout: a, b, c, a_scales
    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage
    var c_tiles_storage: Self.CTileArray.Storage
    var a_scales_tiles_storage: Self.AScalesTileArray.Storage

    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor."""
        return Self.ATileArray(self.a_tiles_storage)

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor."""
        return Self.BTileArray(self.b_tiles_storage)

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        """Get C tile array accessor."""
        return Self.CTileArray(self.c_tiles_storage)

    @always_inline
    fn a_scales_tiles(ref[AddressSpace.SHARED] self) -> Self.AScalesTileArray:
        """Get A-scales tile array accessor."""
        return Self.AScalesTileArray(self.a_scales_tiles_storage)


# ============================================================================
# BlockwiseFP8TilePayload - TileTensor-based payload for blockwise FP8
# ============================================================================

# Import the TilePayload trait from tile_pipeline to ensure compatibility
# with InputPipelineStorage which expects that specific trait
from .tile_pipeline import TilePayload


struct BlockwiseFP8TilePayload[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # A-scales tile dimensions
    a_scales_dim0: Int,
    a_scales_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
](TilePayload, TrivialRegisterType):
    """TileTensor-based tile payload for blockwise FP8 matmul.

    Unlike BlockScaledTilePayload, this only stores A-scales in SMEM.
    B-scales are read directly from global memory during the epilogue phase.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        a_scales_type: Data type for A scale tiles.
        a_dim0: First dimension for A tiles.
        a_dim1: Second dimension for A tiles.
        b_dim0: First dimension for B tiles.
        b_dim1: Second dimension for B tiles.
        a_scales_dim0: First dimension for A scale tiles.
        a_scales_dim1: Second dimension for A scale tiles.
        num_pipeline_stages: Number of input pipeline stages.
    """

    comptime ATileArray = SMemTileArray2D[
        Self.a_type, Self.a_dim0, Self.a_dim1, Self.num_pipeline_stages, 128
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type, Self.b_dim0, Self.b_dim1, Self.num_pipeline_stages, 128
    ]
    comptime AScalesTileArray = SMemTileArray2D[
        Self.a_scales_type,
        Self.a_scales_dim0,
        Self.a_scales_dim1,
        Self.num_pipeline_stages,
        128,
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile
    comptime AScalesTile = Self.AScalesTileArray.Tile

    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray
    var a_scales_tiles: Self.AScalesTileArray

    @always_inline
    fn __init__(
        out self,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
        a_scales_tiles: Self.AScalesTileArray,
    ):
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles
        self.a_scales_tiles = a_scales_tiles

    @always_inline
    fn get_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Tuple[
        Self.ATile, Self.BTile, Self.AScalesTile
    ]:
        """Get A, B, A-scales tiles at the specified stage and k-group index."""
        var idx = stage * UInt32(k_group_size) + UInt32(k_idx)
        return (
            self.a_tiles[idx],
            self.b_tiles[idx],
            self.a_scales_tiles[idx],
        )

    @always_inline
    fn get_a_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified stage and k-group index."""
        return self.a_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    fn get_b_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified stage and k-group index."""
        return self.b_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    fn get_a_scales_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.AScalesTile:
        """Get A-scales tile at the specified stage and k-group index."""
        return self.a_scales_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]
