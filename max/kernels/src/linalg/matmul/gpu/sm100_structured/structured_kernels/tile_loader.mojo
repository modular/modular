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

"""TMA tile loader for SM100 matrix multiplication.

Provides a wrapper around TMA async_multicast_load operations, following
the SM90 TileLoaderTMA pattern. Orchestration logic (k-group iteration,
expect_bytes, barrier management) is handled by the kernel, not the loader.

Usage:
    # In kernel - create separate A and B loaders
    var a_loader = ATileLoaderType(Pointer(to=a_tma_op), ctx.a_multicast_mask)
    var b_loader = BTileLoaderType(Pointer(to=b_tma_op), ctx.b_multicast_mask)

    # Load tiles using the loaders (LayoutTensor or TileTensor)
    a_loader.load(a_tile, barrier, k_coord, m_coord)
    b_loader.load(b_tile, barrier, k_coord, n_coord)

    # TileTensor tiles are passed directly to TMA ops
"""

from gpu.memory import AddressSpace
from layout import Layout as LegacyLayout, LayoutTensor
from layout.tma_async import SharedMemBarrier, TMATensorTile

from linalg.structuring import SMemTile as LTSMemTile

# Import TileTensor types for overloaded load methods
from .tile_types import SMemTile2D, TMATile

# Import variadic types for TileTensor load overload
from builtin.variadics import Variadic
from layout._layout import TensorLayout
from layout._tile_tensor import TileTensor


struct TileLoaderTMA[
    tma_origin: ImmutOrigin,
    dtype: DType,
    gmem_layout: Layout,
    desc_layout: Layout,
    /,
    *,
    cta_group: Int,
](TrivialRegisterPassable):
    """TMA-based tile loader for SM100.

    Wraps a TMA descriptor and multicast mask for efficient tile loading.
    The load method issues async_multicast_load with proper CTA group handling.

    Parameters:
        tma_origin: Origin of the TMA descriptor pointer.
        dtype: Element data type.
        gmem_layout: Global memory tensor layout.
        desc_layout: TMA descriptor layout (tile dimensions).
        cta_group: CTA group size (1 or 2 for SM100 2-SM MMA).
    """

    comptime TmaOp = TMATensorTile[
        Self.dtype, Self.gmem_layout, Self.desc_layout
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]

    # TMA descriptor pointer (referencing grid constant)
    var tma_op: Self.TmaOpPtr
    # Multicast mask for cluster distribution
    var multicast_mask: UInt16

    @always_inline
    fn __init__(out self, tma_op: Self.TmaOpPtr, multicast_mask: UInt16):
        """Initialize the TMA tile loader.

        Args:
            tma_op: Pointer to TMA descriptor (grid constant).
            multicast_mask: Multicast mask for cluster distribution.
        """
        self.tma_op = tma_op
        self.multicast_mask = multicast_mask

    @always_inline
    fn load[
        tile_layout: Layout,
        /,
        alignment: Int = 128,
    ](
        self,
        dest: LTSMemTile[Self.dtype, tile_layout, alignment=alignment],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        k_coord: UInt,
        row_coord: UInt,
    ):
        """Load a tile using TMA hardware acceleration.

        Issues an async multicast load from global memory to shared memory.
        Coordinates are in element units (not tile units).

        Args:
            dest: Destination SMEM tile (already sliced for peer CTA if needed).
            barrier: Memory barrier for TMA completion signaling.
            k_coord: K dimension coordinate in global memory (elements).
            row_coord: Row coordinate (M for A, N for B) in global memory (elements).
        """
        self.tma_op[].async_multicast_load[Self.cta_group](
            dest, barrier, (k_coord, row_coord), self.multicast_mask
        )

    @always_inline
    fn load[
        dim0: Int,
        dim1: Int,
        /,
        alignment: Int = 128,
    ](
        self,
        dest: SMemTile2D[Self.dtype, dim0, dim1, alignment=alignment],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        k_coord: UInt,
        row_coord: UInt,
    ):
        """Load a TileTensor tile using TMA hardware acceleration.

        This overload accepts TileTensor-based tiles and passes them directly
        to the TMA TileTensor overload (no LayoutTensor conversion needed).

        Args:
            dest: Destination SMEM TileTensor tile.
            barrier: Memory barrier for TMA completion signaling.
            k_coord: K dimension coordinate in global memory (elements).
            row_coord: Row coordinate (M for A, N for B) in global memory (elements).
        """
        # TileTensor overload of async_multicast_load - no conversion needed
        self.tma_op[].async_multicast_load[Self.cta_group](
            dest, barrier, (k_coord, row_coord), self.multicast_mask
        )

    @always_inline
    fn load[
        LayoutType: TensorLayout
    ](
        self,
        dest: TileTensor[
            Self.dtype,
            LayoutType,
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        k_coord: UInt,
        row_coord: UInt,
    ):
        """Load a TileTensor tile with variadic shape/stride types using TMA.

        This overload accepts TileTensor tiles with swizzled layouts (created via
        internal_k_major) and passes them to the TMA operation.

        Args:
            dest: Destination SMEM TileTensor tile with swizzled layout.
            barrier: Memory barrier for TMA completion signaling.
            k_coord: K dimension coordinate in global memory (elements).
            row_coord: Row coordinate (M for A, N for B) in global memory (elements).
        """
        self.tma_op[].async_multicast_load[Self.cta_group](
            dest, barrier, (k_coord, row_coord), self.multicast_mask
        )


# =============================================================================
# TileLoader -- TileTensor-native TMA loader (new Layout types)
# =============================================================================


struct TileLoader[
    tma_origin: ImmutOrigin,
    dtype: DType,
    tile_layout: TensorLayout,
    desc_layout: TensorLayout,
    /,
    *,
    cta_group: Int,
](TrivialRegisterPassable):
    """TMA tile loader parameterized on new Layout types.

    Uses TMATile to derive the TMATensorTile type from new Layout.
    Accepts TileTensor destinations.
    """

    comptime TmaOp = TMATile[
        Self.dtype, Self.tile_layout, Self.desc_layout
    ].InnerType
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]

    var tma_op: Self.TmaOpPtr
    var multicast_mask: UInt16

    @always_inline
    fn __init__[
        tma_op_type: AnyType
    ](
        out self,
        tma_op: Pointer[tma_op_type, Self.tma_origin],
        multicast_mask: UInt16,
    ):
        """Accepts any TMA pointer. Rebinds to the loader's derived type."""
        self.tma_op = rebind[Self.TmaOpPtr](tma_op)
        self.multicast_mask = multicast_mask

    @always_inline
    fn load[
        LayoutType: TensorLayout
    ](
        self,
        dest: TileTensor[
            Self.dtype,
            LayoutType,
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        k_coord: UInt,
        row_coord: UInt,
    ):
        """Load a tile using TMA async multicast load."""
        self.tma_op[].async_multicast_load[Self.cta_group](
            dest, barrier, (k_coord, row_coord), self.multicast_mask
        )


# =============================================================================
# ScalesLoader -- TileTensor-native scales loader (new Layout types)
# =============================================================================


struct ScalesLoader[
    tma_origin: ImmutOrigin,
    dtype: DType,
    tile_layout: TensorLayout,
    desc_layout: TensorLayout = tile_layout,
    /,
    *,
    cta_group: Int,
](TrivialRegisterPassable):
    """TMA scales loader parameterized on new Layout types.

    Uses TMATile to derive the TMATensorTile type from new Layout.
    Uses async_copy (no multicast). Coordinate order is
    (row_coord, k_coord) matching scales tensor layout.
    """

    comptime TmaOp = TMATile[
        Self.dtype, Self.tile_layout, Self.desc_layout
    ].InnerType
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]

    var tma_op: Self.TmaOpPtr

    @always_inline
    fn __init__[
        tma_op_type: AnyType
    ](out self, tma_op: Pointer[tma_op_type, Self.tma_origin]):
        """Accepts any TMA pointer. Rebinds to the loader's derived type."""
        self.tma_op = rebind[Self.TmaOpPtr](tma_op)

    @always_inline
    fn load[
        LayoutType: TensorLayout
    ](
        self,
        dest: TileTensor[
            Self.dtype,
            LayoutType,
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        row_coord: Int,
        k_coord: Int,
    ):
        """Load scales using TMA async copy."""
        self.tma_op[].async_copy[Self.cta_group](
            dest, barrier, (row_coord, k_coord)
        )
