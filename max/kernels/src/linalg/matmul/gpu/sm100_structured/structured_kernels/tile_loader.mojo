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

    # TileTensor tiles are automatically converted to LayoutTensor for TMA ops
"""

from gpu.memory import AddressSpace
from layout import Layout as LegacyLayout, LayoutTensor
from layout.tma_async import SharedMemBarrier, TMATensorTile

from linalg.structuring import SMemTile as LTSMemTile

# Import TileTensor types for overloaded load methods
from .tile_types import SMemTile2D

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
](TrivialRegisterType):
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


struct ScalesTileLoader[
    tma_origin: ImmutOrigin,
    dtype: DType,
    gmem_layout: Layout,
    desc_layout: Layout,
    /,
    *,
    cta_group: Int,
](TrivialRegisterType):
    """TMA-based scales tile loader for blockwise FP8.

    Unlike TileLoaderTMA, this loader:
    - Uses async_copy (no multicast) since scales aren't distributed across CTAs
    - Uses (row_coord, k_coord) coordinate order matching scales tensor layout

    Parameters:
        tma_origin: Origin of the TMA descriptor pointer.
        dtype: Element data type (typically float8 for scales).
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

    @always_inline
    fn __init__(out self, tma_op: Self.TmaOpPtr):
        """Initialize the scales tile loader.

        Args:
            tma_op: Pointer to TMA descriptor (grid constant).
        """
        self.tma_op = tma_op

    @always_inline
    fn load[
        tile_layout: Layout,
        /,
        alignment: Int = 128,
    ](
        self,
        dest: LTSMemTile[Self.dtype, tile_layout, alignment=alignment],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        row_coord: Int,
        k_coord: Int,
    ):
        """Load a scales tile using TMA hardware acceleration.

        Issues an async copy from global memory to shared memory.
        Unlike TileLoaderTMA, this uses (row_coord, k_coord) order
        matching the scales tensor layout.

        Args:
            dest: Destination SMEM tile.
            barrier: Memory barrier for TMA completion signaling.
            row_coord: Row coordinate (M for A-scales) in global memory.
            k_coord: K dimension coordinate in global memory.
        """
        self.tma_op[].async_copy[Self.cta_group](
            dest, barrier, (row_coord, k_coord)
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
        row_coord: Int,
        k_coord: Int,
    ):
        """Load a TileTensor scales tile using TMA hardware acceleration.

        This overload accepts TileTensor-based tiles and converts them to
        LayoutTensor internally for the TMA operation. Zero-cost conversion
        via pointer reinterpretation.

        Args:
            dest: Destination SMEM TileTensor tile.
            barrier: Memory barrier for TMA completion signaling.
            row_coord: Row coordinate (M for A-scales) in global memory.
            k_coord: K dimension coordinate in global memory.
        """
        # Construct LayoutTensor from TileTensor pointer for TMA API
        comptime tile_layout = LegacyLayout.row_major(dim0, dim1)
        var lt_dest = LayoutTensor[
            Self.dtype,
            tile_layout,
            address_space = AddressSpace.SHARED,
            alignment=alignment,
        ](dest.ptr)
        self.tma_op[].async_copy[Self.cta_group](
            lt_dest, barrier, (row_coord, k_coord)
        )
