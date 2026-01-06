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

"""TMA tile loader for SM100 matrix multiplication.

Provides a wrapper around TMA async_multicast_load operations, following
the SM90 TileLoaderTMA pattern. Orchestration logic (k-group iteration,
expect_bytes, barrier management) is handled by the kernel, not the loader.

Usage:
    # In kernel - create separate A and B loaders
    var a_loader = ATileLoaderType(Pointer(to=a_tma_op), ctx.a_multicast_mask)
    var b_loader = BTileLoaderType(Pointer(to=b_tma_op), ctx.b_multicast_mask)

    # Load tiles using the loaders
    a_loader.load(a_tile, barrier, k_coord, m_coord)
    b_loader.load(b_tile, barrier, k_coord, n_coord)
"""

from layout import LayoutTensor
from layout.tma_async import SharedMemBarrier, TMATensorTile


@register_passable("trivial")
struct TileLoaderTMA[
    tma_origin: ImmutOrigin,
    dtype: DType,
    gmem_layout: Layout,
    desc_layout: Layout,
    /,
    *,
    cta_group: Int,
]:
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
    fn load(
        self,
        dest: LayoutTensor[
            Self.dtype,
            _,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref [AddressSpace.SHARED]barrier: SharedMemBarrier,
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
