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

"""Tile loader for SM100 convolution with hardware im2col TMA.

This module provides TileLoaderTMAIm2col, which uses TMA's im2col addressing
mode to perform implicit GEMM convolution without explicit im2col buffers.
The TMA descriptor encodes convolution geometry and transforms coordinates
on-the-fly during memory loads.
"""

from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout.tma_async import SharedMemBarrier, TMATensorTileIm2col
from layout._tile_tensor import TileTensor, TensorLayout


struct TileLoaderTMAIm2col[
    tma_origin: ImmutOrigin,
    dtype: DType,
    gmem_layout: Layout,
    desc_layout: Layout,
    /,
    *,
    cta_group: Int,
](TrivialRegisterPassable):
    """TMA tile loader using hardware im2col for implicit GEMM convolution.

    Uses a TMATensorTileIm2col descriptor (cuTensorMapEncodeIm2col) to perform
    coordinate transformation in TMA hardware. Coordinates are in GEMM space:
    - k_coord: K dimension (C * R * S reduction)
    - m_coord: M dimension (batch * H_out * W_out spatial)

    Parameters:
        tma_origin: Origin of the TMA descriptor pointer.
        dtype: Element data type.
        gmem_layout: Global memory layout of activation tensor (NHWC).
        desc_layout: TMA descriptor layout.
        cta_group: CTA group size (1 or 2 for SM100).
    """

    comptime TmaOp = TMATensorTileIm2col[
        Self.dtype, Self.gmem_layout, Self.desc_layout
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]

    var tma_op: Self.TmaOpPtr
    var multicast_mask: UInt16

    @always_inline
    fn __init__(out self, tma_op: Self.TmaOpPtr, multicast_mask: UInt16):
        self.tma_op = tma_op
        self.multicast_mask = multicast_mask

    @always_inline
    fn load[
        tile_layout: Layout,
        /,
        alignment: Int = 128,
    ](
        self,
        dest: LayoutTensor[
            Self.dtype,
            tile_layout,
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=alignment,
        ],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        k_coord: UInt,
        m_coord: UInt,
    ):
        """Load an activation tile using im2col TMA.

        Note: Uses non-multicast TMA because CUTLASS disables multicast im2col
        on SM100/SM120 (SM90_TMA_LOAD_IM2COL_MULTICAST has CUTE_INVALID_CONTROL_PATH).

        Args:
            dest: Destination SMEM tile.
            barrier: Memory barrier for TMA completion signaling.
            k_coord: K dimension coordinate (C * R * S indexing).
            m_coord: M dimension coordinate (batch * H_out * W_out indexing).
        """
        self.tma_op[].async_copy[Self.cta_group](
            dest, barrier, (k_coord, m_coord)
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
        m_coord: UInt,
    ):
        """Load a TileTensor tile using im2col TMA.

        Args:
            dest: Destination SMEM TileTensor tile.
            barrier: Memory barrier for TMA completion signaling.
            k_coord: K dimension coordinate (C * R * S indexing).
            m_coord: M dimension coordinate (batch * H_out * W_out indexing).
        """
        self.tma_op[].async_copy[Self.cta_group](
            dest, barrier, (k_coord, m_coord)
        )
