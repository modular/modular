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

"""TMA loaders for 5D MXFP8 scaling factor tensors.

- ScalingFactorLoader: Async TMA loads for SFA/SFB tiles
"""

from sys import size_of

from gpu.cluster import elect_one_sync
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.mma_sm100 import MMASmemDescriptor
from gpu.tcgen05 import tcgen05_cp
from layout import Layout, LayoutTensor
from layout.tma_async import SharedMemBarrier, TMATensorTile

from linalg.fp4_utils import SF_MN_GROUP_SIZE, SF_ATOM_M, SF_ATOM_K

from .tile_loader import TileLoaderTMA


# =============================================================================
# ScalingFactorLoader - TMA loader for 5D scaling factor tensors
# =============================================================================


@register_passable("trivial")
struct ScalingFactorLoader[
    tma_origin: ImmutOrigin,
    sf_dtype: DType,
    sf_layout: Layout,
    sf_desc_layout: Layout,
    /,
    *,
    cta_group: Int,
    BM_or_MMA_N: Int,  # BM for A scales, MMA_N for B scales
]:
    """TMA loader for 5D MXFP8 scaling factor tensors (SFA or SFB)."""

    comptime TmaOp = TMATensorTile[
        Self.sf_dtype, Self.sf_layout, Self.sf_desc_layout
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]

    var tma_op: Self.TmaOpPtr

    @always_inline
    fn __init__(out self, tma_op: Self.TmaOpPtr):
        """Initialize the scaling factor loader.

        Args:
            tma_op: Pointer to TMA descriptor for scaling factors.
        """
        self.tma_op = tma_op

    @always_inline
    fn load(
        self,
        dest: LayoutTensor[
            Self.sf_dtype,
            _,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref [AddressSpace.SHARED]barrier: SharedMemBarrier,
        k_iter: UInt,
        mn_block_idx: UInt,
        batch_idx: UInt,
    ):
        """Load scaling factors using TMA 5D copy.

        Issues an async load from global memory to shared memory for
        scaling factor tiles.

        Args:
            dest: Destination SMEM tile for scaling factors.
            barrier: Memory barrier for TMA completion signaling.
            k_iter: K iteration index.
            mn_block_idx: M block index (for A) or N block index (for B).
            batch_idx: Batch dimension index.
        """
        self.tma_op[].async_copy_5d[Self.cta_group](
            dest,
            barrier,
            (
                UInt(0),  # SF_ATOM_K start (always 0)
                UInt(0),  # SF_ATOM_M start (always 0)
                k_iter,
                mn_block_idx * UInt(Self.BM_or_MMA_N // SF_MN_GROUP_SIZE),
                batch_idx,
            ),
        )


# TileLoaderTMA is imported above and can be used directly for A/B tiles
