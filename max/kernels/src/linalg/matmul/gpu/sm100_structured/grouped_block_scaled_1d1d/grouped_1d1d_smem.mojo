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
"""Shared memory layout for grouped 1D-1D block-scaled SM100 matmul.

This is a simplified SMEM structure for the 1D-1D kernel variant that uses
offset-based addressing instead of pointer-per-group. Key differences from
the standard GroupedBlockScaledSmem:

1. No tensormap descriptors - TMAs are grid-constant (not updated per-group)
2. No CLC pipeline storage - uses 3-warp specialization (no scheduler warp)
3. Simpler barrier structure optimized for the 1D-1D workload

Tile storage is shared via BlockScaledTileCore from block_scaled_smem.mojo.
"""

from gpu.memory import AddressSpace

from ..block_scaled.block_scaled_smem import BlockScaledTileCore
from ..structured_kernels.config import BlockScaledMatmulConfig
from ..structured_kernels.pipeline_storage import SmemPipelineBundleNoClc


struct Grouped1D1DSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]:
    """SMEM struct for grouped 1D-1D block-scaled GEMM without CLC scheduler.

    Thin wrapper over BlockScaledTileCore + SmemPipelineBundleNoClc.
    Uses 3-warp specialization (Load, MMA, Epilogue) without a scheduler warp.
    """

    # ========== Core (tile storage + constants) ==========
    comptime Core = BlockScaledTileCore[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.transpose_b,
        config = Self.config,
    ]

    # ========== Storage Fields ==========
    var core: Self.Core

    # ========== Pipeline Storage (no CLC) ==========
    comptime Pipelines = SmemPipelineBundleNoClc[
        Self.Core.num_group_pipeline_stages,
        Self.Core.num_accum_pipeline_stages,
        Self.Core.Payload,
    ]
    var pipelines: Self.Pipelines

    # ========== Tile Accessors (forwarding) ==========
    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.ATileArray:
        """Get A tile array accessor."""
        return self.core.a_tiles()

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.BTileArray:
        """Get B tile array accessor."""
        return self.core.b_tiles()

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.CTileArray:
        """Get C tile array accessor."""
        return self.core.c_tiles()

    @always_inline
    fn sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.SFATileArray:
        """Get SFA tile array accessor."""
        return self.core.sfa_tiles()

    @always_inline
    fn sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.SFBTileArray:
        """Get SFB tile array accessor."""
        return self.core.sfb_tiles()

    # ========== Size Utilities (forwarding) ==========
    @staticmethod
    @always_inline
    fn ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.Core.ab_pipeline_size()

    @staticmethod
    @always_inline
    fn sf_pipeline_size() -> Int:
        """Total size of SFA+SFB tiles for all pipeline stages (in elements)."""
        return Self.Core.sf_pipeline_size()

    @staticmethod
    @always_inline
    fn c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.Core.c_output_size()

    @staticmethod
    @always_inline
    fn total_tile_size() -> Int:
        """Total tile storage size (A+B+SFA+SFB+C) in elements."""
        return Self.Core.total_tile_size()
