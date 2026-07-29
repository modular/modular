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
"""HAL Execution Config traits and structs for AMD products.
"""

from .. import (
    ExecutionConfig,
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    ExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
)
from std.collections.type_dict import TypeDict
from std.gpu.host.constant_memory_mapping import ConstantMemoryMapping
from std.gpu.host.dim import Dim

from .gcn1 import GCN1ExecutionConfiguration
from .gcn2 import GCN2ExecutionConfiguration
from .gcn3 import GCN3ExecutionConfiguration
from .gcn4 import GCN4ExecutionConfiguration
from .gcn5 import GCN5ExecutionConfiguration
from .rdna1 import RDNA1ExecutionConfiguration
from .cdna1 import CDNA1ExecutionConfiguration
from .rdna2 import RDNA2ExecutionConfiguration
from .cdna2 import CDNA2ExecutionConfiguration
from .rdna3 import RDNA3ExecutionConfiguration
from .cdna3 import CDNA3ExecutionConfiguration
from .rdna4 import RDNA4ExecutionConfiguration
from .cdna4 import CDNA4ExecutionConfiguration
from .cdna5 import CDNA5ExecutionConfiguration

# Every AMDGPU processor (LLVM's gfx names,
# https://llvm.org/docs/AMDGPUUsage.html#processors) maps to its generation's
# config, so all processors in a generation share one value. The key and value
# lists are parallel and kept in the same generation-grouped order.
#
# Two placements are deliberate:
#   - gfx803 is both Fiji (GCN3) and Polaris (GCN4). A key resolves to a single
#     config, so gfx803 maps to GCN4 (its only ISA); GCN3 keys on its other
#     processors. A Fiji device therefore resolves to the GCN4 config, which is
#     harmless since the two share an ISA and thus launch capabilities.
#   - RDNA3.5 (gfx115x) folds into RDNA3, and RDNA4m (gfx117x, the Samsung
#     Exynos 2600 Xclipse GPU) folds into RDNA4; the repo has no separate
#     config for either mobile variant. Split them out if configs are added.
comptime _ExecutionConfigDictForTarget = TypeDict[
    T=StaticString,
    Trait=ExecutionConfig,
    [
        # GCN1 (Southern Islands)
        "amdgpu:gfx600",
        "amdgpu:gfx601",
        "amdgpu:gfx602",
        # GCN2 (Sea Islands)
        "amdgpu:gfx700",
        "amdgpu:gfx701",
        "amdgpu:gfx702",
        "amdgpu:gfx703",
        "amdgpu:gfx704",
        "amdgpu:gfx705",
        # GCN3 (Volcanic Islands); gfx803 (Fiji) is keyed under GCN4 below.
        "amdgpu:gfx801",
        "amdgpu:gfx802",
        "amdgpu:gfx805",
        "amdgpu:gfx810",
        # GCN4 (Polaris)
        "amdgpu:gfx803",
        # GCN5 (Vega)
        "amdgpu:gfx900",
        "amdgpu:gfx902",
        "amdgpu:gfx904",
        "amdgpu:gfx906",
        "amdgpu:gfx909",
        "amdgpu:gfx90c",
        # RDNA1
        "amdgpu:gfx1010",
        "amdgpu:gfx1011",
        "amdgpu:gfx1012",
        "amdgpu:gfx1013",
        # CDNA1 (MI100)
        "amdgpu:gfx908",
        # RDNA2
        "amdgpu:gfx1030",
        "amdgpu:gfx1031",
        "amdgpu:gfx1032",
        "amdgpu:gfx1033",
        "amdgpu:gfx1034",
        "amdgpu:gfx1035",
        "amdgpu:gfx1036",
        # CDNA2 (MI250X)
        "amdgpu:gfx90a",
        # RDNA3 (GFX11) and RDNA3.5 (gfx115x)
        "amdgpu:gfx1100",
        "amdgpu:gfx1101",
        "amdgpu:gfx1102",
        "amdgpu:gfx1103",
        "amdgpu:gfx1150",
        "amdgpu:gfx1151",
        "amdgpu:gfx1152",
        "amdgpu:gfx1153",
        # CDNA3 (MI300)
        "amdgpu:gfx940",
        "amdgpu:gfx941",
        "amdgpu:gfx942",
        # RDNA4 (GFX12) and RDNA4m (gfx117x, Samsung Exynos 2600 Xclipse GPU)
        "amdgpu:gfx1200",
        "amdgpu:gfx1201",
        "amdgpu:gfx1170",
        "amdgpu:gfx1171",
        "amdgpu:gfx1172",
        # CDNA4 (MI350X)
        "amdgpu:gfx950",
        # CDNA5 (MI400)
        "amdgpu:gfx1250",
        "amdgpu:gfx1251",
    ],
    GCN1ExecutionConfiguration,
    GCN1ExecutionConfiguration,
    GCN1ExecutionConfiguration,
    GCN2ExecutionConfiguration,
    GCN2ExecutionConfiguration,
    GCN2ExecutionConfiguration,
    GCN2ExecutionConfiguration,
    GCN2ExecutionConfiguration,
    GCN2ExecutionConfiguration,
    GCN3ExecutionConfiguration,
    GCN3ExecutionConfiguration,
    GCN3ExecutionConfiguration,
    GCN3ExecutionConfiguration,
    GCN4ExecutionConfiguration,
    GCN5ExecutionConfiguration,
    GCN5ExecutionConfiguration,
    GCN5ExecutionConfiguration,
    GCN5ExecutionConfiguration,
    GCN5ExecutionConfiguration,
    GCN5ExecutionConfiguration,
    RDNA1ExecutionConfiguration,
    RDNA1ExecutionConfiguration,
    RDNA1ExecutionConfiguration,
    RDNA1ExecutionConfiguration,
    CDNA1ExecutionConfiguration,
    RDNA2ExecutionConfiguration,
    RDNA2ExecutionConfiguration,
    RDNA2ExecutionConfiguration,
    RDNA2ExecutionConfiguration,
    RDNA2ExecutionConfiguration,
    RDNA2ExecutionConfiguration,
    RDNA2ExecutionConfiguration,
    CDNA2ExecutionConfiguration,
    RDNA3ExecutionConfiguration,
    RDNA3ExecutionConfiguration,
    RDNA3ExecutionConfiguration,
    RDNA3ExecutionConfiguration,
    RDNA3ExecutionConfiguration,
    RDNA3ExecutionConfiguration,
    RDNA3ExecutionConfiguration,
    RDNA3ExecutionConfiguration,
    CDNA3ExecutionConfiguration,
    CDNA3ExecutionConfiguration,
    CDNA3ExecutionConfiguration,
    RDNA4ExecutionConfiguration,
    RDNA4ExecutionConfiguration,
    RDNA4ExecutionConfiguration,
    RDNA4ExecutionConfiguration,
    RDNA4ExecutionConfiguration,
    CDNA4ExecutionConfiguration,
    CDNA5ExecutionConfiguration,
    CDNA5ExecutionConfiguration,
]

comptime ExecutionConfigForTarget[
    target: StaticString
] = _ExecutionConfigDictForTarget.get[target]


trait _BaseAmdExecutionConfig(
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
):
    """The shared capabilities of all supported AMD GPUs."""

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        ...


trait _AmdExecutionConfigDelegator(
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    _BaseAmdExecutionConfig,
):
    """A delegating `ExecutionConfig` which forwards all calls to an inner
    `_BaseAmdExecutionConfiguration`.
    """

    def _get_inner_config[
        o: Origin
    ](ref[o] self) -> ref[o] _BaseAmdExecutionConfiguration:
        """Gets the inner `_BaseAmdExecutionConfiguration` to which this delegator forwards
        all calls.

        Returns:
            The inner `_BaseAmdExecutionConfiguration`.
        """
        ...

    def __init__(out self, *, block_dim: Dim):
        """Initializes the execution config with the given block dimensions.

        Args:
            block_dim: The block dimensions as a `Dim` value.
        """
        self = Self.__init__(grid_dim=Dim(1, 1, 1), block_dim=block_dim)

    def get_block_dim(self) -> Dim:
        """Gets the block dimensions for the kernel launch.

        Returns:
            The block dimensions as a `Dim` value.
        """
        return self._get_inner_config().get_block_dim()

    def set_block_dim(mut self, var block_dim: Dim):
        """Sets the block dimensions for the kernel launch.

        Args:
            block_dim: The block dimensions as a `Dim` value.
        """
        self._get_inner_config().set_block_dim(block_dim)

    def __init__(out self, *, grid_dim: Dim, block_dim: Dim):
        """Initializes the execution config with the given grid and block dimensions.

        Args:
            grid_dim: The grid dimensions as a `Dim` value.
            block_dim: The block dimensions as a `Dim` value.
        """
        self = Self.__init__(
            grid_dim=grid_dim, block_dim=block_dim, shared_mem_bytes=0
        )

    def get_grid_dim(self) -> Dim:
        """Gets the grid dimensions for the kernel launch.

        Returns:
            The grid dimensions as a `Dim` value.
        """
        return self._get_inner_config().get_grid_dim()

    def set_grid_dim(mut self, var grid_dim: Dim):
        """Sets the grid dimensions for the kernel launch.

        Args:
            grid_dim: The grid dimensions as a `Dim` value.
        """
        self._get_inner_config().set_grid_dim(grid_dim)

    def get_near_compute_scratchpad_usage(self) -> Int:
        """Gets the near-compute scratchpad usage configuration.

        Returns:
            The amount of scratchpad used in bytes.
        """
        return self._get_inner_config().get_near_compute_scratchpad_usage()

    def set_near_compute_scratchpad_usage(mut self, var usage: Int):
        """Sets the near-compute scratchpad usage configuration.

        Args:
            usage: The amount of scratchpad to use in bytes.
        """
        self._get_inner_config().set_near_compute_scratchpad_usage(usage)

    def get_constant_memory_mappings[
        o: ImmOrigin
    ](ref[o] self) -> Span[ConstantMemoryMapping, o]:
        """Gets the constant memory mappings configured for the execution config.

        Returns:
            The constant memory mappings, or `None` if unset.
        """
        return rebind[Span[ConstantMemoryMapping, o]](
            self._get_inner_config().get_constant_memory_mappings()
        )

    def set_constant_memory_mappings(
        mut self,
        var mappings: List[ConstantMemoryMapping],
    ):
        """Sets the constant memory mappings for the execution config.

        Args:
            mappings: The constant memory mappings to apply.
        """
        self._get_inner_config().set_constant_memory_mappings(mappings^)


struct _BaseAmdExecutionConfiguration(
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
):
    """The shared capabilities of all supported AMD GPUs."""

    var grid_dim: Dim
    var block_dim: Dim
    var shared_mem_bytes: Int
    var constant_memory: List[ConstantMemoryMapping]

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.shared_mem_bytes = shared_mem_bytes
        self.constant_memory = constant_memory^

    def __init__(out self, *, block_dim: Dim):
        """Initializes the execution config with the given block dimensions.

        Args:
            block_dim: The block dimensions as a `Dim` value.
        """
        self = Self.__init__(grid_dim=Dim(1, 1, 1), block_dim=block_dim)

    def get_block_dim(self) -> Dim:
        """Gets the block dimensions for the kernel launch.

        Returns:
            The block dimensions as a `Dim` value.
        """
        return self.block_dim

    def set_block_dim(mut self, var block_dim: Dim):
        """Sets the block dimensions for the kernel launch.

        Args:
            block_dim: The block dimensions as a `Dim` value.
        """
        self.block_dim = block_dim

    def __init__(out self, *, grid_dim: Dim, block_dim: Dim):
        """Initializes the execution config with the given grid and block dimensions.

        Args:
            grid_dim: The grid dimensions as a `Dim` value.
            block_dim: The block dimensions as a `Dim` value.
        """
        self = Self.__init__(
            grid_dim=grid_dim, block_dim=block_dim, shared_mem_bytes=0
        )

    def get_grid_dim(self) -> Dim:
        """Gets the grid dimensions for the kernel launch.

        Returns:
            The grid dimensions as a `Dim` value.
        """
        return self.grid_dim

    def set_grid_dim(mut self, var grid_dim: Dim):
        """Sets the grid dimensions for the kernel launch.

        Args:
            grid_dim: The grid dimensions as a `Dim` value.
        """
        self.grid_dim = grid_dim

    def get_near_compute_scratchpad_usage(self) -> Int:
        """Gets the near-compute scratchpad usage configuration.

        Returns:
            The amount of scratchpad used in bytes.
        """
        return self.shared_mem_bytes

    def set_near_compute_scratchpad_usage(mut self, var usage: Int):
        """Sets the near-compute scratchpad usage configuration.

        Args:
            usage: The amount of scratchpad to use in bytes.
        """
        self.shared_mem_bytes = usage

    def get_constant_memory_mappings[
        o: ImmOrigin
    ](ref[o] self) -> Span[ConstantMemoryMapping, o]:
        """Gets the constant memory mappings configured for the execution config.

        Returns:
            The constant memory mappings, or `None` if unset.
        """
        return rebind[Span[ConstantMemoryMapping, o]](
            Span[ConstantMemoryMapping, _](self.constant_memory)
        )

    def set_constant_memory_mappings(
        mut self,
        var mappings: List[ConstantMemoryMapping],
    ):
        """Sets the constant memory mappings for the execution config.

        Args:
            mappings: The constant memory mappings to apply.
        """
        self.constant_memory = mappings^
