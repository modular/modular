# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""HAL Execution Config traits and structs for NVIDIA Products."""

from max.gpu.host import (
    ExecutionConfig,
    BlockExecutionConfig,
    ClusterExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
)
from max.gpu.host.nvidia import LaunchAttributeHolderExecutionConfig
from .. import ConstantMemoryMappingExecutionConfig
from std.collections.type_dict import TypeDict
from max.gpu.host import ConstantMemoryMapping, Dim, LaunchAttribute


comptime _ExecutionConfigDictForTarget = TypeDict[
    T=StaticString,
    Trait=ExecutionConfig,
    [
        "nvidia:sm_52",
        "nvidia:sm_60",
        "nvidia:sm_61",
        "nvidia:sm_61",
        "nvidia:sm_75",
        "nvidia:sm_80",
        "nvidia:sm_86",
        "nvidia:sm_86",
        "nvidia:sm_89",
        "nvidia:sm_90a",
        "nvidia:sm_87",
        "nvidia:sm_89",
        "nvidia:sm_100a",
        "nvidia:sm_120a",
        "nvidia:sm_121a",
        "nvidia:sm_110a",
        "nvidia:sm_103a",
        "nvidia:sm_121",
    ],
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _ClusterCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _ClusterCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
    _ClusterCudaExecutionConfiguration,
    _ClusterCudaExecutionConfiguration,
    _BaseCudaExecutionConfiguration,
]

comptime ExecutionConfigForTarget[
    target: StaticString
] = _ExecutionConfigDictForTarget.get[target]


struct _BaseCudaExecutionConfiguration(
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    LaunchAttributeHolderExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
):
    """The shared capabilities of all supported CUDA-capable GPUs."""

    var grid_dim: Dim
    var block_dim: Dim
    var shared_mem_bytes: Int
    var attributes: List[LaunchAttribute]
    var constant_memory: List[ConstantMemoryMapping]

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.shared_mem_bytes = shared_mem_bytes
        self.attributes = attributes^
        self.constant_memory = constant_memory^

    def __init__(out self, *, block_dim: Dim):
        """Initializes the execution config with the given block dimensions.

        Args:
            block_dim: The block dimensions as a `Dim` value.
        """
        self = Self(grid_dim=Dim(1, 1, 1), block_dim=block_dim)

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
        self = Self(grid_dim=grid_dim, block_dim=block_dim, shared_mem_bytes=0)

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

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Gets the launch attributes configured for kernel launch.

        Returns:
            The configured launch attributes, or `None` if unset.
        """
        return rebind[Span[LaunchAttribute, o]](
            Span[LaunchAttribute, _](self.attributes)
        )

    def set_launch_attributes(mut self, var attributes: List[LaunchAttribute]):
        """Sets launch attributes used for kernel launch.

        Args:
            attributes: Launch attributes to apply, or `None` to clear them.
        """
        self.attributes = attributes^

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


struct _ClusterCudaExecutionConfiguration(
    BlockExecutionConfig,
    ClusterExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    LaunchAttributeHolderExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
):
    """The execution configuration of CUDA GPUs that support thread block
    clusters."""

    var grid_dim: Dim
    var block_dim: Dim
    var cluster_dim: Dim
    var shared_mem_bytes: Int
    var attributes: List[LaunchAttribute]
    var constant_memory: List[ConstantMemoryMapping]

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self = Self(
            block_dim=block_dim,
            grid_dim=grid_dim,
            cluster_dim=Dim(1, 1, 1),
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var cluster_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.cluster_dim = cluster_dim
        self.shared_mem_bytes = shared_mem_bytes
        self.attributes = attributes^
        self.constant_memory = constant_memory^

    def __init__(out self, *, cluster_dim: Dim):
        """Initializes the execution config with the given cluster dimensions.

        Args:
            cluster_dim: The cluster dimensions as a `Dim` value.
        """
        self = Self(
            grid_dim=Dim(1, 1, 1),
            block_dim=Dim(1, 1, 1),
            cluster_dim=cluster_dim,
            shared_mem_bytes=0,
        )

    def __init__(out self, *, block_dim: Dim):
        """Initializes the execution config with the given block dimensions.

        Args:
            block_dim: The block dimensions as a `Dim` value.
        """
        self = Self(grid_dim=Dim(1, 1, 1), block_dim=block_dim)

    def __init__(out self, *, grid_dim: Dim, block_dim: Dim):
        """Initializes the execution config with the given grid and block dimensions.

        Args:
            grid_dim: The grid dimensions as a `Dim` value.
            block_dim: The block dimensions as a `Dim` value.
        """
        self = Self(grid_dim=grid_dim, block_dim=block_dim, shared_mem_bytes=0)

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

    def get_cluster_dim(self) -> Dim:
        """Gets the cluster dimensions for the kernel launch.

        Returns:
            The cluster dimensions as a `Dim` value.
        """
        return self.cluster_dim

    def set_cluster_dim(mut self, var cluster_dim: Dim):
        """Sets the cluster dimensions for the kernel launch.

        Args:
            cluster_dim: The cluster dimensions as a `Dim` value.
        """
        self.cluster_dim = cluster_dim

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

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Gets the launch attributes configured for kernel launch.

        Returns:
            The configured launch attributes, or `None` if unset.
        """
        return rebind[Span[LaunchAttribute, o]](
            Span[LaunchAttribute, _](self.attributes)
        )

    def set_launch_attributes(mut self, var attributes: List[LaunchAttribute]):
        """Sets launch attributes used for kernel launch.

        Args:
            attributes: Launch attributes to apply, or `None` to clear them.
        """
        self.attributes = attributes^

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
