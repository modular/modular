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
"""Placeholder configuration to start to prototype the API with."""

from max.gpu.host import (
    ExecutionConfig,
    BlockExecutionConfig,
    ClusterExecutionConfig,
    ConstantMemoryMapping,
    Dim,
    GridBlockExecutionConfig,
    LaunchAttribute,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
)
from max.gpu.host.nvidia import LaunchAttributeHolderExecutionConfig
from . import ConstantMemoryMappingExecutionConfig


struct GPUExecutionConfiguration[
    *,
    has_cluster_dim: Bool = False,
    has_launch_attributes: Bool = False,
    has_constant_memory: Bool = False,
](
    BlockExecutionConfig,
    ClusterExecutionConfig where has_cluster_dim,
    ConstantMemoryMappingExecutionConfig where has_constant_memory,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    LaunchAttributeHolderExecutionConfig where has_launch_attributes,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
):
    """Describes the execution configuration of a GPU kernel launch.

    Each feature a target supports adds a field, its accessors and the matching
    trait conformance. A feature the target lacks adds nothing, so a launch
    that asks for it fails to compile.

    Parameters:
        has_cluster_dim: Whether the target supports thread block clusters.
            If so, the configuration holds a cluster dimension and conforms to
            `ClusterExecutionConfig`.
        has_launch_attributes: Whether the target supports launch attributes.
            If so, the configuration holds them and conforms to
            `LaunchAttributeHolderExecutionConfig`.
        has_constant_memory: Whether the target supports constant memory
            mappings. If so, the configuration holds them and conforms to
            `ConstantMemoryMappingExecutionConfig`.
    """

    var grid_dim: Dim
    var block_dim: Dim
    var cluster_dim: Dim if Self.has_cluster_dim else NoneType
    var shared_mem_bytes: Int
    var attributes: (
        List[LaunchAttribute] if Self.has_launch_attributes else NoneType
    )
    var constant_memory: (
        List[ConstantMemoryMapping] if Self.has_constant_memory else NoneType
    )

    def __init__(
        out self,
        var grid_dim: Dim,
        var block_dim: Dim,
        var shared_mem_bytes: Int = 0,
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        comptime if Self.has_cluster_dim:
            self.cluster_dim = Dim(1, 1, 1)
        else:
            self.cluster_dim = None
        self.shared_mem_bytes = shared_mem_bytes
        comptime if Self.has_launch_attributes:
            self.attributes = []
        else:
            self.attributes = None
        comptime if Self.has_constant_memory:
            self.constant_memory = []
        else:
            self.constant_memory = None

    def __init__(
        out self,
        *,
        var grid_dim: Dim,
        var block_dim: Dim,
        var cluster_dim: Dim,
        var shared_mem_bytes: Int = 0,
    ) where Self.has_cluster_dim:
        """The 'all members' constructor with a cluster dimension."""
        self = Self(grid_dim, block_dim, shared_mem_bytes)
        self.cluster_dim = cluster_dim

    def __init__(out self, *, cluster_dim: Dim) where Self.has_cluster_dim:
        """Initializes the execution config with the given cluster dimensions.

        Args:
            cluster_dim: The cluster dimensions as a `Dim` value.
        """
        self = Self(grid_dim=Dim(1, 1, 1), block_dim=Dim(1, 1, 1))
        self.cluster_dim = cluster_dim

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

    def get_cluster_dim(self) -> Dim where Self.has_cluster_dim:
        """Gets the cluster dimensions for the kernel launch.

        Returns:
            The cluster dimensions as a `Dim` value.
        """
        return self.cluster_dim

    def set_cluster_dim(
        mut self, var cluster_dim: Dim
    ) where Self.has_cluster_dim:
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
    ](ref[o] self) -> Span[LaunchAttribute, o] where Self.has_launch_attributes:
        """Gets the launch attributes configured for kernel launch.

        Returns:
            The configured launch attributes.
        """
        return rebind[Span[LaunchAttribute, o]](
            Span[LaunchAttribute, _](self.attributes)
        )

    def set_launch_attributes(
        mut self, var attributes: List[LaunchAttribute]
    ) where Self.has_launch_attributes:
        """Sets launch attributes used for kernel launch.

        Args:
            attributes: Launch attributes to apply.
        """
        self.attributes = attributes^

    def get_constant_memory_mappings[
        o: ImmOrigin
    ](ref[o] self) -> Span[
        ConstantMemoryMapping, o
    ] where Self.has_constant_memory:
        """Gets the constant memory mappings configured for the execution config.

        Returns:
            The constant memory mappings.
        """
        return rebind[Span[ConstantMemoryMapping, o]](
            Span[ConstantMemoryMapping, _](self.constant_memory)
        )

    def set_constant_memory_mappings(
        mut self,
        var mappings: List[ConstantMemoryMapping],
    ) where Self.has_constant_memory:
        """Sets the constant memory mappings for the execution config.

        Args:
            mappings: The constant memory mappings to apply.
        """
        self.constant_memory = mappings^
