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
"""HAL Execution Config traits and structs for NVIDIA Products."""

from .. import (
    ExecutionConfig,
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    ExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
)
from std.collections.type_dict import TypeDict
from std.gpu.host.launch_attribute import LaunchAttribute
from std.gpu.host.constant_memory_mapping import ConstantMemoryMapping
from std.gpu.host.dim import Dim

from .maxwell_consumer import MaxwellConsumerExecutionConfiguration
from .tesla_p100 import TeslaP100ExecutionConfiguration
from .pascal_consumer import PascalConsumerExecutionConfiguration
from .pascal_workstation import PascalWorkstationExecutionConfiguration
from .turing_consumer import TuringConsumerExecutionConfiguration
from .a100 import A100ExecutionConfiguration
from .ampere_consumer import AmpereConsumerExecutionConfiguration
from .a10 import A10ExecutionConfiguration
from .ada_consumer import AdaConsumerExecutionConfiguration
from .hopper_datacenter import HopperDatacenterExecutionConfiguration
from .orin_nano import OrinNanoExecutionConfiguration
from .l4 import L4ExecutionConfiguration
from .blackwell_datacenter import BlackwellDatacenterExecutionConfiguration
from .blackwell_consumer import BlackwellConsumerExecutionConfiguration
from .dgx_spark import DgxSparkExecutionConfiguration
from .jetson_thor import JetsonThorExecutionConfiguration

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
    ],
    MaxwellConsumerExecutionConfiguration,
    TeslaP100ExecutionConfiguration,
    PascalConsumerExecutionConfiguration,
    PascalWorkstationExecutionConfiguration,
    TuringConsumerExecutionConfiguration,
    A100ExecutionConfiguration,
    AmpereConsumerExecutionConfiguration,
    A10ExecutionConfiguration,
    AdaConsumerExecutionConfiguration,
    HopperDatacenterExecutionConfiguration,
    OrinNanoExecutionConfiguration,
    L4ExecutionConfiguration,
    BlackwellDatacenterExecutionConfiguration,
    BlackwellConsumerExecutionConfiguration,
    DgxSparkExecutionConfiguration,
    JetsonThorExecutionConfiguration,
]

comptime ExecutionConfigForTarget[
    target: StaticString
] = _ExecutionConfigDictForTarget.get[target]


trait LaunchAttributeHolderExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to hold CUDA launch attributes.
    """

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Gets the launch attributes configured for kernel launch.

        Returns:
            The configured launch attributes, or `None` if unset.
        """
        ...

    def set_launch_attributes(mut self, var attributes: List[LaunchAttribute]):
        """Sets launch attributes used for kernel launch.

        Args:
            attributes: Launch attributes to apply, or `None` to clear them.
        """
        ...


trait _BaseCudaExecutionConfig(
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
        ...


trait _CudaExecutionConfigDelegator(
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    LaunchAttributeHolderExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    _BaseCudaExecutionConfig,
):
    """A delegating `ExecutionConfig` which forwards all calls to an inner
    `_BaseCudaExecutionConfig`.
    """

    def _get_inner_config[
        o: Origin
    ](ref[o] self) -> ref[o] _BaseCudaExecutionConfiguration:
        """Gets the inner `_BaseCudaExecutionConfiguration` to which this delegator forwards
        all calls.

        Returns:
            The inner `_BaseCudaExecutionConfiguration`.
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

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Gets the launch attributes configured for kernel launch.

        Returns:
            The configured launch attributes, or `None` if unset.
        """
        return rebind[Span[LaunchAttribute, o]](
            Span[LaunchAttribute, _](
                self._get_inner_config().get_launch_attributes()
            )
        )

    def set_launch_attributes(mut self, var attributes: List[LaunchAttribute]):
        """Sets launch attributes used for kernel launch.

        Args:
            attributes: Launch attributes to apply, or `None` to clear them.
        """
        self._get_inner_config().set_launch_attributes(attributes^)

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
