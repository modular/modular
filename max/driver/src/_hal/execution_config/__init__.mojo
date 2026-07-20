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

from std.collections.optional import OptionalReg
from std.gpu.host.constant_memory_mapping import ConstantMemoryMapping
from std.gpu.host.dim import Dim
from std.gpu.host.launch_attribute import LaunchAttribute
from std.gpu.host.device_context import _DumpPath
from .gpu_execution_config import GPUExecutionConfiguration


trait ExecutionConfig:
    """Holds the relevant configuration for launching kernels on a device.

    This base trait exists so that generic code can request a generic config without needing to require any hardware capabilities. From there, it can make use of `conforms_to` to being to inspect the config, or it can simply pass it to lower layers of the stack.
    """

    pass


trait BlockExecutionConfig(ExecutionConfig):
    """Represents a type which holds an execution config for one or more
    SIMT devices.

    This enables the abstract SIMT-style "GPU" to be programmed generically
    without creating a dependency on a particular GPU. This does not promise that grids exist, meaning it is also suitable for more generic SIMT devices. This may also act as a "per core" config for devices which expose more of a SIMD model but where the user wishes to use SIMT programming on the vector units. This can be mixed, so a user on x86 with AVX512 may ask for numCores=16, block_dim=(16, 1, 1) to have one active SIMT context for each of 16 cores which spans AVX512's logical vector width for fp32 computations.
    """

    def __init__(out self, *, block_dim: Dim):
        """Initializes the execution config with the given block dimensions.

        Args:
            block_dim: The block dimensions as a `Dim` value.
        """
        ...

    def get_block_dim(self) -> Dim:
        """Gets the block dimensions for the kernel launch.

        Returns:
            The block dimensions as a `Dim` value.
        """
        ...

    def set_block_dim(mut self, var block_dim: Dim):
        """Sets the block dimensions for the kernel launch.

        Args:
            block_dim: The block dimensions as a `Dim` value.
        """
        ...


trait GridBlockExecutionConfig(BlockExecutionConfig):
    """Represents a type which holds an execution config for one or more
    GPU-like devices.

    This extension trait adds the the capability to specify grids, which is sufficient to enable many ops to only require this trait of the execution config.
    """

    def __init__(out self, *, grid_dim: Dim, block_dim: Dim):
        """Initializes the execution config with the given grid and block dimensions.

        Args:
            grid_dim: The grid dimensions as a `Dim` value.
            block_dim: The block dimensions as a `Dim` value.
        """
        ...

    def get_grid_dim(self) -> Dim:
        """Gets the grid dimensions for the kernel launch.

        Returns:
            The grid dimensions as a `Dim` value.
        """
        ...

    def set_grid_dim(mut self, var grid_dim: Dim):
        """Sets the grid dimensions for the kernel launch.

        Args:
            grid_dim: The grid dimensions as a `Dim` value.
        """
        ...


trait ClusterExecutionConfig(GridBlockExecutionConfig):
    """Represents an execution config which includes the notion of clusters
    between the grid and block dimensions.
    """

    def __init__(out self, *, cluster_dim: Dim):
        """Initializes the execution config with the given cluster dimensions.

        Args:
            cluster_dim: The cluster dimensions as a `Dim` value.
        """
        ...

    def get_cluster_dim(self) -> Dim:
        """Gets the cluster dimensions for the kernel launch.

        Returns:
            The cluster dimensions as a `Dim` value.
        """
        ...

    def set_cluster_dim(mut self, var cluster_dim: Dim):
        """Sets the cluster dimensions for the kernel launch.

        Args:
            cluster_dim: The cluster dimensions as a `Dim` value.
        """
        ...


trait AsmDumpableExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to dump kernel ASM to a path.
    """

    comptime asm_dump_path: _DumpPath = False
    """The output path used for ASM dumps."""


trait LlvmDumpableExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to dump LLVM IR to a path."""

    comptime llvm_dump_path: _DumpPath = False
    """The output path used for LLVM IR dumps."""


# NOTE: This is where PTX goes, SASS is ASM for NVIDIA.
trait VendorIRDumpableExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to dump vendor-specific IR to a path.
    """

    comptime vendor_ir_dump_path: _DumpPath = False
    """The output path used for vendor-specific IR dumps."""


trait NearComputeGeneralPurposeScratchpadExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to specify whether to use
    near-compute general-purpose scratchpad memory (e.g. NVIDIA's shared memory).

    Maps to shared memory on NVIDIA. Most CPUs don't have an equivalent. **DOES NOT** map to tmem on NVIDIA since the general purpose compute can't directly load/store from that.
    """

    def get_near_compute_scratchpad_usage(self) -> UInt64:
        """Gets the near-compute scratchpad usage configuration.

        Returns:
            The amount of scratchpad used in bytes.
        """
        ...

    def set_near_compute_scratchpad_usage(mut self, var usage: UInt64):
        """Sets the near-compute scratchpad usage configuration.

        Args:
            usage: The amount of scratchpad to use in bytes.
        """
        ...


trait ConstantMemoryMappingExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to hold constant memory
    mappings."""

    def get_constant_memory_mappings[
        o: ImmOrigin
    ](ref[o] self) -> Span[ConstantMemoryMapping, o]:
        """Gets the constant memory mappings configured for the execution config.

        Returns:
            The constant memory mappings.
        """
        ...

    def set_constant_memory_mappings(
        mut self,
        var mappings: List[ConstantMemoryMapping],
    ):
        """Sets the constant memory mappings for the execution config.

        Args:
            mappings: The constant memory mappings to apply.
        """
        ...


trait LaunchAttributeHolderExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to hold launch attributes."""

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Gets the launch attributes configured for kernel launch.

        Returns:
            The configured launch attributes.
        """
        ...

    def set_launch_attributes(mut self, var attributes: List[LaunchAttribute]):
        """Sets launch attributes used for kernel launch.

        Args:
            attributes: Launch attributes to apply.
        """
        ...
