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
"""Placeholder configuration to start to prototype the API with."""

from std.collections.optional import OptionalReg
from std.gpu.host.constant_memory_mapping import ConstantMemoryMapping
from std.gpu.host.dim import Dim
from std.gpu.host.launch_attribute import LaunchAttribute
from . import (
    ExecutionConfig,
    BlockExecutionConfig,
    GridBlockExecutionConfig,
    ClusterExecutionConfig,
    AsmDumpableExecutionConfig,
    LlvmDumpableExecutionConfig,
    VendorIRDumpableExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
)
from std.gpu.host.device_context import _DumpPath


struct GPUExecutionConfiguration(
    BlockExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
):
    """Describes the execution configuration of a GPU kernel launch."""

    var grid_dim: Dim
    var block_dim: Dim
    var shared_mem_bytes: UInt64

    def __init__(
        out self,
        var grid_dim: Dim,
        var block_dim: Dim,
        var shared_mem_bytes: UInt64 = 0,
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.shared_mem_bytes = shared_mem_bytes

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

    def get_near_compute_scratchpad_usage(self) -> UInt64:
        """Gets the near-compute scratchpad usage configuration.

        Returns:
            The amount of scratchpad used in bytes.
        """
        return self.shared_mem_bytes

    def set_near_compute_scratchpad_usage(mut self, var usage: UInt64):
        """Sets the near-compute scratchpad usage configuration.

        Args:
            usage: The amount of scratchpad to use in bytes.
        """
        self.shared_mem_bytes = usage
