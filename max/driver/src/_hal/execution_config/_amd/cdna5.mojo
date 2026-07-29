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
"""Execution config placeholder for AMD CDNA5 GPUs (Instinct MI400 series:
MI430X, MI440X, MI455X)."""

from std.gpu.host.constant_memory_mapping import ConstantMemoryMapping
from std.gpu.host.dim import Dim
from . import (
    _BaseAmdExecutionConfiguration,
    _BaseAmdExecutionConfig,
    _AmdExecutionConfigDelegator,
)
from .. import (
    ExecutionConfig,
    BlockExecutionConfig,
    GridBlockExecutionConfig,
    ClusterExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
)


struct CDNA5ExecutionConfiguration(
    BlockExecutionConfig,
    ClusterExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    _AmdExecutionConfigDelegator,
    _BaseAmdExecutionConfig,
):
    """Describes the execution configuration of a GPU kernel launch on AMD
    CDNA5 GPUs."""

    var _base: _BaseAmdExecutionConfiguration
    var cluster_dim: Optional[Dim]

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var cluster_dim: Optional[Dim] = Optional[Dim](Dim(1, 1, 1)),
        var shared_mem_bytes: Int = 0,
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self._base = _BaseAmdExecutionConfiguration(
            block_dim=block_dim,
            grid_dim=grid_dim,
            shared_mem_bytes=shared_mem_bytes,
            constant_memory=constant_memory^,
        )
        self.cluster_dim = cluster_dim

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self._base = _BaseAmdExecutionConfiguration(
            block_dim=block_dim,
            grid_dim=grid_dim,
            shared_mem_bytes=shared_mem_bytes,
            constant_memory=constant_memory^,
        )
        self.cluster_dim = Optional[Dim](Dim(1, 1, 1))

    def _get_inner_config[
        o: Origin
    ](ref[o] self) -> ref[o] _BaseAmdExecutionConfiguration:
        """Gets the inner `_BaseAmdExecutionConfiguration` to which this delegator forwards
        all calls.

        Returns:
            The inner `_BaseAmdExecutionConfiguration`.
        """
        return Pointer(to=self._base).unsafe_origin_cast[o]()[]

    def __init__(out self, *, cluster_dim: Optional[Dim]):
        """Initializes the execution config with the given cluster dimensions.

        Args:
            cluster_dim: The cluster dimensions as a `Dim` value, or `None` to unset.
        """
        self = Self.__init__(
            grid_dim=Dim(1, 1, 1),
            block_dim=Dim(1, 1, 1),
            cluster_dim=cluster_dim,
            shared_mem_bytes=0,
        )

    def get_cluster_dim(self) -> Optional[Dim]:
        """Gets the cluster dimensions for the kernel launch.

        Returns:
            The cluster dimensions as a `Dim` value, or `None` if unset.
        """
        return self.cluster_dim

    def set_cluster_dim(mut self, var cluster_dim: Optional[Dim]):
        """Sets the cluster dimensions for the kernel launch.

        Args:
            cluster_dim: The cluster dimensions as a `Dim` value, or `None` to unset.
        """
        self.cluster_dim = cluster_dim
