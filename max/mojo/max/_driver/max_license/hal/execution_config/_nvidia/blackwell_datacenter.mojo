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
"""Execution config placeholder for NVIDIA Blackwell datacenter GPUs (B100, B200, B300)."""

from max.gpu.host import ConstantMemoryMapping, Dim, LaunchAttribute
from . import (
    LaunchAttributeHolderExecutionConfig,
    _CudaExecutionConfigDelegator,
    _BaseCudaExecutionConfig,
    _BaseCudaExecutionConfiguration,
)
from max.gpu.host import (
    ExecutionConfig,
    BlockExecutionConfig,
    GridBlockExecutionConfig,
    ClusterExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
)
from .. import ConstantMemoryMappingExecutionConfig


# Blackwell datacenter (B100/B200/B300) is compute capability 10.0, well past
# the 9.0 floor for thread-block clusters, and clusters are a confirmed core
# feature on this datacenter SKU family, so `ClusterExecutionConfig` is kept.
struct BlackwellDatacenterExecutionConfiguration(
    BlockExecutionConfig,
    ClusterExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    LaunchAttributeHolderExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    _BaseCudaExecutionConfig,
    _CudaExecutionConfigDelegator,
):
    """Describes the execution configuration of a GPU kernel launch on
    NVIDIA Blackwell datacenter GPUs."""

    var _base: _BaseCudaExecutionConfiguration
    var cluster_dim: Dim

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
        self._base = _BaseCudaExecutionConfiguration(
            block_dim=block_dim,
            grid_dim=grid_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )
        self.cluster_dim = cluster_dim

    def _get_inner_config[
        o: Origin
    ](ref[o] self) -> ref[o] _BaseCudaExecutionConfiguration:
        """Gets the inner `_BaseCudaExecutionConfiguration` to which this delegator forwards
        all calls.

        Returns:
            The inner `_BaseCudaExecutionConfiguration`.
        """
        return Pointer(to=self._base).unsafe_origin_cast[o]()[]

    def __init__(out self, *, cluster_dim: Dim):
        """Initializes the execution config with the given grid, block, and cluster dimensions.

        Args:
            cluster_dim: The cluster dimensions as a `Dim` value.
        """
        self = Self(
            grid_dim=Dim(1, 1, 1),
            block_dim=Dim(1, 1, 1),
            cluster_dim=cluster_dim,
            shared_mem_bytes=0,
        )

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
