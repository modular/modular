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
"""Capability-typed kernel launch configurations.

A launch configuration's type says which launch capabilities it uses. The
portable tier (grid, block, and dynamic shared memory) works on every GPU
target. A performance-tier capability, such as thread-block clusters, is a
separate trait, so launching with it on a target that lacks it fails to
compile rather than raising at run time.

A `DeviceContext` publishes `ExecutionConfigType`, the configuration type
with every capability its target supports, so generic code can check for a
capability with `conforms_to` before using it. The HAL `DeviceContext`
accepts the same configuration types.

```mojo
from max.gpu.host import ClusterLaunchConfig, DeviceContext, Dim

def kernel():
    pass

with DeviceContext() as ctx:
    ctx.enqueue_function[kernel](
        ClusterLaunchConfig(grid_dim=4, block_dim=128, cluster_dim=Dim(2, 1, 1))
    )
```
"""

from std.sys.info import CompilationTarget

from ._legacy.dim import Dim


def _supports_cluster_launch[target: CompilationTarget]() -> Bool:
    """Returns whether a target can launch thread block clusters.

    True on sm_90, sm_100, sm_101, sm_103 and sm_110. sm_120 and sm_121 are
    left out to match the HAL per-target configurations, not for lack of
    hardware support.

    Parameters:
        target: The compilation target to check.

    Returns:
        True if `target` supports cluster launch.
    """
    return (
        target._is_sm_9x()
        or target._has_blackwell_tcgen05()
        or target._is_sm_110x()
    )


trait ExecutionConfig:
    """Describes how to launch a kernel.

    Generic code checks which capabilities a configuration has with
    `conforms_to`.
    """

    pass


trait BlockExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` that launches SIMT thread blocks.

    It promises no grid, so it also fits SIMT devices without one.
    """

    def get_block_dim(self) -> Dim:
        """Returns the block dimensions.

        Returns:
            The number of threads in each dimension of a thread block.
        """
        ...


trait GridBlockExecutionConfig(BlockExecutionConfig):
    """An `ExecutionConfig` that launches a grid of thread blocks."""

    def get_grid_dim(self) -> Dim:
        """Returns the grid dimensions.

        Returns:
            The number of thread blocks in each dimension of the grid.
        """
        ...


trait NearComputeGeneralPurposeScratchpadExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` that requests dynamic near-compute scratchpad
    memory, which is shared memory on GPUs."""

    def get_near_compute_scratchpad_usage(self) -> Int:
        """Returns the dynamic scratchpad memory to allocate.

        Returns:
            The number of bytes of dynamic shared memory per thread block.
        """
        ...


trait ClusterExecutionConfig(GridBlockExecutionConfig):
    """An `ExecutionConfig` that groups thread blocks into clusters.

    Launching one on a target without cluster launch fails to compile.
    """

    def get_cluster_dim(self) -> Dim:
        """Returns the cluster dimensions.

        Returns:
            The number of thread blocks in each dimension of a cluster.
        """
        ...


struct GridBlockLaunchConfig(
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    TrivialRegisterPassable,
):
    """A portable launch configuration: grid, block, and dynamic shared memory.

    A dynamic shared memory request above the target's default limit raises
    the kernel's limit, so no `func_attribute` is needed.
    """

    var grid_dim: Dim
    """The grid dimensions."""
    var block_dim: Dim
    """The block dimensions."""
    var shared_mem_bytes: Int
    """The dynamic shared memory per thread block, in bytes."""

    def __init__(
        out self, *, grid_dim: Dim, block_dim: Dim, shared_mem_bytes: Int = 0
    ):
        """Creates a portable launch configuration.

        Args:
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            shared_mem_bytes: The dynamic shared memory per thread block, in
                bytes.
        """
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.shared_mem_bytes = shared_mem_bytes

    def get_grid_dim(self) -> Dim:
        """Returns the grid dimensions.

        Returns:
            The grid dimensions.
        """
        return self.grid_dim

    def get_block_dim(self) -> Dim:
        """Returns the block dimensions.

        Returns:
            The block dimensions.
        """
        return self.block_dim

    def get_near_compute_scratchpad_usage(self) -> Int:
        """Returns the dynamic shared memory per thread block.

        Returns:
            The dynamic shared memory per thread block, in bytes.
        """
        return self.shared_mem_bytes


struct ClusterLaunchConfig(
    ClusterExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    TrivialRegisterPassable,
):
    """A launch configuration that groups thread blocks into clusters.

    Launching with this configuration fails to compile on a target without
    cluster launch.
    """

    var grid_dim: Dim
    """The grid dimensions."""
    var block_dim: Dim
    """The block dimensions."""
    var cluster_dim: Dim
    """The cluster dimensions."""
    var shared_mem_bytes: Int
    """The dynamic shared memory per thread block, in bytes."""

    def __init__(
        out self,
        *,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: Dim,
        shared_mem_bytes: Int = 0,
    ):
        """Creates a cluster launch configuration.

        Args:
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory per thread block, in
                bytes.
        """
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.cluster_dim = cluster_dim
        self.shared_mem_bytes = shared_mem_bytes

    def get_grid_dim(self) -> Dim:
        """Returns the grid dimensions.

        Returns:
            The grid dimensions.
        """
        return self.grid_dim

    def get_block_dim(self) -> Dim:
        """Returns the block dimensions.

        Returns:
            The block dimensions.
        """
        return self.block_dim

    def get_cluster_dim(self) -> Dim:
        """Returns the cluster dimensions.

        Returns:
            The cluster dimensions.
        """
        return self.cluster_dim

    def get_near_compute_scratchpad_usage(self) -> Int:
        """Returns the dynamic shared memory per thread block.

        Returns:
            The dynamic shared memory per thread block, in bytes.
        """
        return self.shared_mem_bytes
