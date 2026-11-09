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
"""NVIDIA launch configurations that carry CUDA launch attributes.

Launch attributes, such as programmatic dependent launch (PDL), are an NVIDIA
performance capability, so they live here rather than on the portable
`GridBlockLaunchConfig` and `ClusterLaunchConfig`. On an NVIDIA target,
`DeviceContext[].ExecutionConfigType` is one of the two configurations below.
Launching one on a target without launch attributes fails to compile.

```mojo
from max.gpu.host import DeviceContext, Dim
from max.gpu.host.nvidia import ClusterAttributeLaunchConfig
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes

def kernel():
    pass

with DeviceContext() as ctx:
    ctx.enqueue_function[kernel](
        ClusterAttributeLaunchConfig(
            grid_dim=4,
            block_dim=128,
            cluster_dim=Dim(2, 1, 1),
            shared_mem_bytes=64 * 1024,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
    )
```
"""

from .._legacy.dim import Dim
from .._legacy.launch_attribute import LaunchAttribute
from ..launch_config import (
    ClusterExecutionConfig,
    ExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
)


trait LaunchAttributeHolderExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` that carries CUDA launch attributes.

    Launching one on a target without launch attributes fails to compile.
    """

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Returns the launch attributes.

        Parameters:
            o: The origin of `self`.

        Returns:
            The attributes to apply to the launch.
        """
        ...


struct GridBlockAttributeLaunchConfig(
    Copyable,
    GridBlockExecutionConfig,
    LaunchAttributeHolderExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
):
    """A launch configuration with grid, block, dynamic shared memory, and
    CUDA launch attributes.

    It works on every NVIDIA target. A dynamic shared memory request above the
    target's default limit raises the kernel's limit, so no `func_attribute` is
    needed.
    """

    var grid_dim: Dim
    """The grid dimensions."""
    var block_dim: Dim
    """The block dimensions."""
    var shared_mem_bytes: Int
    """The dynamic shared memory per thread block, in bytes."""
    var attributes: List[LaunchAttribute]
    """The launch attributes."""

    def __init__(
        out self,
        *,
        grid_dim: Dim,
        block_dim: Dim,
        shared_mem_bytes: Int = 0,
        var attributes: List[LaunchAttribute] = [],
    ):
        """Creates a launch configuration with launch attributes.

        Args:
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            shared_mem_bytes: The dynamic shared memory per thread block, in
                bytes.
            attributes: The launch attributes, for example the result of
                `pdl_launch_attributes()`.
        """
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.shared_mem_bytes = shared_mem_bytes
        self.attributes = attributes^

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

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Returns the launch attributes.

        Parameters:
            o: The origin of `self`.

        Returns:
            The launch attributes.
        """
        return rebind[Span[LaunchAttribute, o]](
            Span[LaunchAttribute, _](self.attributes)
        )


struct ClusterAttributeLaunchConfig(
    ClusterExecutionConfig,
    Copyable,
    LaunchAttributeHolderExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
):
    """A launch configuration with thread block clusters, dynamic shared
    memory, and CUDA launch attributes.

    Launching with this configuration fails to compile on a target without
    cluster launch, such as NVIDIA GPUs before sm_90.
    """

    var grid_dim: Dim
    """The grid dimensions."""
    var block_dim: Dim
    """The block dimensions."""
    var cluster_dim: Dim
    """The cluster dimensions."""
    var shared_mem_bytes: Int
    """The dynamic shared memory per thread block, in bytes."""
    var attributes: List[LaunchAttribute]
    """The launch attributes."""

    def __init__(
        out self,
        *,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: Dim,
        shared_mem_bytes: Int = 0,
        var attributes: List[LaunchAttribute] = [],
    ):
        """Creates a cluster launch configuration with launch attributes.

        Args:
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory per thread block, in
                bytes.
            attributes: The launch attributes, for example the result of
                `pdl_launch_attributes()`.
        """
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.cluster_dim = cluster_dim
        self.shared_mem_bytes = shared_mem_bytes
        self.attributes = attributes^

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

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Returns the launch attributes.

        Parameters:
            o: The origin of `self`.

        Returns:
            The launch attributes.
        """
        return rebind[Span[LaunchAttribute, o]](
            Span[LaunchAttribute, _](self.attributes)
        )
