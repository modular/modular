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
"""Execution config placeholder for AMD GCN 4th generation (Polaris) GPUs:
Polaris10/Polaris11 (gfx803, e.g. Radeon RX 470, RX 480, RX 570, RX 580)."""

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
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
)


# Polaris10/Polaris11 (gfx803) share their LLVM compiler target with GCN3
# Fiji (see gcn3.mojo); the two are split here because they are distinct
# product families, not distinct compiler targets.
struct GCN4ExecutionConfiguration(
    BlockExecutionConfig,
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
    GCN 4th generation GPUs."""

    var _base: _BaseAmdExecutionConfiguration

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

    def _get_inner_config[
        o: Origin
    ](ref[o] self) -> ref[o] _BaseAmdExecutionConfiguration:
        """Gets the inner `_BaseAmdExecutionConfiguration` to which this delegator forwards
        all calls.

        Returns:
            The inner `_BaseAmdExecutionConfiguration`.
        """
        return Pointer(to=self._base).unsafe_origin_cast[o]()[]
