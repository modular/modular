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
"""Execution config placeholder for Apple M3-family GPUs (M3, M3 Pro, M3 Max)."""

from std.gpu.host.constant_memory_mapping import ConstantMemoryMapping
from std.gpu.host.dim import Dim
from . import (
    _BaseMetalExecutionConfiguration,
    _BaseMetalExecutionConfig,
    _MetalExecutionConfigDelegator,
)
from .. import (
    ExecutionConfig,
    BlockExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
)


struct M3ExecutionConfiguration(
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    _BaseMetalExecutionConfig,
    _MetalExecutionConfigDelegator,
):
    """Describes the execution configuration of a GPU kernel launch on Apple
    M3-family GPUs."""

    var _base: _BaseMetalExecutionConfiguration

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self._base = _BaseMetalExecutionConfiguration(
            block_dim=block_dim,
            grid_dim=grid_dim,
            shared_mem_bytes=shared_mem_bytes,
            constant_memory=constant_memory^,
        )

    def _get_inner_config[
        o: Origin
    ](ref[o] self) -> ref[o] _BaseMetalExecutionConfiguration:
        """Gets the inner `_BaseMetalExecutionConfiguration` to which this delegator forwards
        all calls.

        Returns:
            The inner `_BaseMetalExecutionConfiguration`.
        """
        return Pointer(to=self._base).unsafe_origin_cast[o]()[]
