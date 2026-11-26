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
"""The HAL launch path shared by `DeviceContext` and `DeviceStream`.

A compiled launch goes through the plugin's `Launcher`, which checks its dims;
an external launch goes to the enqueuer's stream directly and checks them here.
"""

from std.builtin.device_passable import DevicePassable
from std.collections.optional import OptionalReg
from std.reflection import SourceLocation
from std.sys.info import CompilationTarget

from max._driver.max_license.device_context_dispatch.device_function import (
    DeviceExternalFunction,
    DeviceFunction,
    _DispatchEnqueuer,
)
from max._driver.max_license.device_context_dispatch.device_context import (
    use_legacy_backend,
)
from max.gpu.host._driver_common import _check_dim_in_range
from max.gpu.host._legacy.constant_memory_mapping import ConstantMemoryMapping
from max.gpu.host._legacy.dim import Dim
from max.gpu.host._legacy.launch_attribute import LaunchAttribute
from max.gpu.host.launch_config import (
    ExecutionConfig,
    GridBlockExecutionConfig,
)


@inline(.always)
def _check_grid_block_dims[
    func_name: StringLiteral
](grid_dim: Dim, block_dim: Dim, location: SourceLocation) raises:
    _check_dim_in_range[func_name, "grid_dim"](grid_dim, location=location)
    _check_dim_in_range[func_name, "block_dim"](block_dim, location=location)


@inline(.always)
def _hal_enqueue_compiled[
    E: _DispatchEnqueuer,
    //,
    target: CompilationTarget,
    func_name: StringLiteral,
    *Ts: DevicePassable,
](
    enqueuer: E,
    f: DeviceFunction,
    *args: *Ts,
    grid_dim: Dim,
    block_dim: Dim,
    cluster_dim: OptionalReg[Dim],
    shared_mem_bytes: OptionalReg[Int],
    var attributes: List[LaunchAttribute],
    var constant_memory: List[ConstantMemoryMapping],
    location: SourceLocation,
) raises where not use_legacy_backend():
    """Launches a compiled function with grid/block dims on `enqueuer`."""
    f._call_with_pack_checked(
        enqueuer,
        *args,
        grid_dim=grid_dim,
        block_dim=block_dim,
        cluster_dim=cluster_dim,
        shared_mem_bytes=shared_mem_bytes,
        attributes=attributes^,
        constant_memory=constant_memory^,
        location=location,
    )


@inline(.always)
def _hal_enqueue_compiled[
    E: _DispatchEnqueuer,
    ExecutionConfigType: ExecutionConfig,
    //,
    target: CompilationTarget,
    func_name: StringLiteral,
    *Ts: DevicePassable,
](
    enqueuer: E,
    execution_config: ExecutionConfigType,
    f: DeviceFunction,
    *args: *Ts,
    location: SourceLocation,
) raises where not use_legacy_backend():
    """Launches a compiled function with an execution config on `enqueuer`."""
    f._do_call_with_pack_checked(
        enqueuer._hal_stream(),
        execution_config,
        *args,
        location=location,
    )


@inline(.always)
def _hal_enqueue_external[
    E: _DispatchEnqueuer,
    //,
    target: CompilationTarget,
    func_name: StringLiteral,
    *Ts: AnyType,
](
    enqueuer: E,
    f: DeviceExternalFunction,
    *args: *Ts,
    grid_dim: Dim,
    block_dim: Dim,
    cluster_dim: OptionalReg[Dim],
    shared_mem_bytes: OptionalReg[Int],
    var attributes: List[LaunchAttribute],
    var constant_memory: List[ConstantMemoryMapping],
    location: SourceLocation,
) raises where not use_legacy_backend():
    """Launches an external function with grid/block dims on `enqueuer`.

    An external launch goes to the stream directly, never to the plugin's
    launcher, so its dims are always checked.
    """
    _check_grid_block_dims[func_name](grid_dim, block_dim, location)
    f._call_with_pack(
        enqueuer._hal_stream(),
        *args,
        grid_dim=grid_dim,
        block_dim=block_dim,
        cluster_dim=cluster_dim,
        shared_mem_bytes=shared_mem_bytes,
        attributes=attributes^,
        constant_memory=constant_memory^,
        location=location,
    )


@inline(.always)
def _hal_enqueue_external[
    E: _DispatchEnqueuer,
    ExecutionConfigType: GridBlockExecutionConfig,
    //,
    target: CompilationTarget,
    func_name: StringLiteral,
    *Ts: AnyType,
](
    enqueuer: E,
    execution_config: ExecutionConfigType,
    f: DeviceExternalFunction,
    *args: *Ts,
    location: SourceLocation,
) raises where not use_legacy_backend():
    """Launches an external function with an execution config on `enqueuer`,
    as the grid/block overload does."""
    _check_grid_block_dims[func_name](
        execution_config.get_grid_dim(),
        execution_config.get_block_dim(),
        location,
    )
    f._call_with_pack(
        enqueuer._hal_stream(),
        execution_config,
        *args,
        location=location,
    )
