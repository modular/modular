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
"""The device graph names that `max.gpu.host` exports, selected by backend.

Only the legacy backend implements device graphs. The HAL backend has stubs
that let the graph-compiler primitives compile, so under the HAL backend these
names are the stubs.
"""

from max._driver.max_license.device_context_hal.device_graph_hal import (
    HALDeviceGraph,
    HALDeviceGraphBuilder,
    HALDeviceGraphCache,
    HALDeviceGraphInputKind,
    HALDeviceGraphNode,
    HALStableAddr,
)
from max.gpu.host._legacy.device_graph import (
    LegacyDeviceGraph,
    LegacyDeviceGraphBuilder,
    LegacyDeviceGraphCache,
    LegacyDeviceGraphInputKind,
    LegacyDeviceGraphNode,
    LegacyStableAddr,
)
from max.gpu.host._overlay import HAL_DEVICE_CONTEXT

comptime DeviceGraph = (
    LegacyDeviceGraph if not HAL_DEVICE_CONTEXT else HALDeviceGraph
)
"""A captured sequence of device operations that can be replayed."""

comptime DeviceGraphBuilder[arena_origin: ImmOrigin] = (
    LegacyDeviceGraphBuilder[
        arena_origin
    ] if not HAL_DEVICE_CONTEXT else HALDeviceGraphBuilder
)
"""Records operations into a device graph."""

comptime DeviceGraphCache = (
    LegacyDeviceGraphCache if not HAL_DEVICE_CONTEXT else HALDeviceGraphCache
)
"""Reuses instantiated graphs across calls."""

comptime DeviceGraphInputKind = (
    LegacyDeviceGraphInputKind if not HAL_DEVICE_CONTEXT else HALDeviceGraphInputKind
)
"""How a graph input is stored and keyed."""

comptime DeviceGraphNode[arena_origin: ImmOrigin] = (
    LegacyDeviceGraphNode[
        arena_origin
    ] if not HAL_DEVICE_CONTEXT else HALDeviceGraphNode
)
"""A node of a device graph under construction."""

comptime StableAddr = LegacyStableAddr if not HAL_DEVICE_CONTEXT else HALStableAddr
"""A graph-lifetime slot that holds one device address."""
