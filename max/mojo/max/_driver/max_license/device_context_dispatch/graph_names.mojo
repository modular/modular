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
"""The device graph names that `max.gpu.host` exports.

Both backends use the legacy device graph types. On the HAL backend, which has
no device graphs, the legacy `DeviceGraph` entry points raise instead of
failing to compile.
"""

from max.gpu.host._legacy.device_graph import (
    LegacyDeviceGraph,
    LegacyDeviceGraphBuilder,
    LegacyDeviceGraphCache,
    LegacyDeviceGraphInput,
    LegacyDeviceGraphInputKind,
    LegacyDeviceGraphNode,
    LegacyStableAddr,
)

comptime DeviceGraph = LegacyDeviceGraph
"""A captured sequence of device operations that can be replayed."""

comptime DeviceGraphBuilder[arena_origin: ImmOrigin] = LegacyDeviceGraphBuilder[
    arena_origin
]
"""Records operations into a device graph."""

comptime DeviceGraphCache = LegacyDeviceGraphCache
"""Reuses instantiated graphs across calls."""

comptime DeviceGraphInput = LegacyDeviceGraphInput
"""A device graph input that contributes to the graph's cache key."""

comptime DeviceGraphInputKind = LegacyDeviceGraphInputKind
"""How a graph input is stored and keyed."""

comptime DeviceGraphNode[arena_origin: ImmOrigin] = LegacyDeviceGraphNode[
    arena_origin
]
"""A node of a device graph under construction."""

comptime StableAddr = LegacyStableAddr
"""A graph-lifetime slot that holds one device address."""
