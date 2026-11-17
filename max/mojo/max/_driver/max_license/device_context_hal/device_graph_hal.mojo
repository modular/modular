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

"""HAL stand-in for the device-graph entry point.

The HAL DeviceContext has no device-graph support: graph capture fails in the
C++ runtime shim before any of these values is ever created. `DeviceGraph.create`
and `replay` exist so that code that builds and replays a graph compiles under
the HAL and raises at the call. Every other device-graph name is the legacy type
on both backends.
"""

from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max.gpu.host._driver_handles import _DeviceGraphPtr


struct HALDeviceGraph(ImplicitlyCopyable):
    """Unsupported: the HAL DeviceContext has no device-graph support."""

    var _handle: _DeviceGraphPtr[mut=True]

    @staticmethod
    def create(
        ctx: DeviceContext, build: Some[AnyType]
    ) raises -> HALDeviceGraph:
        """Unsupported: builds and instantiates a device graph.

        Args:
            ctx: Device context for the target device.
            build: Callback that would add nodes to the graph builder.

        Returns:
            Never returns; the HAL `DeviceContext` cannot build device graphs.

        Raises:
            Always, since the HAL backend has no device-graph support.
        """
        raise Error("device graphs are unsupported by the HAL DeviceContext")

    def replay(self) raises:
        """Unsupported: launches the device graph.

        Raises:
            Always, since the HAL backend has no device-graph support.
        """
        raise Error("device graphs are unsupported by the HAL DeviceContext")

    def take_handle(deinit self) -> _DeviceGraphPtr[mut=True]:
        """Surrenders the owning handle net-zero, suppressing the destructor.

        Returns:
            The owning `DeviceGraph*`; the caller must hand it to a runtime owner
            that adopts it without an extra reference.
        """
        return self._handle
