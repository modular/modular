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
"""
HAL Host — the collection of all devices - CPUs and accelerators - on the
local motherboard.
"""
from std.os import getenv

from .device import Device
from .driver import Driver
from std.memory import ArcPointer
from .status import HALError, STATUS_INVALID_ARG
from .driver import _first_accelerator, get_device_spec


def _accelerator_plugin_entry(plugins_spec: String) -> Optional[String]:
    """Returns the first non-`cpu` entry of a `MODULAR_DRIVER_PLUGINS` value.

    An entry with no `name@` prefix counts as an accelerator.

    Args:
        plugins_spec: The `MODULAR_DRIVER_PLUGINS` value.

    Returns:
        The accelerator entry, or `None` if the value names no accelerator.
    """
    for entry in plugins_spec.split(";"):
        var spec = String(entry.strip())
        if not spec:
            continue
        var parts = spec.split("@", maxsplit=1)
        if len(parts) == 1 or String(parts[0]).lower() != "cpu":
            return spec
    return None


struct Host(not Copyable, Movable):
    var drivers: List[ArcPointer[Driver]]

    def __init__(out self) raises HALError:
        var plugins_spec = getenv("MODULAR_DRIVER_PLUGINS")
        if not plugins_spec:
            raise HALError(
                STATUS_INVALID_ARG, message="MODULAR_DRIVER_PLUGINS not set"
            )

        self.drivers = []
        __guard case .Some(accelerator_spec) = _accelerator_plugin_entry(
            plugins_spec
        ) else:
            raise HALError(
                STATUS_INVALID_ARG,
                message="No accelerator plugin in MODULAR_DRIVER_PLUGINS",
            )
        # TODO: support multiple drivers
        var driver = Driver.create(accelerator_spec)
        self.drivers.append(driver^)

    def driver_count(self) -> Int64:
        return Int64(len(self.drivers))

    def driver(self, id: Int64) raises HALError -> ArcPointer[Driver]:
        if id < 0 or id >= self.driver_count():
            raise HALError(STATUS_INVALID_ARG, message="Invalid driver ID")
        return self.drivers[id]

    def accelerator_count(self) raises HALError -> Int64:
        return Int64(self.driver(0)[].device_count())

    def accelerator(
        self, id: Int64
    ) raises HALError -> ArcPointer[
        Device[get_device_spec[_first_accelerator().value()]()]
    ]:
        if id < 0 or id >= self.accelerator_count():
            raise HALError(STATUS_INVALID_ARG, message="Invalid accelerator ID")
        return self.driver(0)[].get_device_dynamic(id)
