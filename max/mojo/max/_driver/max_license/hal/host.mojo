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


struct Host(not Copyable, Movable):
    var drivers: List[ArcPointer[Driver]]

    def __init__(out self) raises HALError:
        var plugins_spec = getenv("MODULAR_DRIVER_PLUGINS")
        if not plugins_spec:
            raise HALError(
                STATUS_INVALID_ARG, message="MODULAR_DRIVER_PLUGINS not set"
            )

        self.drivers = []
        var plugin_specs = plugins_spec.strip().split(":")
        if len(plugin_specs) == 0:
            raise HALError(
                STATUS_INVALID_ARG, message="No plugin specs provided"
            )
        # TODO: support multiple drivers
        var driver = Driver.create(String(plugin_specs[0]))
        self.drivers.append(driver^)

    def driver_count(self) -> Int64:
        return Int64(len(self.drivers))

    def driver(self, id: Int64) raises HALError -> ArcPointer[Driver]:
        if id < 0 or id >= self.driver_count():
            raise HALError(STATUS_INVALID_ARG, message="Invalid driver ID")
        return self.drivers[id]

    def accelerator_count(self) raises HALError -> Int64:
        return Int64(self.driver(0)[].get_device_count())

    def accelerator(
        self, id: Int64
    ) raises HALError -> ArcPointer[
        Device[get_device_spec[_first_accelerator().value()]()]
    ]:
        if id < 0 or id >= self.accelerator_count():
            raise HALError(STATUS_INVALID_ARG, message="Invalid accelerator ID")
        return self.driver(0)[].get_device_dynamic(id)
