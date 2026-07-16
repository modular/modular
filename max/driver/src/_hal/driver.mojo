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
"""HAL Driver — entry point for interacting with hardware via a plugin."""

from .plugin import RawDriver
from .device import Device, get_device_spec, get_machine_definition
from .status import HALError
from std.memory import ArcPointer
from std.memory.arc_pointer import WeakPointer


def _first_accelerator() -> Optional[Int64]:
    var machine = get_machine_definition()
    for i in range(len(machine.devices())):
        ref device = machine.devices()[i]
        if device.spec.info:
            # Currently, only accelerators have corresponding GPUInfo.
            # Once this turns into broader DeviceInfo, this should
            # instead check if `info.is_accelerator()`
            return Optional(Int64(i))

    return Optional[Int64]()


@fieldwise_init
struct Driver(ImplicitlyDeletable, Movable):
    """Top-level driver that owns a loaded plugin and the driver handle.

    Lifecycle: create via `Driver.create(plugin_spec)`, then call methods.
    The driver handle is destroyed when the Driver is destroyed.
    """

    var _raw: ArcPointer[RawDriver]
    var _self_ref: WeakPointer[Self]
    var _device_count: Int64

    @staticmethod
    def create(plugin_spec: String) raises HALError -> ArcPointer[Self]:
        """Create a Driver by loading a plugin and initialising the backend.

        Args:
            plugin_spec: 'name@/path/to/plugin.so' or just the path.
        """
        var raw = RawDriver.load(plugin_spec)
        # If `get_device_count` raises, `raw`'s destructor cleans up the
        # loaded plugin and initialised driver handle.
        var device_count = raw.get_device_count()
        var arc = ArcPointer[Self](
            Driver(
                _raw=ArcPointer(raw^),
                _device_count=device_count,
                _self_ref=WeakPointer[Driver](),
            )
        )
        arc[]._self_ref = WeakPointer(downgrade=arc)
        return arc

    # ===-------------------------------------------------------------------===#
    # Queries
    # ===-------------------------------------------------------------------===#

    def get_name(self) -> String:
        return self._raw[]._raw.name

    def get_device_count(self) -> Int64:
        return self._device_count

    def get_device[
        id: Int64
    ](self) raises HALError -> ArcPointer[Device[get_device_spec[id]()]]:
        """Retrieve a device by ID."""
        return rebind[ArcPointer[Device[get_device_spec[id]()]]](
            self.get_device_dynamic[id](id)
        )

    def get_device_dynamic[
        StaticID: Optional[Int64] = _first_accelerator()
    ](self, dynamic_id: Int64) raises HALError -> ArcPointer[
        Device[get_device_spec[StaticID.value()]()]
    ]:
        """
        Allows type-erased contexts to retrieve devices by a dynamic ID by assuming
        their architecture is equal to (or dynamically compatible with) the ID in `StaticID`.

        Parameters:
            StaticID: The ID to assume for the device at compile time. Must be `Some`, `Optional` allows for inferring ID when not provided.

        Arguments:
            dynamic_id: The ID to assume for runtime execution, regardless of the value of StaticID.
        """

        comptime assert (
            StaticID
        ), "No accelerator could be found for the given StaticID"

        return Device[get_device_spec[StaticID.value()]()]._create(
            self, dynamic_id
        )
