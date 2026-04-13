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

from .plugin import (
    Plugin,
    OutParam,
    DriverHandle,
    DeviceHandle,
    DriverVersion,
    M_DRIVER_INTERFACE_VERSION_MAJOR,
    M_DRIVER_INTERFACE_VERSION_MINOR,
    M_DRIVER_INTERFACE_VERSION_PATCH,
)
from .device import Device
from .status import STATUS_SUCCESS, STATUS_INVALID_ARG, HALError
from std.memory import (
    ArcPointer,
    ImmutPointer,
    MutPointer,
    UnsafePointer,
    UnsafeMaybeUninit,
)


@fieldwise_init
struct Driver(Movable):
    """Top-level driver that owns a loaded plugin and the driver handle.

    Lifecycle: create via `Driver.create(plugin_spec)`, then call methods.
    The driver handle is destroyed when the Driver is destroyed.
    """

    var _plugin: Plugin
    var _handle: DriverHandle
    var _device_count: Int64

    @staticmethod
    def create(plugin_spec: String) raises HALError -> Self:
        """Create a Driver by loading a plugin and initialising the backend.

        Args:
            plugin_spec: 'name@/path/to/plugin.so' or just the path.
        """
        var plugin = Plugin.load(plugin_spec)

        var version = DriverVersion(
            major=M_DRIVER_INTERFACE_VERSION_MAJOR,
            minor=M_DRIVER_INTERFACE_VERSION_MINOR,
            patch=M_DRIVER_INTERFACE_VERSION_PATCH,
        )
        var handle = UnsafeMaybeUninit(DriverHandle(_unsafe_null=()))

        var status = plugin.create.f(
            ImmutPointer(
                to=UnsafePointer[DriverVersion, ImmutAnyOrigin](
                    UnsafePointer(to=version)
                )[]
            ),
            OutParam[DriverHandle](to=handle),
        )
        if status != STATUS_SUCCESS:
            raise HALError(
                status,
                message=String(
                    t"Failed to initialise driver plugin from {plugin.so_path}"
                ),
            )

        var driver_handle = handle.unsafe_assume_init_ref()

        var num_devices = UnsafeMaybeUninit(Int64(0))
        status = plugin.device_count.f(
            driver_handle, OutParam[Int64](to=num_devices)
        )
        if status != STATUS_SUCCESS:
            _ = plugin.destroy.f(driver_handle)
            var err = plugin.get_status_message(driver_handle, status)
            raise HALError(
                err.status,
                message=String(t"Failed to get device count: {err.message}"),
            )

        return Driver(
            _plugin=plugin^,
            _handle=driver_handle,
            _device_count=num_devices.unsafe_assume_init_ref(),
        )

    def __del__(deinit self):
        # Move `_plugin` into a local so its `OwnedDLHandle` stays alive
        # until after the `destroy` call returns — ASAP destruction would
        # otherwise `dlclose` the `.so` (unmapping the code page) before
        # we call through the function pointer.
        var plugin = self._plugin^
        _ = plugin.destroy.f(self._handle)
        _ = plugin^

    # ===-------------------------------------------------------------------===#
    # Queries
    # ===-------------------------------------------------------------------===#

    def get_name(self) -> String:
        return self._plugin.name

    def get_device_count(self) -> Int64:
        return self._device_count

    def get_device(
        mut self, id: Int64
    ) raises HALError -> Device[origin_of(self)]:
        # """Retrieve a device by ID."""
        return Device(self, id)
