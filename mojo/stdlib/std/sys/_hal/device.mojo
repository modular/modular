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
"""HAL Device — a device retrieved from a Driver."""

from .plugin import (
    Plugin,
    OutParam,
    DeviceHandle,
)
from .status import STATUS_SUCCESS, STATUS_INVALID_ARG, HALError

from std.memory import (
    MutPointer,
    ArcPointer,
    UnsafeMaybeUninit,
)


@fieldwise_init
struct Device[driver_origin: MutOrigin](Movable):
    """A device retrieved from a Driver.

    Does not own the device handle — the plugin manages device lifetime
    internally. The Device is only valid while the parent Driver is alive.

    Parameters:
        driver_origin: The origin of the parent Driver pointer.
    """

    var _handle: DeviceHandle
    var id: Int64
    var driver: MutPointer[Driver, Self.driver_origin]

    def __init__[
        o1: MutOrigin
    ](out self: Device[o1], ref[o1] driver: Driver, id: Int64) raises HALError:
        self.driver = MutPointer(to=driver)

        if id < 0 or id >= driver._device_count:
            raise HALError(
                STATUS_INVALID_ARG,
                message="Invalid device ID "
                + String(id)
                + " — range is [0, "
                + String(driver._device_count)
                + ")",
            )

        var device_handle = UnsafeMaybeUninit[DeviceHandle]()
        var status = driver._plugin.device_get.f(
            driver._handle, id, OutParam[DeviceHandle](to=device_handle)
        )
        if status != STATUS_SUCCESS:
            var err = driver._plugin.get_status_message(status)
            raise HALError(
                err.status,
                message="Failed to get device "
                + String(id)
                + ": "
                + err.message,
            )

        self.id = id
        self._handle = device_handle.unsafe_assume_init_ref()
