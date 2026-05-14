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
    RawDriver,
    OutParam,
    DeviceHandle,
)
from .status import STATUS_SUCCESS, STATUS_INVALID_ARG, HALError

from std.memory import (
    ArcPointer,
    UnsafeMaybeUninit,
)
from std.memory.arc_pointer import WeakPointer

from std.gpu.host.compile import get_gpu_target


@fieldwise_init
struct DeviceSpec[target: CompilationTarget[value=get_gpu_target()]](
    Movable, TrivialRegisterPassable
):
    pass


# TODO(Sawyer: Static Device Info, DRIV-4):
# Does not currently handle multiple accelerator architectures, but stubbed
# in anyway. This should pull from machine topo and actually use the `device_id`.
@doc_hidden
def get_device_spec[
    _device_id: Int64
]() -> DeviceSpec[target=CompilationTarget[value=get_gpu_target()]()]:
    return DeviceSpec[target=CompilationTarget[value=get_gpu_target()]()]()


@fieldwise_init
struct Device[spec: DeviceSpec](ImplicitlyDestructible, Movable):
    """A device retrieved from a Driver.

    Does not own the device handle — the plugin manages device lifetime
    internally. The Device is only valid while the parent Driver is alive.

    Parameters:
        spec: The DeviceSpec that describes the architecture and other characteristics
                of this device.
    """

    var _handle: DeviceHandle
    var _driver: ArcPointer[Driver]
    var _raw: ArcPointer[RawDriver]
    var _self_ref: WeakPointer[Self]
    var id: Int64

    @staticmethod
    def _create(
        out _self: ArcPointer[Self], driver: Driver, id: Int64
    ) raises HALError:
        _self = ArcPointer(Self(driver, id))
        _self[]._self_ref = WeakPointer(downgrade=_self)

    @doc_hidden
    def __init__(
        out self: Device[Self.spec], driver: Driver, id: Int64
    ) raises HALError:
        self.id = id

        self._raw = driver._raw
        self._driver = driver._self_ref.try_upgrade().value()
        self._self_ref = WeakPointer[Self]()

        ref raw = self._raw[]

        if self.id < 0 or self.id >= driver._device_count:
            raise HALError(
                STATUS_INVALID_ARG,
                message=String(
                    t"Invalid device ID {self.id} — range is [0,"
                    t" {driver._device_count})"
                ),
            )

        var device_handle = UnsafeMaybeUninit[DeviceHandle]()
        var status = raw._raw.device_get.f(
            raw._driver_handle,
            self.id,
            OutParam[DeviceHandle](to=device_handle),
        )
        if status != STATUS_SUCCESS:
            var err = raw.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get device {self.id}: {err.message}"
                ),
            )

        self._handle = device_handle.unsafe_assume_init_ref()

    def get_context(
        self,
    ) raises HALError -> ArcPointer[Context[Self.spec]]:
        return Context[Self.spec]._create(self)
