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
"""HAL Device — a device retrieved from a Driver."""

from . import Context, Driver
from .plugin import (
    RawDriver,
    OutParam,
    DeviceHandle,
    M_driver_dlpack_device,
)
from ._plugin._overlay import HAL_PLUGINS
from .extensions.traits import SupportsPeerAccessV1
from .status import STATUS_SUCCESS, STATUS_INVALID_ARG, HALError

from std.collections import Array
from std.ffi import c_char

from std.memory import (
    ArcPointer,
    MaybeUninit,
)
from std.memory.arc_pointer import WeakPointer

from ._machine import MachineDefinition, DeviceSpec

# Buffer size for string-valued device properties; must match
# M_DRIVER_DEVICE_PROPERTY_MAX_LEN in SDK/include/HAL/M_driver_device.h.
comptime _DEVICE_PROPERTY_MAX_LEN = 256


def get_machine_definition() -> MachineDefinition:
    comptime shared_def = MachineDefinition(private=())
    return materialize[shared_def]()


# TODO(Sawyer: Static Device Info, DRIV-4):
# Does not currently handle multiple accelerator architectures, but stubbed
# in anyway. This should pull from machine topo and actually use the `device_id`.
@doc_hidden
def get_device_spec[
    device_id: Int64,
    machine: MachineDefinition = get_machine_definition(),
]() -> DeviceSpec:
    # There isn't currently a way of telling the Mojo compiler
    # that multiple devices exist on the target machine.
    # To enable building out the stdlib constructs,
    # we map that down to assuming the device has an unbounded
    # number of accelerators with arch matching the value passed
    # to --target-accelerator.
    comptime if device_id < 0:
        comptime assert False, "Negative device indices are not supported"
    elif len(machine.devices()) == 0:
        comptime assert False, "Can't get device spec if no devices exist"
    else:
        # clamp the max idx so that for the default constructed
        # case we get a spec for machines with N devices (now)
        comptime idx = min(len(machine.devices()) - 1, Int(device_id))
        return materialize[machine.devices()[idx].spec]()


@fieldwise_init
struct Device[spec: DeviceSpec](Deinitable, Movable):
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

    comptime Extensions = HAL_PLUGINS.for_target[
        DeviceSpec._mlir_target[Self.spec]()
    ].HookFor[DeviceSpec.target[Self.spec]()].extensions

    # SAFETY: this field is immutably shared across the HAL
    # object graph, and must not be mutated without
    # universal synchronization over an internally mutable
    # member or member set.
    var _extensions: Self.Extensions

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

        self._extensions = Self.Extensions(
            self._raw[]._raw._lib.borrow(), self._raw[]._raw.so_path
        )

        var device_handle = MaybeUninit[DeviceHandle]()
        var status = raw._raw.base.device_get.get()(
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

        self._handle = device_handle.unsafe_assume_init()

    def extensions(
        imm self,
    ) -> ref[self._extensions] Self.Extensions:
        """Returns a ref to the HAL extension carrier object that
        lives on this `Device`.

        Returns:
            An immutable ref to the extension carrier.
        """
        return self._extensions

    def get_context(
        self,
    ) raises HALError -> ArcPointer[Context[Self.spec]]:
        return Context[Self.spec]._create(self)

    def get_attribute(self, attribute: Int32) raises HALError -> Int32:
        """Queries a numeric device attribute by its CUDA-style ID."""
        return self._raw[].get_device_attribute(self._handle, attribute)

    def can_access_peer(
        self, peer: Device[Self.spec]
    ) raises HALError -> Bool where conforms_to(
        Self.Extensions, SupportsPeerAccessV1
    ):
        """Returns whether this device can access `peer` once enabled."""
        var peer_access = self._extensions.peer_access()
        if not peer_access:
            return False
        return peer_access.value().can_access_peer(
            self._raw[], self._handle, peer._handle
        )

    def get_dlpack_device(
        self, pinned: Bool
    ) raises HALError -> M_driver_dlpack_device:
        """Returns the DLPack `(device_type, device_id)` for this device."""
        if pinned:
            return self._raw[].get_device_property[
                "dlpack_device_pinned", M_driver_dlpack_device
            ](self._handle)
        return self._raw[].get_device_property[
            "dlpack_device", M_driver_dlpack_device
        ](self._handle)

    def is_host_unified(self) raises HALError -> Bool:
        """Whether this device and the host draw from one physical memory pool.
        """
        return (
            self._raw[].get_device_property["host_unified", Int64](self._handle)
            != 0
        )

    def get_name(self) raises HALError -> String:
        """Queries the device's human-readable name."""
        var buf = Array[Int8, _DEVICE_PROPERTY_MAX_LEN](fill=0)
        var buf_ptr: Pointer[Int8, origin_of(buf)] = buf.unsafe_ptr()
        self._raw[].get_device_property_string["name"](self._handle, buf_ptr)
        return String(unsafe_from_utf8_ptr=buf_ptr.unsafe_bitcast[c_char]())

    def get_arch(self) raises HALError -> String:
        """Queries the device's compile-target architecture name.

        For AMD GPUs this is the `gfx*` name (e.g. `gfx942`); vendors whose arch
        is derived host-side (e.g. CUDA `sm_<cc>`) need not implement this
        plugin property.
        """
        var buf = Array[Int8, _DEVICE_PROPERTY_MAX_LEN](fill=0)
        var buf_ptr: Pointer[Int8, origin_of(buf)] = buf.unsafe_ptr()
        self._raw[].get_device_property_string["arch"](self._handle, buf_ptr)
        return String(unsafe_from_utf8_ptr=buf_ptr.unsafe_bitcast[c_char]())
