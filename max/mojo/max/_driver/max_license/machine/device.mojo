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
"""Static Device Information.

Provides types and helpers for describing individual computational
devices to be used within the Mojo metaprogramm.
"""

from std.sys.info import _TargetType, CompilationTarget

from max.gpu.host.info import GPUInfo, TargetAccelerator

from max._plugin._overlay import TARGETS


struct DeviceRef(Copyable, Movable):
    """Uniquely identifies a specific compute device within
    this system."""

    var driver: StaticString
    """The unique identifier for the driver that provides access to
    this device."""

    var driver_device_index: Int
    """The index of this device within the list of enumerated devices
    in the context of the providing driver. This will never be negative."""

    var spec: DeviceSpec
    """The device specification this device conforms to."""

    @doc_hidden
    def __init__(
        out self, var spec: DeviceSpec, device_index: Int, *, private: Tuple[]
    ):
        """
        This exists only as support code for enabling machine definition
        prior to rich target info arguments to compiler.

        IMPORTANT: This constructor should never be used by non-stdlib code.
        """

        # see DRIV-208, avoiding this requires CPU devices
        if case .Some(info) = spec.info:
            self.driver = info.api
        else:
            self.driver = "CPU"
        self.spec = spec^
        self.driver_device_index = device_index


@fieldwise_init
struct DeviceSpec(
    Copyable,
    Movable,
):
    """
    A DeviceSpec represents a single model/sku of device.
    It enumerates identifying information and provides sufficiently
    detailed information to compile for a device and to set up a
    control plane for it at runtime.

    Fields within this should be considered const,
    with DeviceSpec being a _static_ description
    of the device as it exists.
    """

    var name: StaticString
    """The unique name of this specific model of device, as enumerated by the driver."""

    var info: Optional[GPUInfo]
    """
    Extended device information that has not yet been inlined into DeviceSpec.
    This is not expected to be optional "forever", but is necessary to allow the Python
    HAL bindings to not require a non-CPU-device be connected to the system to import them.
    See DRIV-208.
    """

    @staticmethod
    def target[
        comptime_self: Self
    ]() -> CompilationTarget[
        _mlir_value=DeviceSpec._mlir_target[comptime_self]()
    ]:
        """Returns the compilation target for this spec, to be used when setting
        up offload compilation for the device.

        Parameters:
            comptime_self: The `DeviceSpec` instance, available at compile time.

        Returns:
            The `CompilationTarget` instance for this device.
        """
        return {}

    @staticmethod
    def _mlir_target[comptime_self: Self]() -> _TargetType:
        comptime if case .Some(info) = comptime_self.info:
            return comptime (
                TargetAccelerator.from_gpu_name[info.name, TARGETS].mlir_target
            )
        else:
            return CompilationTarget._unknown()._mlir_value
