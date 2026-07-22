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
"""Static Device Information.

Provides types and helpers for describing individual computational
devices to be used within the Mojo metaprogramm.
"""

from std.gpu.host.info import GPUInfo, _get_empty_target
from std.sys.info import _TargetType


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
        out self, var spec: DeviceSpec, device_index: Int, *, private: ()
    ):
        """
        This exists only as support code for enabling machine definition
        prior to rich target info arguments to compiler.

        IMPORTANT: This constructor should never be used by non-stdlib code.
        """

        # see DRIV-208, avoiding this requires CPU devices
        if spec.info:
            self.driver = spec.info.unsafe_value().api
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

    def _mlir_target(self) -> _TargetType:
        """Returns the kgen target for this spec, to be used when setting up
        offload compilation for the device."""
        if self.info:
            return self.info.unsafe_value().target()
        else:
            return _get_empty_target()
