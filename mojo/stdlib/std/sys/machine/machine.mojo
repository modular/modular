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
"""Static Machine Information.

Provides types and helpers for describing a machine,
including its compositional view ("set of devices"),
control plane view (NUMA domains, device network
interconnects, etc), and associated data and metadata.
"""

from .device import DeviceRef
from std.sys.info import _accelerator_arch
from std.gpu.host.info import GPUInfo


struct MachineDefinition:
    """
    Enumerates the computational capabilities of the
    machine that is being compiled for and that the
    application is being run on.

    Fields within this should be considered const,
    with MachineDefinition being a _static_ description
    of the machine as it exists.
    """

    var _devices: List[DeviceRef]

    def devices(self) -> ref[self._devices] List[DeviceRef]:
        """Get the set of devices available to the metaprogram
        in compiling for this machine.

        Returns:
            A reference to the list of devices within this machine.
        """
        return self._devices

    @doc_hidden
    def __init__(out self, *, private: ()):
        """
        This constructor for MachineDefinition is
        introduced as a bridge construct that allows
        for temporarily reusing the target configuration
        flags of the Mojo compiler.

        IMPORTANT: This constructor should never be used by non-stdlib code.
        """
        comptime arch = _accelerator_arch()
        comptime if arch != "":
            comptime info = GPUInfo.from_name[arch]()
            comptime spec = DeviceSpec(arch, info)
            comptime device_ref = DeviceRef(spec, 0, private=())

            self._devices = [materialize[device_ref]()]
        else:
            comptime spec = DeviceSpec(arch, None)
            comptime device_ref = DeviceRef(spec, 0, private=())
            self._devices = [materialize[device_ref]()]
