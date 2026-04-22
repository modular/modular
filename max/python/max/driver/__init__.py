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

from max._core import __version__
from max._core.driver import (
    DeviceEvent,
    enable_all_peer_access,
    get_virtual_device_api,
    get_virtual_device_count,
    get_virtual_device_target_arch,
    is_virtual_device_mode,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max._core_types.driver import DLPackArray

from .buffer import Buffer, DevicePinnedBuffer, load_max_buffer
from .driver import (
    CPU,
    Accelerator,
    Device,
    DeviceSpec,
    DeviceStream,
    accelerator_api,
    accelerator_architecture_name,
    accelerator_count,
    calculate_virtual_device_count,
    calculate_virtual_device_count_from_cli,
    devices_exist,
    load_devices,
    scan_available_devices,
)

del driver  # type: ignore
del buffer  # type: ignore

__all__ = [
    "CPU",
    "Accelerator",
    "Buffer",
    "DLPackArray",
    "Device",
    "DeviceEvent",
    "DevicePinnedBuffer",
    "DeviceSpec",
    "DeviceStream",
    "accelerator_api",
    "accelerator_architecture_name",
    "accelerator_count",
    "calculate_virtual_device_count",
    "calculate_virtual_device_count_from_cli",
    "devices_exist",
    "enable_all_peer_access",
    "get_virtual_device_api",
    "get_virtual_device_count",
    "get_virtual_device_target_arch",
    "is_virtual_device_mode",
    "load_devices",
    "load_max_buffer",
    "scan_available_devices",
    "set_virtual_device_api",
    "set_virtual_device_count",
    "set_virtual_device_target_arch",
]
