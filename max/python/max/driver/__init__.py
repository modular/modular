# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max._core_types.driver import DLPackArray

from .driver import (
    CPU,
    Accelerator,
    Device,
    DeviceSpec,
    DeviceStream,
    accelerator_api,
    accelerator_architecture_name,
    accelerator_count,
    devices_exist,
    load_devices,
    scan_available_devices,
)
from .tensor import Tensor

del driver  # type: ignore
del tensor  # type: ignore

__all__ = [
    "CPU",
    "Accelerator",
    "DLPackArray",
    "Device",
    "DeviceSpec",
    "DeviceStream",
    "Tensor",
    "accelerator_api",
    "accelerator_architecture_name",
    "devices_exist",
    "load_devices",
    "scan_available_devices",
]
