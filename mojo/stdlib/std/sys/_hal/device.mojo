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

from .plugin import DeviceHandle


struct Device(Movable):
    """A device retrieved from a Driver.

    Does not own the device handle — the plugin manages device lifetime
    internally. The Device is only valid while the parent Driver is alive.
    """

    var _handle: DeviceHandle
    var id: Int64

    def __init__(out self, handle: DeviceHandle, id: Int64):
        self._handle = handle
        self.id = id
