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
"""HIP native-handle accessors for the HAL `DeviceContext`."""

from .device_context_hal import (
    HALDeviceContext,
    HALDeviceFunction,
)
from max.gpu.host._vendor_handles import (
    hipDevice_t,
    hipModule_t,
)


@inline(.always)
def HIP(ctx: HALDeviceContext) raises -> hipDevice_t:
    raise Error("hipDevice_t access is not wired for the HAL DeviceContext")


@inline(.always)
def HIP_MODULE(func: HALDeviceFunction) raises -> hipModule_t:
    raise Error(
        "hipModule_t access is not wired for the HAL DeviceContext; the HIP"
        " plugin does not answer M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE"
    )
