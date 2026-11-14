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
"""CUDA native-handle accessors for the HAL `DeviceContext`."""

from .device_context_hal import (
    HALDeviceContext,
    HALDeviceFunction,
)
from max.gpu.host._vendor_handles import (
    CUcontext,
    CUmodule,
)


@inline(.always)
def CUDA(ctx: HALDeviceContext) raises -> CUcontext:
    raise Error(
        "CUcontext access is not wired for the HAL DeviceContext; the plugin"
        " exposes it via M_driver_context_get_current_driver_context"
    )


@inline(.always)
def CUDA_MODULE(func: HALDeviceFunction) raises -> CUmodule:
    raise Error(
        "CUmodule access is not wired for the HAL DeviceContext; the plugin"
        " exposes it via M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE"
    )


def CUDA_get_current_context() raises -> CUcontext:
    raise Error(
        "current-CUcontext access is not wired for the HAL DeviceContext; the"
        " plugin exposes it via M_driver_context_get_current_driver_context"
    )
