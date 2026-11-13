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
"""Backend-dispatching versions of the vendor handle helpers.

`CUDA`, `CUDA_MODULE`, `HIP` and `HIP_MODULE` take a context, stream or function
and return the vendor's native handle. Each forwards to the legacy or the HAL
helper, as `use_legacy_backend` selects. The handle types come from the
overlay, which names the linked backend's.
"""

from .device_context import use_legacy_backend
from max._driver.max_license.device_context_dispatch.function import (
    DeviceFunction,
)
from max._driver.max_license.device_context_dispatch.host_types import (
    DeviceStreamBase,
)
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max._driver.max_license.device_context_hal._amdgpu_hip_hal import (
    HIP as _hal_HIP,
    HIP_MODULE as _hal_HIP_MODULE,
)
from max._driver.max_license.device_context_hal._nvidia_cuda_hal import (
    CUDA as _hal_CUDA,
    CUDA_MODULE as _hal_CUDA_MODULE,
)
from max.gpu.host._legacy._amdgpu_hip import (
    HIP as _legacy_HIP,
    HIP_MODULE as _legacy_HIP_MODULE,
)
from max.gpu.host._legacy._nvidia_cuda import (
    CUDA as _legacy_CUDA,
    CUDA_MODULE as _legacy_CUDA_MODULE,
)
from max.gpu.host._vendor_handles import (
    CUcontext,
    CUmodule,
    CUstream,
    hipDevice_t,
    hipModule_t,
    hipStream_t,
)


def CUDA(ctx: DeviceContext) raises -> CUcontext:
    """Returns the CUDA driver context of a device context.

    Args:
        ctx: The device context to query.

    Returns:
        The native `CUcontext`.

    Raises:
        If the context is not a CUDA context.
    """
    comptime if use_legacy_backend():
        return rebind_var[CUcontext](_legacy_CUDA(ctx.unwrap_legacy()))
    else:
        return rebind_var[CUcontext](_hal_CUDA(ctx.unwrap_hal()))


def CUDA(stream: DeviceStreamBase) raises -> CUstream:
    """Returns the CUDA stream behind a device stream.

    Args:
        stream: The device stream to query.

    Returns:
        The native `CUstream`.

    Raises:
        If the stream is not a CUDA stream.
    """
    comptime if use_legacy_backend():
        return rebind_var[CUstream](_legacy_CUDA(stream.unwrap_legacy()))
    else:
        return rebind_var[CUstream](_hal_CUDA(stream.unwrap_hal()))


def CUDA_MODULE(func: DeviceFunction) raises -> CUmodule:
    """Returns the CUDA module a compiled function lives in.

    Args:
        func: The compiled function to query.

    Returns:
        The native `CUmodule`.

    Raises:
        If the function is not a CUDA function.
    """
    comptime if use_legacy_backend():
        return rebind_var[CUmodule](_legacy_CUDA_MODULE(func.unwrap_legacy()))
    else:
        return rebind_var[CUmodule](_hal_CUDA_MODULE(func.unwrap_hal()))


def HIP(ctx: DeviceContext) raises -> hipDevice_t:
    """Returns the HIP device of a device context.

    Args:
        ctx: The device context to query.

    Returns:
        The native `hipDevice_t`.

    Raises:
        If the context is not a HIP context.
    """
    comptime if use_legacy_backend():
        return rebind_var[hipDevice_t](_legacy_HIP(ctx.unwrap_legacy()))
    else:
        return rebind_var[hipDevice_t](_hal_HIP(ctx.unwrap_hal()))


def HIP(stream: DeviceStreamBase) raises -> hipStream_t:
    """Returns the HIP stream behind a device stream.

    Args:
        stream: The device stream to query.

    Returns:
        The native `hipStream_t`.

    Raises:
        If the stream is not a HIP stream.
    """
    comptime if use_legacy_backend():
        return rebind_var[hipStream_t](_legacy_HIP(stream.unwrap_legacy()))
    else:
        return rebind_var[hipStream_t](_hal_HIP(stream.unwrap_hal()))


def HIP_MODULE(func: DeviceFunction) raises -> hipModule_t:
    """Returns the HIP module a compiled function lives in.

    Args:
        func: The compiled function to query.

    Returns:
        The native `hipModule_t`.

    Raises:
        If the function is not a HIP function.
    """
    comptime if use_legacy_backend():
        return rebind_var[hipModule_t](_legacy_HIP_MODULE(func.unwrap_legacy()))
    else:
        return rebind_var[hipModule_t](_hal_HIP_MODULE(func.unwrap_hal()))
