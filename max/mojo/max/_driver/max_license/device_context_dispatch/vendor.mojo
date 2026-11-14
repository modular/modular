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
from max._driver.max_license.device_context_dispatch.device_function import (
    DeviceFunction,
)
from max._driver.max_license.device_context_dispatch.host_types import (
    DeviceStreamBase,
)
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max.gpu.host._legacy._amdgpu_hip import (
    HIP as _legacy_HIP,
    HIP_MODULE as _legacy_HIP_MODULE,
)
from max.gpu.host._legacy._nvidia_cuda import (
    CUDA as _legacy_CUDA,
    CUDA_MODULE as _legacy_CUDA_MODULE,
)
from std.memory.pointer import unsafe_cast
from max.gpu.host._vendor_handles import (
    _CUstream_st,
    _ihipStream_t,
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
        raise Error(
            "CUcontext access is not wired for the HAL DeviceContext; the"
            " plugin exposes it via M_driver_context_get_current_driver_context"
        )


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
        return unsafe_cast[Type=_CUstream_st, origin=UntrackedOrigin[mut=True]](
            stream._native_stream()
        )


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
        raise Error(
            "CUmodule access is not wired for the HAL DeviceContext; the plugin"
            " exposes it via M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE"
        )


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
        raise Error("hipDevice_t access is not wired for the HAL DeviceContext")


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
        return unsafe_cast[
            Type=_ihipStream_t, origin=UntrackedOrigin[mut=True]
        ](stream._native_stream())


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
        raise Error(
            "hipModule_t access is not wired for the HAL DeviceContext; the HIP"
            " plugin does not answer M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE"
        )
