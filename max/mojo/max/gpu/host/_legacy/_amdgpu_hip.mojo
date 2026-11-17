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

from std.ffi import external_call

from .device_context import (
    LegacyDeviceContext,
    LegacyDeviceFunction,
    LegacyDeviceStream,
)
from max.gpu.host._driver_handles import _CString
from max.gpu.host._driver_common import _checked
from max.gpu.host._driver_handles import _CString
from max.gpu.host._vendor_handles import (
    hipDevice_t,
    hipStream_t,
    hipModule_t,
)
from max.gpu.host._driver_common import _checked


# Accessor function to get access to the underlying hipDevice_t from an abstract DeviceContext.
# Use `var hip_dev: hipDevice_t = HIP(ctx)` where ctx is a `DeviceContext` to get access to the
# underlying hipDevice_t.
@inline(.always)
def HIP(ctx: LegacyDeviceContext) raises -> hipDevice_t:
    var result = hipDevice_t()
    # const char *AsyncRT_DeviceContext_hip_device(hipDevice_t *result, const DeviceContext *ctx)
    _checked(
        external_call["AsyncRT_DeviceContext_hip_device", _CString[]](
            Pointer(to=result),
            ctx._handle,
        )
    )
    return result


# Accessor function to get access to the underlying hipStream_t from an abstract DeviceStream.
# Use `var hip_stream: hipStream_t = HIP(ctx.stream())` where ctx is a `DeviceContext` to get access to the underlying hipStream_t.
@inline(.always)
def HIP(stream: LegacyDeviceStream) raises -> hipStream_t:
    var result = hipStream_t()
    # const char *AsyncRT_DeviceStream_hip_stream(hipStream_t *result, const DeviceStream *stream)
    _checked(
        external_call[
            "AsyncRT_DeviceStream_hip_stream",
            _CString[],
        ](
            Pointer(to=result),
            stream._handle,
        )
    )
    return result


# Accessor function to get access to the underlying hipModule_t from a DeviceFunction.
@inline(.always)
def HIP_MODULE(func: LegacyDeviceFunction) raises -> hipModule_t:
    var result = hipModule_t()
    # const char *AsyncRT_DeviceFunction_hip_module(hipModule_t *result, const DeviceFunction *func)
    _checked(
        external_call[
            "AsyncRT_DeviceFunction_hip_module",
            _CString[],
        ](
            Pointer(to=result),
            func._handle,
        )
    )
    return result
