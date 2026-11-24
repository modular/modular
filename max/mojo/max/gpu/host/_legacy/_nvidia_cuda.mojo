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
    CUcontext,
    CUstream,
    CUmodule,
)
from max.gpu.host._driver_common import _checked


# Accessor function to get access to the underlying CUcontext from a abstract DeviceContext.
# Use `var cuda_ctx: CUcontext = CUDA(ctx)` where ctx is a `DeviceContext` to get access to the underlying CUcontext.
@inline(.always)
def CUDA(ctx: LegacyDeviceContext) raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_context(CUcontext *result, const DeviceContext *ctx)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_cuda_context",
            _CString[],
        ](
            Pointer(to=result),
            ctx._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUstream from a abstract DeviceStream.
# Use `var cuda_stream: CUstream = CUDA(ctx.stream())` where ctx is a `DeviceContext` to get access to the underlying CUstream.
@inline(.always)
def CUDA(stream: LegacyDeviceStream) raises -> CUstream:
    var result = CUstream()
    # const char *AsyncRT_DeviceStream_cuda_stream(CUstream *result, const DeviceStream *stream)
    _checked(
        external_call[
            "AsyncRT_DeviceStream_cuda_stream",
            _CString[],
        ](
            Pointer(to=result),
            stream._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUmodule from a DeviceFunction.
@inline(.always)
def CUDA_MODULE(func: LegacyDeviceFunction) raises -> CUmodule:
    var result = CUmodule()
    # const char *AsyncRT_DeviceFunction_cuda_module(CUmodule *result, const DeviceFunction *func)
    _checked(
        external_call[
            "AsyncRT_DeviceFunction_cuda_module",
            _CString[],
        ](
            Pointer(to=result),
            func._handle,
        )
    )
    return result


def CUDA_get_current_context() raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_current_context(CUcontext *result)
    _checked(
        external_call["AsyncRT_DeviceContext_cuda_current_context", _CString[]](
            Pointer(to=result),
        )
    )
    return result
