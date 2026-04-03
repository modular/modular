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
from std.gpu.host import DeviceContext, DeviceFunction, DeviceStream
from std.gpu.host.device_context import (
    _CString,
    _checked,
    _DeviceBufferPtr,
    _DeviceContextPtr,
    _DeviceFunctionPtr,
    _DeviceStreamPtr,
)


struct _CUctx_st:
    pass


struct _CUstream_st:
    pass


struct _CUmod_st:
    pass


struct _CUevent_st:
    pass


comptime CUcontext = UnsafePointer[_CUctx_st, ExternalOrigin[mut=True]]
comptime CUstream = UnsafePointer[_CUstream_st, ExternalOrigin[mut=True]]
comptime CUmodule = UnsafePointer[_CUmod_st, ExternalOrigin[mut=True]]
comptime CUevent = UnsafePointer[_CUevent_st, ExternalOrigin[mut=True]]


# Accessor function to get access to the underlying CUcontext from a abstract DeviceContext.
# Use `var cuda_ctx: CUcontext = CUDA(ctx)` where ctx is a `DeviceContext` to get access to the underlying CUcontext.
@always_inline
def CUDA(ctx: DeviceContext) raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_context(CUcontext *result, const DeviceContext *ctx)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_cuda_context",
            _CString[],
        ](
            UnsafePointer(to=result),
            ctx._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUstream from a abstract DeviceStream.
# Use `var cuda_stream: CUstream = CUDA(ctx.stream())` where ctx is a `DeviceContext` to get access to the underlying CUstream.
@always_inline
def CUDA(stream: DeviceStream) raises -> CUstream:
    var result = CUstream()
    # const char *AsyncRT_DeviceStream_cuda_stream(CUstream *result, const DeviceStream *stream)
    _checked(
        external_call[
            "AsyncRT_DeviceStream_cuda_stream",
            _CString[],
        ](
            UnsafePointer(to=result),
            stream._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUmodule from a DeviceFunction.
@always_inline
def CUDA_MODULE(func: DeviceFunction) raises -> CUmodule:
    var result = CUmodule()
    # const char *AsyncRT_DeviceFunction_cuda_module(CUmodule *result, const DeviceFunction *func)
    _checked(
        external_call[
            "AsyncRT_DeviceFunction_cuda_module",
            _CString[],
        ](
            UnsafePointer(to=result),
            func._handle,
        )
    )
    return result


def CUDA_get_current_context() raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_current_context(CUcontext *result)
    _checked(
        external_call["AsyncRT_DeviceContext_cuda_current_context", _CString[]](
            UnsafePointer(to=result),
        )
    )
    return result
