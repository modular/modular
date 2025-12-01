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

from memory import LegacyUnsafePointer as UnsafePointer
from sys import external_call
from gpu.host import DeviceContext, DeviceFunction, DeviceStream
from gpu.host.device_context import (
    _ConstCharPtr,
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


comptime CUcontext = UnsafePointer[_CUctx_st]
comptime CUstream = UnsafePointer[_CUstream_st]
comptime CUmodule = UnsafePointer[_CUmod_st]
comptime CUevent = UnsafePointer[_CUevent_st]


# Accessor function to get access to the underlying CUcontext from a abstract DeviceContext.
# Use `var cuda_ctx: CUcontext = CUDA(ctx)` where ctx is a `DeviceContext` to get access to the underlying CUcontext.
@always_inline
fn CUDA(ctx: DeviceContext) raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_context(CUcontext *result, const DeviceContext *ctx)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_cuda_context",
            _ConstCharPtr,
            UnsafePointer[CUcontext],
            _DeviceContextPtr,
        ](
            UnsafePointer(to=result),
            ctx._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUstream from a abstract DeviceStream.
# Use `var cuda_stream: CUstream = CUDA(ctx.stream())` where ctx is a `DeviceContext` to get access to the underlying CUstream.
@always_inline
fn CUDA(stream: DeviceStream) raises -> CUstream:
    var result = CUstream()
    # const char *AsyncRT_DeviceStream_cuda_stream(CUstream *result, const DeviceStream *stream)
    _checked(
        external_call[
            "AsyncRT_DeviceStream_cuda_stream",
            _ConstCharPtr,
            UnsafePointer[CUstream],
            _DeviceStreamPtr,
        ](
            UnsafePointer(to=result),
            stream._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUmodule from a DeviceFunction.
@always_inline
fn CUDA_MODULE(func: DeviceFunction) raises -> CUmodule:
    var result = CUmodule()
    # const char *AsyncRT_DeviceFunction_cuda_module(CUmodule *result, const DeviceFunction *func)
    _checked(
        external_call[
            "AsyncRT_DeviceFunction_cuda_module",
            _ConstCharPtr,
            UnsafePointer[CUmodule],
            _DeviceFunctionPtr,
        ](
            UnsafePointer(to=result),
            func._handle,
        )
    )
    return result


fn CUDA_get_current_context() raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_current_context(CUcontext *result)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_cuda_current_context", _ConstCharPtr
        ](
            UnsafePointer(to=result),
        )
    )
    return result
