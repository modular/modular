# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer

from gpu.host import DeviceContext, DeviceStream

from gpu.host.device_context_v2 import (
    _checked,
    _CharPtr,
    _DeviceContextPtr,
    _DeviceStreamPtr,
)

from sys import external_call


struct _CUctx_st:
    pass


struct _CUstream_st:
    pass


alias CUcontext = UnsafePointer[_CUctx_st]
alias CUstream = UnsafePointer[_CUstream_st]


# Accessor function to get access to the underlying CUcontext from a abstract DeviceContext.
# Use `var cuda_ctx: CUcontext = CUDA(ctx)` where ctx is a `DeviceContext` to get access to the underlying CUcontext.
fn CUDA(ctx: DeviceContext) raises -> CUcontext:
    var result = CUcontext()
    # const char *AsyncRT_DeviceContext_cuda_context(CUcontext *result, const DeviceContext *ctx)
    _checked(
        external_call[
            "AsyncRT_DeviceContext_cuda_context",
            _CharPtr,
            UnsafePointer[CUcontext],
            _DeviceContextPtr,
        ](
            UnsafePointer.address_of(result),
            ctx.v2()._handle,
        )
    )
    return result


# Accessor function to get access to the underlying CUstream from a abstract DeviceStream.
# Use `var cuda_stream: CUstream = CUDA(ctx.stream())` where ctx is a `DeviceContext` to get access to the underlying CUstream.
fn CUDA(stream: DeviceStream) raises -> CUstream:
    var result = CUstream()
    # const char *AsyncRT_DeviceStream_cuda_stream(CUstream *result, const DeviceStream *stream)
    _checked(
        external_call[
            "AsyncRT_DeviceStream_cuda_stream",
            _CharPtr,
            UnsafePointer[CUstream],
            _DeviceStreamPtr,
        ](
            UnsafePointer.address_of(result),
            stream._handle,
        )
    )
    return result
