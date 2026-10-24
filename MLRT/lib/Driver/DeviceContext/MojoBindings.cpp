//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file contains bindings for the C++ DeviceContext
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Driver/DeviceContext/BinaryID.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

// See the static_assert in DeviceContext.cpp.
static_assert(
    !MODULAR_HAL_DEVICE_CONTEXT,
    "the AsyncRT_Device* bindings must compile against the legacy "
    "DeviceContext declarations: -UMODULAR_HAL_DEVICE_CONTEXT in this "
    "target's copts must override Driver/DeviceContextAPI's define");

#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADriver.h"
#include "MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/HIP/HIPDriver.h"
#include "MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.h"
#include "Support/BinaryID.h"
#include "Support/RCRef.h"
#include "Support/SymbolExport.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <chrono>
#include <new>

using namespace M;
using namespace M::Driver;

//===----------------------------------------------------------------------===//
// DeviceContext
//===----------------------------------------------------------------------===//

// Make a malloc'd copy of the string, so that we can safely free the
// pointer from Mojo.
static const char *raiseError(const char *msg) { return strdup(msg); }

static const char *checkSuccess(ErrorOrSuccess errOr) {
  if (errOr) {
    return raiseError(errOr.getError());
  }
  return nullptr;
}

template <typename T, typename R>
const char *checkRef(T **result, ErrorOr<R> errOr) {
  if (errOr) {
    *result = nullptr;
    return raiseError(errOr.getError());
  }
  // Return the pointer, transferring the ownership to Mojo.
  *result = errOr.get().release();
  return nullptr;
}

template <typename T, typename V>
const char *checkVal(T *result, V errVal, ErrorOr<T> errOr) {
  if (errOr) {
    *result = errVal;
    return raiseError(errOr.getError());
  }
  *result = errOr.get();
  return nullptr;
}

MODULAR_EXPORT void AsyncRT_DeviceContext_strfree(const char *ptr) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_release", TRACY_COLOR_GRAY);
  free(const_cast<char *>(ptr));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_create(DeviceContext **result, const char *kind, int id) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_release", TRACY_COLOR_GRAY);
  return checkRef(result, DeviceContext::create(kind, id));
}

MODULAR_EXPORT void AsyncRT_DeviceContext_retain(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceContextRef::lowLevelAddRef(ctx);
}

MODULAR_EXPORT void AsyncRT_DeviceContext_release(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_release", TRACY_COLOR_GRAY);
  // Give the event handlers opportunity to run, which can decrement the
  // reference count.
  [[maybe_unused]] ErrorOrSuccess ignored = ctx->pollPendingEvents();
  // Decrement the low-level reference count.
  DeviceContextRef::lowLevelDropRef(ctx);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_deviceName(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_deviceName", TRACY_COLOR_GRAY);
  return raiseError(ctx->device().name().c_str());
}

MODULAR_EXPORT void AsyncRT_DeviceContext_deviceApi(llvm::StringRef *result,
                                                    DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_deviceApi", TRACY_COLOR_GRAY);
  *result = ctx->device().api();
}

MODULAR_EXPORT void AsyncRT_DeviceContext_archName(llvm::StringRef *result,
                                                   DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_archName", TRACY_COLOR_GRAY);
  ErrorOr<const char *> archName = ctx->device().getArchName();
  if (archName.isError())
    *result = "";
  else
    *result = archName.get();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContextScope_create(const DeviceContextScope **result,
                                  DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContextScope_create", TRACY_COLOR_GRAY);
  DeviceContextScope *scope = new DeviceContextScope(*ctx);
  const char *errMsg = checkSuccess(scope->err());
  if (errMsg == nullptr) {
    *result = scope;
  }
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setAsCurrent(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_setAsCurrent", TRACY_COLOR_GRAY);
  ErrorOrSuccess err = ctx->setAsCurrent();
  return checkSuccess(std::move(err));
}

MODULAR_EXPORT void
AsyncRT_DeviceContextScope_release(const DeviceContextScope *scope) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContextScope_release", TRACY_COLOR_GRAY);
  delete scope;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_cuda_current_context(CUcontext *result) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_cuda_current_context",
                       TRACY_COLOR_GRAY);
  ErrorOr<const CUDADriver &> errDriver = CUDADriver::get();
  if (errDriver.isError()) {
    return raiseError(errDriver.getError());
  }
  if ((*errDriver).cuCtxGetCurrent(result) != CUDADriver::SUCCESS) {
    return raiseError("cuCtxGetCurrent failed.");
  }
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createBuffer_async(DeviceBuffer **result,
                                         void **devicePtr, DeviceContext *ctx,
                                         size_t len, size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createBuffer_async",
                       TRACY_COLOR_GRAY);
  const char *errMsg =
      checkRef(result, ctx->enqueueCreateBuffer(len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createHostBuffer(DeviceBuffer **result, void **devicePtr,
                                       DeviceContext *ctx, size_t len,
                                       size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createHostBuffer",
                       TRACY_COLOR_GRAY);
  const char *errMsg = checkRef(result, ctx->createHostBuffer(len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_wrapHostMemory(DeviceBuffer **result, void **devicePtr,
                                     DeviceContext *ctx, void *hostPtr,
                                     size_t len, size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_wrapHostMemory",
                       TRACY_COLOR_GRAY);
  const char *errMsg =
      checkRef(result, ctx->wrapHostMemory(hostPtr, len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

// TODO: Remove the V1 compatibility.
MODULAR_EXPORT void AsyncRT_DeviceContext_createBuffer_owning(
    DeviceBuffer **result, DeviceContext *ctx, void *devicePtr, size_t len,
    size_t elemSize, bool owning) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createBuffer_owning",
                       TRACY_COLOR_GRAY);
  *result =
      ctx->createBufferOwning(devicePtr, len * elemSize, owning).release();
}
MODULAR_EXPORT const char *
AsyncRT_DeviceBuffer_createSubBuffer(DeviceBuffer **result, void **devicePtr,
                                     DeviceBuffer *buf, size_t offset,
                                     size_t len, size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_createSubBuffer",
                       TRACY_COLOR_GRAY);
  const char *errMsg =
      checkRef(result, buf->createSubBuffer(offset * elemSize, len * elemSize));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_retain(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceBufferRef::lowLevelAddRef(buffer);
}

MODULAR_EXPORT void AsyncRT_DeviceBuffer_release(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceBufferRef::lowLevelDropRef(buffer);
}

// Wraps an already-owning `DeviceBuffer` handle (surrendered net-zero by the
// Mojo side via `DeviceBuffer.take_handle()`) in an
// `AsyncValue<DeviceBufferRef>` and returns the owning `AsyncValue*` (refcount
// 1) for a Mojo `AnyAsyncValueRef` to adopt net-zero.
MODULAR_EXPORT AsyncRT::AsyncValue *
AsyncRT_AsyncValue_createFromDeviceBuffer(DeviceBuffer *handle) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_createFromDeviceBuffer",
                       TRACY_COLOR_GRAY);
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  // TAKE the reference net-zero (no addRef): `take_handle` is `deinit self`, so
  // adopting via `takeRCRef` transfers the single live reference without an
  // extra ref (a copy would leak +1, a fresh owner would double-free).
  DeviceBufferRef buf = takeRCRef(handle);
  AnyAsyncValueRef storage =
      AnyAsyncValueRef::createReady<DeviceBufferRef>(cpuDevice, std::move(buf));
  // Hand the owning `AsyncValue*` (refcount 1) to the Mojo caller;
  // `releasePointer` resets `storage` to null so its destructor does not drop
  // the reference.
  return storage.releasePointer();
}

MODULAR_EXPORT const DeviceContext *
AsyncRT_DeviceBuffer_context(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_context", TRACY_COLOR_GRAY);
  return &buffer->context();
}

// TODO: Remove the V1 compatibility.
MODULAR_EXPORT void AsyncRT_DeviceBuffer_release_ptr(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_release_ptr", TRACY_COLOR_GRAY);
  buffer->releasePtr();
}

MODULAR_EXPORT int64_t AsyncRT_DeviceBuffer_bytesize(DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_bytesize", TRACY_COLOR_GRAY);
  return buffer->byteSize();
}

MODULAR_EXPORT const char *AsyncRT_DeviceBuffer_hostPtr(void **result,
                                                        DeviceBuffer *buffer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceBuffer_hostPtr", TRACY_COLOR_GRAY);
  return checkVal(result, static_cast<void *>(nullptr),
                  buffer->context().getHostAccessiblePointer(buffer->ptr()));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_stream(DeviceStream **result,
                                                        DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_stream", TRACY_COLOR_GRAY);
  return checkRef(result, ctx->stream());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createStream(DeviceStream **stream, int priority,
                                   DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createStream", TRACY_COLOR_GRAY);
  return checkRef(stream, ctx->createStream(priority));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_selectStream(DeviceContext **result, DeviceContext *ctx,
                                   unsigned int streamId) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_selectStream", TRACY_COLOR_GRAY);
  return checkRef(result, ctx->selectStream(streamId));
}

MODULAR_EXPORT int AsyncRT_DeviceContext_numStreams(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_numStreams", TRACY_COLOR_GRAY);
  return ctx->numStreams();
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_createExternalStream(
    DeviceStream **stream, void *externalStream, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createExternalStream",
                       TRACY_COLOR_GRAY);
  return checkRef(stream, ctx->createExternalStream(externalStream));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_streamPriorityRange(
    int *leastPriority, int *greatestPriority, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_streamPriorityRange",
                       TRACY_COLOR_GRAY);
  return checkSuccess(
      ctx->streamPriorityRange(leastPriority, greatestPriority));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_synchronize(DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_synchronize", TRACY_COLOR_GRAY);
  return checkSuccess(stream->synchronize());
}

MODULAR_EXPORT void AsyncRT_DeviceStream_retain(DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceStreamRef::lowLevelAddRef(stream);
}

MODULAR_EXPORT void AsyncRT_DeviceStream_release(DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceStreamRef::lowLevelDropRef(stream);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_waitForEvent(DeviceStream *stream, DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_waitForEvent", TRACY_COLOR_GRAY);
  return checkSuccess(stream->enqueueWaitForEvent(*event));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_eventRecord(DeviceStream *stream, DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_eventRecord", TRACY_COLOR_GRAY);
  return checkSuccess(stream->eventRecord(*event));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_enqueueHostFunc(DeviceStream *stream, void (*fn)(void *),
                                     void *userData) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_enqueueHostFunc",
                       TRACY_COLOR_GRAY);
  return checkSuccess(stream->enqueueHostFunc(fn, userData));
}

MODULAR_EXPORT const char *AsyncRT_DeviceStream_enqueueWaitOnHostValue(
    DeviceStream *stream, CompletionFlag *flag, uint64_t value) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_enqueueWaitOnHostValue",
                       TRACY_COLOR_GRAY);
  return checkSuccess(stream->enqueueWaitOnHostValue(*flag, value));
}

MODULAR_EXPORT uint64_t AsyncRT_CompletionFlag_devicePtr(CompletionFlag *flag) {
  return flag->devicePtr();
}

MODULAR_EXPORT const char *AsyncRT_DeviceEvent_synchronize(DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceEvent_synchronize", TRACY_COLOR_GRAY);
  return checkSuccess(event->synchronize());
}

MODULAR_EXPORT void AsyncRT_DeviceEvent_retain(DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceEvent_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceEventRef::lowLevelAddRef(event);
}

MODULAR_EXPORT void AsyncRT_DeviceEvent_release(DeviceEvent *event) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceEvent_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceEventRef::lowLevelDropRef(event);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_synchronize(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_synchronize", TRACY_COLOR_GRAY);
  return checkSuccess(ctx->synchronize());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_enqueue_wait_for_context(DeviceContext *ctx,
                                               DeviceContext *other) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enqueue_wait_for_context",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueWaitFor(*other));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_eventCreate(DeviceEvent **result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_eventCreate", TRACY_COLOR_GRAY);
  // Mojo events have no elapsed-time query, so timing would be pure overhead.
  return checkRef(result, ctx->eventCreate(/*enableTiming=*/false));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_HtoD_async(DeviceContext *ctx,
                                                            DeviceBuffer *dst,
                                                            const void *src) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_HtoD_async", TRACY_COLOR_GRAY);
  // Compare devices, not context identity: a recording-context view
  // (`DeviceGraphBuilder::recordingContext`) is a distinct DeviceContext object
  // that shares the backing device, and legitimately enqueues copies of buffers
  // owned by that device.
  assert((&ctx->device() == &dst->context().device()) &&
         "ctx device not matching dst");
  return checkSuccess(ctx->enqueueCopyToDevice(*dst, src));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_DtoH_async(DeviceContext *ctx,
                                                            void *dst,
                                                            DeviceBuffer *src) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_DtoH_async", TRACY_COLOR_GRAY);
  // Compare devices, not context identity (see HtoD_async above): a
  // recording-context view shares the backing device but is a distinct object.
  assert((&ctx->device() == &src->context().device()) &&
         "ctx device not matching src");
  return checkSuccess(ctx->enqueueCopyFromDevice(dst, *src));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_HtoD_async_sized(DeviceContext *ctx, void *dst, void *src,
                                       size_t size) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_HtoD_async_sized",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueCopyToDevice(dst, src, size));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoH_async_sized(DeviceContext *ctx, void *dst, void *src,
                                       size_t size) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_DtoH_async_sized",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueCopyFromDevice(dst, src, size));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_DtoD_async(DeviceContext *ctx,
                                                            DeviceBuffer *dst,
                                                            DeviceBuffer *src) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_DtoD_async", TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueCopyDeviceToDevice(*dst, *src));
}

// Same as AsyncRT_DeviceContext_DtoD_async, but the driver does not insert the
// cross-stream synchronization for the copy. The caller is responsible for
// synchronizing the source and destination streams (e.g. the graph compiler
// emits explicit synchronization ops around the copy).
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_DtoD_async_no_cross_stream_sync(DeviceContext *ctx,
                                                      DeviceBuffer *dst,
                                                      DeviceBuffer *src) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_DtoD_async_no_cross_stream_sync",
                       TRACY_COLOR_GRAY);
  return checkSuccess(
      ctx->enqueueCopyDeviceToDevice(*dst, *src, /*crossStreamSync=*/false));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setMemory_async(DeviceContext *ctx, DeviceBuffer *dst,
                                      uint64_t val, size_t valSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_setMemory_async",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueSetMemory(*dst, val, valSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_loadFunction(
    DeviceFunction **result, DeviceContext *ctx, const char *moduleName,
    const char *functionName, const char *data, size_t dataLen,
    int32_t maxDynamicSharedBytes, const char *debugLevel,
    int32_t optimizationLevel) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_loadFunction", TRACY_COLOR_GRAY);
  return checkRef(result, ctx->loadFunction(moduleName, functionName,
                                            StringRef(data, dataLen),
                                            maxDynamicSharedBytes, debugLevel,
                                            optimizationLevel));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_loadFunctionWithDebugName(
    DeviceFunction **result, DeviceContext *ctx, const char *moduleName,
    const char *functionName, const char *data, size_t dataLen,
    int32_t maxDynamicSharedBytes, const char *debugLevel,
    int32_t optimizationLevel, const char *debugName) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_loadFunctionWithDebugName",
                       TRACY_COLOR_GRAY);
  return checkRef(result, ctx->loadFunctionWithDebugName(
                              moduleName, functionName,
                              StringRef(data, dataLen), maxDynamicSharedBytes,
                              debugLevel, optimizationLevel, debugName));
}

MODULAR_EXPORT void AsyncRT_DeviceFunction_retain(DeviceFunction *func) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_retain", TRACY_COLOR_GRAY);
  // Increment the low-level reference count.
  DeviceFunctionRef::lowLevelAddRef(func);
}

MODULAR_EXPORT void AsyncRT_DeviceFunction_release(DeviceFunction *func) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceFunctionRef::lowLevelDropRef(func);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_copyToConstantMemory(DeviceFunction *func,
                                            const char *name, size_t nameSize,
                                            const void *data, size_t dataSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_copyToConstantMemory",
                       TRACY_COLOR_GRAY);
  assert(name != nullptr && "name is required");
  assert(nameSize > 0 && "nameSize is required");
  std::string nameStr(name, nameSize);
  return checkSuccess(
      func->copyToNamedConstant(nameStr.c_str(), data, dataSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_enqueueFunctionDirect(
    DeviceContext *ctx, DeviceFunction *func, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
    uint32_t argCount, const size_t *argSizes) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enqueueFunctionDirect",
                       TRACY_COLOR_GRAY);
  // Compare devices, not context identity: a recording-context view
  // (`DeviceGraphBuilder::recordingContext`) records launches of functions
  // compiled on the underlying context, which shares the same device.
  assert((&ctx->device() == &func->context().device()) &&
         "ctx device not matching func");
  return checkSuccess(ctx->enqueueFunctionExecDirect(
      *func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, attrs,
      numAttrs, args, argCount, argSizes));
}

MODULAR_EXPORT const char *AsyncRT_DeviceStream_enqueueFunctionDirect(
    DeviceStream *stream, DeviceFunction *func, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
    uint32_t argCount, const size_t *argSizes) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_enqueueFunctionDirect",
                       TRACY_COLOR_GRAY);
  return checkSuccess(stream->enqueueFunctionExecDirect(
      *func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, attrs,
      numAttrs, args, argCount, argSizes));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_enqueueHostFunction(
    DeviceContext *ctx, void (*resumeFunc)(int8_t *),
    void (*destroyFunc)(int8_t *), int8_t *coroHandle) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enqueueHostFunction",
                       TRACY_COLOR_GRAY);
  return checkSuccess(
      ctx->enqueueHostFunction(resumeFunc, destroyFunc, coroHandle));
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_enqueueHostFunctionRange(
    DeviceContext *ctx, void (*resumeFunc)(int8_t *),
    void (*destroyFunc)(int8_t *), int8_t **coroHandles, size_t numHandles) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enqueueHostFunctionRange",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->enqueueHostFunctionRange(resumeFunc, destroyFunc,
                                                    coroHandles, numHandles));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_getAttribute(int32_t *result, DeviceFunction *func,
                                    int32_t attrCode) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_getAttribute", TRACY_COLOR_GRAY);
  return checkVal(result, 0, func->getAttribute(attrCode));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_startTimer(DeviceTimer **result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_startTimer", TRACY_COLOR_GRAY);
  return checkRef(result, ctx->startTimer());
}

MODULAR_EXPORT const char *AsyncRT_DeviceContext_stopTimer(int64_t *result,
                                                           DeviceContext *ctx,
                                                           DeviceTimer *timer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_stopTimer", TRACY_COLOR_GRAY);
  std::chrono::nanoseconds elapsedNanos;
  const char *errMsg = checkVal(&elapsedNanos, std::chrono::nanoseconds{0},
                                ctx->stopTimer(*timer));
  *result = elapsedNanos.count();
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_startMetalTraceCapture(DeviceContext *ctx,
                                             const char *path) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_startMetalTraceCapture",
                       TRACY_COLOR_GRAY);
#if MODULAR_MACOS
  MetalDeviceContext *verifiedCtx = MetalDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr)
    return raiseError("Not a MetalContext");
  return checkSuccess(verifiedCtx->startMetalTraceCapture(path));
#else
  return raiseError("Not a MetalContext");
#endif
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_stopMetalTraceCapture(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_stopMetalTraceCapture",
                       TRACY_COLOR_GRAY);
#if MODULAR_MACOS
  MetalDeviceContext *verifiedCtx = MetalDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr)
    return raiseError("Not a MetalContext");
  return checkSuccess(verifiedCtx->stopMetalTraceCapture());
#else
  return raiseError("Not a MetalContext");
#endif
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_setMetalPrintEnabled(DeviceContext *ctx, bool enabled) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_setMetalPrintEnabled",
                       TRACY_COLOR_GRAY);
#if MODULAR_MACOS
  MetalDeviceContext *verifiedCtx = MetalDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr)
    return raiseError("Not a MetalContext");
  return checkSuccess(verifiedCtx->setMetalPrintEnabled(enabled));
#else
  return raiseError("Not a MetalContext");
#endif
}

MODULAR_EXPORT void AsyncRT_DeviceTimer_release(DeviceTimer *timer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceTimer_release", TRACY_COLOR_GRAY);
  // Decrement the low-level reference count.
  DeviceTimerRef::lowLevelDropRef(timer);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getApiVersion(int *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_getApiVersion", TRACY_COLOR_GRAY);
  return checkVal(result, 0, ctx->device().getApiVersion());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getAttribute(int *result, DeviceContext *ctx, int attr) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_getAttribute", TRACY_COLOR_GRAY);
  // Context-level attributes (values >= 1000) are dispatched to the
  // DeviceContext itself; everything else falls through to the Device.
  if (attr >= 1000)
    return checkVal(
        result, 0,
        ctx->getAttribute(static_cast<DeviceContext::Attribute>(attr)));
  return checkVal(
      result, 0,
      ctx->device().getAttribute(static_cast<Device::Attribute>(attr)));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_isCompatible(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_isCompatible", TRACY_COLOR_GRAY);
  return checkSuccess(ctx->device().isCompatible());
}

MODULAR_EXPORT int64_t AsyncRT_DeviceContext_id(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_id", TRACY_COLOR_GRAY);
  return ctx->device().id();
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_computeCapability(int32_t *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_computeCapability",
                       TRACY_COLOR_GRAY);
  int compute;
  const char *errMsg = checkVal(&compute, 0, ctx->device().computeCapability());
  *result = compute;
  return errMsg;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_getMemoryInfo(DeviceContext *ctx, size_t *free,
                                    size_t *total) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_getMemoryInfo", TRACY_COLOR_GRAY);
  Device::MemInfo memInfo;
  const char *errMsg =
      checkVal(&memInfo, Device::MemInfo(), ctx->getMemoryInfo());
  *free = memInfo.free;
  *total = memInfo.total;
  return errMsg;
}

MODULAR_EXPORT const char *AsyncRT_occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, DeviceFunction *func, int blockSize,
    size_t dynamicSharedMemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_occupancyMaxActiveBlocksPerMultiprocessor",
                       TRACY_COLOR_GRAY);
  auto ctx = &func->context();
  return checkSuccess(ctx->occupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, *func, blockSize, dynamicSharedMemSize));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_canAccess(bool *result, DeviceContext *ctx,
                                DeviceContext *peer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_canAccess", TRACY_COLOR_GRAY);
  return checkVal(result, false, ctx->device().canAccess(peer->device()));
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_enablePeerAccess(DeviceContext *ctx,
                                       DeviceContext *peer) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enablePeerAccess",
                       TRACY_COLOR_GRAY);
  return checkSuccess(ctx->device().enablePeerAccess(peer->device()));
}

// Enable peer access for all CUDA devices.
MODULAR_EXPORT const char *AsyncRT_DeviceContext_enableAllPeerAccess() {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_enableAllPeerAccess",
                       TRACY_COLOR_GRAY);
  return checkSuccess(Device::enableAllPeerAccess());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_allPeerAccessEnabled(bool *result) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_allPeerAccessEnabled",
                       TRACY_COLOR_GRAY);
  return checkVal(result, false, Device::allPeerAccessEnabled());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_supportsMulticast(bool *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_supportsMulticast",
                       TRACY_COLOR_GRAY);
  return checkVal(result, false, ctx->device().supportsMulticast());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_isHostUnified(bool *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_isHostUnified", TRACY_COLOR_GRAY);
  return checkVal(result, false, ctx->device().isHostUnified());
}

MODULAR_EXPORT int32_t AsyncRT_DeviceContext_numberOfDevices(const char *kind) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_numberOfDevices",
                       TRACY_COLOR_GRAY);
  return Device::numberOfDevices(kind);
}

MODULAR_EXPORT const char *
AsyncRT_DeviceMulticastBuffer_allocate(DeviceMulticastBuffer **result,
                                       size_t ctxsLen, DeviceContext **ctxs,
                                       size_t len, size_t elemSize) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceMulticastBuffer_allocate",
                       TRACY_COLOR_GRAY);
  std::vector<DeviceContextRef> contexts;
  for (size_t i = 0; i < ctxsLen; ++i) {
    contexts.push_back(DeviceContextRef::copy(ctxs[i]));
  }
  return checkRef(result,
                  DeviceMulticastBuffer::allocate(contexts, len * elemSize));
}

MODULAR_EXPORT const char *AsyncRT_DeviceMulticastBuffer_unicastBufferFor(
    DeviceBuffer **result, void **devicePtr,
    const DeviceMulticastBuffer *multiBuffer, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceMulticastBuffer_unicastBufferFor",
                       TRACY_COLOR_GRAY);
  const char *errMsg = checkRef(result, multiBuffer->unicastBufferFor(*ctx));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

MODULAR_EXPORT const char *AsyncRT_DeviceMulticastBuffer_multicastBufferFor(
    DeviceBuffer **result, void **devicePtr,
    const DeviceMulticastBuffer *multiBuffer, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceMulticastBuffer_multicastBufferFor",
                       TRACY_COLOR_GRAY);
  const char *errMsg = checkRef(result, multiBuffer->multicastBuffer(*ctx));
  if (errMsg == nullptr) {
    *devicePtr = (*result)->ptr();
  }
  return errMsg;
}

//
// Raw CUDA accessors
//
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_cuda_context(CUcontext *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_cuda_context", TRACY_COLOR_GRAY);
  const CUDADeviceContext *verifiedCtx = CUDADeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr) {
    return raiseError("Not a CUDAContext");
  }
  *result = verifiedCtx->cudaContext();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_cuda_stream(CUstream *result, DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_cuda_stream", TRACY_COLOR_GRAY);
  const CUDADeviceStream *verifiedStream =
      CUDADeviceStream::verified_cast(stream);
  if (verifiedStream == nullptr) {
    return raiseError("Not a CUDAStream");
  }
  *result = verifiedStream->cudaStream();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_cuda_module(CUmodule *result, DeviceFunction *func) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_cuda_module", TRACY_COLOR_GRAY);
  // Check if the device context is CUDA
  if (func->context().device().api() != Device::cudaAPI) {
    return raiseError("Not a CUDA DeviceFunction");
  }

  const CUDADeviceFunction &cudaFunc = CUDADeviceFunction::cast(*func);
  *result = cudaFunc.cudaModule();
  return nullptr;
}

MODULAR_EXPORT const char *AsyncRT_cuda_tensorMapEncodeTiled(
    void *tensorMap, int32_t tensorDataType, uint32_t tensorRank,
    DeviceBuffer *globalAddress, const uint64_t *globalDim,
    const uint64_t *globalStrides, const uint32_t *boxDim,
    const uint32_t *elementStrides, int32_t interleave, int32_t swizzle,
    int32_t l2Promotion, int32_t oobFill) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_cuda_tensorMapEncodeTiled", TRACY_COLOR_GRAY);
  // Extract the device context from the device buffer. Resolve through
  // backendContext() so a buffer wrapped by a decorating context (e.g. a
  // device-graph RecordingDeviceContext) encodes against the driver-backed
  // context it decorates; downcasting the decorator itself is UB.
  DeviceContext &bufCtx = globalAddress->context().backendContext();
  if (CUDADeviceContext::verified_cast(&bufCtx) == nullptr) {
    return raiseError("Not a CUDADeviceBuffer");
  }
  CUDADeviceContext &ctx = CUDADeviceContext::cast(bufCtx);
  return checkSuccess(ctx.cudaTensorMapEncodeTiled(
      tensorMap, static_cast<CUtensorMapDataType>(tensorDataType), tensorRank,
      *globalAddress, globalDim, globalStrides, boxDim, elementStrides,
      static_cast<CUtensorMapInterleave>(interleave),
      static_cast<CUtensorMapSwizzle>(swizzle),
      static_cast<CUtensorMapL2promotion>(l2Promotion),
      static_cast<CUtensorMapFloatOOBfill>(oobFill)));
}

MODULAR_EXPORT const char *AsyncRT_cuda_tensorMapEncodeIm2col(
    void *tensorMap, int32_t tensorDataType, uint32_t tensorRank,
    DeviceBuffer *globalAddress, const uint64_t *globalDim,
    const uint64_t *globalStrides, const int32_t *pixelBoxLowerCorner,
    const int32_t *pixelBoxUpperCorner, uint32_t channelsPerPixel,
    uint32_t pixelsPerColumn, const uint32_t *elementStrides,
    int32_t interleave, int32_t swizzle, int32_t l2Promotion, int32_t oobFill) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_cuda_tensorMapEncodeIm2col", TRACY_COLOR_GRAY);
  // Extract the device context from the device buffer. See
  // AsyncRT_cuda_tensorMapEncodeTiled for why this resolves through
  // backendContext().
  DeviceContext &bufCtx = globalAddress->context().backendContext();
  if (CUDADeviceContext::verified_cast(&bufCtx) == nullptr) {
    return raiseError("Not a CUDADeviceBuffer");
  }
  CUDADeviceContext &ctx = CUDADeviceContext::cast(bufCtx);
  return checkSuccess(ctx.cudaTensorMapEncodeIm2col(
      tensorMap, static_cast<CUtensorMapDataType>(tensorDataType), tensorRank,
      *globalAddress, globalDim, globalStrides, pixelBoxLowerCorner,
      pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides,
      static_cast<CUtensorMapInterleave>(interleave),
      static_cast<CUtensorMapSwizzle>(swizzle),
      static_cast<CUtensorMapL2promotion>(l2Promotion),
      static_cast<CUtensorMapFloatOOBfill>(oobFill)));
}

//
// Raw HIP accessors
//
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_hip_device(hipDevice_t *result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_hip_device", TRACY_COLOR_GRAY);
  const HIPDeviceContext *verifiedCtx = HIPDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr) {
    return raiseError("Not a HIPContext");
  }
  *result = verifiedCtx->hipDevice();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceStream_hip_stream(hipStream_t *result, DeviceStream *stream) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceStream_hip_stream", TRACY_COLOR_GRAY);
  const HIPDeviceStream *verifiedStream =
      HIPDeviceStream::verified_cast(stream);
  if (verifiedStream == nullptr) {
    return raiseError("Not a HIPStream");
  }
  *result = verifiedStream->hipStream();
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceFunction_hip_module(hipModule_t *result, DeviceFunction *func) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceFunction_hip_module", TRACY_COLOR_GRAY);
  // Check if the device context is HIP
  if (func->context().device().api() != Device::hipAPI) {
    return raiseError("Not a HIP DeviceFunction");
  }

  const HIPDeviceFunction &hipFunc = HIPDeviceFunction::cast(*func);
  *result = hipFunc.hipModule();
  return nullptr;
}

//
// Raw Metal accessors
//
namespace MTL {
class Device;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_metal_device(MTL::Device **result, DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_metal_device", TRACY_COLOR_GRAY);
#if MODULAR_MACOS
  const MetalDeviceContext *verifiedCtx =
      MetalDeviceContext::verified_cast(ctx);
  if (verifiedCtx == nullptr)
    return raiseError("Not a MetalContext");
  *result = verifiedCtx->metalDevice();
  return nullptr;
#else
  return raiseError("Not a MetalContext");
#endif
}

//===----------------------------------------------------------------------===//
// DeviceGraph / DeviceGraphBuilder
//===----------------------------------------------------------------------===//

MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createGraphBuilder(DeviceGraphBuilder **result,
                                         DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createGraphBuilder",
                       TRACY_COLOR_GRAY);
  return checkRef(result, ctx->createGraphBuilder());
}

// Creates a graph memory pool bound to `ctx`'s device. Infallible; ownership
// of the returned pool (refcount 1) transfers to Mojo, released via
// AsyncRT_DeviceGraphMemoryPool_release.
MODULAR_EXPORT DeviceGraphMemoryPool *
AsyncRT_DeviceContext_createGraphMemoryPool(DeviceContext *ctx) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createGraphMemoryPool",
                       TRACY_COLOR_GRAY);
  return DeviceGraphMemoryPoolRef::create(DeviceContextRef::copy(ctx))
      .release();
}

MODULAR_EXPORT void
AsyncRT_DeviceGraphMemoryPool_retain(DeviceGraphMemoryPool *pool) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphMemoryPool_retain",
                       TRACY_COLOR_GRAY);
  DeviceGraphMemoryPoolRef::lowLevelAddRef(pool);
}

MODULAR_EXPORT void
AsyncRT_DeviceGraphMemoryPool_release(DeviceGraphMemoryPool *pool) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphMemoryPool_release",
                       TRACY_COLOR_GRAY);
  DeviceGraphMemoryPoolRef::lowLevelDropRef(pool);
}

// Variant of createGraphBuilder that allocates from an existing shared pool
// instead of a builder-private one, so multiple graphs share activation
// memory. `pool` is borrowed: the builder takes its own reference, and the
// Mojo caller keeps its handle (typically stored in the device graph cache).
// Graphs sharing a pool must replay serially in recording order.
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_createGraphBuilderWithPool(DeviceGraphBuilder **result,
                                                 DeviceContext *ctx,
                                                 DeviceGraphMemoryPool *pool) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_createGraphBuilderWithPool",
                       TRACY_COLOR_GRAY);
  if (&pool->device() != &ctx->device()) {
    *result = nullptr;
    return raiseError("device graph memory pool belongs to a different device "
                      "than the context");
  }
  return checkRef(
      result, ctx->createGraphBuilder(DeviceGraphMemoryPoolRef::copy(pool)));
}

// Returns a DeviceContext facade that records operations enqueued through it
// into `builder` instead of executing them eagerly. `seedNodeId` is the node
// that recorded operations chain after — the Mojo layer seeds it with the
// ambient/region dependencies, since the builder does not model dependencies
// itself. Ownership of the returned context is transferred to Mojo (released
// via AsyncRT_DeviceContext_release).
MODULAR_EXPORT const char *AsyncRT_DeviceGraphBuilder_recordingContext(
    DeviceContext **result, DeviceGraphBuilder *builder, int32_t seedNodeId) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_recordingContext",
                       TRACY_COLOR_GRAY);
  return checkRef(result,
                  builder->recordingContext(DeviceGraphNode{seedNodeId}));
}

// Reseats the chain root of a memoized recording context before the Mojo
// builder hands it out again; see `DeviceGraphBuilder.recording_context` in
// device_graph.mojo.
MODULAR_EXPORT const char *
AsyncRT_DeviceContext_reseedRecording(DeviceContext *ctx, int32_t seedNodeId) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceContext_reseedRecording",
                       TRACY_COLOR_GRAY);
  RecordingDeviceContext *rc = RecordingDeviceContext::verified_cast(ctx);
  if (!rc)
    return raiseError("reseed target is not a recording device context");
  rc->reseed(DeviceGraphNode{seedNodeId});
  return nullptr;
}

MODULAR_EXPORT void
AsyncRT_DeviceGraphBuilder_release(DeviceGraphBuilder *builder) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_release", TRACY_COLOR_GRAY);
  DeviceGraphBuilderRef::lowLevelDropRef(builder);
}

namespace {
// Converts a (depIds, numDeps) pair from the C ABI into an
// `llvm::ArrayRef<DeviceGraphNode>` suitable for the builder's add-node
// methods. `numDeps` is the (non-negative) number of node ids; when
// `numDeps == 0`, `depIds` may be null (the result is an empty ArrayRef,
// which the builder treats as "graph root, no predecessors").
// `DeviceGraphNode` is a single-`int32_t`-field standard-layout struct, so
// the depIds pointer is `reinterpret_cast` here.
llvm::ArrayRef<DeviceGraphNode> makeDepsArrayRef(const int32_t *depIds,
                                                 int64_t numDeps) {
  static_assert(sizeof(DeviceGraphNode) == sizeof(int32_t),
                "wire-format assumes DeviceGraphNode is layout-compatible "
                "with int32_t");
  static_assert(std::is_standard_layout_v<DeviceGraphNode>,
                "wire-format assumes DeviceGraphNode has standard layout");
  return llvm::ArrayRef<DeviceGraphNode>(
      reinterpret_cast<const DeviceGraphNode *>(depIds),
      static_cast<size_t>(numDeps));
}
} // namespace

// Adds a kernel dispatch node to the graph under construction.
//
// The first 14 parameters mirror AsyncRT_DeviceContext_enqueueFunctionDirect
// and AsyncRT_DeviceStream_enqueueFunctionDirect because the Mojo
// `_FunctionEnqueuer` trait routes all three through the same arg-packing
// path (`DeviceFunction._call_with_pack[_checked]`). argSizes is accepted
// to keep that shared parameter layout but is unused by the graph builder
// (the CUDA/HIP graph APIs do not require it).
//
// The trailing (depIds, numDeps) pair carries the explicit dependency list
// from Mojo into the C++ graph builder; see `makeDepsArrayRef` for the
// wire convention. On the Mojo side these are filled in by a transient
// `_DeviceGraphBuilderEnqueuer` that wraps a borrowed `DeviceGraphBuilder`
// plus a `dependencies` argument passed to `DeviceGraphBuilder.add_function`.
MODULAR_EXPORT const char *AsyncRT_DeviceGraphBuilder_addFunctionDirect(
    DeviceGraphBuilder *builder, DeviceFunction *func, uint32_t gridX,
    uint32_t gridY, uint32_t gridZ, uint32_t blockX, uint32_t blockY,
    uint32_t blockZ, uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs,
    void **args, uint32_t argCount, const size_t * /*argSizes*/,
    const int32_t *depIds, int64_t numDeps) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addFunctionDirect",
                       TRACY_COLOR_GRAY);
  DeviceFunctionLaunchParams params;
  params.gridX = gridX;
  params.gridY = gridY;
  params.gridZ = gridZ;
  params.blockX = blockX;
  params.blockY = blockY;
  params.blockZ = blockZ;
  params.sharedMemBytes = sharedMemBytes;
  params.attrs = attrs;
  params.numAttrs = numAttrs;
  params.args = args;
  params.argCount = argCount;

  // Node handle is intentionally discarded; callers retrieve it via
  // AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone.
  ErrorOr<DeviceGraphNode> nodeOrErr = builder->addFunctionDirect(
      *func, params, makeDepsArrayRef(depIds, numDeps));
  if (nodeOrErr.isError())
    return raiseError(nodeOrErr.getError());
  return nullptr;
}

// Memcpy and memset graph-builder nodes. Node IDs are intentionally discarded
// to match AsyncRT_DeviceGraphBuilder_addFunctionDirect; callers retrieve the
// id via AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone. The trailing (depIds,
// numDeps) pair carries the explicit dependency list — see `makeDepsArrayRef`.

MODULAR_EXPORT const char *AsyncRT_DeviceGraphBuilder_addCopyHostToDevice(
    DeviceGraphBuilder *builder, DeviceBuffer *dst, const void *src,
    const int32_t *depIds, int64_t numDeps) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addCopyHostToDevice",
                       TRACY_COLOR_GRAY);
  ErrorOr<DeviceGraphNode> nodeOrErr = builder->addCopyHostToDevice(
      *dst, src, makeDepsArrayRef(depIds, numDeps));
  if (nodeOrErr.isError())
    return raiseError(nodeOrErr.getError());
  return nullptr;
}

MODULAR_EXPORT const char *AsyncRT_DeviceGraphBuilder_addCopyDeviceToHost(
    DeviceGraphBuilder *builder, void *dst, DeviceBuffer *src,
    const int32_t *depIds, int64_t numDeps) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addCopyDeviceToHost",
                       TRACY_COLOR_GRAY);
  ErrorOr<DeviceGraphNode> nodeOrErr = builder->addCopyDeviceToHost(
      dst, *src, makeDepsArrayRef(depIds, numDeps));
  if (nodeOrErr.isError())
    return raiseError(nodeOrErr.getError());
  return nullptr;
}

MODULAR_EXPORT const char *AsyncRT_DeviceGraphBuilder_addCopyDeviceToDevice(
    DeviceGraphBuilder *builder, DeviceBuffer *dst, DeviceBuffer *src,
    const int32_t *depIds, int64_t numDeps) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addCopyDeviceToDevice",
                       TRACY_COLOR_GRAY);
  ErrorOr<DeviceGraphNode> nodeOrErr = builder->addCopyDeviceToDevice(
      *dst, *src, makeDepsArrayRef(depIds, numDeps));
  if (nodeOrErr.isError())
    return raiseError(nodeOrErr.getError());
  return nullptr;
}

MODULAR_EXPORT const char *AsyncRT_DeviceGraphBuilder_addSetMemory(
    DeviceGraphBuilder *builder, DeviceBuffer *dst, uint64_t val,
    size_t valSize, const int32_t *depIds, int64_t numDeps) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addSetMemory",
                       TRACY_COLOR_GRAY);
  ErrorOr<DeviceGraphNode> nodeOrErr = builder->addSetMemory(
      *dst, val, valSize, makeDepsArrayRef(depIds, numDeps));
  if (nodeOrErr.isError())
    return raiseError(nodeOrErr.getError());
  return nullptr;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceGraphBuilder_addEmpty(DeviceGraphBuilder *builder,
                                    const int32_t *depIds, int64_t numDeps) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addEmpty", TRACY_COLOR_GRAY);
  ErrorOr<DeviceGraphNode> nodeOrErr =
      builder->addEmpty(makeDepsArrayRef(depIds, numDeps));
  if (nodeOrErr.isError())
    return raiseError(nodeOrErr.getError());
  return nullptr;
}

MODULAR_EXPORT void
AsyncRT_DeviceGraphBuilder_addOutput(DeviceGraphBuilder *builder,
                                     AsyncRT::AsyncValue *output) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addOutput",
                       TRACY_COLOR_GRAY);
  // Adopt the reference the Mojo `AnyAsyncValueRef` surrendered net-zero.
  builder->addOutput(AnyAsyncValueRef::take(output));
}

// Returns the number of outputs registered on the device graph. Cannot fail.
MODULAR_EXPORT int64_t
AsyncRT_DeviceGraphBuilder_numOutputs(DeviceGraphBuilder *builder) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_numOutputs",
                       TRACY_COLOR_GRAY);
  return static_cast<int64_t>(builder->numOutputs());
}

MODULAR_EXPORT void
AsyncRT_DeviceGraphBuilder_addInput(DeviceGraphBuilder *builder,
                                    DeviceBuffer *input) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addInput", TRACY_COLOR_GRAY);
  // COPY (addRef), unlike `addOutput`: the handle is borrowed rather than
  // surrendered, because the Mojo caller keeps using the buffer it just
  // allocated as the graph's stable input.
  builder->addInput(DeviceBufferRef::copy(input));
}

// Registers an in-place input marker: the graph records the caller's live
// address at this position and replay must not copy into it. See
// `DeviceGraphBuilder::addInPlaceInput`. Cannot fail.
MODULAR_EXPORT void
AsyncRT_DeviceGraphBuilder_addInPlaceInput(DeviceGraphBuilder *builder) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addInPlaceInput",
                       TRACY_COLOR_GRAY);
  builder->addInPlaceInput();
}

// Returns the number of stable inputs registered on the device graph. Cannot
// fail.
MODULAR_EXPORT int64_t
AsyncRT_DeviceGraphBuilder_numInputs(DeviceGraphBuilder *builder) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_numInputs",
                       TRACY_COLOR_GRAY);
  return static_cast<int64_t>(builder->numInputs());
}

// Records that the graph publishes `ptr` into the stable address slot at
// `stableAddress`; both are device addresses. See
// `DeviceGraphBuilder::addStableAssignment`. Cannot fail.
MODULAR_EXPORT void AsyncRT_DeviceGraphBuilder_addStableAssignment(
    DeviceGraphBuilder *builder, void **stableAddress, void *ptr) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_addStableAssignment",
                       TRACY_COLOR_GRAY);
  builder->addStableAssignment(stableAddress, ptr);
}

// Returns the id of the most recently added node, or -1 if no nodes have
// been added yet. Cannot fail. The -1 sentinel mirrors the lower bound of
// the Mojo-side scope helper which uses (start_id, end_id] to enumerate
// the ids of nodes added during a region; non-scope callers (the various
// `add_*` helpers in `DeviceGraphBuilder.mojo`) call this immediately
// after a successful add and treat -1 as a logic error.
MODULAR_EXPORT int32_t
AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone(DeviceGraphBuilder *builder) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone",
                       TRACY_COLOR_GRAY);
  std::optional<DeviceGraphNode> node = builder->lastNode();
  return node ? node->id : -1;
}

MODULAR_EXPORT const char *
AsyncRT_DeviceGraphBuilder_instantiate(DeviceGraph **result,
                                       DeviceGraphBuilder *builder) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraphBuilder_instantiate",
                       TRACY_COLOR_GRAY);
  return checkRef(result, builder->instantiate());
}

MODULAR_EXPORT void AsyncRT_DeviceGraph_retain(DeviceGraph *graph) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraph_retain", TRACY_COLOR_GRAY);
  DeviceGraphRef::lowLevelAddRef(graph);
}

MODULAR_EXPORT void AsyncRT_DeviceGraph_release(DeviceGraph *graph) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraph_release", TRACY_COLOR_GRAY);
  DeviceGraphRef::lowLevelDropRef(graph);
}

MODULAR_EXPORT const char *AsyncRT_DeviceGraph_replay(DeviceGraph *graph) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraph_replay", TRACY_COLOR_GRAY);
  return checkSuccess(graph->replay());
}

// Enqueues the graph's stable address slot writes on its context; the caller
// synchronizes before launching any graph that reads the slots. See
// `DeviceGraph::patchStableAddresses`.
MODULAR_EXPORT const char *
AsyncRT_DeviceGraph_patchStableAddresses(DeviceGraph *graph) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraph_patchStableAddresses",
                       TRACY_COLOR_GRAY);
  return checkSuccess(graph->patchStableAddresses());
}

MODULAR_EXPORT const char *
AsyncRT_DeviceGraph_createBuffer(DeviceBuffer **result, void **devicePtr,
                                 DeviceGraphBuilder *builder, size_t len,
                                 size_t elemSize, bool isHost) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_DeviceGraph_createBuffer", TRACY_COLOR_GRAY);
  const char *errMsg =
      checkRef(result, builder->createBuffer(len * elemSize, isHost));
  if (errMsg == nullptr)
    *devicePtr = (*result)->ptr();
  return errMsg;
}

//
// BinaryID
//
MODULAR_CXX_EXPORT std::string M::Driver::getMojoBindingsBinaryID() {
  // M::getBinaryID() returns the binary ID of the shared library that contains
  // it. For the purposes of MEF cache invalidation, we need to know when
  // there's been a change in these shared libraries.
  return M::getBinaryID();
}
