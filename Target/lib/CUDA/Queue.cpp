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

#include "Queue.h"
#include "Context.h"
#include "Device.h"
#include "Event.h"

using M::Target::CUDA::cudaErrorStatus;

DriverResult<M_driver_queue_t> M_driver_queue::create(M_driver_context &ctx) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  // Default priority is 0 (lowest); compatible with CUDA specs.
  int constexpr priority = 0;

  // CUDA blocking streams are a legacy feature; interpretation of "blocking"
  // is w.r.t. the default stream and does not mean "synchronous".
  // We want our streams to be independent of any other stream.
  unsigned int constexpr kNonBlocking = 0x1;

  CUstream stream;
  if (CUresult r =
          cuda.cuStreamCreateWithPriority(&stream, kNonBlocking, priority);
      r != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&ctx.device.driver, r, "cuStreamCreateWithPriority"));

  return new M_driver_queue(ctx, stream);
}

DriverResult<M_driver_memory_t> M_driver_queue::allocAsync(uint64_t size) {
  if (size == 0)
    return nullptr;

  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  auto &cuda = context.device.driver.rawCuda;

  CUdeviceptr devicePtr;
  CUresult result = cuda.cuMemAllocAsync(&devicePtr, size, stream);
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&context.device.driver, result, "cuMemAllocAsync"));
  M_driver_memory_t memory =
      new (std::nothrow) M_driver_memory(reinterpret_cast<void *>(devicePtr));
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

M_driver_status_t M_driver_queue::freeAsync(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;

  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;

  CUdeviceptr devicePtr = reinterpret_cast<CUdeviceptr>(memory->ptr);
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;
  if (CUresult r = cuda.cuMemFreeAsync(devicePtr, stream); r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuMemFreeAsync");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst,
                                               uint64_t dstOffset,
                                               const void *src, uint64_t size) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyHtoDAsync(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset, src, size,
          stream);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuMemcpyHtoDAsync");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t srcOffset,
                                                 uint64_t size) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyDtoHAsync(
          dst, reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset, size,
          stream);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuMemcpyDtoHAsync");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyIntraDevice(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  M_driver_memory_t src,
                                                  uint64_t srcOffset,
                                                  uint64_t size) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyDtoDAsync(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset,
          reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset, size, stream);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuMemcpyDtoDAsync");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_queue::copyInterDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                M_driver_memory_t src, uint64_t srcOffset,
                                uint64_t size, M_driver_context_t srcContext) {
  // The peer copy is enqueued on this queue, which belongs to the
  // destination's context, so make that context current.
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyPeerAsync(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset,
          context.primaryCtx,
          reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset,
          srcContext->primaryCtx, size, stream);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuMemcpyPeerAsync");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t dstOffset, uint64_t size,
                                            uint8_t value) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset;
  if (CUresult r = cuda.cuMemsetD8Async(dstPtr, value, size, stream);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuMemsetD8Async");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::fillMemory(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (valueSize == 1)
    return setMemory(dst, dstOffset, size, static_cast<uint8_t>(value));

  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset;
  switch (valueSize) {
  case 2:
    if ((size & 1) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (CUresult r = cuda.cuMemsetD16Async(dstPtr, static_cast<uint16_t>(value),
                                           size >> 1, stream);
        r != CUDA_SUCCESS)
      return cudaErrorStatus(&context.device.driver, r, "cuMemsetD16Async");
    break;
  case 4:
    if ((size & 3) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (CUresult r = cuda.cuMemsetD32Async(dstPtr, static_cast<uint32_t>(value),
                                           size >> 2, stream);
        r != CUDA_SUCCESS)
      return cudaErrorStatus(&context.device.driver, r, "cuMemsetD32Async");
    break;
  case 8: {
    if ((size & 7) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    uint32_t hiBits = static_cast<uint32_t>(value >> 32);
    uint32_t loBits = static_cast<uint32_t>(value & 0xfffffffful);
    if (hiBits == loBits) {
      if (CUresult r = cuda.cuMemsetD32Async(dstPtr, loBits, size >> 2, stream);
          r != CUDA_SUCCESS)
        return cudaErrorStatus(&context.device.driver, r,
                               "cuMemsetD32Async (uint64 broadcast)");
    } else {
      size_t len = size >> 3;
      CUdeviceptr hiPtr = dstPtr + sizeof(uint32_t);
      if (CUresult r =
              cuda.cuMemsetD2D32Async(dstPtr, 8, loBits, 1, len, stream);
          r != CUDA_SUCCESS)
        return cudaErrorStatus(&context.device.driver, r,
                               "cuMemsetD2D32Async (uint64 low half)");
      if (CUresult r =
              cuda.cuMemsetD2D32Async(hiPtr, 8, hiBits, 1, len, stream);
          r != CUDA_SUCCESS)
        return cudaErrorStatus(&context.device.driver, r,
                               "cuMemsetD2D32Async (uint64 high half)");
    }
    break;
  }
  default:
    return M_DRIVER_STATUS_INVALID_ARG;
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::recordEvent(M_driver_event_t event) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = context.device.driver.rawCuda;
  if (CUresult r = cuda.cuEventRecord(event->event, stream); r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuEventRecord");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitForEvents(M_driver_event_t *events,
                                                uint32_t num_events) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = context.device.driver.rawCuda;
  for (uint32_t i = 0; i < num_events; ++i)
    if (CUresult r = cuda.cuStreamWaitEvent(stream, events[i]->event, 0);
        r != CUDA_SUCCESS)
      return cudaErrorStatus(&context.device.driver, r,
                             "cuStreamWaitEvent event " + llvm::Twine(i));
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;

  ScopedContext scope(queue->context.primaryCtx,
                      queue->context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = queue->context.device.driver.rawCuda;
  if (CUresult r = cuda.cuStreamDestroy(queue->stream); r != CUDA_SUCCESS)
    return cudaErrorStatus(&queue->context.device.driver, r, "cuStreamDestroy");

  delete queue;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::synchronize() {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = context.device.driver.rawCuda;
  if (CUresult r = cuda.cuStreamSynchronize(stream); r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuStreamSynchronize");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::launchHostFunc(void (*fn)(void *),
                                                 void *user_data) {
  if (fn == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  if (CUresult r =
          context.device.driver.rawCuda.cuLaunchHostFunc(stream, fn, user_data);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuLaunchHostFunc");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitValue64(uint64_t device_address,
                                              uint64_t value) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  // FLUSH is intentionally omitted, mirroring
  // CUDADeviceStream::enqueueWaitOnHostValue: host writes are visible via the
  // device-mapped pinned alias without it.
  if (CUresult r = context.device.driver.rawCuda.cuStreamWaitValue64(
          stream, static_cast<CUdeviceptr>(device_address), value,
          CU_STREAM_WAIT_VALUE_EQ);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&context.device.driver, r, "cuStreamWaitValue64");
  return M_DRIVER_STATUS_SUCCESS;
}
