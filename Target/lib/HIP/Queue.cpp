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
#include "llvm/ADT/Twine.h"

#include <algorithm>

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_queue_t> M_driver_queue::create(M_driver_context &ctx) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  HIPApi &hip = dev.getDriver().hip();

  int constexpr kDefaultHipPriority = 0;

  hipStream_t stream;
  if (hipError_t r = hip.hipStreamCreateWithPriority(
          &stream, hipStreamNonBlocking, kDefaultHipPriority);
      r != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&dev.getDriver(), r, "hipStreamCreateWithPriority"));

  return new M_driver_queue(ctx, stream);
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;

  M_driver_device &dev = queue->context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipStreamDestroy(queue->stream);
  delete queue;
  if (result != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), result, "hipStreamDestroy");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst,
                                               uint64_t dstOffset,
                                               const void *src, uint64_t size) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  if (hipError_t r =
          hip.hipMemcpyAsync(dstPtr, src, size, hipMemcpyHostToDevice, stream);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpyAsync HtoD");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t srcOffset,
                                                 uint64_t size) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r =
          hip.hipMemcpyAsync(dst, srcPtr, size, hipMemcpyDeviceToHost, stream);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpyAsync DtoH");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyIntraDevice(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  M_driver_memory_t src,
                                                  uint64_t srcOffset,
                                                  uint64_t size) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r = hip.hipMemcpyAsync(dstPtr, srcPtr, size,
                                        hipMemcpyDeviceToDevice, stream);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpyAsync DtoD");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_queue::copyInterDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                M_driver_memory_t src, uint64_t srcOffset,
                                uint64_t size, M_driver_context_t srcContext) {
  // The copy is enqueued on this queue, which belongs to the destination's
  // context, so make that device current.
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r = hip.hipMemcpyPeerAsync(
          dstPtr, dev.id(), srcPtr, srcContext->getDevice().id(), size, stream);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpyPeerAsync");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t dstOffset, uint64_t size,
                                            uint8_t value) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;

  if (hipError_t r =
          hip.hipMemsetAsync(dstPtr, static_cast<int>(value), size, stream);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemsetAsync");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::fillMemory(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (valueSize == 1)
    return setMemory(dst, dstOffset, size, static_cast<uint8_t>(value));

  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;

  switch (valueSize) {
  case 2: {
    if ((size & 1) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (hipError_t r = hip.hipMemsetD16Async(
            dstPtr, static_cast<unsigned short>(value), size >> 1, stream);
        r != hipSuccess)
      return hipErrorStatus(&dev.getDriver(), r, "hipMemsetD16Async");
    break;
  }
  case 4: {
    if ((size & 3) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (hipError_t r = hip.hipMemsetD32Async(dstPtr, static_cast<int>(value),
                                             size >> 2, stream);
        r != hipSuccess)
      return hipErrorStatus(&dev.getDriver(), r, "hipMemsetD32Async");
    break;
  }
  case 8: {
    if ((size & 7) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    uint32_t hiBits = static_cast<uint32_t>(value >> 32);
    uint32_t loBits = static_cast<uint32_t>(value & 0xfffffffful);
    if (hiBits == loBits) {
      if (hipError_t r = hip.hipMemsetD32Async(dstPtr, static_cast<int>(loBits),
                                               size >> 2, stream);
          r != hipSuccess)
        return hipErrorStatus(&dev.getDriver(), r,
                              "hipMemsetD32Async (uint64 broadcast)");
    } else {
      size_t len = size >> 3;
      if (len > 0) {
        // Seed the first 8 bytes with the uint64 pattern by writing its
        // low and high 32-bit halves, then fill the rest via doubling
        // device-to-device copies (log2(len) calls instead of O(len)).
        //
        // `hipMemsetD2D32Async` would do this in 2 calls, but ROCm
        // forwards `width` unchanged into byte-valued internal fields
        // (sizeBytes = width * height * depth), making the documented
        // "width in 32-bit elements" contract unusable: width=1 aborts
        // with "Pattern Size: 4 cannot be greater than fill size: 1",
        // and width=4 (bytes) silently fails to fill rows 1..N-1 on
        // pinned host memory. This bug is unchanged in ROCm 7.1.1,
        // 7.2.0, 7.2.1, and 7.2.2.
        void *hiPtr = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(dstPtr) + sizeof(uint32_t));
        if (hipError_t r = hip.hipMemsetD32Async(dstPtr, loBits, 1, stream);
            r != hipSuccess)
          return hipErrorStatus(&dev.getDriver(), r,
                                "hipMemsetD32Async (uint64 low half seed)");
        if (hipError_t r = hip.hipMemsetD32Async(hiPtr, hiBits, 1, stream);
            r != hipSuccess)
          return hipErrorStatus(&dev.getDriver(), r,
                                "hipMemsetD32Async (uint64 high half seed)");
        for (size_t filled = 1; filled < len;) {
          size_t copyCount = std::min(filled, len - filled);
          void *dstCopy = reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(dstPtr) + filled * sizeof(uint64_t));
          if (hipError_t r = hip.hipMemcpyAsync(
                  dstCopy, dstPtr, copyCount * sizeof(uint64_t),
                  hipMemcpyDeviceToDevice, stream);
              r != hipSuccess)
            return hipErrorStatus(
                &dev.getDriver(), r,
                "hipMemcpyAsync DtoD (uint64 broadcast doubling)");
          filled += copyCount;
        }
      }
    }
    break;
  }
  default:
    return M_DRIVER_STATUS_INVALID_ARG;
  }

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::recordEvent(M_driver_event_t event) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  if (hipError_t r = hip.hipEventRecord(event->event, stream); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipEventRecord");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitForEvents(M_driver_event_t *events,
                                                uint32_t num_events) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  for (uint32_t i = 0; i < num_events; ++i) {
    if (hipError_t r = hip.hipStreamWaitEvent(stream, events[i]->event, 0);
        r != hipSuccess)
      return hipErrorStatus(&dev.getDriver(), r,
                            "hipStreamWaitEvent event " + llvm::Twine(i));
  }

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::synchronize() {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  if (hipError_t r = hip.hipStreamSynchronize(stream); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipStreamSynchronize");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::launchHostFunc(void (*fn)(void *),
                                                 void *user_data) {
  if (fn == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_driver &driver = context.getDevice().getDriver();
  if (hipError_t r = driver.hip().hipLaunchHostFunc(stream, fn, user_data);
      r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipLaunchHostFunc");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitValue64(uint64_t device_address,
                                              uint64_t value) {
  M_driver_driver &driver = context.getDevice().getDriver();
  if (hipError_t r = driver.hip().hipStreamWaitValue64(
          stream, reinterpret_cast<hipDeviceptr_t>(device_address), value,
          hipStreamWaitValueEq, /*mask=*/~0ULL);
      r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipStreamWaitValue64");
  return M_DRIVER_STATUS_SUCCESS;
}
