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

#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "Error.h"
#include "Memory.h"

#include <algorithm>
#include <new>

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_context_t>
M_driver_context::create(M_driver_device &dev) {
  return new M_driver_context(dev);
}

DriverResult<M_driver_memory_t> M_driver_context::allocPinned(uint64_t size) {
  if (size == 0)
    return nullptr;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  HIPApi &hip = dev.getDriver().hip();
  void *ptr = nullptr;
  hipError_t result = hip.hipHostMalloc(&ptr, size, hipHostMallocDefault);
  if (result == hipErrorOutOfMemory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&dev.getDriver(), result, "hipHostMalloc"));

  auto *memory = new (std::nothrow) M_driver_memory(ptr);
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  // Pinned host memory is host-dereferenceable.
  memory->hostAccessible = true;

  return memory;
}

DriverResult<M_driver_memory_t> M_driver_context::allocSync(uint64_t size) {
  if (size == 0)
    return nullptr;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  HIPApi &hip = dev.getDriver().hip();
  void *ptr = nullptr;
  hipError_t result = hip.hipMalloc(&ptr, size);
  if (result == hipErrorOutOfMemory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&dev.getDriver(), result, "hipMalloc"));

  auto *memory = new (std::nothrow) M_driver_memory(ptr);
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  return memory;
}

DriverResult<M_driver_memory_t>
M_driver_context::wrapSync(uint64_t address, uint64_t size, bool owning) {
  (void)size;
  if (address == 0)
    return tl::unexpected(M_DRIVER_STATUS_INVALID_ARG);
  auto *memory = new (std::nothrow)
      M_driver_memory{reinterpret_cast<void *>(address), owning};
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

M_driver_status_t M_driver_context::freePinned(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  void *ptr = memory->ptr;
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;

  HIPApi &hip = dev.getDriver().hip();
  if (hipError_t r = hip.hipHostFree(ptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipHostFree");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::freeSync(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  void *ptr = memory->ptr;
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;

  HIPApi &hip = dev.getDriver().hip();
  if (hipError_t r = hip.hipFree(ptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipFree");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyToDeviceSync(M_driver_memory_t dst,
                                                     uint64_t dstOffset,
                                                     const void *src,
                                                     uint64_t size) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();
  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  if (hipError_t r = hip.hipMemcpy(dstPtr, src, size, hipMemcpyHostToDevice);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpy HtoD");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyFromDeviceSync(void *dst,
                                                       M_driver_memory_t src,
                                                       uint64_t srcOffset,
                                                       uint64_t size) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();
  HIPApi &hip = dev.getDriver().hip();
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r = hip.hipMemcpy(dst, srcPtr, size, hipMemcpyDeviceToHost);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpy DtoH");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyIntraDeviceSync(M_driver_memory_t dst,
                                                        uint64_t dstOffset,
                                                        M_driver_memory_t src,
                                                        uint64_t srcOffset,
                                                        uint64_t size) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();
  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r =
          hip.hipMemcpy(dstPtr, srcPtr, size, hipMemcpyDeviceToDevice);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpy DtoD");
  // hipMemcpy is synchronous, but wait on the default stream so the blocking
  // contract holds by construction (matching the CUDA sync path).
  if (hipError_t r = hip.hipStreamSynchronize(nullptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r,
                          "hipStreamSynchronize (sync D2D)");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyInterDeviceSync(
    M_driver_memory_t dst, uint64_t dstOffset, M_driver_memory_t src,
    uint64_t srcOffset, uint64_t size, M_driver_context_t srcContext) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();
  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r = hip.hipMemcpyPeer(dstPtr, dev.id(), srcPtr,
                                       srcContext->getDevice().id(), size);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpyPeer");
  // Wait on the default stream so the blocking contract holds by construction
  // (matching the CUDA sync peer path).
  if (hipError_t r = hip.hipStreamSynchronize(nullptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r,
                          "hipStreamSynchronize (sync peer)");
  return M_DRIVER_STATUS_SUCCESS;
}

// HIP has no driver-level current-context object; the per-thread state is the
// current device id, which round-trips through the opaque native handle.

M_driver_status_t M_driver_context::setCurrent() {
  if (hipError_t r = device.getDriver().hip().hipSetDevice(device.id());
      r != hipSuccess)
    return hipErrorStatus(&device.getDriver(), r, "hipSetDevice");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::getCurrentDriverContext(
    M_driver_native_handle_t *driver_context) {
  if (driver_context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_driver &driver = device.getDriver();
  int currentId = -1;
  if (hipError_t r = driver.hip().hipGetDevice(&currentId); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipGetDevice");
  *driver_context = static_cast<M_driver_native_handle_t>(currentId);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::setCurrentDriverContext(
    M_driver_native_handle_t driver_context) {
  M_driver_driver &driver = device.getDriver();
  int targetId = static_cast<int>(driver_context);
  if (hipError_t r = driver.hip().hipSetDevice(targetId); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipSetDevice");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::memoryInfo(uint64_t *free,
                                               uint64_t *total) {
  if (free == nullptr || total == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_driver &driver = device.getDriver();
  // hipMemGetInfo reports the *current* device's memory. Make this context's
  // device current for the query, then restore the previous device (mirrors
  // the CUDA plugin's ScopedContext save/restore).
  int prevId = -1;
  if (hipError_t r = driver.hip().hipGetDevice(&prevId); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipGetDevice");
  if (hipError_t r = driver.hip().hipSetDevice(device.id()); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipSetDevice");
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  hipError_t r = driver.hip().hipMemGetInfo(&freeBytes, &totalBytes);
  (void)driver.hip().hipSetDevice(prevId);
  if (r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipMemGetInfo");
  *free = freeBytes;
  *total = totalBytes;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::enablePeerAccess(M_driver_context &peer) {
  M_driver_driver &driver = device.getDriver();
  HIPApi &hip = driver.hip();
  bool canAccess = false;
  if (M_driver_status_t s = device.canAccessPeer(peer.getDevice(), &canAccess);
      s != M_DRIVER_STATUS_SUCCESS)
    return s;
  if (!canAccess)
    return M_DRIVER_STATUS_INCOMPATIBLE;

  int prevId = -1;
  if (hipError_t r = hip.hipGetDevice(&prevId); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipGetDevice");
  if (hipError_t r = hip.hipSetDevice(device.id()); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipSetDevice");

  M_driver_status_t status = M_DRIVER_STATUS_SUCCESS;
  hipError_t r = hip.hipDeviceEnablePeerAccess(peer.getDevice().id(), 0);
  if (r == hipErrorPeerAccessAlreadyEnabled) {
    // Clear the sticky last error, which other HIP users (torch) would raise.
    (void)hip.hipGetLastError();
  } else if (r != hipSuccess) {
    status = hipErrorStatus(&driver, r, "hipDeviceEnablePeerAccess");
  }
  if (status == M_DRIVER_STATUS_SUCCESS) {
    hipMemPool_t pool;
    if (hipError_t r2 =
            hip.hipDeviceGetDefaultMemPool(&pool, peer.getDevice().id());
        r2 != hipSuccess) {
      status = hipErrorStatus(&driver, r2, "hipDeviceGetDefaultMemPool");
    } else {
      hipMemAccessDesc desc{};
      desc.location.type = hipMemLocationTypeDevice;
      desc.location.id = device.id();
      desc.flags = hipMemAccessFlagsProtReadWrite;
      if (hipError_t r3 = hip.hipMemPoolSetAccess(pool, &desc, 1);
          r3 != hipSuccess)
        status = hipErrorStatus(&driver, r3, "hipMemPoolSetAccess");
    }
  }
  if (prevId >= 0)
    (void)hip.hipSetDevice(prevId);
  return status;
}

M_driver_status_t M_driver_context::setMemorySync(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  uint64_t size,
                                                  uint8_t value) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;

  if (hipError_t r =
          hip.hipMemsetAsync(dstPtr, static_cast<int>(value), size, nullptr);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemsetAsync");
  if (hipError_t r = hip.hipStreamSynchronize(nullptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r,
                          "hipStreamSynchronize (sync memset)");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::fillSync(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (valueSize == 1)
    return setMemorySync(dst, dstOffset, size, static_cast<uint8_t>(value));

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;

  switch (valueSize) {
  case 2: {
    if ((size & 1) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (hipError_t r = hip.hipMemsetD16Async(
            dstPtr, static_cast<unsigned short>(value), size >> 1, nullptr);
        r != hipSuccess)
      return hipErrorStatus(&dev.getDriver(), r, "hipMemsetD16Async");
    break;
  }
  case 4: {
    if ((size & 3) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (hipError_t r = hip.hipMemsetD32Async(dstPtr, static_cast<int>(value),
                                             size >> 2, nullptr);
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
                                               size >> 2, nullptr);
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
        if (hipError_t r = hip.hipMemsetD32Async(dstPtr, loBits, 1, nullptr);
            r != hipSuccess)
          return hipErrorStatus(&dev.getDriver(), r,
                                "hipMemsetD32Async (uint64 low half seed)");
        if (hipError_t r = hip.hipMemsetD32Async(hiPtr, hiBits, 1, nullptr);
            r != hipSuccess)
          return hipErrorStatus(&dev.getDriver(), r,
                                "hipMemsetD32Async (uint64 high half seed)");
        for (size_t filled = 1; filled < len;) {
          size_t copyCount = std::min(filled, len - filled);
          void *dstCopy = reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(dstPtr) + filled * sizeof(uint64_t));
          if (hipError_t r = hip.hipMemcpyAsync(
                  dstCopy, dstPtr, copyCount * sizeof(uint64_t),
                  hipMemcpyDeviceToDevice, nullptr);
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

  if (hipError_t r = hip.hipStreamSynchronize(nullptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r,
                          "hipStreamSynchronize (sync fill)");
  return M_DRIVER_STATUS_SUCCESS;
}
