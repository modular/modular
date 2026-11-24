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
#include "Memory.h"

using M::Target::CUDA::cudaErrorStatus;

DriverResult<M_driver_context_t>
M_driver_context::create(M_driver_device &dev) {
  CUcontext primaryCtx;
  if (CUresult r = dev.driver.rawCuda.cuDevicePrimaryCtxRetain(&primaryCtx,
                                                               dev.deviceId);
      r != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&dev.driver, r, "cuDevicePrimaryCtxRetain"));

  return new M_driver_context(dev, primaryCtx);
}

M_driver_status_t M_driver_context::destroy(M_driver_context_t ctx) {
  if (CUresult r = ctx->device.driver.rawCuda.cuDevicePrimaryCtxRelease(
          ctx->device.deviceId);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&ctx->device.driver, r, "cuDevicePrimaryCtxRelease");
  delete ctx;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_memory_t> M_driver_context::allocPinned(uint64_t size) {
  if (size == 0)
    return nullptr;
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  auto &driver = device.driver.rawCuda;
  void *memAlloc = nullptr;
  CUresult result = driver.cuMemAllocHost(&memAlloc, size);
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&device.driver, result, "cuMemAllocHost"));
  M_driver_memory_t memory = new (std::nothrow) M_driver_memory(memAlloc);
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  memory->hostAccessible = true;
  return memory;
}

DriverResult<M_driver_memory_t> M_driver_context::allocSync(uint64_t size) {
  if (size == 0)
    return nullptr;
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  auto &driver = device.driver.rawCuda;
  CUdeviceptr devicePtr;
  CUresult result = driver.cuMemAlloc(&devicePtr, size);
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&device.driver, result, "cuMemAlloc"));
  M_driver_memory_t memory =
      new (std::nothrow) M_driver_memory(reinterpret_cast<void *>(devicePtr));
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

DriverResult<M_driver_memory_t>
M_driver_context::wrapSync(uint64_t address, uint64_t size, bool owning) {
  (void)size;
  if (address == 0)
    return tl::unexpected(M_DRIVER_STATUS_INVALID_ARG);
  M_driver_memory_t memory = new (std::nothrow)
      M_driver_memory{reinterpret_cast<void *>(address), owning};
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

M_driver_status_t M_driver_context::freePinned(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &driver = device.driver.rawCuda;
  void *ptr = memory->ptr;
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;
  if (CUresult r = driver.cuMemFreeHost(ptr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemFreeHost");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::freeSync(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &driver = device.driver.rawCuda;
  CUdeviceptr devicePtr = reinterpret_cast<CUdeviceptr>(memory->ptr);
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;
  if (CUresult r = driver.cuMemFree(devicePtr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemFree");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyToDeviceSync(M_driver_memory_t dst,
                                                     uint64_t dstOffset,
                                                     const void *src,
                                                     uint64_t size) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyHtoD(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset, src, size);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemcpyHtoD");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyFromDeviceSync(void *dst,
                                                       M_driver_memory_t src,
                                                       uint64_t srcOffset,
                                                       uint64_t size) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyDtoH(
          dst, reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset, size);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemcpyDtoH");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyIntraDeviceSync(M_driver_memory_t dst,
                                                        uint64_t dstOffset,
                                                        M_driver_memory_t src,
                                                        uint64_t srcOffset,
                                                        uint64_t size) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyDtoD(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset,
          reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset, size);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemcpyDtoD");
  // The synchronous device-to-device memcpy does NOT host-block (CUDA performs
  // no host-side synchronization for device-to-device transfers), so wait on
  // the default stream it ran on to honor the blocking contract.
  if (CUresult r = cuda.cuStreamSynchronize(nullptr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuStreamSynchronize (sync D2D)");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyInterDeviceSync(
    M_driver_memory_t dst, uint64_t dstOffset, M_driver_memory_t src,
    uint64_t srcOffset, uint64_t size, M_driver_context_t srcContext) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyPeer(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset, primaryCtx,
          reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset,
          srcContext->primaryCtx, size);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemcpyPeer");
  // Like the D2D case above, the synchronous peer copy does not host-block, so
  // wait on the default stream to honor the blocking contract.
  if (CUresult r = cuda.cuStreamSynchronize(nullptr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r,
                           "cuStreamSynchronize (sync peer)");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::setCurrent() {
  if (CUresult r = device.driver.rawCuda.cuCtxSetCurrent(primaryCtx);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuCtxSetCurrent");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::getCurrentDriverContext(
    M_driver_native_handle_t *driver_context) {
  if (driver_context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  CUcontext current = nullptr;
  if (CUresult r = device.driver.rawCuda.cuCtxGetCurrent(&current);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuCtxGetCurrent");
  *driver_context = reinterpret_cast<M_driver_native_handle_t>(current);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::setCurrentDriverContext(
    M_driver_native_handle_t driver_context) {
  if (CUresult r = device.driver.rawCuda.cuCtxSetCurrent(
          reinterpret_cast<CUcontext>(driver_context));
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuCtxSetCurrent");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::memoryInfo(uint64_t *free,
                                               uint64_t *total) {
  if (free == nullptr || total == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // cuMemGetInfo reports the *current* context's memory, so make this context
  // current for the duration of the query (restored on scope exit).
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  if (scope.status != M_DRIVER_STATUS_SUCCESS)
    return scope.status;
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  if (CUresult r = device.driver.rawCuda.cuMemGetInfo(&freeBytes, &totalBytes);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemGetInfo");
  *free = freeBytes;
  *total = totalBytes;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::enablePeerAccess(M_driver_context &peer) {
  M::Target::CUDA::CUDALoader &cuda = device.driver.rawCuda;
  bool canAccess = false;
  if (M_driver_status_t s = device.canAccessPeer(peer.device, &canAccess);
      s != M_DRIVER_STATUS_SUCCESS)
    return s;
  if (!canAccess)
    return M_DRIVER_STATUS_INCOMPATIBLE;

  // cuCtxEnablePeerAccess acts on the current context.
  ScopedContext scope(primaryCtx, cuda);
  if (scope.status != M_DRIVER_STATUS_SUCCESS)
    return scope.status;
  CUresult r = cuda.cuCtxEnablePeerAccess(peer.primaryCtx, /*flags=*/0);
  if (r != CUDA_SUCCESS && r != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
    return cudaErrorStatus(&device.driver, r, "cuCtxEnablePeerAccess");

  // cuCtxEnablePeerAccess does not cover the peer's stream-ordered pool.
  CUmemoryPool pool;
  if (CUresult r2 = cuda.cuDeviceGetMemPool(&pool, peer.device.deviceId);
      r2 != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r2, "cuDeviceGetMemPool");
  CUmemAccessDesc desc{};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device.deviceId;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (CUresult r3 = cuda.cuMemPoolSetAccess(pool, &desc, /*count=*/1);
      r3 != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r3, "cuMemPoolSetAccess");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::setMemorySync(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  uint64_t size,
                                                  uint8_t value) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset;
  if (CUresult r = cuda.cuMemsetD8(dstPtr, value, size); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemsetD8");
  if (CUresult r = cuda.cuStreamSynchronize(nullptr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r,
                           "cuStreamSynchronize (sync set_memory)");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::fillSync(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (valueSize == 1)
    return setMemorySync(dst, dstOffset, size, static_cast<uint8_t>(value));

  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset;
  switch (valueSize) {
  case 2:
    if ((size & 1) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (CUresult r =
            cuda.cuMemsetD16(dstPtr, static_cast<uint16_t>(value), size >> 1);
        r != CUDA_SUCCESS)
      return cudaErrorStatus(&device.driver, r, "cuMemsetD16");
    break;
  case 4:
    if ((size & 3) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    if (CUresult r =
            cuda.cuMemsetD32(dstPtr, static_cast<uint32_t>(value), size >> 2);
        r != CUDA_SUCCESS)
      return cudaErrorStatus(&device.driver, r, "cuMemsetD32");
    break;
  case 8: {
    if ((size & 7) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    uint32_t hiBits = static_cast<uint32_t>(value >> 32);
    uint32_t loBits = static_cast<uint32_t>(value & 0xfffffffful);
    if (hiBits == loBits) {
      if (CUresult r = cuda.cuMemsetD32(dstPtr, loBits, size >> 2);
          r != CUDA_SUCCESS)
        return cudaErrorStatus(&device.driver, r,
                               "cuMemsetD32 (uint64 broadcast)");
    } else {
      size_t len = size >> 3;
      CUdeviceptr hiPtr = dstPtr + sizeof(uint32_t);
      if (CUresult r = cuda.cuMemsetD2D32(dstPtr, 8, loBits, 1, len);
          r != CUDA_SUCCESS)
        return cudaErrorStatus(&device.driver, r,
                               "cuMemsetD2D32 (uint64 low half)");
      if (CUresult r = cuda.cuMemsetD2D32(hiPtr, 8, hiBits, 1, len);
          r != CUDA_SUCCESS)
        return cudaErrorStatus(&device.driver, r,
                               "cuMemsetD2D32 (uint64 high half)");
    }
    break;
  }
  default:
    return M_DRIVER_STATUS_INVALID_ARG;
  }
  if (CUresult r = cuda.cuStreamSynchronize(nullptr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r,
                           "cuStreamSynchronize (sync fill)");
  return M_DRIVER_STATUS_SUCCESS;
}
