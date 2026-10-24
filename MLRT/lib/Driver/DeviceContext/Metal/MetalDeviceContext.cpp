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
// This file implements the Metal device context.
//
//===----------------------------------------------------------------------===//

#include "MetalDeviceContext.h"
#include <limits>
#if MODULAR_MACOS

#include "AsyncRT/Runtime/Globals/Globals.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "MetalDriver.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "Support/max_license/Tracy/Tracy.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#define CHECK_IS_METAL_ERR(dev)                                                \
  do {                                                                         \
    if ((dev).api() != Device::metalAPI)                                       \
      RETURN_ERROR("Not a Metal device.");                                     \
  } while (0)

// Copy paths only, as on CUDA: kernel launches and event waits stay
// asynchronous even under `device-sync-mode`.
#define METAL_MAYBE_SYNC(ctx)                                                  \
  do {                                                                         \
    DeviceContext &metal_ctx = (ctx);                                          \
    if (metal_ctx.syncMode()) {                                                \
      RETURN_IF_ERROR(metal_ctx.synchronize());                                \
    }                                                                          \
  } while (0)

namespace M::Driver {

static ErrorOr<int> getComputeCapability(StringRef name) {
  if (name.empty())
    return 3;

  if (name.contains_insensitive("M6"))
    return 6;

  if (name.contains_insensitive("M5"))
    return 5;

  if (name.contains_insensitive("M4"))
    return 4;

  if (name.contains_insensitive("M3"))
    return 3;

  if (name.contains_insensitive("M2"))
    return 2;

  if (name.contains_insensitive("M1"))
    return 1;

  RETURN_ERROR("Unknown device name");
}

// Return true if logging for Metal kernel is enabled.
static bool isLoggingEnabled(void *attrs, uint32_t numAttrs) {
  assert((attrs != nullptr || numAttrs == 0) &&
         "attrs must be non-null when numAttrs > 0");
  Device::Attribute *deviceAttrs = (Device::Attribute *)attrs;
  for (uint32_t i = 0; i < numAttrs; ++i) {
    if (deviceAttrs[i] == Device::Attribute::METAL_DISABLE_LOGGING)
      return false;
  }
  return true;
}

static ErrorOr<MTL::Buffer *> allocateNewBuffer(MTL::Device *metalDev,
                                                size_t byteSize,
                                                MTL::ResourceOptions options) {
  size_t maxBufferLength = metalDev->maxBufferLength();
  if (byteSize >= maxBufferLength) {
    RETURN_ERROR("Failed to allocate MTL Buffer of a size " +
                 std::to_string(byteSize) +
                 " bytes which is larger than allowed " +
                 std::to_string(maxBufferLength) + " bytes");
  }

  return metalDev->newBuffer(byteSize, options);
}

static ErrorOr<MTL::Buffer *> allocateNewBuffer(MTL::Device *metalDev,
                                                void *ptr, size_t byteSize,
                                                MTL::ResourceOptions options) {
  size_t maxBufferLength = metalDev->maxBufferLength();
  if (byteSize >= maxBufferLength) {
    RETURN_ERROR("Failed to allocate MTL Buffer of a size " +
                 std::to_string(byteSize) +
                 " bytes which is larger than allowed " +
                 std::to_string(maxBufferLength) + " bytes");
  }
  return metalDev->newBuffer(ptr, byteSize, options);
}

// Aliases the caller's pages instead of copying, as the overload above does.
// The null deallocator is what leaves the pages to the caller.
static ErrorOr<MTL::Buffer *> wrapExistingPages(MTL::Device *metalDev,
                                                void *ptr, size_t byteSize,
                                                MTL::ResourceOptions options) {
  size_t maxBufferLength = metalDev->maxBufferLength();
  if (byteSize >= maxBufferLength) {
    RETURN_ERROR("Failed to wrap host memory of size " +
                 std::to_string(byteSize) +
                 " bytes which is larger than allowed " +
                 std::to_string(maxBufferLength) + " bytes");
  }
  return metalDev->newBuffer(ptr, byteSize, options, /*deallocator=*/nullptr);
}

//
//
// ===== MetalDevice ===== //
//
MetalDevice::MetalDevice(int id, NS::SharedPtr<MTL::Device> device,
                         std::string name, Device::MemInfo memInfo,
                         size_t allocAlignment)
    : Device(api, name, id, memInfo, allocAlignment), device_(device),
      memInfo_(memInfo), driverVersion_([]() -> std::string {
        // Metal drivers ship with macOS, Darwin release is the driver version
        struct utsname systemInfo;
        uname(&systemInfo);
        return systemInfo.release;
      }()) {}

ErrorOr<int> MetalDevice::getApiVersion() const { return 0; }

ErrorOr<DriverVersion &> MetalDevice::getDriverVersion() const {
  return driverVersion_;
}

ErrorOr<Device::MemInfo> MetalDevice::getMemInfo(MTL::Device *metalDev) {
  Device::MemInfo info;
  info.total = metalDev->recommendedMaxWorkingSetSize();
  size_t used = metalDev->currentAllocatedSize();
  info.free = info.total > used ? info.total - used : 0;
  return info;
}

ErrorOr<Device::MemInfo> MetalDevice::getMemoryInfo() const {
  return getMemInfo(device_.get());
}

ErrorOr<size_t> MetalDevice::maxSingleAllocationSize() const {
  return static_cast<size_t>(device_->maxBufferLength());
}

ErrorOrSuccess MetalDevice::isCompatible() const { return success(); }

ErrorOr<int> MetalDevice::computeCapability() const {
  // Use the major compute capability as the compute capability.
  // This allows to get something like apple-m1 as arch name.
  return getAttribute(Device::Attribute::COMPUTE_CAPABILITY_MAJOR);
}

ErrorOr<bool> MetalDevice::canAccess(const Device &peer) const { return false; }

ErrorOr<int> MetalDevice::getAttribute(Attribute attr) const {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  // Constants pulled from the Metal feature set tables available in
  // PDF/Numbers spreadsheet at:
  // https://developer.apple.com/documentation/metal/
  // detecting-gpu-features-and-metal-software-versions
  //
  // The families we care about:
  // - M1 -> Apple7
  // - M2 -> Apple8
  // - M3 -> Apple9
  // - M4 -> Apple9
  // - M5 -> Apple10

#define ATTRIBUTE_UNSUPPORTED(ATTR)                                            \
  RETURN_ERROR("Attribute \"" #ATTR "\" not supported for Apple GPU.")

  switch (attr) {
  // 'Maximum threads per threadgroup' from 'Function arguments' section in the
  // 'GPU implementation limits by family', same limit for Apple7 - Apple10.
  case Device::Attribute::MAX_THREADS_PER_BLOCK: {
    return 1024;
  }

  // Block dimensions map to max threads per threadgroup.
  case Device::Attribute::MAX_BLOCK_DIM_X:
    return device_->maxThreadsPerThreadgroup().width;
  case Device::Attribute::MAX_BLOCK_DIM_Y:
    return device_->maxThreadsPerThreadgroup().height;
  case Device::Attribute::MAX_BLOCK_DIM_Z:
    return device_->maxThreadsPerThreadgroup().depth;

  // There is no limit to the size of a grid.
  case Device::Attribute::MAX_GRID_DIM_X:
    return std::numeric_limits<int>::max();
  case Device::Attribute::MAX_GRID_DIM_Y:
    return std::numeric_limits<int>::max();
  case Device::Attribute::MAX_GRID_DIM_Z:
    return std::numeric_limits<int>::max();

  // 'Maximum total threadgroup memory allocation' from 'Function arguments'
  // section in the 'GPU implementation limits by family', same limit for
  // Apple7 - Apple10.
  case Device::Attribute::MAX_SHARED_MEMORY_PER_BLOCK:
    return device_->maxThreadgroupMemoryLength();
  case Device::Attribute::MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
    ATTRIBUTE_UNSUPPORTED(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);

  // The SIMD-group size is a dynamic property which is accessible through the
  // computePipeline->threadExecutionWidth property, i.e. it's tied to a
  // specific kernel invocation rather available as a static/dynamic device
  // property.
  case Device::Attribute::WARP_SIZE:
    ATTRIBUTE_UNSUPPORTED(WARP_SIZE);

  // Max registers per block is not documented.
  case Device::Attribute::MAX_REGISTERS_PER_BLOCK:
    ATTRIBUTE_UNSUPPORTED(MAX_REGISTERS_PER_BLOCK);

  // Clock rate is exposed through performance counters not as a property of
  // the device.
  case Device::Attribute::CLOCK_RATE:
    ATTRIBUTE_UNSUPPORTED(CLOCK_RATE);

  case Device::Attribute::MULTIPROCESSOR_COUNT: {
    static std::once_flag mpcFlag;
    static std::optional<int> mpCount;

// Host is built for MacOS 11.0, but `kIOMainPortDefault` is only available on
// MacOS >= 12.0. However, we support Apple GPU only for MacOS >= 15, so this
// warning can be ignored.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
    std::call_once(mpcFlag, []() {
      io_iterator_t iterator;
      CFMutableDictionaryRef matchDict = IOServiceMatching("AGXAccelerator");

      if (IOServiceGetMatchingServices(kIOMainPortDefault, matchDict,
                                       &iterator) != KERN_SUCCESS)
        return;

      io_service_t service;
      while ((service = IOIteratorNext(iterator)) != 0) {
        CFMutableDictionaryRef properties = nullptr;

        if (IORegistryEntryCreateCFProperties(
                service, &properties, kCFAllocatorDefault, 0) == KERN_SUCCESS) {
          CFNumberRef coreCountRef = (CFNumberRef)CFDictionaryGetValue(
              properties, CFSTR("gpu-core-count"));

          if (coreCountRef &&
              CFGetTypeID(coreCountRef) == CFNumberGetTypeID()) {
            int coreCount = 0;
            CFNumberGetValue(coreCountRef, kCFNumberIntType, &coreCount);
            CFRelease(properties);
            IOObjectRelease(service);
            IOObjectRelease(iterator);
            mpCount = coreCount;
            return;
          }

          if (properties)
            CFRelease(properties);
        }

        IOObjectRelease(service);
      }

      IOObjectRelease(iterator);
    });
#pragma clang diagnostic pop
    if (!mpCount)
      ATTRIBUTE_UNSUPPORTED(MULTIPROCESSOR_COUNT);
    return *mpCount;
  }

  // The max total threads per threadgroup is a dynamic property which is
  // accessible through the computePipeline->maxTotalThreadsPerThreadgroup size
  // property, i.e. it's tied to a specific kernel invocation rather than
  // available as a static/dynamic device property.
  case Device::Attribute::MAX_THREADS_PER_MULTIPROCESSOR:
    ATTRIBUTE_UNSUPPORTED(MAX_THREADS_PER_MULTIPROCESSOR);

  // TODO: Would it make more sense to map the compute capability major/minor
  // to the Apple GPU family instead?
  // Note that M3 and M4 both map to the Apple9 family.
  case Device::Attribute::COMPUTE_CAPABILITY_MAJOR: {
    std::string deviceName;
    if (device_ && device_->name()) {
      deviceName = device_->name()->cString(NS::UTF8StringEncoding);
    }
    return getComputeCapability(deviceName);
  }
  case Device::Attribute::COMPUTE_CAPABILITY_MINOR:
    return 0;

  // Threadgroup memory limit per kernel dispatch, queried directly from the
  // Metal device.  Apple Silicon is 32 KB; this may vary on future hardware.
  case Device::Attribute::MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
    ATTRIBUTE_UNSUPPORTED(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);

  // Max registers per GPU core is not documented.
  case Device::Attribute::MAX_REGISTERS_PER_MULTIPROCESSOR:
    ATTRIBUTE_UNSUPPORTED(MAX_REGISTERS_PER_MULTIPROCESSOR);

  // Cooperative launch is not supported.
  case Device::Attribute::COOPERATIVE_LAUNCH:
    ATTRIBUTE_UNSUPPORTED(COOPERATIVE_LAUNCH);

  // Max threadgroups per GPU core is not documented.
  case Device::Attribute::MAX_BLOCKS_PER_MULTIPROCESSOR:
    ATTRIBUTE_UNSUPPORTED(MAX_BLOCKS_PER_MULTIPROCESSOR);

  // Max access policy window size is not supported.
  case Device::Attribute::MAX_ACCESS_POLICY_WINDOW_SIZE:
    ATTRIBUTE_UNSUPPORTED(MAX_ACCESS_POLICY_WINDOW_SIZE);
  case Device::Attribute::METAL_DISABLE_LOGGING:
    return 0;
  }

#undef ATTRIBUTE_UNSUPPORTED
}

ErrorOr<std::string> MetalDevice::getPciBusId() const {
  RETURN_ERROR("PCI bus address not available for Metal devices");
}

int MetalDevice::deviceCount() const { return 1; }

ErrorOrSuccess MetalDevice::chunkAlloc(void **ptr, size_t byteSize,
                                       bool isHost) {
  assert(byteSize != 0);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::Device *metalDev = metalDevice();
  if (metalDev == nullptr)
    RETURN_ERROR("Invalid Metal device: null pointer");

  // Create Metal buffer with shared storage mode for all allocations
  MTL::ResourceOptions options = MTL::ResourceStorageModeShared;

  // Donate newBuffer's +1 to the tracker; chunkFree releases.
  // NS::SharedPtr + detach() is UB here: ~SharedPtr does nullptr->release().
  DEFINE_OR_RETURN_ERROR(MTL::Buffer *, buf,
                         allocateNewBuffer(metalDev, byteSize, options));
  if (buf == nullptr)
    RETURN_ERROR("Failed to create Metal buffer in chunkAlloc");

  // For both host and device allocations, return the contents pointer
  // The memory manager will manage the memory layout
  void *contents = buf->contents();
  if (contents == nullptr) {
    buf->release();
    RETURN_ERROR("Failed to get contents from Metal buffer");
  }

  *ptr = contents;

  // Track this allocation. The +1 from newBuffer is now owned by the
  // tracker; chunkFree releases it.
  trackBuffer(contents, buf);

  return success();
}

ErrorOrSuccess MetalDevice::chunkFree(void *ptr, bool isHost) {
  if (ptr == nullptr)
    return success();

  // Look up the Metal buffer for this pointer
  MTL::Buffer *bufferToRelease = getTrackedBuffer(ptr);

  if (bufferToRelease == nullptr) {
    // This shouldn't happen if all allocations go through chunkAlloc
    RETURN_ERROR("Attempt to free untracked Metal buffer in chunkFree");
  }

  // Release the buffer and remove from tracking
  bufferToRelease->release();
  untrackBuffer(ptr);

  return success();
}

ErrorOr<void *> MetalDevice::getCurrentDriverContext() const {
  // Not tracking a driver context for Metal (mirrors CPU).
  return nullptr;
}

ErrorOrSuccess MetalDevice::setCurrentDriverContext(void *ctx) {
  // Not tracking a driver context for Metal (mirrors CPU).
  return success();
}

ErrorOr<DeviceContextRef> MetalDevice::createContext() {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  NS::SharedPtr<MTL::CommandQueue> commandQueue =
      NS::TransferPtr(device_->newCommandQueue());
  if (!commandQueue)
    RETURN_ERROR("Failed to create Metal command queue for context.");

  return MetalDeviceContextRef::create(*this, commandQueue);
}

const MetalDevice &MetalDevice::cast(const Device &device) {
  return static_cast<const MetalDevice &>(device);
}

MetalDevice &MetalDevice::cast(Device &device) {
  return static_cast<MetalDevice &>(device);
}

ErrorOr<MetalDevice &> MetalDevice::get(int id) {
  if (id != 0)
    RETURN_ERROR("Apple Metal device only supports id == 0.");

  static RCRef<MetalDevice> metalDeviceInstance;

  if (!metalDeviceInstance) {
    NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
        NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    NS::SharedPtr<MTL::Device> device =
        NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device)
      RETURN_ERROR("Failed to create Metal device.");

    std::string name = device->name()->cString(NS::UTF8StringEncoding);

    Device::MemInfo memInfo;
    bool useMemoryManager = false;
    if (useMemoryManager) {
      DEFINE_OR_RETURN_ERROR(Device::MemInfo, memInfoResult,
                             getMemInfo(device.get()));
      memInfo = memInfoResult;
    }

    size_t allocAlignment = 64;

    metalDeviceInstance =
        RCRef<MetalDevice>::create(id, device, name, memInfo, allocAlignment);
  }

  return *metalDeviceInstance;
}

ErrorOr<int> MetalDevice::numberOfGPUs() {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  int count = 0;
  NS::SharedPtr<NS::Array> devices = NS::TransferPtr(MTL::CopyAllDevices());
  if (devices) {
    count = devices->count();
  }
  return count;
}

//
// Buffer tracking implementation
//
void MetalDevice::trackBuffer(void *ptr, MTL::Buffer *buffer) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  allocations_[addr] = BufferAllocation{addr, buffer->length(), buffer};
}

void MetalDevice::untrackBuffer(void *ptr) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  allocations_.erase(reinterpret_cast<uintptr_t>(ptr));
}

MetalDevice::BufferLookupResult
MetalDevice::findContainingBuffer(const void *ptr) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

  // Find first allocation with base address > addr
  auto it = allocations_.upper_bound(addr);
  if (it != allocations_.begin()) {
    --it; // Step back to candidate allocation
    const BufferAllocation &alloc = it->second;
    if (addr >= alloc.baseAddress && addr < alloc.baseAddress + alloc.size) {
      return {alloc.buffer, static_cast<size_t>(addr - alloc.baseAddress)};
    }
  }
  return {nullptr, 0};
}

MTL::Buffer *MetalDevice::getTrackedBuffer(const void *ptr) const {
  return findContainingBuffer(ptr).buffer;
}

bool MetalDevice::isTrackedBuffer(const void *ptr) const {
  return findContainingBuffer(ptr).isValid();
}

std::map<uintptr_t, MetalDevice::BufferAllocation>
MetalDevice::getTrackedBuffers() const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  // return current copy of the tracked buffers.
  return allocations_;
}

void MetalDevice::forEachTrackedBuffer(
    const std::function<void(const BufferAllocation &)> &fn) const {
  std::lock_guard<std::mutex> lock(bufferMapMutex_);
  // Iterate the live allocation map under the lock instead of copying it, so
  // the per-launch residency sweep (GEX-3761) does not pay a `std::map`
  // value-copy every launch. Iteration order (ascending base address) and the
  // set of buffers are identical to `getTrackedBuffers()`.
  for (const auto &it : allocations_)
    fn(it.second);
}

//
//
// ===== MetalDeviceContext ===== //
//

// Seeded so the queue's first waitForFence has an update to wait on.
static MetalQueue
makeMetalQueue(MTL::Device *device,
               NS::SharedPtr<MTL::CommandQueue> commandQueue) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  NS::SharedPtr<MTL::Fence> fence = NS::TransferPtr(device->newFence());
  if (fence) {
    MTL::CommandBuffer *cmdBuffer = commandQueue->commandBuffer();
    MTL::BlitCommandEncoder *blit =
        cmdBuffer ? cmdBuffer->blitCommandEncoder() : nullptr;
    if (blit) {
      blit->updateFence(fence.get());
      blit->endEncoding();
      cmdBuffer->commit();
    } else {
      fence.reset();
    }
  }
  return MetalQueue{std::move(commandQueue), std::move(fence)};
}

MetalQueuePool::MetalQueuePool(MetalQueue base) {
  queues_.push_back(std::move(base));
}

MetalQueue MetalQueuePool::get(unsigned i) const {
  std::lock_guard<std::mutex> lock(mu_);
  assert(i < queues_.size() && "queue index out of range");
  return queues_[i];
}

unsigned MetalQueuePool::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<unsigned>(queues_.size());
}

unsigned MetalQueuePool::emplace_back(MetalQueue queue) {
  std::lock_guard<std::mutex> lock(mu_);
  queues_.push_back(std::move(queue));
  return static_cast<unsigned>(queues_.size()) - 1;
}

// For processes that cannot call setMetalPrintEnabled, e.g. subprocesses.
static bool metalPrintDisabledByEnv() {
  const char *env = std::getenv("MODULAR_DISABLE_METAL_GPU_PRINT");
  return env && env[0] != '\0' && env[0] != '0';
}

MetalDeviceContext::MetalDeviceContext(
    MetalDevice &device, NS::SharedPtr<MTL::CommandQueue> commandQueue)
    : MetalDeviceContext(device,
                         MetalQueuePoolRef::create(makeMetalQueue(
                             device.metalDevice(), std::move(commandQueue))),
                         /*activeQueueId=*/0) {}

MetalDeviceContext::MetalDeviceContext(MetalDevice &device,
                                       MetalQueuePoolRef queuePool,
                                       unsigned activeQueueId)
    : DeviceContext(device, TypeID::get<MetalDeviceContext>()),
      queuePool_(std::move(queuePool)) {
  MetalQueue active = queuePool_->get(activeQueueId);
  commandQueue_ = std::move(active.commandQueue);
  orderFence_ = std::move(active.orderFence);
  printDisabled_ = metalPrintDisabledByEnv();
}

ErrorOrSuccess MetalDeviceContext::enqueueCopyToDevice(void *dst,
                                                       const void *src,
                                                       size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Validate source
  if (src == nullptr)
    RETURN_ERROR("Source pointer is null");

  if (byteSize == 0)
    return success();

  if (!isMetalBuffer(dst)) {
    // Drain prior GPU work in case `src` or `dst` aliases a shared MTLBuffer
    // the GPU is still writing.
    RETURN_IF_ERROR(drainQueue());
    std::memcpy(dst, src, byteSize);
    return success();
  }

  // Get destination Metal buffer with offset for device buffers (handles
  // sub-buffers)
  MetalDevice::BufferLookupResult dstResult = getMetalBufferWithOffset(dst);
  if (!dstResult.isValid())
    RETURN_ERROR("Invalid Metal buffer pointer");

  NS::UInteger bufferLength = dstResult.buffer->length();
  if (bufferLength == 0)
    RETURN_ERROR("Metal buffer length is 0 - buffer may be invalid");

  // Check that copy fits within the buffer from the offset
  if (dstResult.offset + byteSize > bufferLength)
    RETURN_ERROR("Copy size exceeds Metal buffer length");

  // Check if source is a Metal buffer allocated by our context
  MetalDevice::BufferLookupResult srcResult = getMetalBufferWithOffset(src);
  if (srcResult.isValid()) {
    // Source is a Metal buffer - use blit encoder for copy with offsets
    RETURN_IF_ERROR(enqueueMetalCopy(srcResult.buffer, dstResult.buffer,
                                     srcResult.offset, dstResult.offset,
                                     byteSize));
    METAL_MAYBE_SYNC(*this);
    return success();
  }

  // Source is regular host memory — drain prior GPU work, then memcpy into
  // the shared MTLBuffer's CPU pointer on this thread.
  RETURN_IF_ERROR(drainQueue());

  void *dstContents = dstResult.buffer->contents();
  if (dstContents == nullptr)
    RETURN_ERROR("Destination Metal buffer contents() returned null");
  void *dstPtr = static_cast<char *>(dstContents) + dstResult.offset;
  std::memcpy(dstPtr, src, byteSize);

  return success();
}

ErrorOrSuccess MetalDeviceContext::enqueueCopyFromDevice(void *dst,
                                                         const void *src,
                                                         size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (dst == nullptr)
    RETURN_ERROR("Destination pointer cannot be null");

  if (byteSize == 0)
    return success(); // no-op

  // Get source Metal buffer with offset (handles sub-buffers)
  MetalDevice::BufferLookupResult srcResult = getMetalBufferWithOffset(src);
  if (!srcResult.isValid())
    RETURN_ERROR("Invalid source buffer");

  // When we have 2 Device buffers we can do an async copy via
  // enqueueMetalCopy() similar to what the CUDA implementation does
  // However, when we are using raw pointers we need memcpy and wait for
  // completion otherwise, the next command buffer will be executed before the
  // current one is completed

  // Check if destination is a Metal buffer allocated by our context
  MetalDevice::BufferLookupResult dstResult = getMetalBufferWithOffset(dst);
  if (dstResult.isValid()) {
    // Destination is a Metal buffer - use blit encoder for copy with offsets
    RETURN_IF_ERROR(enqueueMetalCopy(srcResult.buffer, dstResult.buffer,
                                     srcResult.offset, dstResult.offset,
                                     byteSize));
    METAL_MAYBE_SYNC(*this);
    return success();
  }

  // Drain the kernel that produced `src` before reading its shared storage
  // on this thread.
  RETURN_IF_ERROR(drainQueue());

  void *srcContents = srcResult.buffer->contents();
  if (srcContents == nullptr)
    RETURN_ERROR("Source Metal buffer contents() returned null");
  void *srcPtr = static_cast<char *>(srcContents) + srcResult.offset;
  std::memcpy(dst, srcPtr, byteSize);

  return success();
}

ErrorOrSuccess MetalDeviceContext::memCopyDtoD(DeviceBuffer &dst,
                                               DeviceBuffer &src,
                                               size_t srcByteOffset,
                                               size_t srcByteSize,
                                               bool crossStreamSync) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);

  // 1) Basic validation
  if (src.byteSize() == 0 || dst.byteSize() == 0)
    return success(); // no-op for zero-size

  if (dst.byteSize() < src.byteSize())
    RETURN_ERROR("Destination buffer size must be >= source buffer size");

  if (srcByteOffset + srcByteSize < src.byteSize())
    RETURN_ERROR("Requested source slice larger than source buffer");

  // The blit runs on this context's queue, so that is the ordering the caller
  // gets; a destination owned elsewhere would silently get the wrong one.
  if (&dst.context() != this)
    RETURN_ERROR("device context does not match context of dst");

  // 2) Handle Metal buffer to Metal buffer copies (both host and device are
  // Metal buffers now) with offset support
  // This allows sub-buffers and slices to work correctly
  MetalDevice::BufferLookupResult srcResult =
      getMetalBufferWithOffset(src.ptr());
  MetalDevice::BufferLookupResult dstResult =
      getMetalBufferWithOffset(dst.ptr());

  if (!srcResult.isValid() || !dstResult.isValid())
    RETURN_ERROR("Invalid Metal buffer pointer");

  // Metal hazard-tracks within a queue but not across them, so a cross-queue
  // source needs a dependency each way: this queue waits for the producer,
  // then the producer waits for the copy before it may reuse `src`.
  DeviceContext &srcCtx = src.context();
  const bool needsSync = crossStreamSync && &srcCtx != this;

  if (needsSync)
    RETURN_IF_ERROR(enqueueWaitFor(srcCtx));

  RETURN_IF_ERROR(enqueueMetalCopy(srcResult.buffer, dstResult.buffer,
                                   srcResult.offset, dstResult.offset,
                                   src.byteSize()));

  if (needsSync)
    RETURN_IF_ERROR(srcCtx.enqueueWaitFor(*this));

  METAL_MAYBE_SYNC(*this);
  return success();
}

// Fills bytes [begin, end) of `dst` with a pattern that repeats every 16
// bytes: byte `p` gets `pattern` byte `p % 16`. Whole 16-byte blocks take one
// vector store each; only the two edge blocks are written byte by byte, so the
// bytes around the range are left alone.
static constexpr const char kPatternFillSource[] = R"(
#include <metal_stdlib>
using namespace metal;

struct PatternFillArgs {
  uint4 pattern;
  ulong begin;
  ulong end;
  ulong blocks;
};

kernel void pattern_fill(device uint4 *dst [[buffer(0)]],
                         constant PatternFillArgs &args [[buffer(1)]],
                         uint tid [[thread_position_in_grid]],
                         uint threads [[threads_per_grid]]) {
  for (ulong i = tid; i < args.blocks; i += threads) {
    ulong lo = i * 16;
    if (lo >= args.begin && lo + 16 <= args.end) {
      dst[i] = args.pattern;
      continue;
    }
    device uchar *out = reinterpret_cast<device uchar *>(dst + i);
    constant uchar *pat = reinterpret_cast<constant uchar *>(&args.pattern);
    for (uint b = 0; b < 16; ++b)
      if (lo + b >= args.begin && lo + b < args.end)
        out[b] = pat[b];
  }
}
)";

ErrorOrSuccess MetalDeviceContext::enqueueSetMemory(void *dst, size_t byteSize,
                                                    uint64_t val,
                                                    size_t valSize) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (byteSize == 0)
    return success(); // no-op

  if (valSize != 1 && valSize != 2 && valSize != 4 && valSize != 8)
    RETURN_ERROR("unsupported valSize");
  if (byteSize % valSize != 0)
    RETURN_ERROR("byteSize of buffer is misaligned");

  // Get the Metal buffer for the destination with offset (handles sub-buffers)
  MetalDevice::BufferLookupResult dstResult = getMetalBufferWithOffset(dst);
  if (!dstResult.isValid())
    RETURN_ERROR("Destination buffer is null");
  if (dstResult.offset % valSize != 0)
    RETURN_ERROR("destination buffer not properly aligned");
  if (dstResult.offset + byteSize > dstResult.buffer->length())
    RETURN_ERROR("memset size exceeds Metal buffer length");

  // The fill runs on this queue rather than as a host memset of the shared
  // storage, so it orders behind in-flight kernels on the GPU instead of
  // draining the queue first (a drain idles the GPU until the host resubmits).
  // `fillBuffer` takes one byte; a wider pattern whose bytes differ goes to a
  // one-dispatch helper kernel.
  const uint8_t fillByte = static_cast<uint8_t>(val);
  bool byteUniform = true;
  for (size_t i = 1; i < valSize; ++i)
    byteUniform &= static_cast<uint8_t>(val >> (8 * i)) == fillByte;

  if (byteUniform) {
    MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
    if (cmdBuffer == nullptr)
      RETURN_ERROR("Failed to create command buffer for memset");
    MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
    if (blit == nullptr)
      RETURN_ERROR("Failed to create blit encoder for memset");
    if (orderFence_)
      blit->waitForFence(orderFence_.get());
    blit->fillBuffer(dstResult.buffer, NS::Range(dstResult.offset, byteSize),
                     fillByte);
    if (orderFence_)
      blit->updateFence(orderFence_.get());
    blit->endEncoding();
    cmdBuffer->commit();
    METAL_MAYBE_SYNC(*this);
    return success();
  }

  // Must match `PatternFillArgs` in kPatternFillSource.
  struct PatternFillArgs {
    alignas(16) uint8_t pattern[16];
    uint64_t begin;
    uint64_t end;
    uint64_t blocks;
  };
  static_assert(sizeof(PatternFillArgs) == 48, "layout of the MSL struct");

  // Bind at the 16-byte boundary at or below `dst` so the kernel's vector
  // stores are aligned. The pattern size divides 16 and the offset is a
  // multiple of it, so byte `i` of every 16-byte block takes pattern byte
  // `i % valSize`.
  const size_t bindOffset = dstResult.offset & ~size_t(15);
  PatternFillArgs fillArgs;
  for (size_t i = 0; i < sizeof(fillArgs.pattern); ++i)
    fillArgs.pattern[i] = static_cast<uint8_t>(val >> (8 * (i % valSize)));
  fillArgs.begin = dstResult.offset - bindOffset;
  fillArgs.end = fillArgs.begin + byteSize;
  fillArgs.blocks = (fillArgs.end + 15) / 16;

  // Cached in the device's function cache after the first compile.
  DEFINE_OR_RETURN_ERROR(DeviceFunctionRef, fillFn,
                         loadFunction("ModularMetalPatternFill", "pattern_fill",
                                      kPatternFillSource,
                                      /*maxDynamicSharedBytes=*/-1,
                                      /*debugLevel=*/"",
                                      /*optimizationLevel=*/-1));
  MTL::ComputePipelineState *pipeline =
      MetalDeviceFunction::cast(*fillFn).getPipelineState();
  // The kernel loops over the range, so the grid only has to fill the GPU.
  static constexpr uint64_t kMaxThreadgroups = 1 << 16;
  const NS::UInteger threadgroupSize =
      std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
  const NS::UInteger threadgroups = std::min<uint64_t>(
      (fillArgs.blocks + threadgroupSize - 1) / threadgroupSize,
      kMaxThreadgroups);

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer for memset");
  MTL::ComputeCommandEncoder *encoder = cmdBuffer->computeCommandEncoder();
  if (encoder == nullptr)
    RETURN_ERROR("Failed to create compute encoder for memset");
  if (orderFence_)
    encoder->waitForFence(orderFence_.get());
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(dstResult.buffer, bindOffset, 0);
  encoder->setBytes(&fillArgs, sizeof(fillArgs), 1);
  encoder->dispatchThreadgroups(MTL::Size{threadgroups, 1, 1},
                                MTL::Size{threadgroupSize, 1, 1});
  if (orderFence_)
    encoder->updateFence(orderFence_.get());
  encoder->endEncoding();
  cmdBuffer->commit();

  METAL_MAYBE_SYNC(*this);
  return success();
}

ErrorOr<DeviceFunctionRef>
MetalDeviceContext::loadFunction(StringRef moduleName, StringRef functionName,
                                 StringRef code, int32_t maxDynamicSharedBytes,
                                 StringRef debugLevel,
                                 int32_t optimizationLevel) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::loadFunction", TRACY_COLOR_PURPLE);

  // Return the cached compute pipeline state if this exact (module, code,
  // entry point) has already been compiled. Metal pipeline-state creation
  // recompiles the shader and generates GPU machine code, so without this the
  // work would repeat on every launch on the critical path (the CUDA/HIP
  // backends cache the same way).
  Device::FunctionCacheKey key{
      /*name=*/Device::FunctionCacheKey::moduleCacheName(moduleName, code),
      /*functionName=*/functionName.str(),
      /*debugLevel=*/debugLevel.str(),
      /*maxDynamicSharedBytes=*/maxDynamicSharedBytes,
      /*optimizationLevel=*/optimizationLevel};
  DeviceFunctionCacheEntryRef entry = device().findInFunctionCache(key);
  if (entry.getPointer() != nullptr) {
    MetalModuleFunctionRef moduleFunc = MetalModuleFunctionRef::take(
        MetalModuleFunction::cast(entry.release()));
    return MetalDeviceFunctionRef::create(*this, std::move(moduleFunc));
  }

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Get the Metal device
  MTL::Device *mtlDevice = MetalDevice::cast(device()).metalDevice();
  if (mtlDevice == nullptr)
    RETURN_ERROR("Failed to get Metal device");

  NS::SharedPtr<MTL::Library> library;

  // Detect if input is binary or MSL source code
  bool isBinary = false;
  if (code.size() >= 4) {
    // Check if it's NOT MSL text (which would start with valid ASCII/UTF-8)
    // Look for non-printable characters in the first few bytes
    isBinary = false;
    for (size_t i = 0; i < std::min(size_t(8), code.size()); ++i) {
      unsigned char c = static_cast<unsigned char>(code[i]);
      if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
        isBinary = true;
        break;
      }
    }
  }

  if (!isBinary) {
    // Compile from MSL source code
    NS::String *sourceString =
        NS::String::string(code.data(), NS::UTF8StringEncoding);
    NS::SharedPtr<MTL::CompileOptions> options =
        NS::TransferPtr(MTL::CompileOptions::alloc()->init());

    NS::Error *error = nullptr;
    library = NS::TransferPtr(
        mtlDevice->newLibrary(sourceString, options.get(), &error));
    if (error) {
      RETURN_ERROR(
          Twine("Failed to create Metal library (compiling MSL to AIR): ") +
          error->localizedDescription()->cString(NS::UTF8StringEncoding));
    }
  } else {
    // Load from binary metallib data
    dispatch_data_t binaryData = dispatch_data_create(
        code.data(), code.size(), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

    NS::Error *error = nullptr;
    library = NS::TransferPtr(mtlDevice->newLibrary(binaryData, &error));
    if (error) {
      RETURN_ERROR(
          Twine("Failed to create Metal library (loading binary/metallib): ") +
          error->localizedDescription()->cString(NS::UTF8StringEncoding));
    }

    // Release the dispatch data
    dispatch_release(binaryData);
  }

  // Get the function from the library
  NS::String *funcName =
      NS::String::string(functionName.data(), NS::UTF8StringEncoding);
  NS::SharedPtr<MTL::Function> function =
      NS::TransferPtr(library->newFunction(funcName));
  if (!function) {
    RETURN_ERROR("Failed to create Metal function: " + functionName.str());
  }

  NS::SharedPtr<MTL::ComputePipelineDescriptor> pipelineDesc =
      NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipelineDesc->setComputeFunction(function.get());
  pipelineDesc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);

  NS::Error *error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> pipelineState =
      NS::TransferPtr(mtlDevice->newComputePipelineState(
          pipelineDesc.get(), MTL::PipelineOptionNone, nullptr, &error));

  if (error) {
    RETURN_ERROR(
        Twine("Failed to create compute pipeline state (GPU machine code "
              "generation): ") +
        error->localizedDescription()->cString(NS::UTF8StringEncoding));
  }

  if (!pipelineState)
    RETURN_ERROR("Failed to create compute pipeline state (no error returned)");

  // Allocate the module function outside any lock region (the cache takes its
  // own lock). If this key was registered concurrently, insertInFunctionCache
  // returns the existing entry and this ref is released before returning.
  MetalModuleFunctionRef moduleFunc = MetalModuleFunctionRef::create(
      std::string(functionName), std::move(pipelineState));
  entry = device().insertInFunctionCache(key, *moduleFunc);
  return MetalDeviceFunctionRef::create(
      *this,
      MetalModuleFunctionRef::take(MetalModuleFunction::cast(entry.release())));
}

ErrorOrSuccess MetalDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    [[maybe_unused]] const size_t *argSizes) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Metal-specific kernel-launch payload travels via args[0]; per-arg
  // sizes and the buffers-to-mark-resident list live on mtlArgs.
  auto *mtlArgs = static_cast<MetalEnqueueFunctionArgs *>(args[0]);

  const MetalDeviceFunction &metalFunc = MetalDeviceFunction::cast(func);
  MTL::ComputePipelineState *pipeline = metalFunc.getPipelineState();
  const MetalDevice &metalDevice = MetalDevice::cast(device());
  MTL::Device *mtlDevice = metalDevice.metalDevice();

  // 1) Validate everything before opening an encoder — Metal asserts on a
  //    cmdBuffer dropped with an open encoder.
  for (uint32_t i = 0; i < argCount; ++i) {
    if (mtlArgs->args[i] == nullptr)
      RETURN_ERROR("Null argument at index " + std::to_string(i));
    if (mtlArgs->argSizes[i] == 0)
      RETURN_ERROR("Unknown argument size at index " + std::to_string(i));
  }
  if (sharedMemBytes > 0) {
    DEFINE_OR_RETURN_ERROR(uint32_t, maxSharedMemBytes,
                           metalDevice.getAttribute(
                               Device::Attribute::MAX_SHARED_MEMORY_PER_BLOCK));
    if (sharedMemBytes > maxSharedMemBytes)
      RETURN_ERROR(
          "External memory byte count " + std::to_string(sharedMemBytes) +
          " exceeds device limit of " + std::to_string(maxSharedMemBytes));
  }

  // 2) Bind small arguments with `setBytes` (below); only oversized args need
  //    a backing MTLBuffer. Pre-allocate those here, before opening the
  //    encoder, so an allocation error doesn't leave one open. Every MAX
  //    kernel argument is either an 8-byte device address (the compiler adds
  //    an extra indirection for Metal pointer args — see
  //    KGEN LowerArgConventions) or a small scalar/capture, so this fallback
  //    path is effectively never taken; it exists to stay correct if a
  //    by-value argument ever exceeds Metal's 4 KB `setBytes` limit.
  static constexpr uint32_t kMaxSetBytes = 4096;
  MTL::ResourceOptions argBufOpts =
      getResourceOptions(Device::AllocMode::ASYNC);
  llvm::SmallVector<NS::SharedPtr<MTL::Buffer>, 8> argBufs;
  argBufs.resize(argCount);
  for (uint32_t i = 0; i < argCount; ++i) {
    if (mtlArgs->argSizes[i] <= kMaxSetBytes)
      continue;
    DEFINE_OR_RETURN_ERROR(MTL::Buffer *, tmpBuf,
                           allocateNewBuffer(mtlDevice, mtlArgs->args[i],
                                             mtlArgs->argSizes[i], argBufOpts));
    argBufs[i] = NS::TransferPtr(tmpBuf);
  }

  // 3) Attach MTL::LogState only when GPU print is wanted; an active capture
  //    suppresses it so captured command buffers stay replayable.
  MTL::CommandBuffer *cmdBuffer = nullptr;
  const bool wantLog =
      isLoggingEnabled(attrs, numAttrs) && !printDisabled_ && !captureActive_;
  if (MTL::LogState *ls = getOrCreateLogState(wantLog)) {
    NS::SharedPtr<MTL::CommandBufferDescriptor> desc =
        NS::TransferPtr(MTL::CommandBufferDescriptor::alloc()->init());
    desc->setLogState(ls);
    cmdBuffer = commandQueue_->commandBuffer(desc.get());
  } else {
    cmdBuffer = commandQueue_->commandBuffer();
  }
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer");

  // 4) Encode + dispatch + commit. Cross-cmdBuffer ordering is handled by
  //    Metal hazard tracking on the tracked buffers declared via
  //    `useResource`.
  MTL::ComputeCommandEncoder *encoder = cmdBuffer->computeCommandEncoder();
  if (encoder == nullptr)
    RETURN_ERROR("Failed to create compute command encoder");
  if (orderFence_)
    encoder->waitForFence(orderFence_.get());
  encoder->setComputePipelineState(pipeline);
  // Bind each argument's bytes into the encoder's argument buffer directly via
  // `setBytes`, avoiding a fresh MTLBuffer allocation + copy per argument on
  // every launch (the dominant per-launch CPU cost on Metal). Oversized args
  // fall back to the pre-allocated backing buffer bound with `setBuffer`.
  for (uint32_t i = 0; i < argCount; ++i) {
    if (mtlArgs->argSizes[i] <= kMaxSetBytes) {
      encoder->setBytes(mtlArgs->args[i], mtlArgs->argSizes[i], i);
      continue;
    }
    encoder->setBuffer(argBufs[i].get(), 0, i);
  }
  // Mark the buffers reachable through the kernel's pointer arguments as
  // resident so the GPU may access them.
  for (int32_t i = 0; i < mtlArgs->numBuffers; i++)
    encoder->useResource(getMetalBuffer(mtlArgs->buffers[i]->ptr()),
                         MTL::ResourceUsageWrite);
  // FIXME: GEX-3761 only mark resources actually used by the kernel.
  // NOTE: `mtlArgs->buffers` (the arg-derived resident set above) is NOT a
  // complete substitute for this all-tracked-buffers loop: dropping this loop
  // and relying only on the per-arg set produces silently wrong decode output
  // on paged-KV models (e.g. Llama-3.1-8B emits garbage tokens on Metal),
  // because some device buffers a kernel reads are reached indirectly (e.g.
  // through a device-side block/pointer table passed by value) and so are not
  // enumerated as pointer args. Narrowing residency safely requires the kernel
  // to declare its full working set; until then this conservative loop stays.
  // `forEachTrackedBuffer` marks the exact same buffers in the same order as
  // iterating `getTrackedBuffers()`, but without the per-launch `std::map`
  // value-copy (GEX-3761): the `useResource(buffer, Write)` calls are
  // unchanged.
  metalDevice.forEachTrackedBuffer([&](const MetalDevice::BufferAllocation &a) {
    encoder->useResource(a.buffer, MTL::ResourceUsageWrite);
  });
  if (sharedMemBytes > 0)
    encoder->setThreadgroupMemoryLength(sharedMemBytes, /*index=*/0);
  encoder->dispatchThreadgroups(MTL::Size{gridX, gridY, gridZ},
                                MTL::Size{blockX, blockY, blockZ});
  if (orderFence_)
    encoder->updateFence(orderFence_.get());
  encoder->endEncoding();
  cmdBuffer->commit(); // async (DRIV-164)

  return success();
}

ErrorOr<DeviceStreamRef> MetalDeviceContext::stream() const {
  // Streams enqueue through a mutable context.
  return MetalDeviceStreamRef::create(const_cast<MetalDeviceContext &>(*this));
}

unsigned MetalDeviceContext::numStreams() const { return queuePool_->size(); }

ErrorOr<DeviceStreamRef> MetalDeviceContext::createStream(int /*priority*/) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  NS::SharedPtr<MTL::CommandQueue> commandQueue =
      NS::TransferPtr(metalDevice()->newCommandQueue());
  if (!commandQueue)
    RETURN_ERROR("Failed to create Metal command queue for stream");
  unsigned id = queuePool_->emplace_back(
      makeMetalQueue(metalDevice(), std::move(commandQueue)));

  // The stream retains the view.
  MetalDeviceContextRef view = MetalDeviceContextRef::take(
      new MetalDeviceContext(MetalDevice::cast(device()), queuePool_, id));
  return MetalDeviceStreamRef::create(*view);
}

ErrorOr<DeviceContextRef> MetalDeviceContext::selectStream(unsigned streamId) {
  if (streamId >= queuePool_->size())
    return Error("invalid stream id: " + llvm::Twine(streamId));
  if (queuePool_->get(streamId).commandQueue.get() == commandQueue_.get())
    return DeviceContextRef::copy(this);
  return DeviceContextRef::take(new MetalDeviceContext(
      MetalDevice::cast(device()), queuePool_, streamId));
}

MetalDeviceStream::MetalDeviceStream(MetalDeviceContext &ctx)
    : DeviceStream(ctx), metalCtx_(ctx) {}

ErrorOrSuccess MetalDeviceStream::synchronize() {
  return metalCtx_.synchronize();
}

ErrorOr<DeviceEventRef> MetalDeviceStream::enqueueEvent() const {
  return metalCtx_.enqueueEvent();
}

ErrorOrSuccess MetalDeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    const size_t *argSizes) const {
  return metalCtx_.enqueueFunctionExecDirect(
      func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, attrs,
      numAttrs, args, argCount, argSizes);
}

ErrorOrSuccess MetalDeviceStream::eventRecord(const DeviceEvent &event) const {
  return metalCtx_.enqueueEvent(event);
}

ErrorOrSuccess
MetalDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  // enqueueWaitFor only reads the event.
  return metalCtx_.enqueueWaitFor(const_cast<DeviceEvent &>(event));
}

void *MetalDeviceStream::ptr() const { return metalCtx_.commandQueue(); }

ErrorOrSuccess MetalDeviceContext::synchronize() {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::synchronize", TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Check for any pending errors first, similar to CUDA's behavior
  RETURN_IF_ERROR(checkPendingError());

  // Fence-chained because Metal only orders command buffer starts, not
  // completions.
  RETURN_IF_ERROR(drainQueue());

  // Matches CUDA. Without it the deferred host frees drain only from
  // `~DeviceBuffer`, so the last drop of a teardown is never collected.
  RETURN_IF_ERROR(pollPendingEvents());

  return success();
}

ErrorOr<DeviceEventRef> MetalDeviceContext::eventCreate(bool enableTiming) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::eventCreate", TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (enableTiming)
    RETURN_ERROR("Metal events do not support timing");
  if (!orderFence_)
    RETURN_ERROR("Metal ordering fence unavailable");

  NS::SharedPtr<MTL::SharedEvent> event =
      NS::TransferPtr(metalDevice()->newSharedEvent());
  if (!event)
    RETURN_ERROR("Failed to create MTLSharedEvent");
  return MetalDeviceEventRef::create(MetalDevice::cast(device()),
                                     std::move(event));
}

ErrorOr<DeviceEventRef> MetalDeviceContext::enqueueEvent() {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  // TODO(DRIV-419): allocates a fresh MTLSharedEvent per call; add an event
  // cache like CUDA's if this shows up hot.
  DEFINE_OR_RETURN_ERROR(DeviceEventRef, event,
                         eventCreate(/*enableTiming=*/false));
  RETURN_IF_ERROR(enqueueEvent(*event));
  return event;
}

ErrorOrSuccess MetalDeviceContext::enqueueEvent(const DeviceEvent &event) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (!orderFence_)
    RETURN_ERROR("Metal ordering fence unavailable");
  const MetalDeviceEvent &metalEvent = MetalDeviceEvent::cast(event);

  std::lock_guard<std::mutex> lock(metalEvent.recordMutex_);
  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer for event record");
  // Chain to the previous signal so a cross-queue re-record cannot signal
  // ahead of an earlier record still gated behind prior work.
  if (metalEvent.waitValue_ > 0)
    cmdBuffer->encodeWait(metalEvent.event_.get(), metalEvent.waitValue_);
  MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
  if (blit == nullptr)
    RETURN_ERROR("Failed to create blit encoder for event record");
  // The fence wait defers the signal until previously committed work finishes.
  blit->waitForFence(orderFence_.get());
  blit->updateFence(orderFence_.get());
  blit->endEncoding();
  cmdBuffer->encodeSignalEvent(metalEvent.event_.get(),
                               metalEvent.waitValue_ + 1);
  cmdBuffer->commit();
  metalEvent.waitValue_ += 1;
  return success();
}

ErrorOrSuccess MetalDeviceContext::enqueueWaitFor(DeviceEvent &event) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueWaitFor",
                       TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (!orderFence_)
    RETURN_ERROR("Metal ordering fence unavailable");
  MetalDeviceEvent &metalEvent = MetalDeviceEvent::cast(event);
  uint64_t target = metalEvent.waitValue();
  if (target == 0)
    return success();

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer for event wait");
  // The event gates this buffer; the fence update gates later work behind it.
  cmdBuffer->encodeWait(metalEvent.sharedEvent(), target);
  MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
  if (blit == nullptr)
    RETURN_ERROR("Failed to create blit encoder for event wait");
  blit->waitForFence(orderFence_.get());
  blit->updateFence(orderFence_.get());
  blit->endEncoding();
  cmdBuffer->commit();
  return success();
}

ErrorOrSuccess MetalDeviceContext::enqueueWaitFor(DeviceContext &ctx) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueWaitFor",
                       TRACY_COLOR_PURPLE);

  if (&ctx == this)
    return success();
  // Non-Metal peers have no shared event to wait on; drain them instead.
  if (MetalDeviceContext::verified_cast(&ctx) == nullptr)
    return ctx.synchronize();
  DEFINE_OR_RETURN_ERROR(DeviceEventRef, event, ctx.enqueueEvent());
  return enqueueWaitFor(*event);
}

// Like CUDA's timer events, the start and stop markers are ordered behind
// earlier work and the elapsed time is read from the GPU timeline, so it covers
// the work enqueued between them rather than the host's enqueue time.
ErrorOr<DeviceTimerRef> MetalDeviceContext::startTimer() {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::startTimer", TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (!orderFence_)
    RETURN_ERROR("Metal ordering fence unavailable");
  DEFINE_OR_RETURN_ERROR(MTL::CommandBuffer *, startMarker,
                         commitOrderedMarker());
  return MetalDeviceTimerRef::create(*this, startMarker);
}

ErrorOr<std::chrono::nanoseconds>
MetalDeviceContext::stopTimer(const DeviceTimer &timer) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::stopTimer", TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (!orderFence_)
    RETURN_ERROR("Metal ordering fence unavailable");
  MTL::CommandBuffer *startMarker = MetalDeviceTimer::cast(timer).startMarker();
  DEFINE_OR_RETURN_ERROR(MTL::CommandBuffer *, stopMarker,
                         commitOrderedMarker());
  stopMarker->waitUntilCompleted();
  startMarker->waitUntilCompleted();
  if (startMarker->status() != MTL::CommandBufferStatusCompleted ||
      stopMarker->status() != MTL::CommandBufferStatusCompleted)
    RETURN_ERROR("Metal timer marker command buffer failed");

  std::chrono::duration<double> elapsed(stopMarker->GPUEndTime() -
                                        startMarker->GPUEndTime());
  return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
}

ErrorOr<Device::MemInfo> MetalDeviceContext::getDeviceMemoryInfo() const {
  return device().getMemoryInfo();
}

ErrorOrSuccess MetalDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                            Device::AllocMode mode) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memAlloc", TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (ptr == nullptr)
    RETURN_ERROR("Invalid allocation parameters: ptr is null");
  if (byteSize == 0) {
    *ptr = nullptr;
    return success();
  }

  MTL::Device *metalDev = MetalDevice::cast(device()).metalDevice();
  if (metalDev == nullptr)
    RETURN_ERROR("Invalid Metal device: null pointer");

  // create the Metal buffer
  MTL::ResourceOptions options = getResourceOptions(mode);
  // Donate newBuffer's +1 to the tracker; memFree releases.
  // NS::SharedPtr + detach() is UB here: ~SharedPtr does nullptr->release().
  DEFINE_OR_RETURN_ERROR(MTL::Buffer *, buf,
                         allocateNewBuffer(metalDev, byteSize, options));
  if (buf == nullptr)
    RETURN_ERROR("Failed to create Metal buffer");

  // Based on mode of allocation, return the appropriate pointer
  switch (mode) {
  case Device::AllocMode::HOST: {
    // For host buffers, return the contents pointer
    void *contents = buf->contents();
    if (contents == nullptr) {
      buf->release();
      RETURN_ERROR("Failed to get contents from Metal buffer");
    }
    // TODO: Update ptr() to return from MTL::Buffer
    *ptr = contents;

    // Track this allocation at device level. The +1 from newBuffer is
    // now owned by the tracker; memFree releases it.
    MetalDevice::cast(device()).trackBuffer(contents, buf);
    return success();
  }
  case Device::AllocMode::SYNC:
  case Device::AllocMode::ASYNC: {
    // For device buffers, return the GPU address
    uint64_t gpuAddress = buf->gpuAddress();
    if (gpuAddress == 0) {
      buf->release();
      RETURN_ERROR("Failed to get GPU address from Metal buffer");
    }
    *ptr = reinterpret_cast<void *>(gpuAddress);

    // Track this allocation at device level. See HOST branch for
    // ownership semantics.
    // TODO: Remove tracking of device buffers and add a new field to the
    // deviceBuffer class to track the MTL::buffer
    MetalDevice::cast(device()).trackBuffer(
        reinterpret_cast<void *>(gpuAddress), buf);
    return success();
  }
  }
}

ErrorOrSuccess MetalDeviceContext::memWrap(void **devPtr, void *hostPtr,
                                           size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memWrap", TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  void *ptr = hostPtr;
  if (ptr == nullptr)
    RETURN_ERROR("memWrap: ptr is null");
  if (byteSize == 0)
    RETURN_ERROR("memWrap: byteSize is zero");

  // NoCopy aliases whole VM pages; rounding out to one here could hand the GPU
  // memory outside the caller's range, which may not be mapped at all.
  const size_t pageSize = static_cast<size_t>(getpagesize());
  if (reinterpret_cast<uintptr_t>(ptr) % pageSize != 0)
    RETURN_ERROR("memWrap: host pointer is not page-aligned (page size " +
                 std::to_string(pageSize) + ")");
  if (byteSize % pageSize != 0)
    RETURN_ERROR("memWrap: byteSize " + std::to_string(byteSize) +
                 " is not a multiple of the page size " +
                 std::to_string(pageSize));

  MTL::Device *metalDev = MetalDevice::cast(device()).metalDevice();
  if (metalDev == nullptr)
    RETURN_ERROR("Invalid Metal device: null pointer");

  // Default-cached, not `memAlloc`'s write-combined: the caller reads these
  // pages with ordinary CPU accesses.
  MTL::ResourceOptions options =
      static_cast<MTL::ResourceOptions>(MTL::StorageModeShared) |
      static_cast<MTL::ResourceOptions>(MTL::CPUCacheModeDefaultCache);

  // Donate newBuffer's +1 to the tracker; memFree releases.
  DEFINE_OR_RETURN_ERROR(MTL::Buffer *, buf,
                         wrapExistingPages(metalDev, ptr, byteSize, options));
  if (buf == nullptr)
    RETURN_ERROR("Failed to wrap host memory in a Metal buffer");

  // Key by the GPU address, as `memAlloc` does for device memory: that is what
  // kernels dereference and what `memFree` is handed back.
  uint64_t gpuAddress = buf->gpuAddress();
  if (gpuAddress == 0) {
    buf->release();
    RETURN_ERROR("memWrap: wrapped Metal buffer has no GPU address");
  }
  *devPtr = reinterpret_cast<void *>(gpuAddress);

  // Untracked, the range is invisible to the copy paths and memFree rejects it.
  MetalDevice::cast(device()).trackBuffer(*devPtr, buf);
  return success();
}

ErrorOrSuccess MetalDeviceContext::memFree(void *ptr, Device::FreeKind kind) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::memFree", TRACY_COLOR_PURPLE);

  if (ptr == nullptr)
    return success();

  // `kind` is unused: whether releasing the buffer also frees pages was settled
  // by the deallocator passed at creation.
  MTL::Buffer *bufferToRelease = getMetalBuffer(ptr);

  if (bufferToRelease == nullptr) {
    // Not a Metal buffer allocated by us - this is an error
    RETURN_ERROR("Attempt to free non-Metal memory - operation failed");
  }

  // Release the buffer and remove from tracking
  bufferToRelease->release();
  MetalDevice::cast(device()).untrackBuffer(ptr);

  return success();
}

ErrorOr<std::tuple<void *, size_t,
                   std::optional<llvm::sys::fs::mapped_file_region>>>
MetalDeviceContext::memmapFileImpl(
    const std::filesystem::path &filename) const {
  RETURN_ERROR("Metal memmapFileImpl not implemented");
}

void *MetalDeviceContext::driverContext() const {
  return const_cast<MetalDeviceContext *>(this);
}

ErrorOr<MTL::CommandBuffer *> MetalDeviceContext::commitOrderedMarker() const {
  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer for ordered marker");
  if (orderFence_) {
    MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
    if (blit == nullptr)
      RETURN_ERROR("Failed to create blit encoder for ordered marker");
    blit->waitForFence(orderFence_.get());
    blit->updateFence(orderFence_.get());
    blit->endEncoding();
  }
  cmdBuffer->commit();
  return cmdBuffer;
}

ErrorOrSuccess MetalDeviceContext::drainQueue() const {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  DEFINE_OR_RETURN_ERROR(MTL::CommandBuffer *, marker, commitOrderedMarker());
  marker->waitUntilCompleted();
  return success();
}

ErrorOrSuccess MetalDeviceContext::enqueueMetalCopy(MTL::Buffer *src,
                                                    MTL::Buffer *dst,
                                                    size_t srcOffset,
                                                    size_t dstOffset,
                                                    size_t size) const {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueMetalCopy",
                       TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::CommandBuffer *cmdBuffer = commandQueue_->commandBuffer();
  if (cmdBuffer == nullptr) {
    RETURN_ERROR("Failed to create command buffer");
  }

  MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
  if (blit == nullptr)
    RETURN_ERROR("Failed to create blit encoder");

  if (orderFence_)
    blit->waitForFence(orderFence_.get());
  blit->copyFromBuffer(src, srcOffset, dst, dstOffset, size);
  if (orderFence_)
    blit->updateFence(orderFence_.get());
  blit->endEncoding();

  cmdBuffer->commit();
  return success();
}

MetalDevice::BufferLookupResult
MetalDeviceContext::getMetalBufferWithOffset(const void *ptr) const {
  return MetalDevice::cast(device()).findContainingBuffer(ptr);
}

ErrorOr<void *>
MetalDeviceContext::getHostAccessiblePointer(const void *devicePtr) {
  if (devicePtr == nullptr)
    RETURN_ERROR("getHostAccessiblePointer: pointer is null");

  // The range map is keyed by whatever address `memAlloc`/`chunkAlloc` handed
  // out, so this resolves device (`gpuAddress`) pointers, host (`contents`)
  // pointers, sub-buffers, and interior pointers alike. Feeding a host pointer
  // back in is idempotent: its base is already `contents()`.
  MetalDevice::BufferLookupResult found =
      MetalDevice::cast(device()).findContainingBuffer(devicePtr);
  if (!found.isValid())
    RETURN_ERROR(
        "getHostAccessiblePointer: pointer was not allocated by this device");

  // Null `contents()` means the buffer was created without host (shared)
  // access; `getStorageMode` only hands out `StorageModeShared` today, so this
  // guards a future private-storage allocation path.
  void *contents = found.buffer->contents();
  if (contents == nullptr)
    RETURN_ERROR("getHostAccessiblePointer: Metal buffer was not created with "
                 "host (shared) access");

  return static_cast<char *>(contents) + found.offset;
}

//
// Helper methods for buffer management
//
bool MetalDeviceContext::isMetalBuffer(const void *ptr) const {
  return MetalDevice::cast(device()).isTrackedBuffer(ptr);
}

MTL::Buffer *MetalDeviceContext::getMetalBuffer(const void *ptr) const {
  return MetalDevice::cast(device()).getTrackedBuffer(ptr);
}

MTL::StorageMode
MetalDeviceContext::getStorageMode(Device::AllocMode mode) const {
  // Always use Shared storage mode for all allocations
  // Todo(Amir): Make this configurable in the future via an Environment
  // variable
  return MTL::StorageModeShared;
}

MTL::ResourceOptions
MetalDeviceContext::getResourceOptions(Device::AllocMode mode) const {
  // Pick the storage mode
  MTL::StorageMode storage = getStorageMode(mode);
  // We always want write-combined CPU cache for GPU buffers
  MTL::CPUCacheMode cache = MTL::CPUCacheModeWriteCombined;
  // Combine into a ResourceOptions bitmask
  return static_cast<MTL::ResourceOptions>(storage) |
         static_cast<MTL::ResourceOptions>(cache);
}

const MetalDeviceContext *
MetalDeviceContext::verified_cast(const DeviceContext *ctx) {
  return DeviceContext::cast<MetalDeviceContext>(ctx);
}

MetalDeviceContext *MetalDeviceContext::verified_cast(DeviceContext *ctx) {
  return DeviceContext::cast<MetalDeviceContext>(ctx);
}

/// Returns true if the installed SDK supports the indicated Major.Minor
/// revision of the Metal Shading Language.
/// Parses the SDK header once on first call and caches all discovered
/// MTLLanguageVersion values for the process lifetime.
static bool sdkSupportsMSLVersion(int maj, int min) {
  static const auto supportedVersions = [] {
    std::set<std::pair<int, int>> versions;

    // Resolve the active Xcode developer directory.
    std::filesystem::path devDir;
    if (const char *env = std::getenv("DEVELOPER_DIR")) {
      devDir = env;
    } else {
      // Same symlink that `xcode-select -p` resolves.
      std::error_code ec;
      devDir = std::filesystem::read_symlink("/var/db/xcode_select_link", ec);
      if (ec)
        return versions;
    }

    // Header that defines the MTLLanguageVersion enum.
    std::filesystem::path headerPath =
        devDir / "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk" /
        "System/Library/Frameworks/Metal.framework" /
        "Versions/A/Headers/MTLLibrary.h";
    auto bufOrErr = llvm::MemoryBuffer::getFile(headerPath.c_str());
    if (!bufOrErr)
      return versions;

    // Scan for MTLLanguageVersionX_Y enum entries.
    StringRef prefix = "MTLLanguageVersion";
    for (llvm::line_iterator it(**bufOrErr, /*SkipBlanks=*/true, '\0');
         !it.is_at_end(); ++it) {
      StringRef line = *it;
      auto pos = line.find(prefix);
      if (pos == StringRef::npos)
        continue;
      StringRef suffix = line.substr(pos + prefix.size());
      auto [majStr, rest] = suffix.split('_');
      int major = 0, minor = 0;
      // majStr is clean (e.g. "1"); rest may have trailing text
      // (e.g. "1 API_AVAILABLE(...)"), so use consumeInteger.
      if (!majStr.getAsInteger(10, major) && !rest.consumeInteger(10, minor))
        versions.emplace(major, minor);
    }
    return versions;
  }();

  return supportedVersions.count({maj, min}) > 0;
}

ErrorOr<std::string> MetalDevice::getArchName(MTL::Device *d) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  // CreateSystemDefaultDevice returns a +1 (Create-rule) reference that a
  // pool cannot reclaim; own it so the nullptr-device path does not leak.
  NS::SharedPtr<MTL::Device> createdDevice;
  if (d == nullptr)
    createdDevice = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  MTL::Device *device = d ? d : createdDevice.get();
  if (device == nullptr)
    RETURN_ERROR("Failed to create Metal device.");
  std::string name = device->name()->cString(NS::UTF8StringEncoding);

  DEFINE_OR_RETURN_ERROR(int, cc, getComputeCapability(name));
  std::string archName = std::to_string(cc);
  if (device->supportsFamily(MTL::GPUFamilyMetal4) &&
      sdkSupportsMSLVersion(4, 0)) {
    // Prefer to use latest supported Metal version by default, but only if
    // the installed Xcode's Metal compiler can actually handle Metal 4.0 AIR.
    archName += "-metal4";
  }
  return archName;
}

ErrorOr<const char *> MetalDevice::getArchName() const {
  // Lazily compute the arch string on first call and cache it. The static
  // `getArchName(MTL::Device *)` form requires Metal device introspection
  // (autorelease pool + name parsing + Metal4 feature detection) so we avoid
  // paying that cost at construction time. Mirrors HIPDevice's cached
  // `arch_` pattern (HIPDeviceContext.cpp:414), but with lazy population
  // since MetalDevice's constructor does not receive an arch parameter.
  std::lock_guard<std::mutex> lock(archMutex_);
  if (arch_.empty()) {
    DEFINE_OR_RETURN_ERROR(std::string, computed, getArchName(device_.get()));
    arch_ = std::move(computed);
  }
  return arch_.c_str();
}

ErrorOrSuccess MetalDeviceContext::startMetalTraceCapture(const char *path) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::startMetalTraceCapture",
                       TRACY_COLOR_PURPLE);
  if (captureActive_)
    RETURN_ERROR("a Metal capture is already in progress");

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::CaptureManager *mgr = MTL::CaptureManager::sharedCaptureManager();
  if (!mgr->supportsDestination(MTL::CaptureDestinationGPUTraceDocument))
    RETURN_ERROR("Metal GPU trace capture is unavailable; set "
                 "MTL_CAPTURE_ENABLED=1 before launching the process");

  // Work committed before the window must not straddle the capture boundary;
  // Metal only captures command buffers created after startCapture.
  RETURN_IF_ERROR(synchronize());

  NS::SharedPtr<MTL::CaptureDescriptor> desc =
      NS::TransferPtr(MTL::CaptureDescriptor::alloc()->init());
  desc->setCaptureObject(metalDevice());
  desc->setDestination(MTL::CaptureDestinationGPUTraceDocument);
  // NS::String::string() and NS::URL::fileURLWithPath() are autoreleased
  // convenience constructors — use as raw pointers within the pool scope.
  // Wrapping in NS::TransferPtr would double-free when the pool drains.
  NS::String *nsPath = NS::String::string(path, NS::UTF8StringEncoding);
  NS::URL *url = NS::URL::fileURLWithPath(nsPath);
  desc->setOutputURL(url);

  NS::Error *error = nullptr;
  if (!mgr->startCapture(desc.get(), &error)) {
    std::string msg = "failed to start Metal capture";
    if (error && error->localizedDescription()) {
      if (const char *c =
              error->localizedDescription()->cString(NS::UTF8StringEncoding))
        msg += std::string(": ") + c;
    }
    RETURN_ERROR(msg);
  }
  captureActive_ = true;
  return success();
}

ErrorOrSuccess MetalDeviceContext::stopMetalTraceCapture() {
  TRACY_ZONE_SCOPED_NC("MetalDeviceContext::stopMetalTraceCapture",
                       TRACY_COLOR_PURPLE);
  if (!captureActive_)
    RETURN_ERROR("no Metal capture is in progress");
  // Committed command buffers are async; drain them before stopping or the
  // trace is truncated.
  RETURN_IF_ERROR(synchronize());
  // Call stopCapture outside any local autorelease pool: on some macOS
  // versions CaptureManager autoreleases internal objects that become
  // invalid if drained while still referenced by the framework.
  MTL::CaptureManager::sharedCaptureManager()->stopCapture();
  captureActive_ = false;
  return success();
}

ErrorOrSuccess MetalDeviceContext::setMetalPrintEnabled(bool enabled) {
  printDisabled_ = !enabled;
  return success();
}

MTL::LogState *MetalDeviceContext::getOrCreateLogState(bool enable) {
  if (!enable)
    return nullptr;
  if (logState_)
    return logState_.get();

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::Device *mtlDevice = metalDevice();

  NS::SharedPtr<MTL::LogStateDescriptor> desc =
      NS::TransferPtr(MTL::LogStateDescriptor::alloc()->init());
  desc->setLevel(MTL::LogLevelDebug);

  NS::Error *error = nullptr;
  logState_ = NS::TransferPtr(mtlDevice->newLogState(desc.get(), &error));

  if (error || !logState_)
    return nullptr; // LogState not supported — fall back silently.

  // Register a handler that writes GPU print output to stdout so it
  // can be captured by test infrastructure (FileCheck pipes stdout).
  logState_->addLogHandler([](NS::String *subsystem, NS::String *category,
                              MTL::LogLevel /*level*/, NS::String *message) {
    if (message) {
      const char *cstr = message->cString(NS::UTF8StringEncoding);
      if (cstr) {
        fprintf(stdout, "%s", cstr);
        fflush(stdout);
      }
    }
  });

  return logState_.get();
}

//
//
// ===== MetalDeviceEvent ===== //
//

// A signal that never arrives (errored or wedged command buffer) is not
// recoverable on Metal; time out instead of parking forever.
static constexpr uint64_t kSynchronizeTimeoutMs = 60000;

MetalDeviceEvent::MetalDeviceEvent(MetalDevice &device,
                                   NS::SharedPtr<MTL::SharedEvent> event,
                                   uint64_t waitValue)
    : DeviceEvent(device), event_(std::move(event)), waitValue_(waitValue) {}

uint64_t MetalDeviceEvent::waitValue() const {
  std::lock_guard<std::mutex> lock(recordMutex_);
  return waitValue_;
}

ErrorOrSuccess MetalDeviceEvent::synchronize() {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  uint64_t target = waitValue();
  if (target == 0)
    return success();
  // Block like cuEventSynchronize; the timeout is the only error to propagate.
  if (!event_->waitUntilSignaledValue(target, kSynchronizeTimeoutMs))
    RETURN_ERROR("Timed out waiting for Metal shared event");
  return success();
}

ErrorOr<bool> MetalDeviceEvent::isReady() const {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  uint64_t target = waitValue();
  // Never recorded is always ready (CUDA semantics).
  if (target == 0)
    return true;
  return event_->signaledValue() >= target;
}

const MetalDeviceEvent &MetalDeviceEvent::cast(const DeviceEvent &event) {
  return static_cast<const MetalDeviceEvent &>(event);
}

MetalDeviceEvent &MetalDeviceEvent::cast(DeviceEvent &event) {
  return static_cast<MetalDeviceEvent &>(event);
}

//
//
// ===== MetalDeviceTimer ===== //
//

// The marker is autoreleased by the pool in startTimer; retain it until
// stopTimer reads its end time.
MetalDeviceTimer::MetalDeviceTimer(MetalDeviceContext &context,
                                   MTL::CommandBuffer *startMarker)
    : DeviceTimer(context), startMarker_(NS::RetainPtr(startMarker)) {}

const MetalDeviceTimer &MetalDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const MetalDeviceTimer &>(timer);
}

//
//
// ===== MetalModuleFunction ===== //
MetalModuleFunction::MetalModuleFunction(
    std::string name, NS::SharedPtr<MTL::ComputePipelineState> pipelineState)
    : DeviceFunctionCacheEntry(std::move(name)),
      pipelineState_(std::move(pipelineState)) {}

const MetalModuleFunction *
MetalModuleFunction::cast(const DeviceFunctionCacheEntry *entry) {
  return static_cast<const MetalModuleFunction *>(entry);
}

// ===== MetalDeviceFunction ===== //
//

MetalDeviceFunction::MetalDeviceFunction(MetalDeviceContext &context,
                                         MetalModuleFunctionRef modFunc)
    : DeviceFunction(context), moduleFunction_(std::move(modFunc)) {}

const std::string &MetalDeviceFunction::name() const {
  return moduleFunction_->name();
}

ErrorOrSuccess MetalDeviceFunction::copyToNamedConstant(const char *name,
                                                        const void *data,
                                                        size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("MetalDeviceFunction::copyToNamedConstant",
                       TRACY_COLOR_PURPLE);
  // Metal doesn't support named constants in the same way as CUDA/HIP
  RETURN_ERROR("Named constants not supported in Metal");
}

ErrorOr<int32_t> MetalDeviceFunction::getAttribute(int32_t code) {
  // Metal doesn't support the same attributes as CUDA/HIP
  RETURN_ERROR("Attributes not supported in Metal");
}

const MetalDeviceFunction &
MetalDeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const MetalDeviceFunction &>(func);
}

} // namespace M::Driver

#endif // MODULAR_MACOS
