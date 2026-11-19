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
// This file implements the allocator interfaces over the legacy driver, and
// the factories that build a `MemoryManager` over a `Device` from the driver
// config.
//
//===----------------------------------------------------------------------===//

#include "DeviceAllocator.h"
#include "MLRT/lib/Driver/MemoryManager/MemoryManager.h"
#include "Support/StringExtras.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

static constexpr int KB = 1024;                  // NOLINT
static constexpr size_t GB = 1024 * 1024 * 1024; // NOLINT

// Ceiling on memory_manager_vmm_drain_budget_ms; see where it is read.
static constexpr int64_t kMaxDrainBudgetMs = 60 * 1000;

namespace M::Driver {

// The allocator enums share their values with the legacy driver's, so that
// conversion is a cast. See `AllocatorContext`.
static_assert(static_cast<uint8_t>(AllocatorContext::AllocMode::HOST) ==
              static_cast<uint8_t>(Device::AllocMode::HOST));
static_assert(static_cast<uint8_t>(AllocatorContext::AllocMode::SYNC) ==
              static_cast<uint8_t>(Device::AllocMode::SYNC));
static_assert(static_cast<uint8_t>(AllocatorContext::AllocMode::ASYNC) ==
              static_cast<uint8_t>(Device::AllocMode::ASYNC));
static_assert(static_cast<uint8_t>(AllocatorContext::FreeKind::HOST) ==
              static_cast<uint8_t>(Device::FreeKind::HOST));
static_assert(static_cast<uint8_t>(AllocatorContext::FreeKind::DEVICE) ==
              static_cast<uint8_t>(Device::FreeKind::DEVICE));

AllocatorContext::AllocMode toAllocatorAllocMode(Device::AllocMode mode) {
  return static_cast<AllocatorContext::AllocMode>(mode);
}

Device::MemInfo toDeviceMemInfo(const AllocatorMemInfo &info) {
  Device::MemInfo out;
  out.total = info.total;
  out.free = info.free;
  return out;
}

static AllocatorMemInfo toAllocatorMemInfo(const Device::MemInfo &info) {
  AllocatorMemInfo out;
  out.total = info.total;
  out.free = info.free;
  return out;
}

ErrorOrSuccess DeviceAllocatorChunks::chunkAlloc(void **ptr, size_t byteSize,
                                                 bool isHost) {
  return device_.chunkAlloc(ptr, byteSize, isHost);
}

ErrorOrSuccess DeviceAllocatorChunks::chunkFree(void *ptr, bool isHost) {
  return device_.chunkFree(ptr, isHost);
}

ErrorOr<AllocatorMemInfo> DeviceAllocatorChunks::getMemoryInfo() const {
  ErrorOr<Device::MemInfo> info = device_.getMemoryInfo();
  if (info.isError())
    return info.takeError();
  return toAllocatorMemInfo(*info);
}

ErrorOr<void *> DeviceAllocatorVmm::vmmAddressReserve(size_t byteSize) {
  return device_.vmmAddressReserve(byteSize);
}

ErrorOrSuccess DeviceAllocatorVmm::vmmAddressFree(void *va, size_t byteSize) {
  return device_.vmmAddressFree(va, byteSize);
}

ErrorOr<uint64_t> DeviceAllocatorVmm::vmmCreate(size_t byteSize) {
  return device_.vmmCreate(byteSize);
}

ErrorOrSuccess DeviceAllocatorVmm::vmmRelease(uint64_t handle) {
  return device_.vmmRelease(handle);
}

ErrorOrSuccess DeviceAllocatorVmm::vmmMapAndSetAccess(void *va, size_t byteSize,
                                                      uint64_t handle) {
  return device_.vmmMapAndSetAccess(va, byteSize, handle);
}

ErrorOrSuccess DeviceAllocatorVmm::vmmUnmap(void *va, size_t byteSize) {
  return device_.vmmUnmap(va, byteSize);
}

ErrorOrSuccess DeviceAllocatorVmm::vmmMap(void *va, size_t byteSize,
                                          uint64_t handle) {
  return device_.vmmMap(va, byteSize, handle);
}

ErrorOrSuccess DeviceAllocatorVmm::vmmSetAccess(void *va, size_t byteSize) {
  return device_.vmmSetAccess(va, byteSize);
}

namespace {

// A `DeviceEvent` as the memory manager sees it.
class DeviceAllocatorEvent final : public AllocatorEvent {
public:
  // Returns a null ref for a null `event`, never a wrapper around one: the
  // free list reads a null base-stream event as a stream that is its own base.
  static ErrorOr<AllocatorEventRef> wrap(ErrorOr<DeviceEventRef> event) {
    if (event.isError())
      return event.takeError();
    DeviceEventRef ref = event.takeValue();
    if (!ref)
      return AllocatorEventRef();
    return AllocatorEventRef::take(new DeviceAllocatorEvent(std::move(ref)));
  }

  // REQUIRES: `event` came from `wrap`.
  static DeviceEvent &unwrap(AllocatorEvent &event) {
    return *static_cast<DeviceAllocatorEvent &>(event).event_;
  }

  ErrorOr<bool> isReady() const override { return event_->isReady(); }
  ErrorOrSuccess synchronize() override { return event_->synchronize(); }

private:
  explicit DeviceAllocatorEvent(DeviceEventRef event)
      : event_(std::move(event)) {}

  DeviceEventRef event_;
};

} // namespace

// Forwards each call to the `DeviceContext`, which befriends it for the
// protected `memAlloc` and `memFree`.
class DeviceAllocatorContext final : public AllocatorContext {
public:
  explicit DeviceAllocatorContext(DeviceContext &context) : context_(context) {}

  ErrorOr<AllocatorEventRef> enqueueEvent() override {
    return DeviceAllocatorEvent::wrap(context_.enqueueEvent());
  }
  ErrorOr<AllocatorEventRef> enqueueBaseStreamEvent() override {
    return DeviceAllocatorEvent::wrap(context_.enqueueBaseStreamEvent());
  }
  ErrorOrSuccess enqueueWaitFor(AllocatorEvent &event) override {
    return context_.enqueueWaitFor(DeviceAllocatorEvent::unwrap(event));
  }

  bool isGraphCapturing() const override { return context_.isGraphCapturing(); }
  bool isBaseStreamCapturing() const override {
    return context_.isBaseStreamCapturing();
  }

  ErrorOrSuccess pollPendingEvents() override {
    return context_.pollPendingEvents();
  }
  bool flushIdleStreamViews() override {
    return context_.flushIdleStreamViews();
  }

  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          AllocMode mode) override {
    return context_.memAlloc(ptr, byteSize,
                             static_cast<Device::AllocMode>(mode));
  }
  ErrorOrSuccess memFree(void *ptr, FreeKind kind) override {
    return context_.memFree(ptr, static_cast<Device::FreeKind>(kind));
  }

  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t val,
                                  size_t valSize) override {
    return context_.enqueueSetMemory(dst, byteSize, val, valSize);
  }
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) override {
    return context_.enqueueCopyFromDevice(dst, src, byteSize);
  }
  ErrorOrSuccess synchronize() override { return context_.synchronize(); }

private:
  DeviceContext &context_;
};

std::unique_ptr<MemoryManagerFreeList>
createFreeListForContext(DeviceContext &context, MemoryManager &memoryManager,
                         MemoryManagerFreeList *parent, bool disableFlush,
                         std::string label) {
  bool redzones = context.isDebugAllocatorEnabled("out-of-bounds");
  // `poison-all` is the allocator-layer counterpart to the graph-driver's
  // type-aware `uninitialized-poison` (see Device::createDeviceMemory): it has
  // no dtype information, so it fills *every* allocation body with a raw byte
  // pattern rather than a type-specific sentinel. See the MemoryManagerOverview
  // doc (`MODULAR_DEBUG_DEVICE_ALLOCATOR`) for how the two tiers relate.
  bool poison = context.isDebugAllocatorEnabled("poison-all");
  return std::make_unique<MemoryManagerFreeList>(
      std::make_unique<DeviceAllocatorContext>(context), memoryManager, parent,
      disableFlush, std::move(label), redzones, poison);
}

static bool memoryManagerOnlyMode(const Device &device) {
  return (device.api() != Device::cpuAPI) &&
         DeviceContext::configAsBool("memory_manager_only", false);
}

static std::string deviceLabel(const Device &device, bool isHost) {
  return (device.api() + "[" + llvm::Twine(device.id()) +
          (isHost ? " - host]" : "]"))
      .str();
}

// The options every factory takes from the config. The factory fills in the
// geometry and the label.
static MemoryManagerOptions readOptionsFromConfig(size_t alignment) {
  MemoryManagerOptions options;
  options.alignment = alignment;

  // Zero disables the drain retry; anything else is a pass count.
  options.drainRetries = std::max<int64_t>(
      0, DeviceContext::configAsInt("memory_manager_vmm_drain_retries",
                                    options.drainRetries));
  // Converting to the clock's nanoseconds overflows a signed 64-bit count
  // somewhere past 9.2e12 ms, and a wrapped deadline lands in the past, so an
  // operator setting a huge budget to mean "no time limit" would silently get
  // a single pass. A minute is already far past anything useful here.
  options.drainBudgetMs = std::clamp<int64_t>(
      DeviceContext::configAsInt("memory_manager_vmm_drain_budget_ms",
                                 options.drainBudgetMs),
      1, kMaxDrainBudgetMs);

  options.log = DeviceContext::configAsBool("memory_manager_log", options.log);
  options.selfCheck = DeviceContext::configAsBool("memory_manager_self_check",
                                                  options.selfCheck);

  options.redzoneSize = static_cast<size_t>(DeviceContext::configAsInt(
      "memory_manager_redzone_size", options.redzoneSize));
  options.redzonePattern = static_cast<uint8_t>(DeviceContext::configAsInt(
      "memory_manager_redzone_pattern", options.redzonePattern));
  options.poisonPattern = static_cast<uint8_t>(DeviceContext::configAsInt(
      "memory_manager_poison_pattern", options.poisonPattern));

  options.vmmVaFactor = static_cast<size_t>(std::max<int64_t>(
      1,
      DeviceContext::configAsInt("memory_manager_vmm_va_factor",
                                 static_cast<int64_t>(options.vmmVaFactor))));
  return options;
}

std::unique_ptr<MemoryManager> createDisabledMemoryManager(Device &device,
                                                           size_t alignment) {
  MemoryManagerOptions options = readOptionsFromConfig(alignment);
  // TODO (GEX-3475): useOnlyMemoryManagerMode should not be required.
  options.useOnlyMemoryManagerMode = memoryManagerOnlyMode(device);
  options.label = deviceLabel(device, /*isHost=*/false);
  return MemoryManager::create(std::move(options),
                               std::make_unique<DeviceAllocatorChunks>(device));
}

std::unique_ptr<MemoryManager>
createDeviceMemoryManager(Device &device, Device::MemInfo memInfo,
                          size_t alignment, int64_t memoryLimit,
                          int64_t largeAllocAlignmentOverride) {
  auto disabledMemoryManager = [&device, alignment]() {
    return createDisabledMemoryManager(device, alignment);
  };

  if (memoryLimit == 0)
    return disabledMemoryManager();

  if (!memInfo.free)
    llvm::errs() << "DeviceContext.MemoryManager: free device memory is "
                    "unknown, sizing against total memory ("
                 << humanMemorySize(memInfo.total) << ").\n";
  size_t freeMemory = memInfo.free.value_or(memInfo.total);

  // Determine cache size
  bool isDefaultCacheSize = false;
  size_t maxCacheSize;
  if (memoryLimit > 0) // Try to use parameter value first
    maxCacheSize = static_cast<size_t>(memoryLimit);
  else // memoryLimit is < 0: look in the config
    maxCacheSize = static_cast<size_t>(DeviceContext::configAsInt(
        "memory_manager_size", 0, &isDefaultCacheSize));

  // Disable the memory manager if the requested cache size is 0
  if (maxCacheSize == 0 && !isDefaultCacheSize) // User has explicitly set to 0
    return disabledMemoryManager();

  // If there's no provided size or if the provided size is larger than
  // the free memory, use a percentage of the available memory
  if (isDefaultCacheSize || maxCacheSize > freeMemory) {
    // By default use 95% of the available memory. The serving memory ledgers
    // measure what forwards allocate, so they need less unmanaged headroom.
    int64_t percentMemoryLimit =
        DeviceContext::configAsInt("memory_manager_size_percent", 95);
    if (percentMemoryLimit == 0) // User has explicitly set to 0 => disable
      return disabledMemoryManager();

    if (maxCacheSize > freeMemory)
      llvm::errs() << "DeviceContext.MemoryManager: Requested maximum memory "
                      "manager size ("
                   << humanMemorySize(maxCacheSize)
                   << ") does not fit into available free memory ("
                   << humanMemorySize(freeMemory) << "), using "
                   << percentMemoryLimit << "% of all free memory instead.\n";

    maxCacheSize = (freeMemory * percentMemoryLimit) / 100;
  }

  // Ensure the cache size is aligned
  // TODO: Check that freeMemory is not violated by this alignment.
  maxCacheSize = llvm::alignToPowerOf2(maxCacheSize, 256 * KB);

  // Get chunk size as percentage of max cache size. Limit to between 1 and 100.
  // GEX-3330
  size_t defaultChunkSizePercent = 100;
  size_t chunkSizePercent = std::max(
      static_cast<size_t>(1),
      std::min(static_cast<size_t>(100),
               static_cast<size_t>(DeviceContext::configAsInt(
                   "memory_manager_chunk_percent", defaultChunkSizePercent))));

  size_t chunkSize = std::min(
      maxCacheSize,
      llvm::alignToPowerOf2((maxCacheSize * chunkSizePercent) / 100, 256 * KB));

  // VMM (CUDA Virtual Memory Management) defragmenting allocator. The default
  // is per-backend (device.vmmEnabledByDefault()); set
  // `memory_manager_vmm=0`/`1` to override, and it auto-disables where
  // vmmSupported() is false. When enabled we serve from a single oversized VA
  // arena backed by remappable physical granules, so a large contiguous
  // allocation can be assembled from physically-scattered free memory
  // (ensureRegionBacked / reclaimMappedGranules). Granule (page) size is
  // rounded up from `memory_manager_vmm_page_mb` (default 256 MiB) to the
  // device VMM granularity, and all VMM sizes are rounded to it.
  // Check vmmSupported() first so we skip the config lookup on backends that
  // cannot use VMM at all.
  bool vmmEnabled = device.vmmSupported() &&
                    DeviceContext::configAsBool("memory_manager_vmm",
                                                device.vmmEnabledByDefault());
  // On AMD (HIP) VMM is opt-in: hipEnabledByDefault() is false, so it takes an
  // explicit `memory_manager_vmm=1`. HIP caps how much memory a process may
  // have VMM-mapped at once -- past roughly 1.7 TiB summed across devices (8x
  // MI355X, ROCm 7.2.4) hipMemSetAccess starts returning hipErrorInvalidValue,
  // and the ceiling falls further as more access descriptors are granted per
  // mapping, since every P2P peer gets access to every granule. Four GPUs at
  // the default budget stay under it; eight do not, and need
  // `memory_manager_size` clamped to roughly 192 GiB per device. GEX-4056 has
  // the measurements and a standalone reproducer for re-measuring the ceiling
  // on a given node.
  //
  // Enabling it was previously refused outright, on the belief that VMM leaked
  // VRAM until the node was rebooted. That does not reproduce: on ROCm 7.2.4 /
  // amdgpu 6.16.13, SIGKILLing both a standalone VMM workload (1.6 TiB across 8
  // GPUs) and a VMM-enabled MAX serve returns every byte within seconds.
  if (vmmEnabled && device.api() == Device::hipAPI) {
    llvm::errs() << "DeviceContext.MemoryManager: VMM is enabled on an AMD GPU "
                    "via memory_manager_vmm. HIP limits total VMM-mapped "
                    "memory per process; if mapping fails with "
                    "hipErrorInvalidValue, clamp memory_manager_size so the "
                    "budget summed across devices stays under ~1.6 TiB "
                    "(GEX-4056).\n";
  }
  size_t vmmPageSize = 0;
  if (vmmEnabled) {
    // vmmSupported() has already verified the granularity query succeeds, so
    // this does not error.
    ErrorOr<size_t> granularity = device.vmmGranularity();
    assert(!granularity.isError() &&
           "vmmSupported() implies vmmGranularity() succeeds");
    size_t pageMb = static_cast<size_t>(std::max<int64_t>(
        1, DeviceContext::configAsInt("memory_manager_vmm_page_mb", 256)));
    vmmPageSize = llvm::alignTo(pageMb * 1024 * KB, *granularity);
    // Round the budget to whole granules.
    maxCacheSize = llvm::alignTo(maxCacheSize, vmmPageSize);
    // Single-arena VMM: the whole budget is one lazily-backed VA arena, so
    // `memory_manager_chunk_percent` is ignored (no incremental chunking).
    chunkSize = maxCacheSize;
  }

  // Start address large blocks get so a NIC can pin them; see
  // `largeAllocAlignment_`. On by default for HIP, off elsewhere; set to 0 to
  // disable.
  size_t largeAllocAlignment =
      largeAllocAlignmentOverride >= 0
          ? static_cast<size_t>(largeAllocAlignmentOverride)
          : static_cast<size_t>(std::max<int64_t>(
                0, DeviceContext::configAsInt(
                       "memory_manager_large_alloc_alignment",
                       device.api() == Device::hipAPI ? 2 * 1024 * KB : 0)));
  // The free-list arithmetic needs a power of 2. Round up rather than assert,
  // so a mistyped config costs a little more padding instead of placing blocks
  // at addresses nothing checks in a release build.
  if (largeAllocAlignment != 0 && !llvm::isPowerOf2_64(largeAllocAlignment)) {
    size_t rounded = llvm::PowerOf2Ceil(largeAllocAlignment);
    llvm::errs() << "DeviceContext.MemoryManager: "
                    "memory_manager_large_alloc_alignment ("
                 << largeAllocAlignment << ") is not a power of 2, using "
                 << rounded << " instead.\n";
    largeAllocAlignment = rounded;
  }

  MemoryManagerOptions options = readOptionsFromConfig(alignment);
  options.maxSize = maxCacheSize;
  options.chunkSize = chunkSize;
  options.useOnlyMemoryManagerMode = memoryManagerOnlyMode(device);
  options.vmmEnabled = vmmEnabled;
  options.vmmPageSize = vmmPageSize;
  options.largeAllocAlignment = largeAllocAlignment;
  options.label = deviceLabel(device, /*isHost=*/false);
  auto result = MemoryManager::create(
      std::move(options),
      vmmEnabled
          ? AllocatorBacking(std::make_unique<DeviceAllocatorVmm>(device))
          : AllocatorBacking(std::make_unique<DeviceAllocatorChunks>(device)));

  result->logIfEnabled("initial memInfo(free: " + humanMemorySize(freeMemory) +
                       ", total: " + humanMemorySize(memInfo.total) + ")");
  return result;
}

std::unique_ptr<MemoryManager> createHostMemoryManager(Device &device,
                                                       size_t alignment) {

  size_t maxCacheSize = llvm::alignToPowerOf2(
      DeviceContext::configAsInt("host_memory_manager_size", 16 * GB),
      256 * KB);

  // Disable host memory manager for Metal devices due to pointer tracking
  // issues. Metal tracks buffers by exact pointer match, but the memory manager
  // uses sub-allocation with offset pointers that aren't in the tracking map.
  // TODO: GEX-2688 - Enable host memory manager for Metal.
  if (device.api() == Device::metalAPI) {
    maxCacheSize = 0;
  }

  // Bump up the default chunk percent so that when the host memory manager is
  // primed by the Python layer (see `_prime_pinned_memory_cache`), then the
  // first allocation does not cause the entire cache to be allocated at once.
  // 8% of the default 16 GiB pool = 1.28 GiB, which is large enough to fit the
  // 1 GiB priming allocation without tripping the `useMaxSize` heuristic in
  // MemoryManagerFreeList::allocate. On 8x B200 this drops `max serve` startup
  // priming from ~55s to ~4s.
  size_t defaultChunkPercent = 8;

  size_t chunkSize = std::min(
      maxCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                        (maxCacheSize * DeviceContext::configAsInt(
                                            "host_memory_manager_chunk_percent",
                                            defaultChunkPercent)) /
                            100,
                        256 * KB)));

  MemoryManagerOptions options = readOptionsFromConfig(alignment);
  options.maxSize = maxCacheSize;
  options.chunkSize = chunkSize;
  options.isHost = true;
  // We disable useOnlyMemoryManagerMode for host memory manager since it is
  // not useful by bazel to better binpack tests on a runner.
  options.useOnlyMemoryManagerMode = false;
  options.label = deviceLabel(device, /*isHost=*/true);
  return MemoryManager::create(std::move(options),
                               std::make_unique<DeviceAllocatorChunks>(device));
}

} // namespace M::Driver
