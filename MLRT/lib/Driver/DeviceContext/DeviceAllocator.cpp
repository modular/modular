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
// This file implements the allocator interfaces over the legacy driver.
//
//===----------------------------------------------------------------------===//

#include "DeviceAllocator.h"
#include "MemoryManager.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

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

} // namespace M::Driver
