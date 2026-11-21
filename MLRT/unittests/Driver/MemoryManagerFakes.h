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
// Fake physical backing and fake streams for unit-testing a `MemoryManager` on
// the CPU, built with `MemoryManager::create`. The fakes check the
// driver contract the manager relies on and report each violation with
// ADD_FAILURE, so a violation fails the test with context instead of aborting
// the binary.
//
// No fake address is backed by host memory. The manager never dereferences
// device memory, and host memory cannot map one physical page at two
// addresses, which a VMM remap does.
//
// The fakes are not thread-safe. A test drives them, and the manager, from one
// thread.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_UNITTESTS_DRIVER_MEMORYMANAGERFAKES_H
#define MLRT_UNITTESTS_DRIVER_MEMORYMANAGERFAKES_H

#include "MLRT/lib/Driver/MemoryManager/AllocatorBacking.h"
#include "MLRT/lib/Driver/MemoryManager/AllocatorContext.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace M::Driver {

namespace fake_detail {

// Disjoint fake address ranges, far from anything a test dereferences. Each
// reservation starts on its own boundary, so two never touch and an address
// names the range it came from.
constexpr uintptr_t kChunkBase = uintptr_t{1} << 44;
constexpr uintptr_t kVmmBase = uintptr_t{1} << 45;
constexpr uintptr_t kRangeAlignment = uintptr_t{1} << 32;

inline uintptr_t nextRange(uintptr_t cursor, size_t byteSize) {
  const uintptr_t end = cursor + byteSize + kRangeAlignment;
  return end & ~(kRangeAlignment - 1);
}

inline const void *asPtr(uintptr_t address) {
  return reinterpret_cast<const void *>(address);
}

} // namespace fake_detail

// Plain chunk backing and the memory budget.
class FakeAllocatorChunks final : public AllocatorChunks {
public:
  FakeAllocatorChunks() {
    memInfo_.total = size_t{64} << 30;
    memInfo_.free = memInfo_.total;
  }

  // Controls.

  void refuseChunkAllocs(bool refuse) { refuseChunkAllocs_ = refuse; }
  void setMemoryInfo(size_t total, std::optional<size_t> free) {
    memInfo_.total = total;
    memInfo_.free = free;
  }

  // Observations.

  // The size of every chunk handed out, in order.
  const std::vector<size_t> &chunkAllocSizes() const { return allocSizes_; }
  size_t refusedChunkAllocs() const { return refused_; }

  void expectNothingLive() const {
    for (const auto &[address, chunk] : live_)
      ADD_FAILURE() << "chunk " << fake_detail::asPtr(address) << " of "
                    << chunk.byteSize << " bytes is still live";
  }

  // AllocatorChunks.

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize, bool isHost) override {
    if (byteSize == 0) {
      ADD_FAILURE() << "chunkAlloc(0) violates its byteSize > 0 precondition";
      return Error("fake: chunkAlloc(0)");
    }
    if (refuseChunkAllocs_) {
      ++refused_;
      return Error("fake: chunkAlloc refused");
    }
    live_[next_] = {byteSize, isHost};
    *ptr = reinterpret_cast<void *>(next_);
    next_ = fake_detail::nextRange(next_, byteSize);
    allocSizes_.push_back(byteSize);
    return success();
  }

  // Never returns an error: the manager aborts on a failed chunkFree, and the
  // failure is more useful as a test failure.
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) override {
    auto it = live_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == live_.end()) {
      ADD_FAILURE() << "chunkFree(" << ptr << ") of an address that is not a "
                    << "live chunk";
      return success();
    }
    if (it->second.isHost != isHost)
      ADD_FAILURE() << "chunkFree(" << ptr << ", isHost=" << isHost
                    << ") of a chunk allocated with isHost="
                    << it->second.isHost;
    live_.erase(it);
    return success();
  }

  ErrorOr<AllocatorMemInfo> getMemoryInfo() const override { return memInfo_; }

private:
  struct Chunk {
    size_t byteSize;
    bool isHost;
  };

  AllocatorMemInfo memInfo_;
  bool refuseChunkAllocs_ = false;
  uintptr_t next_ = fake_detail::kChunkBase;
  std::map<uintptr_t, Chunk> live_;
  std::vector<size_t> allocSizes_;
  size_t refused_ = 0;
};

// Virtual memory backing: a state machine over reserved ranges, physical
// handles and the mappings between them.
class FakeAllocatorVmm final : public AllocatorVmm {
public:
  explicit FakeAllocatorVmm(size_t granularity) : granularity_(granularity) {}

  // Controls.

  // Lets `count` more calls succeed and refuses every one after; a negative
  // count never refuses. Maps count vmmMap and vmmMapAndSetAccess together.
  void refuseCreatesAfter(int64_t count) { createsLeft_ = count; }
  void refuseMapsAfter(int64_t count) { mapsLeft_ = count; }
  // The same, for access grants: vmmSetAccess and the access half of
  // vmmMapAndSetAccess. As the CUDA and HIP drivers do, a refused
  // vmmMapAndSetAccess leaves nothing mapped, and a refused vmmSetAccess
  // leaves the range mapped.
  void refuseAccessAfter(int64_t count) { accessLeft_ = count; }
  // Refuses every map while `capturing` returns true, as the driver does
  // during a graph capture. An unmap then fails the test: the manager must
  // not unmap mid-capture.
  void capturingWhen(std::function<bool()> capturing) {
    capturing_ = std::move(capturing);
  }
  // Refuses every reservation, as when the address space has no room.
  void refuseReserves() { refuseReserves_ = true; }

  // Observations.

  // Every reservation requested, refused or not, in order.
  const std::vector<size_t> &reserveRequests() const {
    return reserveRequests_;
  }
  size_t createCount() const { return creates_; }
  size_t refusedCreates() const { return refusedCreates_; }
  size_t refusedMaps() const { return refusedMaps_; }
  size_t refusedAccess() const { return refusedAccess_; }
  size_t unmapCount() const { return unmaps_; }
  size_t liveHandles() const { return handles_.size(); }
  size_t liveMappings() const { return mappings_.size(); }

  // The handle mapped at `va`, or 0 when nothing is.
  uint64_t handleAt(const void *va) const {
    auto it = mappingContaining(reinterpret_cast<uintptr_t>(va));
    return it == mappings_.end() ? 0 : it->second.handle;
  }
  // Handles mapped at more than one address right now.
  size_t aliasedHandles() const {
    size_t count = 0;
    for (const auto &[handle, info] : handles_)
      count += info.mappings > 1 ? 1 : 0;
    return count;
  }
  // Mappings no vmmSetAccess or vmmMapAndSetAccess has made accessible.
  size_t mappingsWithoutAccess() const {
    size_t count = 0;
    for (const auto &[va, mapping] : mappings_)
      count += mapping.access ? 0 : 1;
    return count;
  }

  void expectNothingLive() const {
    for (const auto &[va, mapping] : mappings_)
      ADD_FAILURE() << "handle " << mapping.handle << " is still mapped at "
                    << fake_detail::asPtr(va);
    for (const auto &[handle, info] : handles_)
      ADD_FAILURE() << "handle " << handle << " of " << info.byteSize
                    << " bytes was never released";
    for (const auto &[va, byteSize] : reservations_)
      ADD_FAILURE() << "reservation " << fake_detail::asPtr(va) << " of "
                    << byteSize << " bytes was never freed";
  }

  // AllocatorVmm.

  ErrorOr<void *> vmmAddressReserve(size_t byteSize) override {
    reserveRequests_.push_back(byteSize);
    if (!checkSize("vmmAddressReserve", 0, byteSize))
      return Error("fake: misaligned reservation");
    if (refuseReserves_)
      return Error("fake: vmmAddressReserve refused");
    const uintptr_t va = nextVa_;
    reservations_[va] = byteSize;
    nextVa_ = fake_detail::nextRange(nextVa_, byteSize);
    return reinterpret_cast<void *>(va);
  }

  ErrorOrSuccess vmmAddressFree(void *ptr, size_t byteSize) override {
    const uintptr_t va = reinterpret_cast<uintptr_t>(ptr);
    auto it = reservations_.find(va);
    if (it == reservations_.end() || it->second != byteSize) {
      ADD_FAILURE() << "vmmAddressFree(" << ptr << ", " << byteSize
                    << ") does not name a live reservation";
      return Error("fake: unknown reservation");
    }
    if (overlapsMapping(va, va + byteSize)) {
      ADD_FAILURE() << "vmmAddressFree(" << ptr << ", " << byteSize
                    << ") while a mapping inside it is live";
      return Error("fake: reservation still mapped");
    }
    reservations_.erase(it);
    return success();
  }

  ErrorOr<uint64_t> vmmCreate(size_t byteSize) override {
    if (!checkSize("vmmCreate", 0, byteSize))
      return Error("fake: misaligned create");
    if (takeRefusal(createsLeft_)) {
      ++refusedCreates_;
      return Error("fake: vmmCreate refused");
    }
    const uint64_t handle = nextHandle_++;
    handles_[handle] = {byteSize, 0};
    ++creates_;
    return handle;
  }

  ErrorOrSuccess vmmRelease(uint64_t handle) override {
    auto it = handles_.find(handle);
    if (it == handles_.end()) {
      ADD_FAILURE() << "vmmRelease(" << handle << ") of a handle that is not "
                    << "live";
      return Error("fake: unknown handle");
    }
    if (it->second.mappings != 0) {
      ADD_FAILURE() << "vmmRelease(" << handle << ") while it is still mapped "
                    << "at " << it->second.mappings << " address(es)";
      return Error("fake: handle still mapped");
    }
    handles_.erase(it);
    return success();
  }

  ErrorOrSuccess vmmMapAndSetAccess(void *va, size_t byteSize,
                                    uint64_t handle) override {
    return map("vmmMapAndSetAccess", va, byteSize, handle, /*access=*/true);
  }

  ErrorOrSuccess vmmMap(void *va, size_t byteSize, uint64_t handle) override {
    return map("vmmMap", va, byteSize, handle, /*access=*/false);
  }

  ErrorOrSuccess vmmUnmap(void *ptr, size_t byteSize) override {
    const uintptr_t va = reinterpret_cast<uintptr_t>(ptr);
    if (!checkSize("vmmUnmap", va, byteSize))
      return Error("fake: misaligned unmap");
    if (capturing_ && capturing_()) {
      ADD_FAILURE() << "vmmUnmap(" << ptr << ", " << byteSize
                    << ") during a graph capture";
      return Error("fake: unmap during capture");
    }
    if (!tilesMappings(va, va + byteSize)) {
      ADD_FAILURE() << "vmmUnmap(" << ptr << ", " << byteSize
                    << ") does not cover whole live mappings";
      return Error("fake: unmap of unmapped range");
    }
    for (auto it = mappings_.find(va);
         it != mappings_.end() && it->first < va + byteSize;) {
      --handles_[it->second.handle].mappings;
      it = mappings_.erase(it);
    }
    ++unmaps_;
    return success();
  }

  ErrorOrSuccess vmmSetAccess(void *ptr, size_t byteSize) override {
    const uintptr_t va = reinterpret_cast<uintptr_t>(ptr);
    if (!checkSize("vmmSetAccess", va, byteSize))
      return Error("fake: misaligned set access");
    if (!tilesMappings(va, va + byteSize)) {
      ADD_FAILURE() << "vmmSetAccess(" << ptr << ", " << byteSize
                    << ") covers a range that is not wholly mapped";
      return Error("fake: set access on unmapped range");
    }
    if (takeRefusal(accessLeft_)) {
      ++refusedAccess_;
      return Error("fake: access refused");
    }
    for (auto it = mappings_.find(va);
         it != mappings_.end() && it->first < va + byteSize; ++it)
      it->second.access = true;
    return success();
  }

private:
  struct Handle {
    size_t byteSize;
    // Live mappings of the handle. More than one is legal: a remap maps a
    // stream's pages at a second address before it retires the first.
    size_t mappings;
  };
  struct Mapping {
    size_t byteSize;
    uint64_t handle;
    bool access;
  };

  bool checkSize(const char *call, uintptr_t va, size_t byteSize) const {
    if (byteSize != 0 && byteSize % granularity_ == 0 && va % granularity_ == 0)
      return true;
    ADD_FAILURE() << call << "(" << fake_detail::asPtr(va) << ", " << byteSize
                  << ") is not a nonzero multiple of the " << granularity_
                  << "-byte granularity";
    return false;
  }

  // Whether this call is refused, consuming one allowed call otherwise.
  static bool takeRefusal(int64_t &left) {
    if (left < 0)
      return false;
    if (left == 0)
      return true;
    --left;
    return false;
  }

  std::map<uintptr_t, Mapping>::const_iterator
  mappingContaining(uintptr_t va) const {
    auto it = mappings_.upper_bound(va);
    if (it == mappings_.begin())
      return mappings_.end();
    --it;
    return va < it->first + it->second.byteSize ? it : mappings_.end();
  }

  bool overlapsMapping(uintptr_t lo, uintptr_t hi) const {
    if (mappingContaining(lo) != mappings_.end())
      return true;
    auto it = mappings_.lower_bound(lo);
    return it != mappings_.end() && it->first < hi;
  }

  // Whether live mappings cover [lo, hi) exactly, with no gap and no mapping
  // reaching outside it.
  bool tilesMappings(uintptr_t lo, uintptr_t hi) const {
    uintptr_t cursor = lo;
    for (auto it = mappings_.find(lo);
         cursor < hi && it != mappings_.end() && it->first == cursor; ++it)
      cursor += it->second.byteSize;
    return cursor == hi;
  }

  bool insideOneReservation(uintptr_t lo, uintptr_t hi) const {
    auto it = reservations_.upper_bound(lo);
    if (it == reservations_.begin())
      return false;
    --it;
    return hi <= it->first + it->second;
  }

  ErrorOrSuccess map(const char *call, void *ptr, size_t byteSize,
                     uint64_t handle, bool access) {
    const uintptr_t va = reinterpret_cast<uintptr_t>(ptr);
    if (!checkSize(call, va, byteSize))
      return Error("fake: misaligned map");
    if (!insideOneReservation(va, va + byteSize)) {
      ADD_FAILURE() << call << "(" << ptr << ", " << byteSize
                    << ") maps outside reserved address space";
      return Error("fake: map outside a reservation");
    }
    if (overlapsMapping(va, va + byteSize)) {
      ADD_FAILURE() << call << "(" << ptr << ", " << byteSize
                    << ") maps over a slot that is already mapped";
      return Error("fake: slot already mapped");
    }
    auto handleIt = handles_.find(handle);
    if (handleIt == handles_.end()) {
      ADD_FAILURE() << call << "(" << ptr << ", " << byteSize << ", handle "
                    << handle << ") maps a handle that is not live";
      return Error("fake: unknown handle");
    }
    if (handleIt->second.byteSize != byteSize) {
      ADD_FAILURE() << call << "(" << ptr << ", " << byteSize << ", handle "
                    << handle << ") maps a handle of "
                    << handleIt->second.byteSize << " bytes";
      return Error("fake: map size differs from the handle");
    }
    if ((capturing_ && capturing_()) || takeRefusal(mapsLeft_)) {
      ++refusedMaps_;
      return Error("fake: map refused");
    }
    if (access && takeRefusal(accessLeft_)) {
      ++refusedAccess_;
      return Error("fake: access refused");
    }
    mappings_[va] = {byteSize, handle, access};
    ++handleIt->second.mappings;
    return success();
  }

  const size_t granularity_;
  int64_t createsLeft_ = -1;
  int64_t mapsLeft_ = -1;
  int64_t accessLeft_ = -1;
  std::function<bool()> capturing_;
  bool refuseReserves_ = false;

  uintptr_t nextVa_ = fake_detail::kVmmBase;
  uint64_t nextHandle_ = 1;
  std::map<uintptr_t, size_t> reservations_;
  std::map<uint64_t, Handle> handles_;
  std::map<uintptr_t, Mapping> mappings_;

  std::vector<size_t> reserveRequests_;
  size_t creates_ = 0;
  size_t refusedCreates_ = 0;
  size_t refusedMaps_ = 0;
  size_t refusedAccess_ = 0;
  size_t unmaps_ = 0;
};

class FakeStream;

// An event on a FakeStream: ready once the stream has run past it.
class FakeEvent final : public AllocatorEvent {
public:
  FakeEvent(std::shared_ptr<FakeStream> stream, uint64_t seq)
      : stream_(std::move(stream)), seq_(seq) {}

  ErrorOrSuccess synchronize() override;
  ErrorOr<bool> isReady() const override;

private:
  std::shared_ptr<FakeStream> stream_;
  const uint64_t seq_;
};

// The work queued on one fake stream. Shared by the stream's AllocatorContext,
// its events and the test, because events outlive the free list that owns the
// stream: the manager keeps them in its pending records and retiring ranges.
//
// Events complete in the order recorded, when the test runs the stream or when
// a synchronize waits for them.
class FakeStream : public std::enable_shared_from_this<FakeStream> {
public:
  // Controls.

  bool capturing = false;
  // Run by poll; a context drains its own free list there.
  std::function<ErrorOrSuccess()> onPoll;

  // Completes every event recorded so far, as if the device ran past them.
  void runAll() { completed_ = recorded_; }

  // Observations.

  size_t eventsRecorded() const { return recorded_; }

  // Used by FakeAllocatorContext and FakeEvent.

  AllocatorEventRef record() {
    return RCRef<FakeEvent>::create(shared_from_this(), ++recorded_);
  }

  ErrorOrSuccess poll() { return onPoll ? onPoll() : success(); }

  // As the CUDA and HIP contexts do, ends by polling pending events, whose
  // handlers take the manager's lock.
  ErrorOrSuccess synchronize() {
    runAll();
    return poll();
  }

  bool isComplete(uint64_t seq) const { return completed_ >= seq; }

  // Events complete in order, so waiting for one completes those before it.
  void completeThrough(uint64_t seq) {
    if (completed_ < seq)
      completed_ = seq;
  }

private:
  uint64_t recorded_ = 0;
  uint64_t completed_ = 0;
};

inline ErrorOrSuccess FakeEvent::synchronize() {
  stream_->completeThrough(seq_);
  return success();
}

inline ErrorOr<bool> FakeEvent::isReady() const {
  return stream_->isComplete(seq_);
}

// The AllocatorContext a free list owns, forwarding to a FakeStream the test
// keeps. The stream is its own base stream, and the tests enable neither
// redzones nor fallthrough, so those calls fail the test.
class FakeAllocatorContext final : public AllocatorContext {
public:
  explicit FakeAllocatorContext(std::shared_ptr<FakeStream> stream)
      : stream_(std::move(stream)) {}

  ErrorOr<AllocatorEventRef> enqueueEvent() override {
    return stream_->record();
  }
  ErrorOr<AllocatorEventRef> enqueueBaseStreamEvent() override {
    return AllocatorEventRef();
  }
  ErrorOrSuccess enqueueWaitFor(AllocatorEvent &) override { return success(); }

  bool isGraphCapturing() const override { return stream_->capturing; }
  bool isBaseStreamCapturing() const override { return false; }

  ErrorOrSuccess pollPendingEvents() override { return stream_->poll(); }
  bool flushIdleStreamViews() override { return false; }

  ErrorOrSuccess memAlloc(void **, size_t byteSize,
                          AllocatorContext::AllocMode) override {
    ADD_FAILURE() << "memAlloc(" << byteSize << ") falls through to the driver";
    return Error("fake: no fallthrough");
  }
  ErrorOrSuccess memFree(void *ptr, AllocatorContext::FreeKind) override {
    ADD_FAILURE() << "memFree(" << ptr << ") of a fallthrough allocation";
    return Error("fake: no fallthrough");
  }

  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t,
                                  size_t) override {
    ADD_FAILURE() << "enqueueSetMemory(" << dst << ", " << byteSize
                  << ") with redzones and poison off";
    return Error("fake: no device writes");
  }
  ErrorOrSuccess enqueueCopyFromDevice(void *, const void *src,
                                       size_t byteSize) override {
    ADD_FAILURE() << "enqueueCopyFromDevice(" << src << ", " << byteSize
                  << ") with redzones off";
    return Error("fake: no device reads");
  }
  ErrorOrSuccess synchronize() override { return stream_->synchronize(); }

private:
  std::shared_ptr<FakeStream> stream_;
};

} // namespace M::Driver

#endif // MLRT_UNITTESTS_DRIVER_MEMORYMANAGERFAKES_H
