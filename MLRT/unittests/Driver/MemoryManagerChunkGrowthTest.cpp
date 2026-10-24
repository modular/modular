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

#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "MLRT/lib/Driver/DeviceContext/Virtual/VirtualDeviceContext.h"
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace M::Driver;

namespace {

constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * KB;
constexpr size_t kChunkAlignment = 256;

class StubCompilationDevice final : public CompilationDevice {
public:
  StubCompilationDevice() : CompilationDevice(Device::cudaAPI, "sm_stub") {}

  M::ErrorOr<int> getApiVersion() const override { return 0; }

  M::ErrorOr<DriverVersion &> getDriverVersion() const override {
    static DriverVersion version("0.0.0");
    return version;
  }

  M::ErrorOr<const char *> getArchName() const override { return "sm_stub"; }

  M::ErrorOr<M::BufferRef> compileFunction(llvm::StringRef, llvm::StringRef,
                                           int32_t,
                                           llvm::StringRef) const override {
    return M::Error("StubCompilationDevice does not compile code");
  }
};

class TestMemoryPressureDevice final : public VirtualDevice {
public:
  TestMemoryPressureDevice(size_t freeBytes, size_t totalBytes)
      : VirtualDevice(/*id=*/0, /*name=*/"test-cuda",
                      std::make_unique<StubCompilationDevice>()),
        freeBytes_(freeBytes), totalBytes_(totalBytes) {}

  ~TestMemoryPressureDevice() override {
    for (const auto &[ptr, _] : allocations_)
      std::free(ptr);
  }

  void reserveExternalPressure(size_t byteSize) {
    ASSERT_LE(byteSize, freeBytes_);
    freeBytes_ -= byteSize;
  }

  void failNextChunkAlloc(llvm::StringRef error) {
    nextChunkAllocError_ = error.str();
  }

  const std::vector<size_t> &chunkAllocRequests() const {
    return chunkAllocRequests_;
  }

  M::ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize,
                               bool isHost) override {
    if (isHost)
      return M::Error("unexpected host allocation");
    if (!nextChunkAllocError_.empty()) {
      std::string error = std::move(nextChunkAllocError_);
      nextChunkAllocError_.clear();
      return M::Error(error);
    }
    if (byteSize > freeBytes_)
      return M::Error("out of fake device memory");

    void *allocation = std::aligned_alloc(kChunkAlignment, byteSize);
    if (allocation == nullptr)
      return M::Error("aligned_alloc failed");

    *ptr = allocation;
    allocations_.emplace(allocation, byteSize);
    chunkAllocRequests_.push_back(byteSize);
    freeBytes_ -= byteSize;
    return M::success();
  }

  M::ErrorOrSuccess chunkFree(void *ptr, bool isHost) override {
    if (isHost)
      return M::Error("unexpected host free");

    auto it = allocations_.find(ptr);
    if (it == allocations_.end())
      return M::Error("unknown allocation");

    freeBytes_ += it->second;
    std::free(ptr);
    allocations_.erase(it);
    return M::success();
  }

  M::ErrorOr<Device::MemInfo> getMemoryInfo() const override {
    Device::MemInfo info;
    info.free = freeBytes_;
    info.total = totalBytes_;
    return info;
  }

private:
  size_t freeBytes_;
  size_t totalBytes_;
  std::string nextChunkAllocError_;
  std::unordered_map<void *, size_t> allocations_;
  std::vector<size_t> chunkAllocRequests_;
};

TEST(MemoryManagerChunkGrowthTest, CapsChunkGrowthToCurrentFreeMemory) {
  ScopedEnvVar chunkPercent(
      "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "25");

  constexpr size_t kInitialFree = MB;
  constexpr size_t kTotalMemory = 2 * MB;
  constexpr size_t kMaxCacheSize = MB;
  constexpr size_t kExternalPressure = 256 * KB;
  constexpr size_t kRequestSize = 512 * KB;
  constexpr size_t kExpectedChunkSize = kInitialFree - kExternalPressure;

  // Use a dedicated test process because useMaxSize_ is process-global.
  auto device =
      M::RCRef<TestMemoryPressureDevice>::create(kInitialFree, kTotalMemory);
  ASSIGN_NO_ERR(Device::MemInfo, initialMemInfo, device->getMemoryInfo());
  auto manager = std::unique_ptr<MemoryManager>(MemoryManager::onDevice(
      *device, initialMemInfo, device->allocationAlignment(), kMaxCacheSize));

  // Simulate unrelated allocations after the memory manager is created.
  device->reserveExternalPressure(kExternalPressure);

  MemoryManagerFreeList &freeList = *manager->freeList();
  ASSIGN_NO_ERR(void *, ptr,
                freeList.allocate(kRequestSize, Device::AllocMode::SYNC));

  ASSERT_EQ(device->chunkAllocRequests().size(), 1u);
  EXPECT_EQ(device->chunkAllocRequests().front(), kExpectedChunkSize);

  EXPECT_NO_ERR(freeList.free(ptr));
  manager->clear();
}

TEST(MemoryManagerChunkGrowthTest, ReportsChunkAllocFailureDetails) {
  ScopedEnvVar chunkPercent(
      "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "25");

  constexpr size_t kInitialFree = MB;
  constexpr size_t kTotalMemory = 2 * MB;
  constexpr size_t kMaxCacheSize = MB;

  auto device =
      M::RCRef<TestMemoryPressureDevice>::create(kInitialFree, kTotalMemory);
  ASSIGN_NO_ERR(Device::MemInfo, initialMemInfo, device->getMemoryInfo());
  auto manager = std::unique_ptr<MemoryManager>(MemoryManager::onDevice(
      *device, initialMemInfo, device->allocationAlignment(), kMaxCacheSize));

  device->failNextChunkAlloc("fake oom");

  MemoryManagerFreeList &freeList = *manager->freeList();
  auto err = freeList.allocate(/*size=*/512, Device::AllocMode::SYNC);
  ASSERT_TRUE(err.isError());
  EXPECT_THAT(err.getError(),
              testing::HasSubstr("last_chunk_alloc_failure: chunkAlloc("));
  EXPECT_THAT(err.getError(), testing::HasSubstr("failed with fake oom"));

  manager->clear();
}

} // namespace
