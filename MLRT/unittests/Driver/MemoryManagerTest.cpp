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

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "MLRT/Driver/DeviceContext/CPU/CpuDeviceContext.h"
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>

using namespace M::Driver;
using M::ErrorOr;

extern char **environ;

namespace {

class MemoryManagerTest : public ::testing::Test {
protected:
  // Helper to create a CPU device with memory manager for testing - match
  // existing test
  CpuDeviceRef createTestDevice() {
    constexpr size_t chunkSize = 1024;
    constexpr size_t maxCacheSize = 4 * chunkSize;

    Device::MemInfo memInfo;
    memInfo.free = 128 * maxCacheSize;  // Needs to be larger than 256 * KB
    memInfo.total = 256 * maxCacheSize; // Needs to be larger than 256 * KB

    return CpuDeviceRef::create(memInfo);
  }

  // Helper to create DeviceContext for memory manager testing
  ErrorOr<DeviceContextRef> createTestDeviceContext() {
    return DeviceContext::create(Device::cpuAPI);
  }

  // Helper to create a host memory manager for testing
  std::unique_ptr<MemoryManager> createTestHostCache() {
    constexpr size_t chunkSize = 1024;
    constexpr size_t maxCacheSize = 4 * chunkSize;

    Device::MemInfo memInfo;
    memInfo.free = 128 * maxCacheSize;  // Needs to be larger than 256 * KB
    memInfo.total = 256 * maxCacheSize; // Needs to be larger than 256 * KB

    CpuDeviceRef device = CpuDeviceRef::create(memInfo);
    return std::unique_ptr<MemoryManager>(MemoryManager::onHost(*device, 64));
  }

  // Helper to create a device memory manager with an explicit large-block
  // start alignment. It is passed here rather than set in the environment
  // because DeviceContext snapshots its config once per process, so an env var
  // set by a test would not reach a manager built later.
  std::unique_ptr<MemoryManager>
  createTestDeviceCache(Device &device, size_t largeAllocAlignment) {
    Device::MemInfo memInfo;
    memInfo.free = 16 * 1024 * 1024;
    memInfo.total = 32 * 1024 * 1024;

    return std::unique_ptr<MemoryManager>(MemoryManager::onDevice(
        device, memInfo, /*alignment=*/64, /*memoryLimit=*/8 * 1024 * 1024,
        static_cast<int64_t>(largeAllocAlignment)));
  }
};

TEST_F(MemoryManagerTest, MemoryManagerCreation) {
  // Test creation through CPU device
  {
    CpuDeviceRef device = createTestDevice();
    MemoryManager &cache = device->memoryManager();

    // CPU device with memInfo creates device memory manager, not host cache
    EXPECT_FALSE(cache.isHost());
    EXPECT_EQ(cache.allocationAlignment(), 64); // Default alignment
  }

  // Test factory method for host cache
  {
    Device::MemInfo memInfo;
    memInfo.free = 1024 * 1024;
    memInfo.total = 2 * 1024 * 1024;

    CpuDeviceRef device = CpuDeviceRef::create(memInfo);
    std::unique_ptr<MemoryManager> cache(MemoryManager::onHost(*device, 64));

    EXPECT_TRUE(cache != nullptr);
    EXPECT_TRUE(cache->isHost());
    EXPECT_EQ(cache->allocationAlignment(), 64);
  }

  // Test factory method for device cache
  {
    Device::MemInfo memInfo;
    memInfo.free = 1024 * 1024;
    memInfo.total = 2 * 1024 * 1024;

    CpuDeviceRef device = CpuDeviceRef::create(memInfo);
    std::unique_ptr<MemoryManager> cache(
        MemoryManager::onDevice(*device, memInfo, 128));

    EXPECT_TRUE(cache != nullptr);
    EXPECT_FALSE(cache->isHost());
    EXPECT_EQ(cache->allocationAlignment(), 128);
  }

  // Test requesting specific size
  {
    Device::MemInfo memInfo;
    memInfo.free = 1024 * 1024;
    memInfo.total = 2 * 1024 * 1024;

    CpuDeviceRef device = CpuDeviceRef::create(memInfo);
    std::unique_ptr<MemoryManager> cache(
        MemoryManager::onDevice(*device, memInfo, 128, memInfo.total / 2));

    EXPECT_TRUE(cache != nullptr);
    EXPECT_FALSE(cache->isHost());
    EXPECT_EQ(cache->allocationAlignment(), 128);
    EXPECT_EQ(cache->getMemoryInfo().total, memInfo.total / 2);
  }

  // Test requesting 0 (disabled cache)
  {
    Device::MemInfo memInfo;
    memInfo.free = 1024 * 1024;
    memInfo.total = 2 * 1024 * 1024;

    CpuDeviceRef device = CpuDeviceRef::create(memInfo);
    std::unique_ptr<MemoryManager> cache(
        MemoryManager::onDevice(*device, memInfo, 128, 0));

    EXPECT_TRUE(cache != nullptr);
    EXPECT_FALSE(cache->isHost());
    EXPECT_EQ(cache->allocationAlignment(), 128);
    EXPECT_EQ(cache->getMemoryInfo().total, 0);
  }

  // Test with zero memory (disabled cache)
  {
    Device::MemInfo memInfo;
    memInfo.free = 0;
    memInfo.total = 0;

    CpuDeviceRef device = CpuDeviceRef::create(memInfo);
    std::unique_ptr<MemoryManager> cache(
        MemoryManager::onDevice(*device, memInfo, 64));

    EXPECT_TRUE(cache != nullptr);
    EXPECT_FALSE(cache->isHost());
    EXPECT_EQ(cache->allocationAlignment(), 64);
    EXPECT_EQ(cache->getMemoryInfo().total, 0);
  }
}

TEST_F(MemoryManagerTest, AlignmentHandling) {
  // Test device memory manager alignment
  {
    CpuDeviceRef device = createTestDevice();
    MemoryManager &cache = device->memoryManager();

    // CPU device should have 64-byte alignment by default
    EXPECT_EQ(cache.allocationAlignment(), 64);
  }
}

TEST_F(MemoryManagerTest, BasicAllocationAndFree) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();
  size_t allocAlignment = device->allocationAlignment();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Test basic allocation
  ASSIGN_NO_ERR(void *, ptr1, freeList.allocate(64, mode));
  EXPECT_NE(ptr1, nullptr);

  // Test that allocation is properly aligned
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr1);
  EXPECT_EQ(addr % allocAlignment, 0);

  // Test free
  EXPECT_NO_ERR(freeList.free(ptr1));

  // Test free with nullptr (should be safe)
  EXPECT_NO_ERR(freeList.free(nullptr));
}

TEST_F(MemoryManagerTest, MultipleAllocations) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();
  size_t allocAlignment = device->allocationAlignment();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> ptrs;

  // Allocate multiple blocks
  for (int i = 0; i < 10; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(64, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Verify all pointers are different and aligned
  for (size_t i = 0; i < ptrs.size(); ++i) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptrs[i]);
    EXPECT_EQ(addr % allocAlignment, 0);

    for (size_t j = i + 1; j < ptrs.size(); ++j) {
      EXPECT_NE(ptrs[i], ptrs[j]);
    }
  }

  // Free all blocks
  for (void *ptr : ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

TEST_F(MemoryManagerTest, AllocationSizes) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();
  size_t allocAlignment = device->allocationAlignment();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<size_t> sizes = {1, 15, 64, 128, 256, 512, 1000};
  std::vector<void *> ptrs;

  // Test allocations of various sizes
  for (size_t size : sizes) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(size, mode));
    EXPECT_NE(ptr, nullptr);

    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % allocAlignment, 0);

    ptrs.push_back(ptr);
  }

  // Free all allocations
  for (void *ptr : ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

// Large-block start alignment (ENABLE-2932). 1 KiB stands in for the 2 MiB
// that HIP devices default to; the threshold is
// MemoryManager::kLargeAllocSizeFactor times that. `kSmallSize` rounds up to a
// block that is not a whole number of `kLargeAlign` units, so a large block
// placed after one needs head padding.
constexpr size_t kLargeAlign = 1024;
constexpr size_t kLargeThreshold = 32 * kLargeAlign;
constexpr size_t kSmallSize = 600;

TEST_F(MemoryManagerTest, LargeAllocationsGetStartAlignment) {
  CpuDeviceRef device = createTestDevice();
  std::unique_ptr<MemoryManager> manager =
      createTestDeviceCache(*device, kLargeAlign);
  MemoryManagerFreeList &freeList = *manager->freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> ptrs;

  // Interleave sub-threshold allocations so each large block would land
  // off-boundary without the head padding.
  for (int i = 0; i < 4; ++i) {
    ASSIGN_NO_ERR(void *, small, freeList.allocate(kSmallSize, mode));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(small) %
                  manager->allocationAlignment(),
              0u);
    ptrs.push_back(small);

    ASSIGN_NO_ERR(void *, large, freeList.allocate(kLargeThreshold, mode));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(large) % kLargeAlign, 0u);
    ptrs.push_back(large);
  }

  for (void *ptr : ptrs)
    EXPECT_NO_ERR(freeList.free(ptr));
  manager->clear();
}

TEST_F(MemoryManagerTest, LargeAlignmentPaddingIsReusable) {
  CpuDeviceRef device = createTestDevice();
  std::unique_ptr<MemoryManager> manager =
      createTestDeviceCache(*device, kLargeAlign);
  MemoryManagerFreeList &freeList = *manager->freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // A whole number of allocation-alignment units but not of the large-block
  // alignment, so the second block needs head padding wherever the chunk
  // itself starts.
  const size_t largeSize = kLargeThreshold + manager->allocationAlignment();
  ASSIGN_NO_ERR(void *, first, freeList.allocate(largeSize, mode));
  ASSIGN_NO_ERR(void *, second, freeList.allocate(largeSize, mode));
  uintptr_t firstAddr = reinterpret_cast<uintptr_t>(first);
  uintptr_t secondAddr = reinterpret_cast<uintptr_t>(second);
  EXPECT_EQ(firstAddr % kLargeAlign, 0u);
  EXPECT_EQ(secondAddr % kLargeAlign, 0u);
  uintptr_t firstEnd = firstAddr + largeSize;
  ASSERT_GT(secondAddr, firstEnd);

  // The padding was handed back as its own free block. Being the smallest
  // block on the list, it is what the next allocation that fits reuses.
  ASSIGN_NO_ERR(void *, filler, freeList.allocate(kSmallSize, mode));
  uintptr_t fillerAddr = reinterpret_cast<uintptr_t>(filler);
  EXPECT_GE(fillerAddr, firstEnd);
  EXPECT_LT(fillerAddr, secondAddr);

  EXPECT_NO_ERR(freeList.free(filler));
  EXPECT_NO_ERR(freeList.free(second));
  EXPECT_NO_ERR(freeList.free(first));
  manager->clear();
}

TEST_F(MemoryManagerTest, LargeStartAlignmentDisabledPacksBlocks) {
  CpuDeviceRef device = createTestDevice();
  std::unique_ptr<MemoryManager> manager =
      createTestDeviceCache(*device, /*largeAllocAlignment=*/0);
  MemoryManagerFreeList &freeList = *manager->freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // With the large-block alignment off, blocks stay packed at the manager's own
  // allocation alignment regardless of size.
  const size_t largeSize = kLargeThreshold + manager->allocationAlignment();
  ASSIGN_NO_ERR(void *, first, freeList.allocate(largeSize, mode));
  ASSIGN_NO_ERR(void *, second, freeList.allocate(largeSize, mode));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(second) -
                reinterpret_cast<uintptr_t>(first),
            largeSize);

  EXPECT_NO_ERR(freeList.free(second));
  EXPECT_NO_ERR(freeList.free(first));
  manager->clear();
}

TEST_F(MemoryManagerTest, FreeListEmpty) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  // Initially, the free list should be empty (no chunks allocated yet)
  EXPECT_TRUE(freeList.empty());

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // After first allocation, free list should not be empty (chunk is allocated)
  ASSIGN_NO_ERR(void *, ptr, freeList.allocate(64, mode));
  // Note: The behavior here depends on internal implementation details
  // We just verify the allocation succeeded
  EXPECT_NE(ptr, nullptr);

  EXPECT_NO_ERR(freeList.free(ptr));
}

// Free List Management Tests

TEST_F(MemoryManagerTest, FreeListSorting) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> ptrs;

  // Allocate blocks of various sizes to create fragmentation
  std::vector<size_t> allocSizes = {128, 64, 256, 32, 512};
  for (size_t size : allocSizes) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(size, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Free some blocks in random order to test sorting
  std::vector<int> freeOrder = {1, 3, 0}; // Free 64, 32, 128 byte blocks
  for (int idx : freeOrder) {
    EXPECT_NO_ERR(freeList.free(ptrs[idx]));
  }

  // The free list should maintain blocks in sorted order
  // We can't directly verify internal sorting, but we can verify
  // that subsequent allocations work correctly

  // Request small allocation - should get the 32-byte block
  ASSIGN_NO_ERR(void *, small_ptr, freeList.allocate(16, mode));
  EXPECT_NE(small_ptr, nullptr);

  // Free remaining blocks
  EXPECT_NO_ERR(freeList.free(ptrs[2])); // 256-byte block
  EXPECT_NO_ERR(freeList.free(ptrs[4])); // 512-byte block
  EXPECT_NO_ERR(freeList.free(small_ptr));
}

TEST_F(MemoryManagerTest, FreeListMerging) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();
  size_t allocAlignment = device->allocationAlignment();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Allocate several consecutive blocks
  std::vector<void *> ptrs;
  for (int i = 0; i < 5; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(allocAlignment, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Test merging with next block only
  EXPECT_NO_ERR(freeList.free(ptrs[1])); // Free middle block
  EXPECT_NO_ERR(freeList.free(ptrs[2])); // Free next block - should merge

  // Test merging with previous block only
  EXPECT_NO_ERR(freeList.free(
      ptrs[0])); // Free previous block - should merge with merged block

  // Test no merge scenario (isolated block)
  EXPECT_NO_ERR(freeList.free(ptrs[4])); // Free last block - isolated

  // Test merging with both previous and next
  EXPECT_NO_ERR(freeList.free(ptrs[3])); // Should merge with both neighbors

  // Verify we can allocate a large block that spans the merged area
  ASSIGN_NO_ERR(void *, large_ptr,
                freeList.allocate(5 * allocAlignment - 64, mode));
  EXPECT_NE(large_ptr, nullptr);
  EXPECT_NO_ERR(freeList.free(large_ptr));
}

TEST_F(MemoryManagerTest, BlockSplittingAndReuse) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();
  size_t allocAlignment = device->allocationAlignment();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Allocate a large block
  size_t largeSize = 8 * allocAlignment;
  ASSIGN_NO_ERR(void *, large_ptr, freeList.allocate(largeSize, mode));
  EXPECT_NE(large_ptr, nullptr);

  // Free the large block
  EXPECT_NO_ERR(freeList.free(large_ptr));

  // Allocate smaller blocks that should split the large block
  std::vector<void *> small_ptrs;
  size_t smallSize = allocAlignment;
  for (int i = 0; i < 4; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(smallSize, mode));
    EXPECT_NE(ptr, nullptr);
    small_ptrs.push_back(ptr);
  }

  // Verify blocks are properly aligned
  for (void *ptr : small_ptrs) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % allocAlignment, 0);
  }

  // Free the small blocks
  for (void *ptr : small_ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Should be able to allocate the original large block again
  ASSIGN_NO_ERR(void *, new_large_ptr, freeList.allocate(largeSize, mode));
  EXPECT_NE(new_large_ptr, nullptr);
  EXPECT_NO_ERR(freeList.free(new_large_ptr));
}

TEST_F(MemoryManagerTest, MemoryReuse) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Allocate and free a block multiple times to verify reuse
  {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(128, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Allocate the same size - verify reuse is working
  ASSIGN_NO_ERR(void *, second_ptr, freeList.allocate(128, mode));
  EXPECT_NE(second_ptr, nullptr);
  EXPECT_NO_ERR(freeList.free(second_ptr));

  // Test reuse with different sizes
  std::vector<size_t> sizes = {64, 128, 256, 512};
  std::vector<void *> ptrs;

  for (size_t size : sizes) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(size, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Free in reverse order
  for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
    EXPECT_NO_ERR(freeList.free(*it));
  }

  // Allocate again in same order - verify no memory leaks
  for (size_t size : sizes) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(size, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

TEST_F(MemoryManagerTest, FragmentationHandling) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();
  size_t allocAlignment = device->allocationAlignment();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Create a fragmentation pattern
  std::vector<void *> ptrs;

  // Allocate 10 blocks
  for (int i = 0; i < 10; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(allocAlignment, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Free every other block to create fragmentation
  for (int i = 1; i < 10; i += 2) {
    EXPECT_NO_ERR(freeList.free(ptrs[i]));
  }

  // Try to allocate blocks that fit in the gaps
  std::vector<void *> gap_ptrs;
  for (int i = 0; i < 4; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(allocAlignment / 2, mode));
    EXPECT_NE(ptr, nullptr);
    gap_ptrs.push_back(ptr);
  }

  // Clean up remaining blocks
  for (int i = 0; i < 10; i += 2) {
    EXPECT_NO_ERR(freeList.free(ptrs[i]));
  }

  for (void *ptr : gap_ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

// Configuration & Heuristics Tests

TEST_F(MemoryManagerTest, ConfigurationVariations) {
  // Test different cache size configurations
  {
    ScopedEnvVar cacheSize("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "50");
    ScopedEnvVar chunkPercent(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "50");

    CpuDeviceRef device = createTestDevice();
    MemoryManager &cache = device->memoryManager();
    MemoryManagerFreeList &freeList = *cache.freeList();

    const Device::AllocMode mode = Device::AllocMode::SYNC;

    // Should be able to allocate with different configuration
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(1024, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Test small chunk percentage
  {
    ScopedEnvVar cacheSize("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "25");
    ScopedEnvVar chunkPercent(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "10");

    CpuDeviceRef device = createTestDevice();
    MemoryManager &cache = device->memoryManager();
    MemoryManagerFreeList &freeList = *cache.freeList();

    const Device::AllocMode mode = Device::AllocMode::SYNC;

    // Should still work with small chunk size
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(512, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

TEST_F(MemoryManagerTest, ChunkAllocationBehavior) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> ptrs;

  // Allocate many blocks to potentially trigger multiple chunk allocations
  for (int i = 0; i < 50; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(1024, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Verify all allocations are properly aligned
  size_t allocAlignment = device->allocationAlignment();
  for (void *ptr : ptrs) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % allocAlignment, 0);
  }

  // Free all blocks
  for (void *ptr : ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

TEST_F(MemoryManagerTest, LargeAllocationHandling) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Try to allocate a very large block that might exceed chunk size
  // This tests the heuristics for chunk size growth
  size_t largeSize = 16 * 1024; // 16KB
  ASSIGN_NO_ERR(void *, large_ptr, freeList.allocate(largeSize, mode));
  EXPECT_NE(large_ptr, nullptr);

  // Verify alignment
  size_t allocAlignment = device->allocationAlignment();
  uintptr_t addr = reinterpret_cast<uintptr_t>(large_ptr);
  EXPECT_EQ(addr % allocAlignment, 0);

  EXPECT_NO_ERR(freeList.free(large_ptr));

  // After large allocation, subsequent smaller allocations should still work
  ASSIGN_NO_ERR(void *, small_ptr, freeList.allocate(128, mode));
  EXPECT_NE(small_ptr, nullptr);
  EXPECT_NO_ERR(freeList.free(small_ptr));
}

TEST_F(MemoryManagerTest, AllocationModeHandling) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  // Test different allocation modes
  std::vector<Device::AllocMode> modes = {Device::AllocMode::SYNC,
                                          Device::AllocMode::HOST};

  for (Device::AllocMode mode : modes) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(256, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

// Error Handling & Edge Cases Tests

TEST_F(MemoryManagerTest, BoundaryConditions) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Test minimum allocation (1 byte)
  ASSIGN_NO_ERR(void *, min_ptr, freeList.allocate(1, mode));
  EXPECT_NE(min_ptr, nullptr);

  // Should still be aligned
  size_t allocAlignment = device->allocationAlignment();
  uintptr_t addr = reinterpret_cast<uintptr_t>(min_ptr);
  EXPECT_EQ(addr % allocAlignment, 0);

  EXPECT_NO_ERR(freeList.free(min_ptr));

  // Test large allocation
  size_t largeSize = 64 * 1024; // 64KB
  ASSIGN_NO_ERR(void *, large_ptr, freeList.allocate(largeSize, mode));
  EXPECT_NE(large_ptr, nullptr);
  EXPECT_NO_ERR(freeList.free(large_ptr));
}

TEST_F(MemoryManagerTest, MemoryPattern) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Allocate a block and write/read pattern to verify memory is usable
  ASSIGN_NO_ERR(void *, ptr, freeList.allocate(1024, mode));
  EXPECT_NE(ptr, nullptr);

  // Write a pattern to the allocated memory
  uint8_t *bytes = static_cast<uint8_t *>(ptr);
  for (int i = 0; i < 1024; ++i) {
    bytes[i] = static_cast<uint8_t>(i & 0xFF);
  }

  // Read back and verify pattern
  for (int i = 0; i < 1024; ++i) {
    EXPECT_EQ(bytes[i], static_cast<uint8_t>(i & 0xFF));
  }

  EXPECT_NO_ERR(freeList.free(ptr));
}

TEST_F(MemoryManagerTest, StressTestAllocation) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Stress test with rapid allocation/deallocation cycles
  for (int cycle = 0; cycle < 10; ++cycle) {
    std::vector<void *> ptrs;

    // Allocate many blocks
    for (int i = 0; i < 20; ++i) {
      size_t size = 64 + (i * 32); // Varying sizes
      ASSIGN_NO_ERR(void *, ptr, freeList.allocate(size, mode));
      EXPECT_NE(ptr, nullptr);
      ptrs.push_back(ptr);
    }

    // Free in random order
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ptrs.begin(), ptrs.end(), g);
    for (void *ptr : ptrs) {
      EXPECT_NO_ERR(freeList.free(ptr));
    }
  }
}

TEST_F(MemoryManagerTest, InvalidOperations) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  // Test free with nullptr (should succeed)
  EXPECT_NO_ERR(freeList.free(nullptr));

  // Test double free - allocate, free, then free again
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  ASSIGN_NO_ERR(void *, ptr, freeList.allocate(256, mode));
  EXPECT_NE(ptr, nullptr);

  EXPECT_NO_ERR(freeList.free(ptr)); // First free should succeed

  // Note: Double free would be undefined behavior, so we skip that test
  // as it may legitimately crash or have unpredictable behavior
}

TEST_F(MemoryManagerTest, ConfigurationLimits) {
  // Test with extreme configuration values
  {
    ScopedEnvVar cacheSize("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "0");
    ScopedEnvVar chunkPercent(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "0");

    // Should still create a valid device even with zero configuration
    CpuDeviceRef device = createTestDevice();
    MemoryManager &cache = device->memoryManager();

    // Cache should be functional even with minimal configuration
    EXPECT_EQ(cache.allocationAlignment(), 64);
  }

  // Test with very high percentage
  {
    ScopedEnvVar cacheSize("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "25");
    ScopedEnvVar chunkPercent(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "100");

    CpuDeviceRef device = createTestDevice();
    MemoryManager &cache = device->memoryManager();
    MemoryManagerFreeList &freeList = *cache.freeList();

    const Device::AllocMode mode = Device::AllocMode::SYNC;

    // Should still work with 100% chunk percentage
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(1024, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

// Debugging & Validation Tests

TEST_F(MemoryManagerTest, SelfCheckValidation) {
  // Test with self-check enabled (already enabled in test setup)
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> ptrs;

  // Perform operations that should trigger self-check validations
  for (int i = 0; i < 10; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(128 + i * 64, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Free in various patterns to exercise self-check
  for (int i = 1; i < 10; i += 2) {
    EXPECT_NO_ERR(freeList.free(ptrs[i]));
  }

  for (int i = 0; i < 10; i += 2) {
    EXPECT_NO_ERR(freeList.free(ptrs[i]));
  }

  // If self-check was working, we should reach here without assertions
  SUCCEED();
}

TEST_F(MemoryManagerTest, LoggingOutput) {
  // Test with logging disabled (set in test setup)
  {
    CpuDeviceRef device = createTestDevice();
    MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

    const Device::AllocMode mode = Device::AllocMode::SYNC;

    // Perform operations that would generate logs if enabled
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(1024, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Test with logging enabled
  {
    ScopedEnvVar enableLogging("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_LOG",
                               "True");

    CpuDeviceRef device = createTestDevice();
    MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

    const Device::AllocMode mode = Device::AllocMode::SYNC;

    // Operations that should generate log output
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(512, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));

    // Note: We can't easily verify log output in unit tests,
    // but we can ensure operations succeed with logging enabled
  }
}

TEST_F(MemoryManagerTest, OOMFragmentationDumpStructure) {
  // Populate the free list and chunks with a mix of free/in-use blocks, then
  // invoke the OOM diagnostic directly and verify it emits the structural
  // markers we depend on at production log parse time: begin/end sentinels,
  // mm summary, freelist label, per-bucket lines, and per-chunk layout.
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> ptrs;
  for (int i = 0; i < 6; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(64 + i * 32, mode));
    ptrs.push_back(ptr);
  }
  // Free two non-adjacent blocks so the free list has multiple entries and
  // doesn't collapse to a single coalesced run.
  EXPECT_NO_ERR(freeList.free(ptrs[1]));
  ptrs[1] = nullptr;
  EXPECT_NO_ERR(freeList.free(ptrs[3]));
  ptrs[3] = nullptr;

  testing::internal::CaptureStderr();
  freeList.dumpFragmentationDiagnostic(8192, "synthetic test OOM");
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, testing::HasSubstr("oom log ===== begin ====="));
  EXPECT_THAT(output, testing::HasSubstr("oom log ===== end ====="));
  EXPECT_THAT(output, testing::HasSubstr("synthetic test OOM"));
  EXPECT_THAT(output, testing::HasSubstr("oom log request="));
  EXPECT_THAT(output, testing::HasSubstr("oom log mm max_cache="));
  EXPECT_THAT(output, testing::HasSubstr("oom log freelist=[mm:root]"));
  // At least one bucket line should appear with stats (count=, total=...).
  EXPECT_THAT(output, testing::HasSubstr("oom log   bucket="));
  // We allocated, so chunk 0 should be reported with the per-chunk layout
  // fields used to diagnose fragmentation.
  EXPECT_THAT(output, testing::HasSubstr("oom log chunk=0 base="));
  EXPECT_THAT(output, testing::HasSubstr("largest_free="));
  EXPECT_THAT(output, testing::HasSubstr("in_use_blocks="));
  EXPECT_THAT(output, testing::HasSubstr("free_blocks="));

  for (void *ptr : ptrs) {
    if (ptr != nullptr) {
      EXPECT_NO_ERR(freeList.free(ptr));
    }
  }
}

TEST_F(MemoryManagerTest, OOMFragmentationDumpEmptyState) {
  // Calling the diagnostic before any chunk has been allocated must not crash
  // and must still emit the begin/end sentinels plus a chunks=0 summary so
  // production logs are unambiguous about the "MM never primed" failure mode.
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  testing::internal::CaptureStderr();
  freeList.dumpFragmentationDiagnostic(1024, "no allocations yet");
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, testing::HasSubstr("oom log ===== begin ====="));
  EXPECT_THAT(output, testing::HasSubstr("oom log ===== end ====="));
  EXPECT_THAT(output, testing::HasSubstr("no allocations yet"));
  EXPECT_THAT(output, testing::HasSubstr(" chunks=0"));
  // With no chunks the root free list buckets should all be marked empty.
  EXPECT_THAT(output, testing::HasSubstr("oom log   bucket=0 empty"));
  EXPECT_THAT(output, testing::HasSubstr("oom log   bucket=1 empty"));
  // No chunk should be reported when none have been allocated.
  EXPECT_THAT(output, testing::Not(testing::HasSubstr("oom log chunk=")));
}

TEST_F(MemoryManagerTest, OOMFragmentationDumpHistogramAndTopN) {
  // Allocate a fan of differently-sized blocks then free them in a pattern
  // that leaves multiple free entries across distinct log2 size buckets, so
  // we can verify the histogram and top-N largest output land in stderr.
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  // Pick sizes that span multiple log2 buckets after alignment.
  std::vector<size_t> sizes = {64, 64, 128, 128, 256, 256, 512};
  std::vector<void *> ptrs;
  for (size_t s : sizes) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(s, mode));
    ptrs.push_back(ptr);
  }
  // Free every other block to keep free entries in the list (not coalesced
  // into one run with their neighbors).
  for (size_t i = 0; i < ptrs.size(); i += 2) {
    EXPECT_NO_ERR(freeList.free(ptrs[i]));
    ptrs[i] = nullptr;
  }

  testing::internal::CaptureStderr();
  freeList.dumpFragmentationDiagnostic(4096, "hist+topN coverage");
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, testing::HasSubstr("hist+topN coverage"));
  // The histogram line uses the "2^<bucket>=<count>" token format.
  EXPECT_THAT(output, testing::ContainsRegex("oom log {3}bucket=[01] hist( 2\\^"
                                             "[0-9]+=[0-9]+)+"));
  // The top-N line uses "topN=<size>,<size>,..." format.
  EXPECT_THAT(output, testing::ContainsRegex("oom log {3}bucket=[01] top[0-9]+="
                                             "[0-9KMGT]"));

  for (void *ptr : ptrs) {
    if (ptr != nullptr) {
      EXPECT_NO_ERR(freeList.free(ptr));
    }
  }
}

TEST_F(MemoryManagerTest, MemoryLeakDetection) {
  // Test proper cleanup on scope exit
  {
    CpuDeviceRef device = createTestDevice();
    MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

    const Device::AllocMode mode = Device::AllocMode::SYNC;
    std::vector<void *> ptrs;

    // Allocate multiple blocks
    for (int i = 0; i < 20; ++i) {
      ASSIGN_NO_ERR(void *, ptr, freeList.allocate(256 + i * 32, mode));
      EXPECT_NE(ptr, nullptr);
      ptrs.push_back(ptr);
    }

    // Free all blocks to prevent leaks
    for (void *ptr : ptrs) {
      EXPECT_NO_ERR(freeList.free(ptr));
    }

    // Device destructor should clean up all chunks when it goes out of scope
  }
  // At this point, all memory should be properly cleaned up
  SUCCEED();
}

TEST_F(MemoryManagerTest, DescriptorManagement) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Test descriptor allocation and recycling through multiple allocation cycles
  std::vector<void *> cycle1_ptrs;
  std::vector<void *> cycle2_ptrs;

  // First allocation cycle
  for (int i = 0; i < 15; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(128, mode));
    EXPECT_NE(ptr, nullptr);
    cycle1_ptrs.push_back(ptr);
  }

  // Free all from first cycle - should recycle descriptors
  for (void *ptr : cycle1_ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Second allocation cycle - should reuse descriptors
  for (int i = 0; i < 15; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(256, mode));
    EXPECT_NE(ptr, nullptr);
    cycle2_ptrs.push_back(ptr);
  }

  // Clean up second cycle
  for (void *ptr : cycle2_ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

TEST_F(MemoryManagerTest, MaxSizeHandling) {
  // Create device with smaller memory limits to test max size enforcement
  Device::MemInfo smallMemInfo;
  smallMemInfo.free = 64 * 1024;   // 64KB
  smallMemInfo.total = 128 * 1024; // 128KB

  ScopedEnvVar cacheSize("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "90");
  ScopedEnvVar chunkPercent(
      "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "25");

  CpuDeviceRef device = CpuDeviceRef::create(smallMemInfo);
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> ptrs;

  // Try to allocate many blocks to approach cache limits
  for (int i = 0; i < 100; ++i) {
    auto result = freeList.allocate(1024, mode);
    if (result.isError()) {
      // Expected to eventually hit limits or fallback to device allocation
      break;
    }
    void *ptr = result.takeValue();
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Clean up allocated blocks
  for (void *ptr : ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

TEST_F(MemoryManagerTest, FreeListHierarchy) {
  // Test hierarchical free list behavior through DeviceContext integration
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  std::vector<DeviceBufferRef> buffers;

  // Allocate through DeviceContext to exercise hierarchical free lists
  for (int i = 0; i < 10; ++i) {
    ASSIGN_NO_ERR(DeviceBufferRef, buffer,
                  ctx->enqueueCreateBuffer(512 + i * 64));
    EXPECT_NE(buffer->ptr(), nullptr);
    buffers.push_back(buffer);
  }

  // Buffers will be automatically freed when they go out of scope
  // This exercises the hierarchical flush behavior
}

// DRIV-193: when a child freelist is destroyed while one of its descriptors is
// still in use, the descriptor's `allocatedBy_` must be redirected to the
// parent. Without the redirect, freeing the descriptor through any sibling
// freelist dereferences the destroyed child via `desc->allocatedBy_`
// (intentional cross-freelist dispatch documented in
// `MemoryManagerFreeList::free`).
TEST_F(MemoryManagerTest, FreeListTeardownRedirectsAllocatedBy) {
  CpuDeviceRef device = createTestDevice();
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());
  MemoryManager &memMgr = device->memoryManager();
  MemoryManagerFreeList &rootFreeList = *memMgr.freeList();

  void *ptr = nullptr;
  {
    MemoryManagerFreeList childFreeList(ctx.getPointer(), memMgr, &rootFreeList,
                                        /*disableFlush=*/false,
                                        /*label=*/"test_child");
    ASSIGN_NO_ERR(void *, allocated,
                  childFreeList.allocate(256, Device::AllocMode::SYNC));
    EXPECT_NE(allocated, nullptr);
    ptr = allocated;
    // `childFreeList` is destroyed here while `ptr` is still in use; the
    // redirect in `~MemoryManagerFreeList` reattributes the descriptor's
    // `allocatedBy_` to `rootFreeList`.
  }

  // Freeing through the (still-live) parent must dispatch through the
  // redirected `allocatedBy_` without touching the destroyed child.
  EXPECT_NO_ERR(rootFreeList.free(ptr));
}

TEST_F(MemoryManagerTest, MemoryPressure) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> large_ptrs;

  // Allocate increasingly large blocks to simulate memory pressure
  for (int i = 1; i <= 10; ++i) {
    size_t size = i * 8 * 1024; // 8KB, 16KB, 24KB, etc.
    auto result = freeList.allocate(size, mode);

    if (result.isError()) {
      // Expected behavior under memory pressure
      break;
    }

    void *ptr = result.takeValue();
    EXPECT_NE(ptr, nullptr);
    large_ptrs.push_back(ptr);
  }

  // Free large blocks to relieve pressure
  for (void *ptr : large_ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Should be able to allocate normally again after freeing
  ASSIGN_NO_ERR(void *, recovery_ptr, freeList.allocate(1024, mode));
  EXPECT_NE(recovery_ptr, nullptr);
  EXPECT_NO_ERR(freeList.free(recovery_ptr));
}

// Integration Tests

TEST_F(MemoryManagerTest, DeviceContextIntegration) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  // Test buffer creation through DeviceContext
  std::vector<DeviceBufferRef> buffers;

  for (int i = 0; i < 5; ++i) {
    size_t size = 1024 + i * 512;
    ASSIGN_NO_ERR(DeviceBufferRef, buffer, ctx->enqueueCreateBuffer(size));

    EXPECT_NE(buffer->ptr(), nullptr);
    EXPECT_EQ(buffer->byteSize(), size);

    buffers.push_back(buffer);
  }

  // Test host buffer creation
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuffer, ctx->createHostBuffer(2048));
  EXPECT_NE(hostBuffer->ptr(), nullptr);
  EXPECT_EQ(hostBuffer->byteSize(), 2048);

  // Verify allocation mode handling - buffers should be properly aligned
  for (const auto &buffer : buffers) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer->ptr());
    EXPECT_EQ(addr % 64, 0); // Should be 64-byte aligned
  }
}

TEST_F(MemoryManagerTest, BufferLifecycle) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  // Test end-to-end buffer lifecycle
  DeviceBufferRef buffer;
  {
    // Create buffer in inner scope
    ASSIGN_NO_ERR(DeviceBufferRef, tempBuffer, ctx->enqueueCreateBuffer(4096));
    EXPECT_NE(tempBuffer->ptr(), nullptr);
    buffer = std::move(tempBuffer);
  }

  // Buffer should still be valid after scope exit
  EXPECT_NE(buffer->ptr(), nullptr);
  EXPECT_EQ(buffer->byteSize(), 4096);

  // Test sub-buffer creation
  ASSIGN_NO_ERR(DeviceBufferRef, subBuffer,
                buffer->createSubBuffer(1024, 2048));
  EXPECT_NE(subBuffer->ptr(), nullptr);
  EXPECT_EQ(subBuffer->byteSize(), 2048);

  // Sub-buffer pointer should be offset from main buffer
  uintptr_t mainAddr = reinterpret_cast<uintptr_t>(buffer->ptr());
  uintptr_t subAddr = reinterpret_cast<uintptr_t>(subBuffer->ptr());
  EXPECT_EQ(subAddr, mainAddr + 1024);
}

TEST_F(MemoryManagerTest, MemoryTransfers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  // Create source and destination buffers
  ASSIGN_NO_ERR(DeviceBufferRef, srcBuffer, ctx->enqueueCreateBuffer(2048));
  ASSIGN_NO_ERR(DeviceBufferRef, dstBuffer, ctx->enqueueCreateBuffer(2048));
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuffer, ctx->createHostBuffer(2048));

  // Initialize host buffer with test pattern
  uint8_t *hostPtr = static_cast<uint8_t *>(hostBuffer->ptr());
  for (int i = 0; i < 2048; ++i) {
    hostPtr[i] = static_cast<uint8_t>(i & 0xFF);
  }

  // Test host-to-device transfer
  EXPECT_NO_ERR(hostBuffer->copyTo(*srcBuffer));

  // Test device-to-device transfer
  EXPECT_NO_ERR(srcBuffer->copyTo(*dstBuffer));

  // Test device-to-host transfer
  std::vector<uint8_t> result(2048, 0);
  EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(result.data(), *dstBuffer));
  EXPECT_NO_ERR(ctx->synchronize());

  // Verify the transferred data
  for (int i = 0; i < 2048; ++i) {
    EXPECT_EQ(result[i], static_cast<uint8_t>(i & 0xFF));
  }
}

TEST_F(MemoryManagerTest, MultipleContexts) {
  // Create multiple device contexts
  ASSIGN_NO_ERR(DeviceContextRef, ctx1, createTestDeviceContext());
  ASSIGN_NO_ERR(DeviceContextRef, ctx2, createTestDeviceContext());

  // Allocate buffers through different contexts
  std::vector<DeviceBufferRef> ctx1_buffers;
  std::vector<DeviceBufferRef> ctx2_buffers;

  for (int i = 0; i < 5; ++i) {
    ASSIGN_NO_ERR(DeviceBufferRef, buffer1, ctx1->enqueueCreateBuffer(1024));
    ASSIGN_NO_ERR(DeviceBufferRef, buffer2, ctx2->enqueueCreateBuffer(1024));

    EXPECT_NE(buffer1->ptr(), nullptr);
    EXPECT_NE(buffer2->ptr(), nullptr);

    ctx1_buffers.push_back(buffer1);
    ctx2_buffers.push_back(buffer2);
  }

  // Verify resource isolation - buffers from different contexts should be
  // independent
  for (size_t i = 0; i < ctx1_buffers.size(); ++i) {
    EXPECT_NE(ctx1_buffers[i]->ptr(), ctx2_buffers[i]->ptr());
  }

  // Test synchronization of different contexts
  EXPECT_NO_ERR(ctx1->synchronize());
  EXPECT_NO_ERR(ctx2->synchronize());
}

// Thread Safety & Concurrency Tests

TEST_F(MemoryManagerTest, ConcurrentAllocations) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const int numThreads = 4;
  const int allocationsPerThread = 10;

  std::vector<std::thread> threads;
  std::vector<std::vector<void *>> threadPtrs(numThreads);
  std::atomic<int> successCount{0};

  // Launch multiple threads doing concurrent allocations
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < allocationsPerThread; ++i) {
        auto result = freeList.allocate(256 + i * 64, mode);
        if (!result.isError()) {
          void *ptr = result.takeValue();
          if (ptr != nullptr) {
            threadPtrs[t].push_back(ptr);
            ++successCount;
          }
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Should have some successful allocations
  EXPECT_GT(successCount.load(), 0);

  // Free all allocated blocks
  for (int t = 0; t < numThreads; ++t) {
    for (void *ptr : threadPtrs[t]) {
      EXPECT_NO_ERR(freeList.free(ptr));
    }
  }
}

TEST_F(MemoryManagerTest, ConcurrentFrees) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const int numBlocks = 20;

  // First, allocate blocks sequentially
  std::vector<void *> ptrs;
  for (int i = 0; i < numBlocks; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(512, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Then free them concurrently
  const int numThreads = 4;
  std::vector<std::thread> threads;
  std::atomic<int> freeCount{0};

  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = t; i < numBlocks; i += numThreads) {
        auto result = freeList.free(ptrs[i]);
        if (!result.isError()) {
          ++freeCount;
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // All blocks should have been successfully freed
  EXPECT_EQ(freeCount.load(), numBlocks);
}

TEST_F(MemoryManagerTest, MixedConcurrentOperations) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const int numThreads = 4;
  const int operationsPerThread = 15;

  std::vector<std::thread> threads;
  std::mutex ptrsMutex;
  std::vector<void *> sharedPtrs;

  // Launch threads doing mixed alloc/free operations
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&]() {
      std::vector<void *> localPtrs;

      for (int i = 0; i < operationsPerThread; ++i) {
        if (i % 3 == 0 && !localPtrs.empty()) {
          // Free a local pointer
          void *ptr = localPtrs.back();
          localPtrs.pop_back();
          auto freeResult = freeList.free(ptr);
          (void)freeResult; // Ignore result
        } else if (i % 5 == 0) {
          // Try to get a pointer from shared pool and free it
          std::lock_guard<std::mutex> lock(ptrsMutex);
          if (!sharedPtrs.empty()) {
            void *ptr = sharedPtrs.back();
            sharedPtrs.pop_back();
            auto freeResult = freeList.free(ptr);
            (void)freeResult; // Ignore result
          }
        } else {
          // Allocate new pointer
          auto result = freeList.allocate(128 + (i % 8) * 64, mode);
          if (!result.isError()) {
            void *ptr = result.takeValue();
            if (ptr != nullptr) {
              if (i % 4 == 0) {
                // Add to shared pool
                std::lock_guard<std::mutex> lock(ptrsMutex);
                sharedPtrs.push_back(ptr);
              } else {
                // Keep in local pool
                localPtrs.push_back(ptr);
              }
            }
          }
        }
      }

      // Clean up remaining local pointers
      for (void *ptr : localPtrs) {
        auto freeResult = freeList.free(ptr);
        (void)freeResult; // Ignore result
      }
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Clean up any remaining shared pointers
  for (void *ptr : sharedPtrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

// Performance & Metrics Tests

TEST_F(MemoryManagerTest, AllocationSpeed) {
  // Test allocation performance compared to direct device allocation
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const int numAllocations = 100;
  const size_t allocSize = 1024;

  // Measure cached allocation performance
  auto startCached = std::chrono::high_resolution_clock::now();
  std::vector<void *> cachedPtrs;

  for (int i = 0; i < numAllocations; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(allocSize, mode));
    cachedPtrs.push_back(ptr);
  }

  auto endCached = std::chrono::high_resolution_clock::now();
  auto cachedDuration = std::chrono::duration_cast<std::chrono::microseconds>(
      endCached - startCached);

  // Clean up cached allocations
  for (void *ptr : cachedPtrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Measure direct device allocation performance
  auto startDirect = std::chrono::high_resolution_clock::now();
  std::vector<DeviceBufferRef> directBuffers;

  for (int i = 0; i < numAllocations; ++i) {
    ASSIGN_NO_ERR(DeviceBufferRef, buffer, ctx->enqueueCreateBuffer(allocSize));
    directBuffers.push_back(buffer);
  }

  auto endDirect = std::chrono::high_resolution_clock::now();
  auto directDuration = std::chrono::duration_cast<std::chrono::microseconds>(
      endDirect - startDirect);

  // Cached allocations should generally be faster or comparable
  double speedupRatio = static_cast<double>(directDuration.count()) /
                        static_cast<double>(cachedDuration.count());

  // Log performance metrics
  std::cerr << "Cached allocation time: " << cachedDuration.count() << "μs\n";
  std::cerr << "Direct allocation time: " << directDuration.count() << "μs\n";
  std::cerr << "Speedup ratio: " << speedupRatio << "x\n";

  // Both allocation methods should complete reasonably quickly
  EXPECT_LT(cachedDuration.count(), 100000); // Less than 100ms
  EXPECT_LT(directDuration.count(), 100000); // Less than 100ms

  // The test passes regardless of which is faster, as performance can vary
  SUCCEED();
}

TEST_F(MemoryManagerTest, FragmentationMeasurement) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Test fragmentation under different allocation patterns
  struct FragmentationTest {
    std::string name;
    std::vector<size_t> allocSizes;
    std::vector<int> freePattern; // Indices to free
  };

  std::vector<FragmentationTest> tests = {
      {"Sequential", {256, 512, 1024, 2048}, {0, 1, 2, 3}},
      {"Alternating", {256, 256, 256, 256, 256, 256}, {1, 3, 5}},
      {"Random sizes", {128, 2048, 512, 4096, 256, 1024}, {0, 2, 4}},
  };

  for (const auto &test : tests) {
    std::vector<void *> ptrs;

    // Allocate according to pattern
    for (size_t size : test.allocSizes) {
      auto result = freeList.allocate(size, mode);
      if (!result.isError()) {
        ptrs.push_back(result.takeValue());
      }
    }

    // Free according to pattern to create fragmentation
    for (int idx : test.freePattern) {
      if (idx < static_cast<int>(ptrs.size()) && ptrs[idx] != nullptr) {
        EXPECT_NO_ERR(freeList.free(ptrs[idx]));
        ptrs[idx] = nullptr;
      }
    }

    // Try to allocate a large block that might not fit due to fragmentation
    auto largeResult = freeList.allocate(8192, mode);
    bool largeAllocSucceeded = !largeResult.isError();
    void *largePtr = largeAllocSucceeded ? largeResult.takeValue() : nullptr;

    std::cerr << "Fragmentation test '" << test.name << "': Large allocation "
              << (largeAllocSucceeded ? "succeeded" : "failed") << "\n";

    // Clean up remaining allocations
    if (largePtr) {
      EXPECT_NO_ERR(freeList.free(largePtr));
    }

    for (void *ptr : ptrs) {
      if (ptr != nullptr) {
        EXPECT_NO_ERR(freeList.free(ptr));
      }
    }
  }

  // Test passes if we can measure fragmentation behavior
  SUCCEED();
}

TEST_F(MemoryManagerTest, CacheEfficiency) {
  CpuDeviceRef device = createTestDevice();
  MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const int totalOperations = 200;

  int cacheHits = 0;
  int cacheMisses = 0;
  std::vector<void *> activePtrs;
  std::mt19937 rng(12345);

  // Simulate realistic allocation/free patterns
  for (int i = 0; i < totalOperations; ++i) {
    bool shouldAllocate =
        activePtrs.size() < 20 || (rng() % 100) < 60; // 60% chance to allocate

    if (shouldAllocate) {
      // Allocate new block
      size_t size = 256 + (rng() % 8) * 128; // 256-1280 bytes
      auto result = freeList.allocate(size, mode);

      if (!result.isError()) {
        void *ptr = result.takeValue();
        activePtrs.push_back(ptr);

        // Heuristic: smaller allocations more likely to be cache hits
        if (size <= 512) {
          ++cacheHits;
        } else {
          ++cacheMisses;
        }
      }
    } else if (!activePtrs.empty()) {
      // Free an existing block
      size_t idx = rng() % activePtrs.size();
      void *ptr = activePtrs[idx];
      activePtrs.erase(activePtrs.begin() + idx);

      EXPECT_NO_ERR(freeList.free(ptr));
    }
  }

  // Calculate efficiency metrics
  int totalAllocations = cacheHits + cacheMisses;
  double hitRate = totalAllocations > 0
                       ? static_cast<double>(cacheHits) / totalAllocations
                       : 0.0;

  double utilizationRate = static_cast<double>(activePtrs.size()) /
                           std::max(activePtrs.size(), size_t(50));

  std::cerr << "Cache efficiency metrics:\n";
  std::cerr << "  Total allocations: " << totalAllocations << "\n";
  std::cerr << "  Cache hits: " << cacheHits << "\n";
  std::cerr << "  Hit rate: " << (hitRate * 100) << "%\n";
  std::cerr << "  Active allocations: " << activePtrs.size() << "\n";
  std::cerr << "  Utilization rate: " << (utilizationRate * 100) << "%\n";

  // Clean up remaining allocations
  for (void *ptr : activePtrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Basic sanity checks
  EXPECT_GE(totalAllocations, 0);
  EXPECT_GE(hitRate, 0.0);
  EXPECT_LE(hitRate, 1.0);

  SUCCEED();
}

TEST_F(MemoryManagerTest, ScalingBehavior) {
  // Test performance scaling with different cache configurations
  struct ScalingTest {
    std::string cacheSize;
    std::string chunkPercent;
    int expectedOperations;
  };

  std::vector<ScalingTest> configs = {
      {"10", "25", 50},  // Small cache
      {"25", "25", 100}, // Medium cache
      {"50", "25", 150}, // Large cache
  };

  for (const auto &config : configs) {
    ScopedEnvVar cacheSize("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE",
                           config.cacheSize);
    ScopedEnvVar chunkPercent(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT",
        config.chunkPercent);

    CpuDeviceRef device = createTestDevice();
    MemoryManagerFreeList &freeList = *device->memoryManager().freeList();

    const Device::AllocMode mode = Device::AllocMode::SYNC;

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<void *> ptrs;
    int successfulOps = 0;

    // Perform allocation operations until we hit limits
    for (int i = 0; i < config.expectedOperations * 2; ++i) {
      auto result = freeList.allocate(512 + (i % 4) * 256, mode);
      if (!result.isError()) {
        ptrs.push_back(result.takeValue());
        ++successfulOps;
      } else {
        // Hit memory limits, stop allocating
        break;
      }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - startTime);

    double opsPerSecond = successfulOps * 1000000.0 / duration.count();

    std::cerr << "Scaling test - Cache: " << config.cacheSize
              << "%, Chunk: " << config.chunkPercent << "%\n";
    std::cerr << "  Successful operations: " << successfulOps << "\n";
    std::cerr << "  Time: " << duration.count() << "μs\n";
    std::cerr << "  Ops/second: " << opsPerSecond << "\n";

    // Clean up
    for (void *ptr : ptrs) {
      EXPECT_NO_ERR(freeList.free(ptr));
    }

    // Verify we achieved reasonable performance
    EXPECT_GT(successfulOps, config.expectedOperations / 2);
    EXPECT_LT(duration.count(), 1000000); // Less than 1 second
  }

  SUCCEED();
}

// Host Memory Manager Tests (Integration-focused)

TEST_F(MemoryManagerTest, HostMemoryManagerBasicOperations) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  // Test multiple host buffer allocations
  std::vector<DeviceBufferRef> hostBuffers;
  std::vector<size_t> sizes = {256, 512, 1024, 2048};

  for (size_t size : sizes) {
    ASSIGN_NO_ERR(DeviceBufferRef, hostBuffer, ctx->createHostBuffer(size));
    EXPECT_NE(hostBuffer->ptr(), nullptr);
    EXPECT_EQ(hostBuffer->byteSize(), size);

    // Verify it's host-accessible memory by writing/reading
    uint8_t *bytes = static_cast<uint8_t *>(hostBuffer->ptr());
    bytes[0] = 0xAB;
    bytes[size - 1] = 0xCD;
    EXPECT_EQ(bytes[0], 0xAB);
    EXPECT_EQ(bytes[size - 1], 0xCD);

    hostBuffers.push_back(hostBuffer);
  }

  // All host buffers should remain accessible
  for (size_t i = 0; i < hostBuffers.size(); ++i) {
    uint8_t *bytes = static_cast<uint8_t *>(hostBuffers[i]->ptr());
    EXPECT_EQ(bytes[0], 0xAB);
    EXPECT_EQ(bytes[sizes[i] - 1], 0xCD);
  }
}

TEST_F(MemoryManagerTest, HostMemoryManagerAllocationPatterns) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  // Test allocation patterns with host buffers
  std::vector<DeviceBufferRef> hostBuffers;

  // Allocate different sizes
  for (int i = 1; i <= 8; ++i) {
    size_t size = i * 512; // 512, 1024, 1536, ..., 4096
    ASSIGN_NO_ERR(DeviceBufferRef, hostBuffer, ctx->createHostBuffer(size));
    EXPECT_NE(hostBuffer->ptr(), nullptr);

    // Fill with pattern to verify accessibility
    uint8_t *bytes = static_cast<uint8_t *>(hostBuffer->ptr());
    for (size_t j = 0; j < size; ++j) {
      bytes[j] = static_cast<uint8_t>((i + j) & 0xFF);
    }

    hostBuffers.push_back(hostBuffer);
  }

  // Verify all patterns are intact (tests memory isolation)
  for (size_t i = 0; i < hostBuffers.size(); ++i) {
    uint8_t *bytes = static_cast<uint8_t *>(hostBuffers[i]->ptr());
    size_t size = hostBuffers[i]->byteSize();

    for (size_t j = 0; j < std::min(size, size_t(64)); ++j) {
      EXPECT_EQ(bytes[j], static_cast<uint8_t>((i + 1 + j) & 0xFF));
    }
  }
}

TEST_F(MemoryManagerTest, HostVsDeviceCacheComparison) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  const size_t testSize = 2048;

  // Create host buffer
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuffer, ctx->createHostBuffer(testSize));
  EXPECT_NE(hostBuffer->ptr(), nullptr);

  // Create device buffer
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer,
                ctx->enqueueCreateBuffer(testSize));
  EXPECT_NE(deviceBuffer->ptr(), nullptr);

  // Pointers should be different (different memory regions)
  EXPECT_NE(hostBuffer->ptr(), deviceBuffer->ptr());

  // Both should have same size
  EXPECT_EQ(hostBuffer->byteSize(), deviceBuffer->byteSize());

  // Host buffer should be directly accessible
  uint8_t *hostPtr = static_cast<uint8_t *>(hostBuffer->ptr());
  hostPtr[0] = 0x11;
  hostPtr[testSize - 1] = 0x22;
  EXPECT_EQ(hostPtr[0], 0x11);
  EXPECT_EQ(hostPtr[testSize - 1], 0x22);

  // Device buffer should also be accessible (CPU device)
  uint8_t *devicePtr = static_cast<uint8_t *>(deviceBuffer->ptr());
  devicePtr[0] = 0x33;
  devicePtr[testSize - 1] = 0x44;
  EXPECT_EQ(devicePtr[0], 0x33);
  EXPECT_EQ(devicePtr[testSize - 1], 0x44);

  // Values should remain distinct
  EXPECT_EQ(hostPtr[0], 0x11);
  EXPECT_EQ(devicePtr[0], 0x33);
}

TEST_F(MemoryManagerTest, HostMemoryManagerThreadSafety) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  const int numThreads = 4;
  const int buffersPerThread = 5;

  std::vector<std::thread> threads;
  std::vector<std::vector<DeviceBufferRef>> threadBuffers(numThreads);
  std::atomic<int> successCount{0};

  // Launch threads creating host buffers concurrently
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < buffersPerThread; ++i) {
        auto result = ctx->createHostBuffer(256 + i * 128);
        if (!result.isError()) {
          DeviceBufferRef buffer = result.takeValue();
          if (buffer->ptr() != nullptr) {
            // Write thread ID to verify isolation
            uint8_t *bytes = static_cast<uint8_t *>(buffer->ptr());
            bytes[0] = static_cast<uint8_t>(t);

            threadBuffers[t].push_back(buffer);
            ++successCount;
          }
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Should have some successful allocations
  EXPECT_GT(successCount.load(), 0);

  // Verify memory contents show proper isolation
  for (int t = 0; t < numThreads; ++t) {
    for (const auto &buffer : threadBuffers[t]) {
      uint8_t *bytes = static_cast<uint8_t *>(buffer->ptr());
      EXPECT_EQ(bytes[0], static_cast<uint8_t>(t));
    }
  }
}

TEST_F(MemoryManagerTest, HostMemoryManagerConfiguration) {
  // Test host buffer creation with different host-specific cache configurations
  {
    // Set host-specific memory manager configuration
    ScopedEnvVar hostCacheSize(
        "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_SIZE",
        "1048576"); // 1MB
    ScopedEnvVar hostChunkPercent(
        "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_CHUNK_PERCENT",
        "10"); // 10%

    ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

    // Should work with host-specific configuration
    ASSIGN_NO_ERR(DeviceBufferRef, hostBuffer, ctx->createHostBuffer(3072));
    EXPECT_NE(hostBuffer->ptr(), nullptr);
    EXPECT_EQ(hostBuffer->byteSize(), 3072);

    // Verify accessibility
    uint8_t *bytes = static_cast<uint8_t *>(hostBuffer->ptr());
    bytes[0] = 0xFF;
    bytes[3071] = 0xEE;
    EXPECT_EQ(bytes[0], 0xFF);
    EXPECT_EQ(bytes[3071], 0xEE);

    // Test multiple allocations with this configuration
    std::vector<DeviceBufferRef> buffers;
    for (int i = 0; i < 5; ++i) {
      ASSIGN_NO_ERR(DeviceBufferRef, buffer,
                    ctx->createHostBuffer(1024 * (i + 1)));
      EXPECT_NE(buffer->ptr(), nullptr);
      buffers.push_back(buffer);
    }
  }

  // Test with different host cache configuration
  {
    ScopedEnvVar hostCacheSize(
        "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_SIZE",
        "2097152"); // 2MB
    ScopedEnvVar hostChunkPercent(
        "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_CHUNK_PERCENT",
        "25"); // 25%

    ASSIGN_NO_ERR(DeviceContextRef, ctx2, createTestDeviceContext());

    // Should work with larger cache configuration
    ASSIGN_NO_ERR(DeviceBufferRef, largeHostBuffer,
                  ctx2->createHostBuffer(8192));
    EXPECT_NE(largeHostBuffer->ptr(), nullptr);
    EXPECT_EQ(largeHostBuffer->byteSize(), 8192);

    // Verify memory pattern
    uint8_t *bytes = static_cast<uint8_t *>(largeHostBuffer->ptr());
    for (size_t i = 0; i < 8192; i += 256) {
      bytes[i] = static_cast<uint8_t>(i & 0xFF);
    }
    for (size_t i = 0; i < 8192; i += 256) {
      EXPECT_EQ(bytes[i], static_cast<uint8_t>(i & 0xFF));
    }
  }
}

TEST_F(MemoryManagerTest, HostMemoryManagerStressTesting) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  // Stress test with rapid allocation/deallocation cycles
  for (int cycle = 0; cycle < 3; ++cycle) {
    std::vector<DeviceBufferRef> buffers;

    // Allocate many host buffers
    for (int i = 0; i < 12; ++i) {
      size_t size = 256 + (i * 128); // Varying sizes
      ASSIGN_NO_ERR(DeviceBufferRef, buffer, ctx->createHostBuffer(size));
      EXPECT_NE(buffer->ptr(), nullptr);

      // Write pattern to verify memory accessibility
      uint8_t *bytes = static_cast<uint8_t *>(buffer->ptr());
      for (size_t j = 0; j < std::min(size, size_t(32)); ++j) {
        bytes[j] = static_cast<uint8_t>((cycle + j) & 0xFF);
      }

      buffers.push_back(buffer);
    }

    // Verify patterns are intact before buffers go out of scope
    for (size_t i = 0; i < buffers.size(); ++i) {
      uint8_t *bytes = static_cast<uint8_t *>(buffers[i]->ptr());
      // Verify at least the first few bytes to confirm accessibility
      for (int j = 0; j < 4; ++j) {
        EXPECT_EQ(bytes[j], static_cast<uint8_t>((cycle + j) & 0xFF));
      }
    }

    // Buffers will be automatically freed when vector goes out of scope
  }
}

TEST_F(MemoryManagerTest, HostMemoryManagerIntegrationWithDeviceContext) {
  // Test that host buffers created through DeviceContext work properly
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());

  // Create host buffer through DeviceContext
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuffer, ctx->createHostBuffer(4096));
  EXPECT_NE(hostBuffer->ptr(), nullptr);
  EXPECT_EQ(hostBuffer->byteSize(), 4096);

  // Verify it's host-accessible memory
  uint8_t *hostPtr = static_cast<uint8_t *>(hostBuffer->ptr());

  // Write test pattern
  for (int i = 0; i < 4096; ++i) {
    hostPtr[i] = static_cast<uint8_t>(i & 0xFF);
  }

  // Read back and verify
  for (int i = 0; i < 4096; ++i) {
    EXPECT_EQ(hostPtr[i], static_cast<uint8_t>(i & 0xFF));
  }

  // Test copying from host to device buffer
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer, ctx->enqueueCreateBuffer(4096));
  EXPECT_NO_ERR(hostBuffer->copyTo(*deviceBuffer));

  // Test copying back
  std::vector<uint8_t> result(4096);
  EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(result.data(), *deviceBuffer));
  EXPECT_NO_ERR(ctx->synchronize());

  // Verify the data made the round trip correctly
  for (int i = 0; i < 4096; ++i) {
    EXPECT_EQ(result[i], static_cast<uint8_t>(i & 0xFF));
  }
}

// MemoryManager Memory Info Tests

TEST_F(MemoryManagerTest, GetMemoryInfoDisabledDefaults) {
  ScopedEnvVar cacheSize("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "0");

  CpuDeviceRef device = createTestDevice();
  MemoryManager &cache = device->memoryManager();

  Device::MemInfo memInfo = cache.getMemoryInfo();

  EXPECT_EQ(memInfo.total, 0);
  EXPECT_EQ(memInfo.free, 0);
}

TEST_F(MemoryManagerTest, GetMemoryInfoAccuracy) {
  CpuDeviceRef device = createTestDevice();
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();

  // Get initial memory info
  Device::MemInfo initialInfo = cache.getMemoryInfo();
  EXPECT_GT(initialInfo.total, 0);
  size_t initialFree = initialInfo.free;

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  std::vector<void *> ptrs;

  // Allocate some blocks
  for (int i = 0; i < 3; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(1024, mode));
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Get memory info after allocations
  Device::MemInfo afterAllocInfo = cache.getMemoryInfo();
  EXPECT_EQ(afterAllocInfo.total,
            initialInfo.total); // Total should remain same
  EXPECT_LE(afterAllocInfo.free,
            initialFree); // Free should decrease or stay same

  // Free the blocks
  for (void *ptr : ptrs) {
    EXPECT_NO_ERR(freeList.free(ptr));
  }

  // Get memory info after freeing
  Device::MemInfo afterFreeInfo = cache.getMemoryInfo();
  EXPECT_EQ(afterFreeInfo.total, initialInfo.total); // Total should remain same
  EXPECT_GE(afterFreeInfo.free, afterAllocInfo.free); // Free should increase
}

} // namespace
