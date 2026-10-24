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
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

using namespace M::Driver;
using M::ErrorOr;

extern char **environ;

namespace {

class GPUMemoryManagerTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Check if GPU is available before running tests.
    // Recall that device objects are memoized so subsequent calls will only
    // retrieve the device object that the first call creates.
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError()) {
      GTEST_SKIP() << "GPU not available, skipping GPU memory manager tests";
    }
#if MODULAR_MACOS
    // GEX-2567: Inside MetalDeviceContext::get(id), useMemoryManager = false
    GTEST_SKIP()
        << "MemoryManager not used in MacOS, skipping GPU memory manager tests";
#endif
  }

  ErrorOr<DeviceRef> createTestDevice() {
    return Device::create(Device::gpuAPI);
  }

  // Helper to create DeviceContext for memory manager testing
  ErrorOr<DeviceContextRef> createTestDeviceContext() {
    return DeviceContext::create(Device::gpuAPI);
  }

  // Helper to create a host memory manager for testing with CUDA device
  std::unique_ptr<MemoryManager> createTestHostCache() {
    auto deviceOr = createTestDevice();
    if (deviceOr.isError()) {
      return nullptr;
    }
    DeviceRef device = deviceOr.takeValue();
    return std::unique_ptr<MemoryManager>(MemoryManager::onHost(*device, 64));
  }
};

// Keep this test as the first in the file in order to ensure that the free list
// is empty before any other tests are run. All tests allocate (and free)
// chunks. Since `createTestDevice()` retrieves a memoized device object, the
// free list may not be empty as post-condition. Since this test checks for
// pre-conditions right after the device is initialized, it must run first.
TEST_F(GPUMemoryManagerTest, FreeListEmpty) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();

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

TEST_F(GPUMemoryManagerTest, MemoryManagerCreation) {
  // Test creation through CUDA device
  {
    ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
    MemoryManager &cache = device->memoryManager();

    // CUDA device should create device memory manager, not host cache
    EXPECT_FALSE(cache.isHost());
    EXPECT_EQ(cache.allocationAlignment(), device->allocationAlignment());
  }

  // Test factory method for host cache with CUDA device
  {
    std::unique_ptr<MemoryManager> cache = createTestHostCache();

    EXPECT_TRUE(cache != nullptr);
    EXPECT_TRUE(cache->isHost());
    EXPECT_EQ(cache->allocationAlignment(), 64);
  }

  // Test factory method for device cache with CUDA device
  {
    ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
    ASSIGN_NO_ERR(Device::MemInfo, memInfo, device->getMemoryInfo());

    std::unique_ptr<MemoryManager> cache(
        MemoryManager::onDevice(*device, memInfo, 128));

    EXPECT_TRUE(cache != nullptr);
    EXPECT_FALSE(cache->isHost());
    EXPECT_EQ(cache->allocationAlignment(), 128);
  }
}

TEST_F(GPUMemoryManagerTest, AlignmentHandling) {
  // Test device memory manager alignment
  {
    ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
    MemoryManager &cache = device->memoryManager();

    // CUDA device should have specific alignment requirements
    size_t expectedAlignment = device->allocationAlignment();
    EXPECT_EQ(cache.allocationAlignment(), expectedAlignment);
    EXPECT_GT(expectedAlignment, 0);
  }
}

TEST_F(GPUMemoryManagerTest, BasicAllocationAndFree) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();
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

TEST_F(GPUMemoryManagerTest, MultipleAllocations) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();
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

TEST_F(GPUMemoryManagerTest, AllocationSizes) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();
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

TEST_F(GPUMemoryManagerTest, CUDASpecificMemoryPatterns) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Allocate a block of GPU memory
  ASSIGN_NO_ERR(void *, ptr, freeList.allocate(1024, mode));
  EXPECT_NE(ptr, nullptr);

  // For CUDA devices, we can't directly write to device memory from host
  // without proper transfers, so we just verify the allocation succeeded
  // and the pointer is valid
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  size_t allocAlignment = device->allocationAlignment();
  EXPECT_EQ(addr % allocAlignment, 0);

  EXPECT_NO_ERR(freeList.free(ptr));
}

TEST_F(GPUMemoryManagerTest, LargeAllocationHandling) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Try to allocate a larger block suitable for GPU
  size_t largeSize = 64 * 1024; // 64KB
  ASSIGN_NO_ERR(void *, largePtr, freeList.allocate(largeSize, mode));
  EXPECT_NE(largePtr, nullptr);

  // Verify alignment
  size_t allocAlignment = device->allocationAlignment();
  uintptr_t addr = reinterpret_cast<uintptr_t>(largePtr);
  EXPECT_EQ(addr % allocAlignment, 0);

  EXPECT_NO_ERR(freeList.free(largePtr));

  // After large allocation, subsequent smaller allocations should still work
  ASSIGN_NO_ERR(void *, smallPtr, freeList.allocate(128, mode));
  EXPECT_NE(smallPtr, nullptr);
  EXPECT_NO_ERR(freeList.free(smallPtr));
}

TEST_F(GPUMemoryManagerTest, AllocationModeHandling) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();

  // Test different allocation modes relevant to CUDA
  std::vector<Device::AllocMode> modes = {Device::AllocMode::SYNC};

  for (Device::AllocMode mode : modes) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(256, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

TEST_F(GPUMemoryManagerTest, ConfigurationVariations) {
  // Test different cache size configurations for CUDA
  {
    ScopedEnvVar cacheSize("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE",
                           "50000");
    ScopedEnvVar chunkPercent(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "50");

    ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
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

    ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
    MemoryManager &cache = device->memoryManager();
    MemoryManagerFreeList &freeList = *cache.freeList();

    const Device::AllocMode mode = Device::AllocMode::SYNC;

    // Should still work with small chunk size
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(512, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }
}

TEST_F(GPUMemoryManagerTest, DeviceContextIntegration) {
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

  // Verify allocation alignment for GPU device
  for (const auto &buffer : buffers) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer->ptr());
    // GPU devices typically have specific alignment requirements
    EXPECT_EQ(addr % ctx->device().allocationAlignment(), 0);
  }
}

TEST_F(GPUMemoryManagerTest, HostVsDeviceCacheComparison) {
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

  // Device buffer memory access depends on CUDA context and cannot be
  // directly accessed from host code like CPU memory
}

TEST_F(GPUMemoryManagerTest, MemoryTransfers) {
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

TEST_F(GPUMemoryManagerTest, ConcurrentAllocations) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  MemoryManager &cache = device->memoryManager();
  MemoryManagerFreeList &freeList = *cache.freeList();

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

TEST_F(GPUMemoryManagerTest, GetMemoryInfoAccuracy) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
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

TEST_F(GPUMemoryManagerTest, BufferUnderflowTest) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());
  ctx->freeList().enableRedzones();

  ASSIGN_NO_ERR(void *, vptr,
                ctx->freeList().allocate(1024, Device::AllocMode::SYNC));
  ASSERT_NE(vptr, nullptr);

  uint8_t *ptr = static_cast<uint8_t *>(vptr);
  ptr -= 10;

  uint8_t dead[2] = {0xDE, 0xAD};
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(ptr, dead, sizeof(dead)));
  EXPECT_NO_ERR(ctx->synchronize());

  // Free the pointer which results in a std::abort() on buffer underflow
  ASSERT_DEATH(EXPECT_NO_ERR(ctx->freeList().free(vptr));, ".*");
}

TEST_F(GPUMemoryManagerTest, BufferOverflowTest) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());
  ctx->freeList().enableRedzones();

  constexpr size_t size = 1024;
  ASSIGN_NO_ERR(void *, vptr,
                ctx->freeList().allocate(size, Device::AllocMode::SYNC));
  ASSERT_NE(vptr, nullptr);

  uint8_t *ptr = static_cast<uint8_t *>(vptr);
  ptr += size + 9;

  uint8_t dead[2] = {0xDE, 0xAD};
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(ptr, dead, sizeof(dead)));
  EXPECT_NO_ERR(ctx->synchronize());

  // Free the pointer which results in a std::abort() on buffer overflow
  ASSERT_DEATH(EXPECT_NO_ERR(ctx->freeList().free(vptr));, ".*");
}

TEST_F(GPUMemoryManagerTest, HostBufferUnderflowTest) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());
  ctx->hostFreeList().enableRedzones();

  ASSIGN_NO_ERR(void *, vptr,
                ctx->hostFreeList().allocate(1024, Device::AllocMode::HOST));
  ASSERT_NE(vptr, nullptr);

  uint8_t *ptr = static_cast<uint8_t *>(vptr);
  ASSERT_EQ(0xBA, *(ptr - 1));

  *(ptr - 10) = 0xDE;
  *(ptr - 9) = 0xAD;

  // Free the pointer which results in a std::abort() on buffer underflow
  ASSERT_DEATH(EXPECT_NO_ERR(ctx->hostFreeList().free(vptr)), ".*");
}

TEST_F(GPUMemoryManagerTest, HostBufferOverflowTest) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());
  ctx->hostFreeList().enableRedzones();

  constexpr size_t size = 1024;
  ASSIGN_NO_ERR(void *, vptr,
                ctx->hostFreeList().allocate(size, Device::AllocMode::HOST));
  ASSERT_NE(vptr, nullptr);

  uint8_t *ptr = static_cast<uint8_t *>(vptr);
  ASSERT_EQ(0xBA, *(ptr - 1));

  *(ptr + size + 9) = 0xDE;
  *(ptr + size + 10) = 0xAD;

  // Free the pointer which results in a std::abort() on buffer overflow
  ASSERT_DEATH(EXPECT_NO_ERR(ctx->hostFreeList().free(vptr)), ".*");
}

} // namespace
