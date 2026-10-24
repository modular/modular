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

#include "Fixture.h"
#include <cstdint>
#include <cstring>
#include <vector>

#include "llvm/ADT/ScopeExit.h"

TEST_F(DriverPluginTest, StandardStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_SUCCESS, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_GT(strlen(errorMessage), 0);

  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_INCOMPATIBLE_VERSION,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Incompatible version");

  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNKNOWN_ERROR,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_GT(strlen(errorMessage), 0);
}

TEST_F(DriverPluginTest, BadStatusCode) {
  char errorMessage[1024];
  M_driver_status_t kPositiveStatusCode = 10; // Arbitrarily chosen
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(driver, kPositiveStatusCode, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(driver, -M_DRIVER_NUM_STANDARD_STATUS_CODES,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(driver, M_DRIVER_STATUS_CUSTOM_CODES_BASE + 1,
                                 errorMessage, sizeof(errorMessage)));
  // Conservative number that we expect no plugin to reach or exceed
  int64_t constexpr assumedNumCustomStatusCodes = 50000;
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(driver,
                                 M_DRIVER_STATUS_CUSTOM_CODES_BASE -
                                     assumedNumCustomStatusCodes,
                                 errorMessage, sizeof(errorMessage)));
}

TEST_F(DriverPluginTest, DriverPropertyVersion) {
  // Good version
  M_driver_version version = {
      .major = 0xFFFFFFFFu, .minor = 0xFFFFFFFFu, .patch = 0xFFFFFFFFu};
  ASSERT_EQ(
      M_DRIVER_STATUS_SUCCESS,
      this->property(driver, M_DRIVER_PROPERTY_VERSION, (void *)&version));
  EXPECT_EQ(M_DRIVER_INTERFACE_VERSION_MAJOR, version.major);
  EXPECT_EQ(M_DRIVER_INTERFACE_VERSION_MINOR, version.minor);
  EXPECT_EQ(M_DRIVER_INTERFACE_VERSION_PATCH, version.patch);
  std::cerr << "Version: " << version.major << "." << version.minor << "."
            << version.patch << "\n";

  // Bad version
  version.major = 0xFFFFFFFFu;
  version.minor = 0xFFFFFFFFu;
  version.patch = 0xFFFFFFFFu;
  M_driver_driver_t noDriver = nullptr;
  EXPECT_EQ(M_DRIVER_STATUS_INCOMPATIBLE_VERSION,
            this->create(&version, &noDriver));
  EXPECT_EQ(noDriver, nullptr);
}

TEST_F(DriverPluginTest, DriverPropertyAPI) {
  const char *api = nullptr;
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->property(driver, M_DRIVER_PROPERTY_API, &api));
  ASSERT_NE(nullptr, api);
  std::cerr << api << "\n";
}

TEST_F(DriverPluginTest, DeviceCount) {
  int64_t deviceCount = 0;
  M_driver_status_t status = this->device_count(driver, &deviceCount);
  EXPECT_EQ(status, M_DRIVER_STATUS_SUCCESS);
  cerr << "Device count: " << deviceCount << endl;
  EXPECT_GT(deviceCount, 0);
}

TEST_F(DriverPluginTest, DeviceGet) {
  int64_t deviceCount = 0;
  RUN(device_count(driver, &deviceCount));
  for (int64_t i = 0; i < deviceCount; i++) {
    M_driver_device_t device;
    M_driver_status_t status = this->device_get(driver, i, &device);
    EXPECT_EQ(status, M_DRIVER_STATUS_SUCCESS);
    cerr << "Got device 0x" << std::hex << device << endl;
  }
}

TEST_F(DriverPluginTest, DriverCreateNullOut) {
  M_driver_version version = {.major = M_DRIVER_INTERFACE_VERSION_MAJOR,
                              .minor = M_DRIVER_INTERFACE_VERSION_MINOR,
                              .patch = M_DRIVER_INTERFACE_VERSION_PATCH};
  EXPECT_EQ(this->create(&version, nullptr), M_DRIVER_STATUS_INVALID_ARG);

  M_driver_driver_t noDriver = nullptr;
  EXPECT_EQ(this->create(nullptr, &noDriver), M_DRIVER_STATUS_INVALID_ARG);
  EXPECT_EQ(noDriver, nullptr);
}

TEST_F(DriverPluginTest, DeviceCountNullOut) {
  EXPECT_EQ(device_count(driver, nullptr), M_DRIVER_STATUS_INVALID_ARG);
}

TEST_F(DriverPluginTest, DeviceGetNullOut) {
  int64_t deviceCount = 0;
  ASSERT_SUCCESS(device_count(driver, &deviceCount));
  ASSERT_GT(deviceCount, 0);
  EXPECT_EQ(device_get(driver, 0, nullptr), M_DRIVER_STATUS_INVALID_ARG);
}

TEST_F(DriverPluginTest, ContextCreateNullOut) {
  int64_t deviceCount = 0;
  ASSERT_SUCCESS(device_count(driver, &deviceCount));
  ASSERT_GT(deviceCount, 0);
  M_driver_device_t device = nullptr;
  ASSERT_SUCCESS(device_get(driver, 0, &device));

  EXPECT_EQ(context_create(device, nullptr), M_DRIVER_STATUS_INVALID_ARG);

  M_driver_context_t context = nullptr;
  EXPECT_EQ(context_create(nullptr, &context), M_DRIVER_STATUS_BAD_DEVICE);
  EXPECT_EQ(context, nullptr);
}

TEST_F(DriverPluginTest, DeviceContext) {
  int64_t deviceCount = 0;
  EXPECT_SUCCESS(device_count(driver, &deviceCount));
  ASSERT_GT(deviceCount, 0);
  deviceCount = testDeviceCount(deviceCount);
  std::vector<M_driver_context_t> contexts;
  for (int64_t i = 0; i < deviceCount; i++) {
    M_driver_device_t device = nullptr;
    RUN(device_get(driver, i, &device));
    M_driver_context_t context = nullptr;
    DO_TEST(context_create(device, &context));
    cerr << "Created context for device 0x" << std::hex << device << ": 0x"
         << context << endl;
    contexts.push_back(context);
  }
  for (int64_t i = deviceCount - 1; i >= 0; i--) {
    DO_TEST(context_destroy(contexts[i]));
    cerr << "Destroyed context 0x" << contexts[i] << endl;
  }
}

TEST_F(DriverPluginDeviceTest, PropertyArch) {
  if (api == "cpu")
    GTEST_SKIP() << "\"arch\" device property not supported on " << api;
  for (M_driver_device_t device : devices) {
    char arch[M_DRIVER_DEVICE_PROPERTY_MAX_LEN] = {};
    ASSERT_SUCCESS(
        device_property(device, M_DRIVER_DEVICE_PROPERTY_ARCH, &arch));
    ASSERT_NE(0, std::strlen(arch));
    std::cout << arch << "\n";
  }
}

TEST_F(DriverPluginContextTest, ContextMemoryInfo) {
  for (M_driver_context_t context : contexts) {
    uint64_t freeBytes = 0;
    uint64_t totalBytes = 0;
    ASSERT_SUCCESS(context_memory_info(context, &freeBytes, &totalBytes));
    EXPECT_GT(totalBytes, 0u);
    EXPECT_LE(freeBytes, totalBytes);
  }
}

TEST_F(DriverPluginContextTest, ContextMemoryInfoBadArgs) {
  // A null context must be rejected. Plugins without an implementation may
  // report their unimplemented status instead; either way it is not SUCCESS.
  uint64_t freeBytes = 0;
  uint64_t totalBytes = 0;
  EXPECT_NE(M_DRIVER_STATUS_SUCCESS,
            context_memory_info(nullptr, &freeBytes, &totalBytes));
  for (M_driver_context_t context : contexts) {
    EXPECT_NE(M_DRIVER_STATUS_SUCCESS,
              context_memory_info(context, nullptr, nullptr));
  }
}

TEST_F(DriverPluginContextTest, CurrentContextRoundTrip) {
  if (api == "Metal")
    GTEST_SKIP() << "current-context state is not implemented for Metal";
  for (M_driver_context_t context : contexts) {
    // Save whatever driver-level context is active (may be null), make this
    // context current, then restore the saved state.
    M_driver_native_handle_t saved = 0;
    ASSERT_SUCCESS(context_get_current_driver_context(context, &saved));
    ASSERT_SUCCESS(context_set_current(context));
    M_driver_native_handle_t active = 0;
    ASSERT_SUCCESS(context_get_current_driver_context(context, &active));
    ASSERT_SUCCESS(context_set_current_driver_context(context, saved));
  }
}

TEST_F(DriverPluginContextTest, CurrentContextBadArgs) {
  EXPECT_NE(M_DRIVER_STATUS_SUCCESS, context_set_current(nullptr));
  M_driver_native_handle_t driverContext = 0;
  EXPECT_NE(M_DRIVER_STATUS_SUCCESS,
            context_get_current_driver_context(nullptr, &driverContext));
  EXPECT_NE(M_DRIVER_STATUS_SUCCESS,
            context_set_current_driver_context(nullptr, 0));
}

TEST_F(DriverPluginContextTest, MemoryAllocSync) {
  for (M_driver_context_t context : contexts) {
    M_driver_memory_t memory;
    ASSERT_SUCCESS(memory_alloc_sync(context, 128, &memory));
    ASSERT_SUCCESS(memory_free_sync(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocPinned) {
  for (M_driver_context_t context : contexts) {
    M_driver_memory_t memory;
    ASSERT_SUCCESS(memory_alloc_pinned(context, 128, &memory));
    ASSERT_SUCCESS(memory_free_pinned(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocReturnsNonNull) {
  for (M_driver_context_t context : contexts) {
    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, 128, &memory));
    EXPECT_NE(memory, nullptr);
    ASSERT_SUCCESS(memory_free_sync(context, memory));

    memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_pinned(context, 128, &memory));
    EXPECT_NE(memory, nullptr);
    ASSERT_SUCCESS(memory_free_pinned(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocZeroSize) {
  for (M_driver_context_t context : contexts) {
    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_pinned(context, 0, &memory));
    EXPECT_EQ(memory, nullptr);

    memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, 0, &memory));
    EXPECT_EQ(memory, nullptr);
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocBadContext) {
  EXPECT_EQ(memory_alloc_sync(nullptr, 128, nullptr),
            M_DRIVER_STATUS_BAD_CONTEXT);
  EXPECT_EQ(memory_alloc_pinned(nullptr, 128, nullptr),
            M_DRIVER_STATUS_BAD_CONTEXT);
}

TEST_F(DriverPluginContextTest, MemoryAllocNullOut) {
  for (M_driver_context_t context : contexts) {
    EXPECT_EQ(memory_alloc_sync(context, 128, nullptr),
              M_DRIVER_STATUS_INVALID_ARG);
    EXPECT_EQ(memory_alloc_pinned(context, 128, nullptr),
              M_DRIVER_STATUS_INVALID_ARG);
  }
}

TEST_F(DriverPluginContextTest, MemoryFreeNull) {
  for (M_driver_context_t context : contexts) {
    ASSERT_SUCCESS(memory_free_pinned(context, nullptr));
    ASSERT_SUCCESS(memory_free_sync(context, nullptr));
  }
}

TEST_F(DriverPluginContextTest, MemoryFreeBadContext) {
  EXPECT_EQ(memory_free_pinned(nullptr, nullptr), M_DRIVER_STATUS_BAD_CONTEXT);
  EXPECT_EQ(memory_free_sync(nullptr, nullptr), M_DRIVER_STATUS_BAD_CONTEXT);
}

TEST_F(DriverPluginContextTest, QueueCreateDestroy) {
  for (M_driver_context_t context : contexts) {
    M_driver_queue_t queue = nullptr;
    ASSERT_SUCCESS(queue_create(context, &queue));
    ASSERT_NE(nullptr, queue);
    ASSERT_SUCCESS(queue_destroy(context, queue));
  }
}

TEST_F(DriverPluginContextTest, QueueCreateNullOut) {
  // A null `queue` out-param must be rejected before any real queue
  // allocation happens, not null-deref after doing the work.
  for (M_driver_context_t context : contexts) {
    EXPECT_EQ(queue_create(context, nullptr), M_DRIVER_STATUS_INVALID_ARG);
  }
}

TEST_F(DriverPluginContextTest, QueueCopySync) {
  constexpr size_t size = 128;
  std::array<uint8_t, size> src;
  for (size_t i = 0; i < size; i++)
    src[i] = static_cast<uint8_t>(i);

  for (M_driver_context_t context : contexts) {
    M_driver_queue_t queue = nullptr;
    ASSERT_SUCCESS(queue_create(context, &queue));

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    ASSERT_NE(memory, nullptr);

    M_driver_memory_view toView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView, src.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> dst;
    dst.fill(0);
    M_driver_memory_view fromView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, dst.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    EXPECT_EQ(src, dst);

    ASSERT_SUCCESS(memory_free_sync(context, memory));
    ASSERT_SUCCESS(queue_destroy(context, queue));
  }
}

TEST_F(DriverPluginContextTest, QueueAsyncAllocSync) {
  if (api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;

  constexpr size_t size = 128;
  std::array<uint8_t, size> src;
  for (size_t i = 0; i < size; i++)
    src[i] = static_cast<uint8_t>(i);

  for (M_driver_context_t context : contexts) {
    M_driver_queue_t queue = nullptr;
    ASSERT_SUCCESS(queue_create(context, &queue));

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_async(queue, size, &memory));
    ASSERT_NE(memory, nullptr);
    ASSERT_SUCCESS(queue_synchronize(queue));

    M_driver_memory_view toView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView, src.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> dst;
    dst.fill(0);
    M_driver_memory_view fromView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, dst.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    EXPECT_EQ(src, dst);

    ASSERT_SUCCESS(memory_free_async(queue, memory));
    ASSERT_SUCCESS(queue_synchronize(queue));
    ASSERT_SUCCESS(queue_destroy(context, queue));
  }
}

TEST_F(DriverPluginContextTest, EventCreateNullOut) {
  for (M_driver_context_t context : contexts) {
    EXPECT_EQ(event_create(context, M_DRIVER_EVENT_FLAG_NONE, nullptr),
              M_DRIVER_STATUS_INVALID_ARG);
  }
}

TEST_F(DriverPluginContextTest, EventCreateDestroy) {
  for (M_driver_context_t ctx : contexts) {
    M_driver_event_t event = nullptr;
    ASSERT_SUCCESS(event_create(ctx, M_DRIVER_EVENT_FLAG_NONE, &event));
    ASSERT_NE(event, nullptr);
    EXPECT_SUCCESS(event_destroy(ctx, event));
  }
}

TEST_F(DriverPluginContextTest, EventRecordWait) {
  constexpr size_t numElems = 256;
  constexpr size_t byteSize = numElems * sizeof(unsigned int);
  std::array<unsigned int, numElems> src;
  for (unsigned int i = 0; i < numElems; i++)
    src[i] = i;

  for (M_driver_context_t ctx : contexts) {
    M_driver_queue_t queueA = nullptr;
    M_driver_queue_t queueB = nullptr;
    ASSERT_SUCCESS(queue_create(ctx, &queueA));
    ASSERT_SUCCESS(queue_create(ctx, &queueB));

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(ctx, byteSize, &memory));

    M_driver_memory_view toView{memory, /*offset=*/0, byteSize};
    ASSERT_SUCCESS(queue_copy_to_device(queueA, &toView, src.data()));
    M_driver_event_t event = nullptr;
    ASSERT_SUCCESS(event_create(ctx, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
    ASSERT_SUCCESS(queue_record_event(queueA, event));

    ASSERT_SUCCESS(queue_wait_for_events(queueB, &event, 1));
    std::array<unsigned int, numElems> dst;
    dst.fill(0);
    M_driver_memory_view fromView{memory, /*offset=*/0, byteSize};
    ASSERT_SUCCESS(queue_copy_from_device(queueB, dst.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queueB));

    EXPECT_EQ(src, dst);

    bool isReady = false;
    EXPECT_SUCCESS(is_event_ready(ctx, event, &isReady));
    EXPECT_TRUE(isReady);

    EXPECT_SUCCESS(event_destroy(ctx, event));
    EXPECT_SUCCESS(memory_free_sync(ctx, memory));
    EXPECT_SUCCESS(queue_destroy(ctx, queueB));
    EXPECT_SUCCESS(queue_destroy(ctx, queueA));
  }
}

TEST_F(DriverPluginQueueTest, CopyToDeviceThenFromDevice) {
  constexpr size_t size = 256;
  std::array<uint8_t, size> content;
  for (size_t i = 0; i < content.size(); i++)
    content[i] = i;

  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, memory); });

    M_driver_memory_view dstView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_to_device(queue, /*dst=*/&dstView,
                                        /*src=*/content.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> result;
    result.fill(0);
    M_driver_memory_view srcView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, /*dst=*/result.data(),
                                          /*src=*/&srcView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    ASSERT_EQ(content, result);
  }
}

TEST_F(DriverPluginQueueTest, SetMemory) {
  constexpr size_t size = 128;
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, memory); });

    M_driver_memory_view setView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_set_memory(queue, /*dst=*/&setView, uint8_t(42)));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> result;
    M_driver_memory_view srcView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, /*dst=*/result.data(),
                                          /*src=*/&srcView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> expected;
    expected.fill(42);

    ASSERT_EQ(expected, result);
  }
}

TEST_F(DriverPluginQueueTest, Fill) {
  constexpr size_t size = 128;
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, memory); });

    // 16-bit pattern.
    {
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(
          queue_fill(queue, &fillView, uint16_t(0xABCD), sizeof(uint16_t)));
      ASSERT_SUCCESS(queue_synchronize(queue));

      std::array<uint16_t, size / sizeof(uint16_t)> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(queue_copy_from_device(queue, result.data(), &srcView));
      ASSERT_SUCCESS(queue_synchronize(queue));
      for (uint16_t v : result)
        ASSERT_EQ(v, uint16_t(0xABCD));
    }

    // 32-bit pattern.
    {
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(
          queue_fill(queue, &fillView, uint32_t(0xDEADBEEF), sizeof(uint32_t)));
      ASSERT_SUCCESS(queue_synchronize(queue));

      std::array<uint32_t, size / sizeof(uint32_t)> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(queue_copy_from_device(queue, result.data(), &srcView));
      ASSERT_SUCCESS(queue_synchronize(queue));
      for (uint32_t v : result)
        ASSERT_EQ(v, uint32_t(0xDEADBEEF));
    }

    // 64-bit pattern with distinct halves (exercises the non-broadcast path).
    {
      constexpr uint64_t pattern = 0x0123456789ABCDEFull;
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(queue_fill(queue, &fillView, pattern, sizeof(uint64_t)));
      ASSERT_SUCCESS(queue_synchronize(queue));

      std::array<uint64_t, size / sizeof(uint64_t)> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(queue_copy_from_device(queue, result.data(), &srcView));
      ASSERT_SUCCESS(queue_synchronize(queue));
      for (uint64_t v : result)
        ASSERT_EQ(v, pattern);
    }

    // value_size == 1 routes through the single-byte set_memory path.
    {
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(
          queue_fill(queue, &fillView, uint8_t(0x5A), sizeof(uint8_t)));
      ASSERT_SUCCESS(queue_synchronize(queue));

      std::array<uint8_t, size> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(queue_copy_from_device(queue, result.data(), &srcView));
      ASSERT_SUCCESS(queue_synchronize(queue));
      std::array<uint8_t, size> expected;
      expected.fill(0x5A);
      ASSERT_EQ(expected, result);
    }

    // An unsupported value_size is still rejected.
    {
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_EQ(M_DRIVER_STATUS_INVALID_ARG,
                queue_fill(queue, &fillView, uint64_t(0), /*value_size=*/3));
    }
  }
}

TEST_F(DriverPluginQueueTest, CopyIntraDevice) {
  constexpr size_t size = 256;
  std::array<uint8_t, size> content;
  for (size_t i = 0; i < size; i++)
    content[i] = static_cast<uint8_t>(i);

  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t srcMem = nullptr;
    M_driver_memory_t dstMem = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &srcMem));
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &dstMem));
    auto defer = llvm::scope_exit([&]() {
      memory_free_sync(context, dstMem);
      memory_free_sync(context, srcMem);
    });

    // Sync between each pair so the test is correct regardless of whether
    // the plugin's raw queue is a stream.
    M_driver_memory_view toView{srcMem, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView, content.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    M_driver_memory_view intraDstView{dstMem, /*offset=*/0, size};
    M_driver_memory_view intraSrcView{srcMem, /*offset=*/0, size};
    ASSERT_SUCCESS(
        queue_copy_intra_device(queue, &intraDstView, &intraSrcView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> result;
    result.fill(0);
    M_driver_memory_view fromView{dstMem, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, result.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    ASSERT_EQ(content, result);
  }
}

TEST_F(DriverPluginContextTest, ContextCopySync) {
  constexpr size_t size = 256;
  std::array<uint8_t, size> content;
  for (size_t i = 0; i < size; i++)
    content[i] = static_cast<uint8_t>(i);

  for (M_driver_context_t context : contexts) {
    M_driver_memory_t srcMem = nullptr;
    M_driver_memory_t dstMem = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &srcMem));
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &dstMem));
    auto defer = llvm::scope_exit([&]() {
      memory_free_sync(context, dstMem);
      memory_free_sync(context, srcMem);
    });

    M_driver_memory_view toView{srcMem, /*offset=*/0, size};
    ASSERT_SUCCESS(copy_to_device_sync(context, &toView, content.data()));

    M_driver_memory_view intraDstView{dstMem, /*offset=*/0, size};
    M_driver_memory_view intraSrcView{srcMem, /*offset=*/0, size};
    ASSERT_SUCCESS(
        copy_intra_device_sync(context, &intraDstView, &intraSrcView));

    std::array<uint8_t, size> result;
    result.fill(0);
    M_driver_memory_view fromView{dstMem, /*offset=*/0, size};
    ASSERT_SUCCESS(copy_from_device_sync(context, result.data(), &fromView));

    ASSERT_EQ(content, result);
  }
}

TEST_F(DriverPluginContextTest, ContextFillSync) {
  constexpr size_t size = 128;
  for (M_driver_context_t context : contexts) {
    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, memory); });

    // set_memory_sync: single-byte fill. Blocking, so no synchronize needed.
    {
      M_driver_memory_view setView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(set_memory_sync(context, &setView, uint8_t(42)));

      std::array<uint8_t, size> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(copy_from_device_sync(context, result.data(), &srcView));
      std::array<uint8_t, size> expected;
      expected.fill(42);
      ASSERT_EQ(expected, result);
    }

    // 16-bit pattern.
    {
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(
          fill_sync(context, &fillView, uint16_t(0xABCD), sizeof(uint16_t)));

      std::array<uint16_t, size / sizeof(uint16_t)> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(copy_from_device_sync(context, result.data(), &srcView));
      for (uint16_t v : result)
        ASSERT_EQ(v, uint16_t(0xABCD));
    }

    // 32-bit pattern.
    {
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(fill_sync(context, &fillView, uint32_t(0xDEADBEEF),
                               sizeof(uint32_t)));

      std::array<uint32_t, size / sizeof(uint32_t)> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(copy_from_device_sync(context, result.data(), &srcView));
      for (uint32_t v : result)
        ASSERT_EQ(v, uint32_t(0xDEADBEEF));
    }

    // 64-bit pattern with distinct halves (exercises the non-broadcast path).
    {
      constexpr uint64_t pattern = 0x0123456789ABCDEFull;
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(fill_sync(context, &fillView, pattern, sizeof(uint64_t)));

      std::array<uint64_t, size / sizeof(uint64_t)> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(copy_from_device_sync(context, result.data(), &srcView));
      for (uint64_t v : result)
        ASSERT_EQ(v, pattern);
    }

    // value_size == 1 routes through the single-byte set_memory_sync path.
    {
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(
          fill_sync(context, &fillView, uint8_t(0x5A), sizeof(uint8_t)));

      std::array<uint8_t, size> result;
      M_driver_memory_view srcView{memory, /*offset=*/0, size};
      ASSERT_SUCCESS(copy_from_device_sync(context, result.data(), &srcView));
      std::array<uint8_t, size> expected;
      expected.fill(0x5A);
      ASSERT_EQ(expected, result);
    }

    // An unsupported value_size is still rejected.
    {
      M_driver_memory_view fillView{memory, /*offset=*/0, size};
      ASSERT_EQ(M_DRIVER_STATUS_INVALID_ARG,
                fill_sync(context, &fillView, uint64_t(0), /*value_size=*/3));
    }
  }
}

TEST_F(DriverPluginQueueTest, CopyInterDevice) {
  if (contexts.size() < 2 || queues.size() < 2)
    GTEST_SKIP() << "peer copy requires at least two devices";

  constexpr size_t length = 256;
  constexpr size_t bufSize = length * sizeof(uint8_t);

  std::vector<uint8_t> inHost(length);
  std::vector<uint8_t> outHost(length, 0);
  for (size_t i = 0; i < length; i++)
    inHost[i] = static_cast<uint8_t>(i & 0xFF);

  // src lives on device 0, dst on device 1 (the queue's device).
  M_driver_memory_t srcDev = nullptr;
  M_driver_memory_t dstDev = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bufSize, &srcDev));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[1], bufSize, &dstDev));

  // Stage the payload onto device 0 and wait for it: the peer copy does not
  // order itself against the producer queue.
  M_driver_memory_view srcView{srcDev, /*offset=*/0, bufSize};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &srcView, inHost.data()));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));

  // Peer-copy device 0 -> device 1 on device 1's queue; contexts[0] is the
  // source (peer) context. EXPECT (not ASSERT) so the frees below still run.
  M_driver_memory_view dstView{dstDev, /*offset=*/0, bufSize};
  EXPECT_SUCCESS(
      queue_copy_inter_device(queues[1], &dstView, &srcView, contexts[0]));

  // Read device 1 back on the same queue (ordered after the peer copy) and
  // verify the payload arrived intact.
  EXPECT_SUCCESS(queue_copy_from_device(queues[1], outHost.data(), &dstView));
  EXPECT_SUCCESS(queue_synchronize(queues[1]));

  for (size_t i = 0; i < length; i++)
    EXPECT_EQ(outHost[i], static_cast<uint8_t>(i & 0xFF));

  memory_free_sync(contexts[1], dstDev);
  memory_free_sync(contexts[0], srcDev);
}

TEST_F(DriverPluginContextTest, ContextCopyInterDeviceSync) {
  if (contexts.size() < 2)
    GTEST_SKIP() << "peer copy requires at least two devices";

  constexpr size_t length = 256;
  constexpr size_t bufSize = length * sizeof(uint8_t);

  std::vector<uint8_t> inHost(length);
  std::vector<uint8_t> outHost(length, 0);
  for (size_t i = 0; i < length; i++)
    inHost[i] = static_cast<uint8_t>((i * 7) & 0xFF);

  M_driver_memory_t srcDev = nullptr;
  M_driver_memory_t dstDev = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bufSize, &srcDev));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[1], bufSize, &dstDev));

  M_driver_memory_view srcView{srcDev, /*offset=*/0, bufSize};
  ASSERT_SUCCESS(copy_to_device_sync(contexts[0], &srcView, inHost.data()));

  M_driver_memory_view dstView{dstDev, /*offset=*/0, bufSize};
  EXPECT_SUCCESS(
      copy_inter_device_sync(contexts[1], &dstView, &srcView, contexts[0]));

  EXPECT_SUCCESS(copy_from_device_sync(contexts[1], outHost.data(), &dstView));

  for (size_t i = 0; i < length; i++)
    EXPECT_EQ(outHost[i], static_cast<uint8_t>((i * 7) & 0xFF));

  memory_free_sync(contexts[1], dstDev);
  memory_free_sync(contexts[0], srcDev);
}

namespace {
// Destroys the shared contexts once every test has run.
struct SharedPluginSessionEnvironment : ::testing::Environment {
  void TearDown() override { SharedPluginSession::drop(); }
};
[[maybe_unused]] ::testing::Environment *const sharedPluginSessionEnvironment =
    ::testing::AddGlobalTestEnvironment(new SharedPluginSessionEnvironment);
} // namespace

static void bumpHostFlag(void *userData) {
  *static_cast<uint64_t *>(userData) = 42;
}

TEST_F(DriverPluginQueueTest, QueueNativeHandle) {
  if (api != "CUDA" && api != "HIP")
    GTEST_SKIP() << "native stream handle requires a CUDA or HIP driver";
  for (M_driver_queue_t queue : queues) {
    uint64_t handle = 0;
    ASSERT_SUCCESS(
        queue_property(queue, M_DRIVER_QUEUE_PROPERTY_NATIVE_HANDLE, &handle));
    // The backing native stream is never null.
    EXPECT_NE(handle, 0u);
  }
}

TEST_F(DriverPluginQueueTest, QueueLaunchHostFunc) {
  if (api != "CUDA" && api != "HIP")
    GTEST_SKIP() << "host-function callbacks require a CUDA or HIP driver";
  uint64_t flag = 0;
  ASSERT_SUCCESS(queue_launch_host_func(queues[0], bumpHostFlag, &flag));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  EXPECT_EQ(flag, 42u);
}

TEST_F(DriverPluginQueueTest, QueueWaitValue64) {
  if (api != "CUDA" && api != "HIP")
    GTEST_SKIP() << "stream wait-value requires a CUDA or HIP driver";
  // Pinned host memory is device-visible; pre-set the value so the equality
  // wait is already satisfied when the queue reaches it.
  M_driver_memory_t mem = nullptr;
  ASSERT_SUCCESS(memory_alloc_pinned(contexts[0], sizeof(uint64_t), &mem));
  void *addr = nullptr;
  ASSERT_SUCCESS(memory_property(mem, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &addr));
  *static_cast<uint64_t *>(addr) = 42;
  ASSERT_SUCCESS(
      queue_wait_value64(queues[0], reinterpret_cast<uint64_t>(addr), 42));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));
  ASSERT_SUCCESS(memory_free_pinned(contexts[0], mem));
}

TEST_F(DriverPluginContextTest, MemoryWrapUnwrapErrors) {
  for (M_driver_context_t context : contexts) {
    // A live allocation gives us a valid address and handle for the argument
    // checks below.
    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, 128, &memory));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, memory); });

    uint64_t address = 0;
    ASSERT_SUCCESS(
        memory_property(memory, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));
    ASSERT_NE(address, 0u);

    M_driver_memory_t wrapped = nullptr;
    EXPECT_EQ(memory_wrap(nullptr, address, 128, /*owning=*/false, &wrapped),
              M_DRIVER_STATUS_BAD_CONTEXT);
    EXPECT_EQ(memory_wrap(context, address, 128, /*owning=*/false, nullptr),
              M_DRIVER_STATUS_INVALID_ARG);
    // A zero address has no valid region to wrap.
    EXPECT_EQ(
        memory_wrap(context, /*address=*/0, 128, /*owning=*/false, &wrapped),
        M_DRIVER_STATUS_INVALID_ARG);

    uint64_t out = 0;
    EXPECT_EQ(memory_unwrap(nullptr, memory, &out),
              M_DRIVER_STATUS_BAD_CONTEXT);
    EXPECT_EQ(memory_unwrap(context, nullptr, &out),
              M_DRIVER_STATUS_BAD_MEMORY);
    EXPECT_EQ(memory_unwrap(context, memory, nullptr),
              M_DRIVER_STATUS_INVALID_ARG);
  }
}

TEST_F(DriverPluginContextTest, MemoryWrapNonOwningAliasesAllocation) {
  constexpr size_t size = 256;
  std::array<uint8_t, size> content;
  for (size_t i = 0; i < size; i++)
    content[i] = static_cast<uint8_t>(i);

  for (M_driver_context_t context : contexts) {
    M_driver_memory_t orig = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &orig));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, orig); });

    uint64_t address = 0;
    ASSERT_SUCCESS(
        memory_property(orig, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));

    M_driver_memory_view origView{orig, /*offset=*/0, size};
    ASSERT_SUCCESS(copy_to_device_sync(context, &origView, content.data()));

    // A non-owning wrapper of the same address must alias the same region.
    M_driver_memory_t wrapped = nullptr;
    ASSERT_SUCCESS(
        memory_wrap(context, address, size, /*owning=*/false, &wrapped));
    ASSERT_NE(wrapped, nullptr);

    uint64_t wrappedAddress = 0;
    ASSERT_SUCCESS(memory_property(wrapped, M_DRIVER_MEMORY_PROPERTY_ADDRESS,
                                   &wrappedAddress));
    EXPECT_EQ(wrappedAddress, address);

    std::array<uint8_t, size> viaWrapped;
    viaWrapped.fill(0);
    M_driver_memory_view wrappedView{wrapped, /*offset=*/0, size};
    ASSERT_SUCCESS(
        copy_from_device_sync(context, viaWrapped.data(), &wrappedView));
    EXPECT_EQ(content, viaWrapped);

    // Freeing the non-owning wrapper releases only the handle, not the region.
    ASSERT_SUCCESS(memory_free_sync(context, wrapped));

    // The original allocation is therefore still alive and readable.
    std::array<uint8_t, size> viaOrig;
    viaOrig.fill(0);
    ASSERT_SUCCESS(copy_from_device_sync(context, viaOrig.data(), &origView));
    EXPECT_EQ(content, viaOrig);
  }
}

TEST_F(DriverPluginContextTest, MemoryUnwrapReleasesOwnershipRoundTrip) {
  constexpr size_t size = 256;
  std::array<uint8_t, size> content;
  for (size_t i = 0; i < size; i++)
    content[i] = static_cast<uint8_t>((i * 3 + 1) & 0xFF);

  for (M_driver_context_t context : contexts) {
    M_driver_memory_t orig = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &orig));

    uint64_t address = 0;
    ASSERT_SUCCESS(
        memory_property(orig, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));

    M_driver_memory_view origView{orig, /*offset=*/0, size};
    ASSERT_SUCCESS(copy_to_device_sync(context, &origView, content.data()));

    // Release ownership: unwrap returns the region's address and leaves `orig`
    // non-owning, so freeing `orig` will no longer free the region.
    uint64_t unwrapped = 0;
    ASSERT_SUCCESS(memory_unwrap(context, orig, &unwrapped));
    EXPECT_EQ(unwrapped, address);

    // Re-adopt the released address as an owning handle. This round-trip is the
    // documented owning-wrap use: the address came from this plugin's own
    // allocator, so freeing the handle frees the region through the device
    // path.
    M_driver_memory_t rewrapped = nullptr;
    ASSERT_SUCCESS(
        memory_wrap(context, unwrapped, size, /*owning=*/true, &rewrapped));
    ASSERT_NE(rewrapped, nullptr);

    // Data survives the ownership hand-off.
    std::array<uint8_t, size> result;
    result.fill(0);
    M_driver_memory_view rewrappedView{rewrapped, /*offset=*/0, size};
    ASSERT_SUCCESS(
        copy_from_device_sync(context, result.data(), &rewrappedView));
    EXPECT_EQ(content, result);

    // Freeing the now non-owning original releases only bookkeeping; the region
    // stays alive under the re-wrapped owning handle.
    ASSERT_SUCCESS(memory_free_sync(context, orig));
    result.fill(0);
    ASSERT_SUCCESS(
        copy_from_device_sync(context, result.data(), &rewrappedView));
    EXPECT_EQ(content, result);

    // The owning handle frees the region.
    ASSERT_SUCCESS(memory_free_sync(context, rewrapped));
  }
}

TEST_F(DriverPluginContextTest, MemoryUnwrapIsIdempotent) {
  constexpr size_t size = 128;
  for (M_driver_context_t context : contexts) {
    M_driver_memory_t orig = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &orig));

    uint64_t address = 0;
    ASSERT_SUCCESS(
        memory_property(orig, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &address));

    uint64_t first = 0;
    ASSERT_SUCCESS(memory_unwrap(context, orig, &first));
    EXPECT_EQ(first, address);

    // Unwrapping an already non-owning handle just returns the address again;
    // the handle stays valid.
    uint64_t second = 0;
    ASSERT_SUCCESS(memory_unwrap(context, orig, &second));
    EXPECT_EQ(second, address);

    // `orig` is non-owning after unwrap, so it can no longer free the region.
    // Reclaim ownership through a fresh owning wrap so the region is freed
    // exactly once.
    M_driver_memory_t owner = nullptr;
    ASSERT_SUCCESS(
        memory_wrap(context, address, size, /*owning=*/true, &owner));
    ASSERT_SUCCESS(memory_free_sync(context, orig));  // bookkeeping only
    ASSERT_SUCCESS(memory_free_sync(context, owner)); // frees the region
  }
}

namespace {
// Peer access is an optional extension, outside the fixture's core symbols.
struct PeerAccessSymbols {
  M_driver_device_can_access_peer_func_ptr_t device_can_access_peer;
  M_driver_context_enable_peer_access_func_ptr_t context_enable_peer_access;
};

PeerAccessSymbols lookupPeerAccess(void *soHandle) {
  return {(M_driver_device_can_access_peer_func_ptr_t)dlsym(
              soHandle, "M_driver_device_can_access_peer"),
          (M_driver_context_enable_peer_access_func_ptr_t)dlsym(
              soHandle, "M_driver_context_enable_peer_access")};
}
} // namespace

TEST_F(DriverPluginContextTest, PeerAccessCapabilityIsSymmetricAndSelfFalse) {
  auto [device_can_access_peer, context_enable_peer_access] =
      lookupPeerAccess(soHandle);
  if (device_can_access_peer == nullptr)
    GTEST_SKIP() << "plugin does not export peer access";
  if (devices.size() < 2)
    GTEST_SKIP() << "peer access requires at least two devices";

  bool selfAccess = true;
  ASSERT_SUCCESS(device_can_access_peer(devices[0], devices[0], &selfAccess));
  EXPECT_FALSE(selfAccess);

  bool zeroToOne = false, oneToZero = true;
  ASSERT_SUCCESS(device_can_access_peer(devices[0], devices[1], &zeroToOne));
  ASSERT_SUCCESS(device_can_access_peer(devices[1], devices[0], &oneToZero));
  EXPECT_EQ(zeroToOne, oneToZero);
}

TEST_F(DriverPluginContextTest, PeerAccessEnableIsIdempotent) {
  auto [device_can_access_peer, context_enable_peer_access] =
      lookupPeerAccess(soHandle);
  if (context_enable_peer_access == nullptr)
    GTEST_SKIP() << "plugin does not export peer access";
  if (contexts.size() < 2)
    GTEST_SKIP() << "peer access requires at least two devices";
  bool canAccess = false;
  ASSERT_SUCCESS(device_can_access_peer(devices[0], devices[1], &canAccess));
  if (!canAccess)
    GTEST_SKIP() << "hardware pair does not support peer access";

  EXPECT_SUCCESS(context_enable_peer_access(contexts[0], contexts[1]));
  EXPECT_SUCCESS(context_enable_peer_access(contexts[0], contexts[1]));
  EXPECT_SUCCESS(context_enable_peer_access(contexts[1], contexts[0]));
}

TEST_F(DriverPluginContextTest, PeerAccessSelfPairRejected) {
  auto [device_can_access_peer, context_enable_peer_access] =
      lookupPeerAccess(soHandle);
  if (context_enable_peer_access == nullptr)
    GTEST_SKIP() << "plugin does not export peer access";
  EXPECT_NE(context_enable_peer_access(contexts[0], contexts[0]),
            M_DRIVER_STATUS_SUCCESS);
}

TEST_F(DriverPluginContextTest, PeerAccessNullArgs) {
  auto [device_can_access_peer, context_enable_peer_access] =
      lookupPeerAccess(soHandle);
  if (device_can_access_peer == nullptr)
    GTEST_SKIP() << "plugin does not export peer access";
  bool unused = false;
  EXPECT_EQ(device_can_access_peer(nullptr, devices[0], &unused),
            M_DRIVER_STATUS_BAD_DEVICE);
  EXPECT_EQ(device_can_access_peer(devices[0], devices[0], nullptr),
            M_DRIVER_STATUS_INVALID_ARG);
  EXPECT_EQ(context_enable_peer_access(nullptr, contexts[0]),
            M_DRIVER_STATUS_BAD_CONTEXT);
}
