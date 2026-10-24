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

#include "../lib/CPU/Common.h"
#include "Fixture.h"
#include "HAL/M_driver_api.h"

#include "llvm/ADT/ScopeExit.h"

#include <dlpack/dlpack.h>

#include <array>
#include <cstdint>

// CPU-plugin-specific cases beyond the shared ones in PluginTest.cpp.

TEST_F(DriverPluginTest, CustomStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNIMPLEMENTED,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Unimplemented");
}

TEST_F(DriverPluginTest, ApiIsCpu) { ASSERT_EQ(api, "cpu"); }

TEST_F(DriverPluginTest, DlpackDevice) {
  M_driver_device_t device = nullptr;
  ASSERT_SUCCESS(device_get(driver, 0, &device));

  for (const char *name : {M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE,
                           M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED}) {
    M_driver_dlpack_device dl{};
    ASSERT_SUCCESS(device_property(device, name, &dl));
    EXPECT_EQ(dl.device_type, kDLCPU);
    EXPECT_EQ(dl.device_id, 0);
  }
}

TEST_F(DriverPluginDeviceTest, HostUnifiedPropertyIsTrue) {
  for (M_driver_device_t device : devices) {
    int64_t unified = 0;
    ASSERT_SUCCESS(device_property(
        device, M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED, &unified));
    EXPECT_EQ(unified, 1);
  }
}

TEST_F(DriverPluginDeviceTest, NameMatchesLegacyCpuDevice) {
  for (M_driver_device_t device : devices) {
    char name[M_DRIVER_DEVICE_PROPERTY_MAX_LEN] = {};
    ASSERT_SUCCESS(
        device_property(device, M_DRIVER_DEVICE_PROPERTY_NAME, name));
    EXPECT_STREQ(name, "cpu");
  }
}

TEST_F(DriverPluginContextTest, CurrentDriverContextIsNull) {
  // The host tracks no driver-level context: set_current is a no-op and the
  // active handle always reads back null.
  for (M_driver_context_t context : contexts) {
    ASSERT_SUCCESS(context_set_current(context));
    M_driver_native_handle_t active = 1;
    ASSERT_SUCCESS(context_get_current_driver_context(context, &active));
    EXPECT_EQ(active, 0);
  }
}

TEST_F(DriverPluginQueueTest, QueueIsStream) {
  for (M_driver_queue_t queue : queues) {
    bool isStream = false;
    ASSERT_SUCCESS(queue_is_stream(queue, &isStream));
    EXPECT_TRUE(isStream);
  }
}

// Regression test for filling at a destination offset that is not aligned to
// value_size. The fill contract only requires dst->size to be a multiple of
// value_size, not that dst->offset is aligned to it, so the CPU plugin must
// handle a misaligned destination. This is CPU-specific: GPU backends' fills
// (e.g. cuMemsetD32) require aligned device pointers, so an unaligned offset is
// only meaningful here.
TEST_F(DriverPluginQueueTest, FillUnalignedOffset) {
  constexpr size_t size = 64;
  constexpr uint64_t fillBytes = 16; // A multiple of every value_size used.
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, memory); });

    struct Case {
      uint64_t offset; // Deliberately not aligned to value_size.
      uint64_t value;
      uint64_t valueSize;
    };
    const Case cases[] = {
        {/*offset=*/1, /*value=*/0xABCDu, /*valueSize=*/2},
        {/*offset=*/2, /*value=*/0xDEADBEEFu, /*valueSize=*/4},
        {/*offset=*/3, /*value=*/0x0123456789ABCDEFull, /*valueSize=*/8},
    };

    for (const Case &c : cases) {
      M_driver_memory_view fillView{memory, c.offset, fillBytes};
      ASSERT_SUCCESS(queue_fill(queue, &fillView, c.value, c.valueSize));
      ASSERT_SUCCESS(queue_synchronize(queue));

      std::array<uint8_t, fillBytes> result;
      M_driver_memory_view srcView{memory, c.offset, fillBytes};
      ASSERT_SUCCESS(queue_copy_from_device(queue, result.data(), &srcView));
      ASSERT_SUCCESS(queue_synchronize(queue));

      // The region must hold the value's low `valueSize` bytes (little-endian)
      // repeated, byte for byte.
      for (uint64_t b = 0; b < fillBytes; ++b) {
        auto expected =
            static_cast<uint8_t>(c.value >> (8 * (b % c.valueSize)));
        ASSERT_EQ(result[b], expected)
            << "value_size=" << c.valueSize << " offset=" << c.offset
            << " byte=" << b;
      }
    }
  }
}
