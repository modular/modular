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
// OOM reporting when the driver backdoor is ENABLED, which is the default
// configuration (MEMORY_MANAGER_ONLY is unset here; common.bazelrc turns it on
// for every other test, so this target switches it back off).
//
// With the backdoor on, a manager miss does not surface an error at all -- the
// allocation falls through to the device driver, and only if the driver also
// refuses does the caller see one. What that error should say is the manager's
// own reason: a backdoor that had succeeded would not be reporting anything,
// so the useful part is why the manager missed, with the driver's refusal
// carried as a note.
//
// Two shapes are covered:
//
//   1. A request past the manager's chunk size. The manager rejects it without
//      consulting the arena, so the field-list message applies and the note is
//      appended to it. Cheap and deterministic.
//   2. A request the manager does consult the arena for, where the device has
//      been exhausted out of band. The manager's miss is classified and the
//      note is appended to that. Needs the whole device, so it is a separate,
//      resource-heavy target sharing this file.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;

namespace {

constexpr size_t kMiB = size_t(1024) * 1024;
constexpr size_t kGiB = size_t(1024) * kMiB;

class MemoryManagerOomBackdoorTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available, skipping OOM tests";
#if MODULAR_MACOS
    GTEST_SKIP() << "VMM not supported on macOS, skipping OOM tests";
#else
    if (!(*device)->vmmSupported())
      GTEST_SKIP() << "device does not support VMM, skipping OOM tests";
#endif
  }
};

// Shape 1: no device can serve this, and it is past the manager's chunk size,
// so the manager rejects it on size alone. The surfaced error must still be the
// manager's, with the driver's refusal noted rather than replacing it.
TEST_F(MemoryManagerOomBackdoorTest, DriverRefusalIsNotedOnTheManagersReason) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  const size_t absurd = size_t(4096) * kGiB; // 4 TiB, beyond any device

  ErrorOr<DeviceBufferRef> buf = ctx->enqueueCreateBuffer(absurd);
  ASSERT_TRUE(buf.isError()) << "expected the request to fail on both paths";

  // The manager's own reason leads.
  EXPECT_THAT(buf.getError(),
              ::testing::HasSubstr("Memory manager cannot satisfy allocation"));
  EXPECT_THAT(buf.getError(), ::testing::HasSubstr(
                                  "request exceeds memory manager chunk size"));
  // The backdoor attempt is recorded, carrying the driver's own text.
  EXPECT_THAT(buf.getError(),
              ::testing::HasSubstr("direct device allocation also failed"));
  EXPECT_THAT(buf.getError(), ::testing::HasSubstr("out of memory"));
}

// Shape 2: the manager consults the arena and its backing step is refused
// because the device was exhausted out of band, so the miss is classified;
// the backdoor then fails too and is noted alongside. This is the combination
// the flag exists to make legible.
TEST_F(MemoryManagerOomBackdoorTest, ClassifiedMissCarriesTheBackdoorNote) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  Device &device = ctx->device();

  ASSIGN_NO_ERR(Device::MemInfo, mem, device.getMemoryInfo());
  if (mem.free < (mem.total * 8) / 10)
    GTEST_SKIP() << "GPU is not near-empty (free=" << mem.free
                 << ", total=" << mem.total
                 << "); another process is resident, skipping to avoid a "
                    "spurious failure";

  // Well inside the manager's chunk size, so the arena is actually consulted
  // and the miss gets classified rather than rejected on size.
  const size_t request = 64 * kMiB;
  const size_t margin = 8 * kMiB; // less than the request

  auto freeNow = [&device]() -> size_t {
    ErrorOr<Device::MemInfo> m = device.getMemoryInfo();
    return m.isError() ? 0 : m->free;
  };

  // Consume the device out of band, stepping the size down as the driver starts
  // refusing so the tail is actually reached.
  std::vector<void *> sinks;
  for (size_t chunk = 4 * kGiB; chunk >= 2 * kMiB; chunk /= 2) {
    while (freeNow() > margin) {
      void *p = nullptr;
      if (device.chunkAlloc(&p, chunk, /*isHost=*/false).isError())
        break;
      sinks.push_back(p);
    }
  }

  const size_t freeAfterConsume = freeNow();
  if (freeAfterConsume >= request) {
    for (void *p : sinks)
      EXPECT_NO_ERR(device.chunkFree(p, /*isHost=*/false));
    GTEST_SKIP() << "could not drive device free memory below the request "
                    "(free="
                 << freeAfterConsume << ", request=" << request
                 << "); cannot set up the condition";
  }

  ErrorOr<DeviceBufferRef> blocked = ctx->enqueueCreateBuffer(request);
  EXPECT_TRUE(blocked.isError())
      << "expected both the manager and the driver to fail with the device "
         "exhausted out of band";

  if (blocked.isError()) {
    // Classified, not a bare driver error.
    EXPECT_THAT(blocked.getError(),
                ::testing::HasSubstr("Memory manager cannot satisfy "
                                     "allocation"));
    EXPECT_THAT(blocked.getError(), ::testing::HasSubstr("(unaccounted)"));
    EXPECT_THAT(blocked.getError(), ::testing::HasSubstr("vmm_oom"));
    // ... with the backdoor attempt noted alongside it.
    EXPECT_THAT(blocked.getError(),
                ::testing::HasSubstr("direct device allocation also failed"));
  } else {
    blocked.takeValue().reset();
  }

  for (void *p : sinks)
    EXPECT_NO_ERR(device.chunkFree(p, /*isHost=*/false));
  EXPECT_NO_ERR(ctx->synchronize());
}

} // namespace
