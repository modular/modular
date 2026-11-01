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
// Concurrent first use of an accelerator, in a test binary of its own.
//
// The driver's device cache is populated lazily, and that first population is
// the window a concurrent consumer races in. That makes this test
// order-sensitive: in a binary where any earlier test has already created an
// accelerator, the singleton is populated before these threads start and the
// race is unreachable. A separate test binary guarantees a fresh process, and
// this is the only test in it, so the window is always live here.
//
// Regression for MetalDevice::get, which published its singleton under a bare
// `if (!instance)` with no lock (CUDA and HIP guard theirs with a mutex): two
// threads entering it together would each build a device and assign to the same
// counted reference, and the loser's release freed the device the winner was
// about to return. Before the fix this binary died -- SIGABRT or SIGTRAP, no
// test result line -- in roughly half of its runs.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/CAPI/DriverCAPI.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

TEST(DriverCApiConcurrency, ConcurrentAcceleratorCreationIsSafe) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  constexpr int kThreads = 16;
  constexpr int kRounds = 32;

  // Start every thread from a spin so they all enter the lazy creation path
  // together, rather than waking up one at a time.
  std::atomic<bool> go{false};
  std::atomic<int> failures{0};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&go, &failures] {
      while (!go.load(std::memory_order_acquire)) {
      }
      for (int round = 0; round < kRounds; ++round) {
        M_DriverDevice *device = nullptr;
        if (M_DriverAcceleratorDeviceCreate(0, &device) != M_DRIVER_OK ||
            device == nullptr) {
          ++failures;
          return;
        }
        // The handle has to be usable the moment it is returned: a device the
        // driver freed underneath the caller would fault here, not return
        // false.
        uint64_t id = ~uint64_t(0);
        if (M_DriverDeviceId(device, &id) != M_DRIVER_OK || id != 0)
          ++failures;
        M_DriverDeviceDestroy(device);
      }
    });
  }

  go.store(true, std::memory_order_release);
  for (std::thread &thread : threads)
    thread.join();
  EXPECT_EQ(failures.load(), 0);
}
