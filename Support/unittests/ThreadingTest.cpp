//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Support/Threading/ThreadAffinity.h"

#include "Support/ErrorOr.h"
#include "Support/Threading/SignalAltStack.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "llvm/ADT/DenseMap.h"
#include "gtest/gtest.h"
#include <thread>

using namespace M;

TEST(Threading, haveThreadAffinity) {
#if defined(__linux__)
  // On linux, we expect haveThreadAffinity() to return True.
  // This may not be the case on every linux system in the world (presumably,
  // it's possible to build the kernel without affinity support), but all linux
  // systems of interest at this time should support this. If this test fails,
  // it could be a recurrence of the bug documented in GEX-3070.
  EXPECT_TRUE(haveThreadAffinity()) << "linux system without thread affinity?";
#endif

#if defined(__APPLE__)
  // On macos, we don't have/use thread affinity.
  EXPECT_FALSE(haveThreadAffinity());
#endif
}

TEST(Threading, signalAltStack) {
  if (!signalAltStackEnabled())
    GTEST_SKIP() << "No altstack in this build";

  // A freshly spawned thread has no alternate stack of its own. The test main
  // thread does, from LLVM's crash handler registration.
  bool preinstalled = false;
  std::thread([&]() {
    preinstalled = hasSignalAltStack();
    if (preinstalled)
      return;

    {
      ScopedSignalAltStack altStack;
      EXPECT_TRUE(hasSignalAltStack());

      // A nested guard must not disturb the outer one when it goes away.
      {
        ScopedSignalAltStack nested;
        EXPECT_TRUE(hasSignalAltStack());
      }
      EXPECT_TRUE(hasSignalAltStack());
    }
    EXPECT_FALSE(hasSignalAltStack());
  }).join();

  if (preinstalled)
    GTEST_SKIP() << "threads on this configuration already have one";
}

TEST(Threading, threadLocalCacheCopiesOriginalByDefault) {
  llvm::DenseMap<int, int> original = {{1, 10}};
  ThreadLocalCache<llvm::DenseMap<int, int>> caches(original,
                                                    /*maxNumThreads=*/1);

  llvm::DenseMap<int, int> &threadCache = caches.getThreadLocalCache();
  EXPECT_EQ(threadCache.lookup(1), 10);
  threadCache[2] = 20;
  EXPECT_FALSE(original.contains(2));
  EXPECT_EQ(&caches.getThreadLocalCache(), &threadCache);
}

namespace {
/// A cache whose thread caches record the cache they were made from. A copy
/// would not record it.
struct FactoryCache {
  static FactoryCache makeThreadCache(const FactoryCache &original) {
    return FactoryCache{&original};
  }
  const FactoryCache *madeFrom = nullptr;
};
} // namespace

static_assert(HasThreadCacheFactory<FactoryCache>);
static_assert(!HasThreadCacheFactory<llvm::DenseMap<int, int>>);

TEST(Threading, threadLocalCacheUsesMakeThreadCache) {
  FactoryCache original;
  ThreadLocalCache<FactoryCache> caches(original, /*maxNumThreads=*/1);

  FactoryCache &threadCache = caches.getThreadLocalCache();
  EXPECT_EQ(threadCache.madeFrom, &original);
  EXPECT_EQ(&caches.getThreadLocalCache(), &threadCache);
}
