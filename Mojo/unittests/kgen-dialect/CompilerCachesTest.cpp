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

#include "Mojo/KGENDialect/CompilerCaches.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace M::KGEN;

namespace {
const void *ptr(uintptr_t i) { return reinterpret_cast<const void *>(i << 4); }
} // namespace

TEST(CompilerCachesTest, GetOrCreateAndRelease) {
  mlir::MLIRContext context;
  EXPECT_EQ(CompilerCaches::get(&context), nullptr);
  {
    std::shared_ptr<CompilerCaches> caches =
        CompilerCaches::getOrCreate(&context);
    EXPECT_EQ(CompilerCaches::get(&context), caches.get());
  }
  EXPECT_EQ(CompilerCaches::get(&context), nullptr);
}

TEST(CompilerCachesTest, ContextsAreIndependent) {
  mlir::MLIRContext context1, context2;
  std::shared_ptr<CompilerCaches> caches1 =
      CompilerCaches::getOrCreate(&context1);
  std::shared_ptr<CompilerCaches> caches2 =
      CompilerCaches::getOrCreate(&context2);
  EXPECT_NE(caches1.get(), caches2.get());
  EXPECT_EQ(CompilerCaches::get(&context1), caches1.get());
  EXPECT_EQ(CompilerCaches::get(&context2), caches2.get());
  caches1.reset();
  EXPECT_EQ(CompilerCaches::get(&context1), nullptr);
  EXPECT_EQ(CompilerCaches::get(&context2), caches2.get());
}

// Several owners on one context share the caches, which stay until the last
// owner releases them.
TEST(CompilerCachesTest, OwnersOnOneContextShare) {
  mlir::MLIRContext context;
  std::shared_ptr<CompilerCaches> first = CompilerCaches::getOrCreate(&context);
  std::shared_ptr<CompilerCaches> second =
      CompilerCaches::getOrCreate(&context);
  EXPECT_EQ(first.get(), second.get());
  first.reset();
  EXPECT_EQ(CompilerCaches::get(&context), second.get());
  second.reset();
  EXPECT_EQ(CompilerCaches::get(&context), nullptr);
}

TEST(CompilerCachesTest, RecreatedAfterRelease) {
  mlir::MLIRContext context;
  CompilerCaches::getOrCreate(&context)->canonical.insert(ptr(1), ptr(2));
  std::shared_ptr<CompilerCaches> caches =
      CompilerCaches::getOrCreate(&context);
  EXPECT_EQ(caches->canonical.lookup(ptr(1)), nullptr);
}

// Threads cache their last lookup, so a change must be visible to them.
TEST(CompilerCachesTest, OtherThreadSeesChanges) {
  mlir::MLIRContext context;
  std::shared_ptr<CompilerCaches> caches =
      CompilerCaches::getOrCreate(&context);
  std::atomic<int> step{0};
  CompilerCaches *seenWhileOwned = nullptr, *seenAfterRelease = caches.get();
  std::thread reader([&] {
    seenWhileOwned = CompilerCaches::get(&context);
    step = 1;
    while (step != 2)
      std::this_thread::yield();
    seenAfterRelease = CompilerCaches::get(&context);
  });
  while (step != 1)
    std::this_thread::yield();
  CompilerCaches *expected = caches.get();
  caches.reset();
  step = 2;
  reader.join();
  EXPECT_EQ(seenWhileOwned, expected);
  EXPECT_EQ(seenAfterRelease, nullptr);
}

// Every key maps to one value, so any hit must be that value whichever thread
// inserted it first.
TEST(CompilerCachesTest, ConcurrentInsertAndLookup) {
  ShardedCache<const void *> cache;
  constexpr unsigned numThreads = 8;
  constexpr uintptr_t keysPerThread = 50000;
  constexpr uintptr_t stride = 10000;
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t] {
      uintptr_t first = 1 + t * stride;
      for (uintptr_t i = first; i < first + keysPerThread; ++i) {
        cache.insert(ptr(i), ptr(i * 3));
        if (cache.lookup(ptr(i)) != ptr(i * 3))
          failed = true;
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();
  EXPECT_FALSE(failed);
  EXPECT_EQ(cache.lookup(ptr(1)), ptr(3));
  EXPECT_EQ(cache.lookup(ptr(uintptr_t(1) << 40)), nullptr);
}
