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

#include "Support/Compiler/Threading.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

#include <atomic>
#include <numeric>
#include <vector>

using namespace mlir;

namespace {

std::vector<int> makeItems(size_t numItems) {
  std::vector<int> items(numItems);
  std::iota(items.begin(), items.end(), 0);
  return items;
}

/// Runs each test with and without multithreading, because the helpers must
/// give the same result in both modes.
class ThreadingTest : public ::testing::TestWithParam<bool> {
protected:
  MLIRContext context{GetParam() ? MLIRContext::Threading::ENABLED
                                 : MLIRContext::Threading::DISABLED};
};

TEST_P(ThreadingTest, ForEachChunkVisitsEachItemOnce) {
  constexpr size_t numItems = 2000;
  std::vector<int> items = makeItems(numItems);
  std::vector<std::atomic<int>> numVisits(numItems);
  std::atomic<size_t> numChunks = 0;
  std::atomic<bool> interleaved = true;
  M::parallelForEachChunk(&context, items, [&](ArrayRef<int> chunk) {
    ++numChunks;
    for (size_t i = 0; i < chunk.size(); ++i) {
      if (static_cast<size_t>(chunk[i]) != chunk[0] + i * M::kMaxParallelChunks)
        interleaved = false;
      ++numVisits[chunk[i]];
    }
  });

  EXPECT_EQ(numChunks, M::kMaxParallelChunks);
  EXPECT_TRUE(interleaved);
  EXPECT_TRUE(llvm::all_of(
      numVisits, [](const std::atomic<int> &count) { return count == 1; }));
}

TEST_P(ThreadingTest, ForEachChunkGivesFewItemsOneChunkEach) {
  SmallVector<int, 8> items = {0, 1, 2, 3, 4};
  std::atomic<size_t> numChunks = 0;
  std::atomic<bool> oneItemEach = true;
  M::parallelForEachChunk(&context, items, [&](ArrayRef<int> chunk) {
    ++numChunks;
    if (chunk.size() != 1)
      oneItemEach = false;
  });

  EXPECT_EQ(numChunks, items.size());
  EXPECT_TRUE(oneItemEach);
}

TEST_P(ThreadingTest, MapChunksReturnsResultsInChunkOrder) {
  constexpr size_t numItems = 2000;
  std::vector<int> items = makeItems(numItems);
  auto chunks = M::parallelMapChunks(&context, items, [](ArrayRef<int> chunk) {
    return std::vector<int>(chunk.begin(), chunk.end());
  });

  ASSERT_EQ(chunks.size(), M::kMaxParallelChunks);
  for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
    std::vector<int> expected;
    for (size_t i = chunk; i < numItems; i += M::kMaxParallelChunks)
      expected.push_back(static_cast<int>(i));
    EXPECT_EQ(chunks[chunk], expected);
  }
}

TEST_P(ThreadingTest, MapReturnsResultsInItemOrder) {
  std::vector<int> items = makeItems(10000);
  auto squares =
      M::parallelMap(&context, items, [](int item) { return item * item; });

  ASSERT_EQ(squares.size(), items.size());
  EXPECT_TRUE(llvm::all_of(
      items, [&](int item) { return squares[item] == item * item; }));
}

TEST_P(ThreadingTest, MapStoresBoolResults) {
  std::vector<int> items = makeItems(10000);
  auto isEven = M::parallelMap(&context, ArrayRef<int>(items),
                               [](int item) { return item % 2 == 0; });

  ASSERT_EQ(isEven.size(), items.size());
  EXPECT_TRUE(llvm::all_of(
      items, [&](int item) { return isEven[item] == (item % 2 == 0); }));
}

TEST_P(ThreadingTest, AnyOfFindsAMatch) {
  std::vector<int> items = makeItems(5000);
  EXPECT_TRUE(
      M::parallelAnyOf(&context, items, [](int item) { return item == 1234; }));
  EXPECT_FALSE(
      M::parallelAnyOf(&context, items, [](int item) { return item < 0; }));
}

TEST_P(ThreadingTest, EmptyRangeCallsNothing) {
  std::vector<int> items;
  bool called = false;
  M::parallelForEachChunk(&context, items,
                          [&](ArrayRef<int>) { called = true; });
  EXPECT_TRUE(M::parallelMapChunks(&context, items, [&](ArrayRef<int>) {
                called = true;
                return 0;
              }).empty());
  EXPECT_TRUE(M::parallelMap(&context, items, [&](int) {
                called = true;
                return 0;
              }).empty());
  EXPECT_FALSE(M::parallelAnyOf(&context, items, [&](int) {
    called = true;
    return true;
  }));
  EXPECT_FALSE(called);
}

INSTANTIATE_TEST_SUITE_P(Multithreading, ThreadingTest, ::testing::Bool());

TEST(SerialThreadingTest, AnyOfStopsAfterAMatch) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  std::vector<int> items = makeItems(100);
  int numCalls = 0;
  EXPECT_TRUE(M::parallelAnyOf(&context, items, [&](int item) {
    ++numCalls;
    return item == 3;
  }));
  EXPECT_EQ(numCalls, 4);
}

} // namespace
