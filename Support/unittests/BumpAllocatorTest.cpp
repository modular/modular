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
//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Support/BumpAllocator.h"

#include "gtest/gtest.h"
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace {

struct alignas(32) OverAligned {
  int value;
};

struct CopyTracked {
  CopyTracked(int value, int &copyCount)
      : value(value), copyCount(&copyCount) {}
  CopyTracked(const CopyTracked &other)
      : value(other.value), copyCount(other.copyCount) {
    ++*copyCount;
  }

  int value;
  int *copyCount;
};

struct MoveOnlyTracked {
  MoveOnlyTracked(int value, int &moveCount)
      : value(value), moveCount(&moveCount) {}
  MoveOnlyTracked(const MoveOnlyTracked &) = delete;
  MoveOnlyTracked(MoveOnlyTracked &&other)
      : value(other.value), moveCount(other.moveCount) {
    ++*moveCount;
  }

  int value;
  int *moveCount;
};

static_assert(!std::is_trivially_copyable_v<CopyTracked>);
static_assert(std::is_trivially_destructible_v<CopyTracked>);
static_assert(!std::is_copy_constructible_v<MoveOnlyTracked>);
static_assert(std::is_trivially_destructible_v<MoveOnlyTracked>);

TEST(BumpAllocator, allocationsHonorAlignment) {
  M::BumpAllocator allocator;

  auto *integer = allocator.allocate<int>();
  auto *overAligned = allocator.allocate<OverAligned>();

  EXPECT_EQ(reinterpret_cast<uintptr_t>(integer) % alignof(int), 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(overAligned) % alignof(OverAligned),
            0u);
  EXPECT_EQ(allocator.getNumSlabs(), 1u);
}

TEST(BumpAllocator, createsObjectsAndInternsValues) {
  M::BumpAllocator allocator;

  auto *created = allocator.create<std::pair<int, int>>(3, 5);
  int copyCount = 0;
  CopyTracked value(7, copyCount);
  CopyTracked *interned = allocator.intern(value);
  int moveCount = 0;
  MoveOnlyTracked moveOnly(9, moveCount);
  MoveOnlyTracked *moved = allocator.intern(std::move(moveOnly));
  CopyTracked values[] = {{2, copyCount}, {4, copyCount}, {6, copyCount}};
  llvm::ArrayRef<CopyTracked> internedElements =
      allocator.internElements<CopyTracked>(values);
  const int source[] = {2, 4, 6};
  llvm::ArrayRef<int> internedIntegers = allocator.internElements<int>(source);
  std::string sourceString = "hello";
  llvm::StringRef internedString = allocator.internString(sourceString);
  sourceString[0] = 'j';

  EXPECT_EQ(created->first, 3);
  EXPECT_EQ(created->second, 5);
  EXPECT_EQ(interned->value, 7);
  EXPECT_NE(interned, &value);
  EXPECT_EQ(moved->value, 9);
  EXPECT_NE(moved, &moveOnly);
  EXPECT_EQ(moveCount, 1);
  ASSERT_EQ(internedElements.size(), 3u);
  EXPECT_EQ(internedElements[0].value, 2);
  EXPECT_EQ(internedElements[1].value, 4);
  EXPECT_EQ(internedElements[2].value, 6);
  EXPECT_EQ(copyCount, 4);
  EXPECT_EQ(internedIntegers, llvm::ArrayRef(source));
  EXPECT_NE(internedIntegers.data(), source);
  EXPECT_EQ(internedString, "hello");
  EXPECT_NE(internedString.data(), sourceString.data());
}

TEST(BumpAllocator, largeAllocationsUseDedicatedSlabs) {
  M::BumpAllocator allocator;

  void *small = allocator.allocate(16, alignof(std::max_align_t));
  void *large =
      allocator.allocate(M::BumpAllocator::sizeThreshold + 1, alignof(char));

  EXPECT_EQ(allocator.getNumSlabs(), 2u);
  EXPECT_GE(allocator.getTotalMemory(),
            M::BumpAllocator::slabSize + M::BumpAllocator::sizeThreshold + 1);
  EXPECT_GE(allocator.identifyKnownObject(small), 0);
  EXPECT_LT(allocator.identifyKnownObject(large), 0);
}

TEST(BumpAllocator, resetRetainsOnlyFirstRegularSlab) {
  M::BumpAllocator allocator;

  allocator.allocate(16, alignof(std::max_align_t));
  allocator.allocate(M::BumpAllocator::sizeThreshold + 1, alignof(char));
  allocator.reset();

  EXPECT_EQ(allocator.getNumSlabs(), 1u);
  EXPECT_EQ(allocator.getTotalMemory(), M::BumpAllocator::slabSize);
  EXPECT_NE(allocator.allocate(0, alignof(char)), nullptr);
}

TEST(BumpAllocator, moveTransfersOwnership) {
  M::BumpAllocator source;
  void *allocation = source.allocate(16, alignof(std::max_align_t));
  int64_t identifier = source.identifyKnownObject(allocation);

  M::BumpAllocator destination(std::move(source));

  EXPECT_EQ(source.getNumSlabs(), 0u);
  EXPECT_EQ(destination.identifyKnownObject(allocation), identifier);
}

TEST(BumpAllocator, checkpointRollsBackWithinCurrentSlab) {
  M::BumpAllocator allocator;
  allocator.allocate(16, alignof(std::max_align_t));

  void *discarded;
  {
    auto checkpoint = allocator.checkpoint();
    discarded = allocator.allocate(32, alignof(std::max_align_t));
  }

  EXPECT_EQ(allocator.allocate(32, alignof(std::max_align_t)), discarded);
}

TEST(BumpAllocator, checkpointRetainsAndReusesLaterSlabs) {
  M::BumpAllocator allocator;
  allocator.allocate(3000, alignof(std::max_align_t));

  {
    auto checkpoint = allocator.checkpoint();
    allocator.allocate(2000, alignof(std::max_align_t));
    EXPECT_EQ(allocator.getNumSlabs(), 2u);
  }

  EXPECT_EQ(allocator.getNumSlabs(), 2u);
  allocator.allocate(2000, alignof(std::max_align_t));
  EXPECT_EQ(allocator.getNumSlabs(), 2u);
}

TEST(BumpAllocator, checkpointRetainsAndReusesDedicatedSlabs) {
  M::BumpAllocator allocator;
  void *discarded;

  {
    auto checkpoint = allocator.checkpoint();
    discarded =
        allocator.allocate(M::BumpAllocator::sizeThreshold + 1, alignof(char));
  }

  EXPECT_EQ(allocator.getNumSlabs(), 1u);
  EXPECT_EQ(
      allocator.allocate(M::BumpAllocator::sizeThreshold + 1, alignof(char)),
      discarded);
  EXPECT_EQ(allocator.getNumSlabs(), 1u);
}

} // namespace
