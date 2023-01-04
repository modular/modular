//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/AlignedAlloc.h"

#include "gtest/gtest.h"

TEST(AlignedAlloc, uniquePtrAlignedShouldUseAlignedFreeDeleter) {
  auto expectedDeleter = &M::alignedFree;
  auto size = 32;
  auto actual =
      M::makeAlignedUniquePtr<int>(M::kPreferredMemoryAlignment, size);
  EXPECT_EQ(expectedDeleter, actual.get_deleter());
}
