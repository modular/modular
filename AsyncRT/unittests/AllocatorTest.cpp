//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Allocator.h"

#include "Support/AlignedAlloc.h"

#include "gtest/gtest.h"

using namespace M::AsyncRT;

namespace {

#if defined(USE_TCMALLOC)
TEST(AllocatorTest, Use_TCMalloc) {
  auto allocator = createTCMallocAllocator();
  int *ptr1 = allocator->allocate<int>();
  int *ptr2 = allocator->allocate<int>();

  // Expect that the pointers are aligned according to the default/preferred
  // alignment.
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) &
                (M::kPreferredMemoryAlignment - 1),
            0UL)
      << ptr1;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) &
                (M::kPreferredMemoryAlignment - 1),
            0UL)
      << ptr2;

  *ptr1 = 42;
  *ptr2 = 43;
  EXPECT_EQ(*ptr1, 42);
  allocator->deallocate(ptr1);
  allocator->deallocate(ptr2);

  // Use an alignment which is larger than the default.
  constexpr size_t largeAlignment = 512;
  ptr1 = reinterpret_cast<int *>(
      allocator->allocateBytes(sizeof(int), largeAlignment));
  ptr2 = reinterpret_cast<int *>(
      allocator->allocateBytes(sizeof(int), largeAlignment));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) & (largeAlignment - 1), 0UL)
      << ptr1;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) & (largeAlignment - 1), 0UL)
      << ptr2;
  allocator->deallocateBytes(ptr1, sizeof(int));
  allocator->deallocateBytes(ptr2, sizeof(int));
}
#endif // defined(USE_TCMALLOC)

} // namespace
