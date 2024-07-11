//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Allocator.h"

#include "gtest/gtest.h"

using namespace M::LLCL;

namespace {

#if defined(HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR)
TEST(UseAfterFreeAllocator, Detects) {
  auto allocator = createUseAfterFreeAllocator();
  int *ptr1 = allocator->allocate<int>();
  int *ptr2 = allocator->allocate<int>();
  *ptr1 = 42;
  *ptr2 = 43;
  allocator->deallocate(ptr1);
  EXPECT_DEATH(*ptr1 = 44, ".*");
  allocator->deallocate(ptr2);
}
#endif

} // namespace
