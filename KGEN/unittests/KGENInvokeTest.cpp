//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/Invoke.h"
#include "gtest/gtest.h"

using namespace M;

static void someKGENKernel(ssize_t, void *, uint8_t, ssize_t, void *, uint8_t) {
}

static float someOtherKernel(int a, ssize_t, void *, ssize_t, float b) {
  return a + b;
}

TEST(KGENInvokeTest, testinvokeKGENFunction) {

  // Test that we can call a function that returns no arguments.
  KGEN::invoke(someKGENKernel, llvm::makeArrayRef<int32_t>(nullptr, 1),
               llvm::makeArrayRef<int32_t>(nullptr, 1));

  // Test that we can call the function with a value.
  auto dummyArray = llvm::makeArrayRef<int32_t>(nullptr, 1);
  KGEN::invoke(someKGENKernel, std::forward<decltype(dummyArray)>(dummyArray),
               std::forward<decltype(dummyArray)>(dummyArray));

  // Test that we can call the function with a value and get the correct result.
  EXPECT_EQ(KGEN::invoke(someOtherKernel, 1,
                         llvm::makeArrayRef<float>(nullptr, 1), 2.0f),
            3.0f);
}
