//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "test_kernels.h"
#include "gtest/gtest.h"

TEST(KGENKernelTest, testArrayArgument) {
  int32_t values[4] = {11, 22, 33, 44};
  int32_t result = array_index(values);
  EXPECT_EQ(result, values[2]);
}
