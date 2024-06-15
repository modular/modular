//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Tests for ASSERT_STREAM
//===----------------------------------------------------------------------===//

#include "Support/AssertStream.h"

#include "gtest/gtest.h"

TEST(Assert, Aborts) {
  EXPECT_EXIT(ASSERT_STREAM(false, << "Error message"),
              testing::KilledBySignal(6), "Error message");
}
