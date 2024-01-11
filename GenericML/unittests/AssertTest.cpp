//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Tests for GENERICML_ASSERT
//===----------------------------------------------------------------------===//

#include "GenericML/Support/Assert.h"

#include "gtest/gtest.h"

TEST(Assert, Aborts) {
  EXPECT_EXIT(GENERICML_ASSERT(false) << "Error message",
              testing::KilledBySignal(6), "Error message");
}
