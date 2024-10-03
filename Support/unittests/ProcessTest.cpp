//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Process.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using ::testing::HasSubstr;

TEST(ProcessTest, GetProcessExecutablePathWorks) {
  // The full path will vary based on the test sandbox, so only try to match
  // part of the path that we know will stay consistent.
  EXPECT_THAT(getProcessExecutablePath(),
              HasSubstr("Support/unittests/BaseTest"));
}
