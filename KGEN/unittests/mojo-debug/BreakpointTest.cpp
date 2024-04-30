//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(BreakpointTest, testBreakpoint) {
  // Test the breakpoint() intrinsic.

  StopContext ctx = buildAndLaunch("breakpoint.mojo");
  SBValue sum = ctx.frame.FindVariable("sum");
  EXPECT_EQ((int)sum.GetValueAsSigned(), 36);
}
