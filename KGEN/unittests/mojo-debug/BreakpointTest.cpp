//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;
using namespace ::testing::internal;

TEST(BreakpointTest, testBreakpoint) {
  // Test the breakpoint() intrinsic.

  StopContext ctx = buildAndLaunch("breakpoint.mojo");
  SBValue sum = ctx.frame.FindVariable("sum");
  EXPECT_EQ((int)sum.GetValueAsSigned(), 36);
}

TEST(BreakpointTest, testBreakOnRaise) {
  // Test the break-on-raise feature.

  StopContext ctx = buildAndLaunch("raise.mojo");
  ctx.runCommand("mojo break-on-raise");
  ctx.resume();

  EXPECT_EQ(ctx.binary.getSource().findLinesWithText("# raises")[0],
            (int)ctx.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testDontAutomaticallyBreakOnRaise) {
  // Test that we don't break on raise without enabling the feature.

  StopContext ctx = buildAndLaunch("raise.mojo");
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}
