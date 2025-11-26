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

  // FIXME: We're getting a different line number on ARM hosts for some reason.
#if defined(__arm__) || defined(__aarch64__)
  GTEST_SKIP() << "ARM hosts get different line numbers for some reason.";
#endif

  EXPECT_EQ(ctx.binary.getSource().findLinesWithText("# raises")[0],
            (int)ctx.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testDontAutomaticallyBreakOnRaise) {
  // Test that we don't break on raise without enabling the feature.

  StopContext ctx = buildAndLaunch("raise.mojo");
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}

TEST(BreakpointTest, testSymbolBreakpoints) {
  StopContext ctx = buildAndLaunch("symbol_breakpoints.mojo");
  ctx.runCommand("b simple_fn");
  ctx.runCommand("b parametrized_fn");
  ctx.runCommand("b parametrized_method");
  ctx.resume();

  int expectedLine =
      ctx.binary.getSource().findLinesWithText("# simple_fn stop")[0];
  EXPECT_EQ(ctx.frame.GetLineEntry().GetLine(), expectedLine);

  ctx.resume();

  expectedLine =
      ctx.binary.getSource().findLinesWithText("# parametrized_fn stop")[0];
  EXPECT_EQ(ctx.frame.GetLineEntry().GetLine(), expectedLine);

  ctx.resume();

  expectedLine =
      ctx.binary.getSource().findLinesWithText("# parametrized_method stop")[0];
  EXPECT_EQ(ctx.frame.GetLineEntry().GetLine(), expectedLine);
}
