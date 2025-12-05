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

TEST(BreakpointTest, testBreakOnRaiseInlined) {
  // Test the break-on-raise feature in an @always_inline function.

  StopContext ctx = buildAndLaunch("raise_inlined.mojo");
  ctx.runCommand("mojo break-on-raise");
  ctx.resume();

  EXPECT_EQ(ctx.binary.getSource().findLinesWithText("# raises")[0],
            (int)ctx.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testBreakOnRaiseSecondTarget) {
  // Test that we break on raise on a second target as well as the first.
  //
  // Test is intended to be equivalent to the below lldb command sequence:
  //
  // target create raise
  // mojo break-on-raise
  // run  # Expect this to break on the raise
  //
  // target create raise
  // mojo break-on-raise
  // run  # Expect this to break on the raise

  StopContext ctx1 = buildAndLaunch("raise.mojo");
  ctx1.runCommand("mojo break-on-raise");
  ctx1.process.Continue();
  EXPECT_EQ(ctx1.binary.getSource().findLinesWithText("# raises")[0],
            (int)ctx1.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());

  StopContext ctx2 = buildAndLaunch("raise.mojo");
  ctx2.runCommand("mojo break-on-raise");
  ctx2.process.Continue();
  EXPECT_EQ(ctx2.binary.getSource().findLinesWithText("# raises")[0],
            (int)ctx2.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testDontAutomaticallyBreakOnRaise) {
  // Test that we don't break on raise without enabling the feature.

  StopContext ctx = buildAndLaunch("raise.mojo");
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}

TEST(BreakpointTest, testDontAutomaticallyBreakOnRaiseSecondTarget) {
  // Test that we don't break on raise on a second target just because we
  // enabled it for a first.
  //
  // Test is intended to be equivalent to the below lldb command sequence:
  //
  // target create raise
  // mojo break-on-raise
  // target create raise
  // run  # Expect this to not break on the raise
  // target select 0
  // run  # Expect this to break on the raise

  StopContext ctx1 = buildAndLaunch("raise.mojo");
  ctx1.runCommand("mojo break-on-raise enable");

  StopContext ctx2 = buildAndLaunch("raise.mojo");
  ctx2.process.Continue();
  EXPECT_EQ(ctx2.process.GetState(), lldb::eStateExited);

  ctx1.process.Continue();
  EXPECT_EQ(ctx1.binary.getSource().findLinesWithText("# raises")[0],
            (int)ctx1.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
}

TEST(BreakpointTest, testDontBreakOnRaiseIfDisabled) {
  // Test that we don't break on raise if we disabled it.

  StopContext ctx = buildAndLaunch("raise.mojo");
  ctx.runCommand("mojo break-on-raise enable");
  ctx.runCommand("mojo break-on-raise disable");
  ctx.process.Continue();
  EXPECT_EQ(ctx.process.GetState(), lldb::eStateExited);
}

TEST(BreakpointTest, testDontBreakOnRaiseIfDisabledSecondTarget) {
  // Test that we don't break on raise if we disabled it for a single target
  // when we have two.
  //
  // Test is intended to be equivalent to the below lldb command sequence:
  //
  // target create raise
  // mojo break-on-raise
  // target create raise
  // mojo break-on-raise
  // target select 0
  // mojo break-on-raise disable
  // run  # Expect this to not break on the raise
  // target select 1
  // run  # Expect this to break on the raise

  StopContext ctx1 = buildAndLaunch("raise.mojo");
  ctx1.runCommand("mojo break-on-raise enable");

  StopContext ctx2 = buildAndLaunch("raise.mojo");
  ctx2.runCommand("mojo break-on-raise enable");

  ctx1.runCommand("mojo break-on-raise disable");
  ctx1.process.Continue();
  EXPECT_EQ(ctx1.process.GetState(), lldb::eStateExited);

  ctx2.process.Continue();
  EXPECT_EQ(ctx2.binary.getSource().findLinesWithText("# raises")[0],
            (int)ctx2.thread.GetFrameAtIndex(0).GetLineEntry().GetLine());
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
