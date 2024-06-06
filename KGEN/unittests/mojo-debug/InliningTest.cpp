//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

llvm::StringRef getFunctionName(StopContext &ctx) {
  return ctx.thread.GetSelectedFrame().GetFunctionName();
}

TEST(InliningTest, testBreakingOnInlinedCalsite) {
  // Tests that setting breakpoints on callsites that were inlined works.

  StopContext ctx = buildAndLaunch("inlined_callsite.mojo");
  std::vector<int> expectedBreakingLines =
      ctx.binary.getSource().findLinesWithText("# breakpoint");

  EXPECT_TRUE(getFunctionName(ctx).contains("main"));
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedBreakingLines[1]);

  ctx.resume();
  EXPECT_TRUE(getFunctionName(ctx).contains("callee_regular"));
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedBreakingLines[0]);

  ctx.resume();
  EXPECT_TRUE(getFunctionName(ctx).contains("main"));
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedBreakingLines[2]);

  ctx.resume();
  EXPECT_TRUE(getFunctionName(ctx).contains("callee_regular"));
  EXPECT_EQ((int)ctx.frame.GetLineEntry().GetLine(), expectedBreakingLines[0]);
}

TEST(InliningTest, testInlinedVariableCalledFromNoDebug) {
  // Tests that debug functions inlined into no-debug functions are still
  // debuggable after being inlined again into a regular function.

  StopContext ctx = buildAndLaunch("inlined_variable.mojo");

  SBValue number = ctx.frame.FindVariable("nested_var");
  EXPECT_EQ((int)number.GetValueAsSigned(), 2);
}

TEST(InliningTest, testLiftedInlinedInoutArgModification) {
  // Tests that modifications to inlined inout args that are lifted by mem2reg
  // show up.

  StopContext ctx = buildAndLaunch("inlined_argument.mojo");

  SBValue number = ctx.frame.FindVariable("m");
  // Pre-req: Make sure the value was actually lifted by mem2reg.
  // Due to convertDbgValueToDeclare, the value is actually stored back to stack
  // memory for debug builds.
  EXPECT_STRNE(number.GetLocation(), "scalar");
  // Check the value is the updated value from the inlined callee.
  EXPECT_EQ((int)number.GetValueAsSigned(), 42);
}

TEST(InliningTest, testLiftedInlinedInoutArgPartialModification) {
  // Tests that modifications to inlined inout args that are lifted by mem2reg
  // show up when the argument is not the full variable from the caller side.

  StopContext ctx = buildAndLaunch("inlined_partial_argument.mojo");

  SBValue pair = ctx.frame.FindVariable("p");
  // Pre-req: Make sure the value was actually lifted by mem2reg.
  EXPECT_FALSE(pair.GetAddress().IsValid());
  // Check the value is the updated value from the inlined callee.
  ASSERT_EQ((int)pair.GetNumChildren(), 2);
  SBValue firstField = pair.GetChildAtIndex(0);
  EXPECT_EQ((int)firstField.GetValueAsSigned(), 42);
  SBValue secondField = pair.GetChildAtIndex(1);
  EXPECT_EQ((int)secondField.GetValueAsSigned(), 4);
}
