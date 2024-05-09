//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

llvm::StringRef getFunctionName(std::optional<StopContext> &ctx) {
  return ctx->thread.GetSelectedFrame().GetFunctionName();
}

TEST(InlinedCallsiteTest, testBreakingOnInlinedCalsite) {
  // Tests that setting breakpoints on callsites that were inlined works.

  std::optional<StopContext> ctx = buildAndLaunch("inlined_callsite.mojo");
  std::vector<int> expectedBreakingLines =
      ctx->binary.getSource().findLinesWithText("# breakpoint");

  EXPECT_TRUE(getFunctionName(ctx).contains("main"));
  EXPECT_EQ((int)ctx->frame.GetLineEntry().GetLine(), expectedBreakingLines[1]);

  ctx.emplace(ctx->resume());
  EXPECT_TRUE(getFunctionName(ctx).contains("callee_regular"));
  EXPECT_EQ((int)ctx->frame.GetLineEntry().GetLine(), expectedBreakingLines[0]);

  ctx.emplace(ctx->resume());
  EXPECT_TRUE(getFunctionName(ctx).contains("main"));
  EXPECT_EQ((int)ctx->frame.GetLineEntry().GetLine(), expectedBreakingLines[2]);
}
