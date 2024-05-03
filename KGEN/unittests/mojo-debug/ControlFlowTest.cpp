//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(ControlFlowTest, testSteppingIntoInlinedNoDebugInfo) {
  // Checks line info for inlined function with no debuginfo.
  std::optional<StopContext> ctx =
      buildAndLaunch("step_into_inlined_no_debug_info.mojo");
  ctx.emplace(ctx->stepInto());

  int expectedLine = ctx->binary.getSource().findLinesWithText(
      "# expected after step-into")[0];
  EXPECT_EQ((int)ctx->frame.GetLineEntry().GetLine(), expectedLine);
}

TEST(ControlFlowTest, testStepStraightLine) {
  // Checks stepping straight line code.
  std::optional<StopContext> ctx = buildAndLaunch("step_straight_line.mojo");
  int functionHeaderLine =
      ctx->binary.getSource().findLinesWithText("fn main()")[0];

  int line = ctx->frame.GetLineEntry().GetLine();
  int prevLine = line;
  while (line != functionHeaderLine) {
    ASSERT_GE(line, prevLine);

    ctx.emplace(ctx->stepOver());
    prevLine = line;
    line = ctx->frame.GetLineEntry().GetLine();
  }
}

static void assertSI64(StopContext &ctx, StringRef name, int64_t expected) {
  SBValue var = ctx.frame.FindVariable(name.data());
  EXPECT_EQ(var.GetValue(), std::to_string(expected));
  EXPECT_EQ(var.GetTypeName(), std::string("si64"));
  EXPECT_EQ(var.GetDisplayTypeName(), std::string("si64"));
  EXPECT_TRUE(var.GetType().GetTypeFlags() | lldb::eTypeIsInteger);
  EXPECT_EQ(var.GetValueAsSigned(expected - 1), expected);
}

TEST(ControlFlowTest, testAssignment) {
  // Make sure basic var mutation assignment is tracked.
  std::optional<StopContext> ctx =
      buildAndLaunch("var_mutation_assignment.mojo");

  assertSI64(*ctx, "i", 5);
  assertSI64(*ctx, "j", 7);

  ctx.emplace(ctx->resume());
  assertSI64(*ctx, "i", 15);
  assertSI64(*ctx, "j", 7);

  ctx.emplace(ctx->resume());
  assertSI64(*ctx, "i", 15);
  assertSI64(*ctx, "j", 13);

  ctx.emplace(ctx->resume());
  assertSI64(*ctx, "i", 2);
  assertSI64(*ctx, "j", 13);
}

// TODO(https://linear.app/modularml/issue/MOTO-429/)
TEST(DISABLED_ControlFlowTest, testIteration) {
  // Make sure changes to basic loop index variable is tracked.
  std::optional<StopContext> ctx =
      buildAndLaunch("var_mutation_iteration.mojo");

  assertSI64(*ctx, "i", 0);
  ctx.emplace(ctx->resume());
  assertSI64(*ctx, "i", 1);
  ctx.emplace(ctx->resume());
  assertSI64(*ctx, "i", 2);
}
