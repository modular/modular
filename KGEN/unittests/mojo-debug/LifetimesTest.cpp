//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(LifetimesTest, testInlinedUser) {
  /// Ensures that the lifetime for a variable with an inlined last user is
  /// correct.
  StopContext ctx = buildAndLaunch("eager_destruction_inlined_user.mojo");
  SBValue foo = ctx.frame.FindVariable("foo");
  EXPECT_EQ(foo.GetSummary(), std::string("\"42\""));
}

static void assertVarNotAvailable(StopContext &ctx, StringRef varName) {
  SBValue var = ctx.frame.FindVariable(varName.data());
  ASSERT_TRUE(StringRef(var.GetError().GetCString())
                  .contains("variable not available"));
}

TEST(LifetimesTest, testFullEagerDestruction) {
  /// Ensures that if a variable is completely destroyed eagerly, the
  /// lifetime of the value is reflected in DWARF.
  std::optional<StopContext> ctx =
      buildAndLaunch("full_eager_destruction.mojo");
  SBValue text = ctx->frame.FindVariable("text");
  EXPECT_EQ(text.GetSummary(), std::string("\"hello\""));
  assertVarNotAvailable(*ctx, "number");
  assertVarNotAvailable(*ctx, "simd");

  for (size_t i = 0; i < 2; ++i) {
    ctx.emplace(ctx->resume());
    SBValue number = ctx->frame.FindVariable("number");
    EXPECT_EQ((int)number.GetValueAsSigned(), 8);
    assertVarNotAvailable(*ctx, "text");
    assertVarNotAvailable(*ctx, "simd");
  }

  // Nothing is alive coming out of the loop.
  ctx.emplace(ctx->resume());
  assertVarNotAvailable(*ctx, "text");
  assertVarNotAvailable(*ctx, "number");
  assertVarNotAvailable(*ctx, "simd");

  // Nothing is alive in the else-block as it's past the last use of all
  // variables.
  ctx.emplace(ctx->resume());
  assertVarNotAvailable(*ctx, "text");
  assertVarNotAvailable(*ctx, "number");
  assertVarNotAvailable(*ctx, "simd");

  // `text_moved` should be alive when breaking on the call.
  ctx.emplace(ctx->resume());
  SBValue text_moved = ctx->frame.FindVariable("text_moved");
  EXPECT_EQ(text_moved.GetSummary(), std::string("\"hello\""));

  // This breakpoint is inside `take_string`.  `s` should be alive when breaking
  // on the print call.
  ctx.emplace(ctx->resume());
  SBValue s = ctx->frame.FindVariable("s");
  EXPECT_EQ(s.GetSummary(), std::string("\"hello\""));

  // `text_moved` should be dead now.
  // `text_copied` should be alive when breaking on the call.
  ctx.emplace(ctx->resume());
  assertVarNotAvailable(*ctx, "text_moved");
  SBValue text_copied = ctx->frame.FindVariable("text_copied");
  EXPECT_EQ(text_copied.GetSummary(), std::string("\"hello\""));

  // This breakpoint is inside `take_string`. `s` should be alive when breaking
  // on the print call.
  ctx.emplace(ctx->resume());
  s = ctx->frame.FindVariable("s");
  EXPECT_EQ(s.GetSummary(), std::string("\"hello\""));

  // `text_before` should be dead after the move.
  ctx.emplace(ctx->resume());
  assertVarNotAvailable(*ctx, "text_copied");
  assertVarNotAvailable(*ctx, "text_before");
  SBValue text_after = ctx->frame.FindVariable("text_after");
  EXPECT_EQ(text_after.GetSummary(), std::string("\"hello\""));

  // `text_after` should be dead now.
  // `number2` should be alive when breaking on the call.
  ctx.emplace(ctx->resume());
  assertVarNotAvailable(*ctx, "text_after");
  SBValue number2 = ctx->frame.FindVariable("number2");
  EXPECT_EQ(number2.GetValueAsSigned(), 8);
}
