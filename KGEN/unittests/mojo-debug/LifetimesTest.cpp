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
  EXPECT_STREQ(foo.GetSummary(), R"("42")");
}

static void assertVarNotAvailable(StopContext &ctx, StringRef varName) {
  SBValue var = ctx.frame.FindVariable(varName.data());
  ASSERT_TRUE(StringRef(var.GetError().GetCString())
                  .contains("variable not available"));
}

TEST(LifetimesTest, testFullEagerDestruction) {
  /// Ensures that if a variable is completely destroyed eagerly, the
  /// lifetime of the value is reflected in DWARF.
  StopContext ctx = buildAndLaunch("full_eager_destruction.mojo");
  SBValue text = ctx.frame.FindVariable("text");
  EXPECT_STREQ(text.GetSummary(), R"("hello")");
  assertVarNotAvailable(ctx, "number");
  assertVarNotAvailable(ctx, "simd");

  for (size_t i = 0; i < 2; ++i) {
    ctx.resume();
    SBValue number = ctx.frame.FindVariable("number");
    EXPECT_EQ((int)number.GetValueAsSigned(), 8);
    assertVarNotAvailable(ctx, "text");
    assertVarNotAvailable(ctx, "simd");
  }

  // Nothing is alive coming out of the loop.
  ctx.resume();
  assertVarNotAvailable(ctx, "text");
  assertVarNotAvailable(ctx, "number");
  assertVarNotAvailable(ctx, "simd");

  // Nothing is alive in the else-block as it's past the last use of all
  // variables.
  ctx.resume();
  assertVarNotAvailable(ctx, "text");
  assertVarNotAvailable(ctx, "number");
  assertVarNotAvailable(ctx, "simd");

  // `text_moved` should be alive when breaking on the call.
  ctx.resume();
  SBValue textMoved = ctx.frame.FindVariable("text_moved");
  EXPECT_STREQ(textMoved.GetSummary(), R"("hello")");

  // This breakpoint is inside `take_string`.  `s` should be alive when breaking
  // on the print call.
  ctx.resume();
  SBValue s = ctx.frame.FindVariable("s");
  EXPECT_STREQ(s.GetSummary(), R"("hello")");

  // `text_moved` should be dead now.
  // `text_copied` should be alive when breaking on the call.
  ctx.resume();
  assertVarNotAvailable(ctx, "text_moved");
  SBValue textCopied = ctx.frame.FindVariable("text_copied");
  EXPECT_STREQ(textCopied.GetSummary(), R"("hello")");

  // This breakpoint is inside `take_string`. `s` should be alive when breaking
  // on the print call.
  ctx.resume();
  s = ctx.frame.FindVariable("s");
  EXPECT_STREQ(s.GetSummary(), R"("hello")");

  // `text_before` should be dead after the move.
  ctx.resume();

  // TODO: Why? assertVarNotAvailable(ctx, "text_before");
  SBValue textAfter = ctx.frame.FindVariable("text_after");
  EXPECT_STREQ(textAfter.GetSummary(), R"("hello")");

  // `text_after` should be dead now.
  // `number2` should be alive when breaking on the call.
  ctx.resume();
  assertVarNotAvailable(ctx, "text_after");
  SBValue number2 = ctx.frame.FindVariable("number2");
  EXPECT_EQ(number2.GetValueAsSigned(), 8);
}

TEST(LifetimesTest, testResurrection) {
  /// Ensures that if a variable is killed and re-initialized again, it is
  /// visible.
  StopContext ctx = buildAndLaunch("resurrection.mojo");

  SBValue text2 = ctx.frame.FindVariable("text2");
  EXPECT_STREQ(text2.GetSummary(), R"("hello")");
  // TODO: Why? assertVarNotAvailable(ctx, "text1");

  ctx.resume();
  SBValue text1 = ctx.frame.FindVariable("text1");
  EXPECT_STREQ(text1.GetSummary(), R"("hello")");
  assertVarNotAvailable(ctx, "text2");
}

TEST(LifetimesTest, testRedefined) {
  /// Ensures that if a variable is redefined, it is visible.
  StopContext ctx = buildAndLaunch("redefined.mojo");

  SBValue x = ctx.frame.FindVariable("x");
  EXPECT_EQ((int)x.GetValueAsSigned(), 468);

  ctx.resume();
  SBValue y = ctx.frame.FindVariable("y");
  EXPECT_STREQ(y.GetSummary(), R"("world")");
}
