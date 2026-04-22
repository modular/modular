//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(StdlibTypesTest, testList) {
  /// Tests that List can be parsed correctly and its data formatter works
  /// correctly as well.
  StopContext ctx = buildAndLaunch("list.mojo");
  SBValue var = ctx.frame.FindVariable("point_vec");
  EXPECT_STREQ(var.GetSummary(), "(size 3)");
  EXPECT_STREQ(var.GetValueForExpressionPath("[0].x").GetValue(), "1");
  EXPECT_STREQ(var.GetValueForExpressionPath("[1].y").GetValue(), "-2");
  EXPECT_STREQ(var.GetValueForExpressionPath("[2].x").GetValue(), "3");

  ctx.resume();
  var = ctx.frame.FindVariable("int_vec");
  EXPECT_STREQ(var.GetSummary(), "(size 3)[1, 2, 3]");

  ctx.resume();
  var = ctx.frame.FindVariable("int_vec");
  EXPECT_STREQ(var.GetSummary(),
               "(size 103)[1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, ...]");
}

TEST(StdlibTypesTest, testUnsafePointerSummary) {
  StopContext ctx = buildAndLaunch("unsafe_pointer.mojo");

  // p_int = UnsafePointer[Int] pointing to 42
  SBValue p_int = ctx.frame.FindVariable("p_int");
  EXPECT_STREQ(p_int.GetSummary(), "42");

  ctx.resume();

  // p_neg = UnsafePointer[Int] pointing to -5 — exercises correct signed
  // display.
  SBValue p_neg = ctx.frame.FindVariable("p_neg");
  EXPECT_STREQ(p_neg.GetSummary(), "-5");

  ctx.resume();

  // p_bool = UnsafePointer[Bool] pointing to True — exercises Bool summary
  // path.
  SBValue p_bool = ctx.frame.FindVariable("p_bool");
  EXPECT_STREQ(p_bool.GetSummary(), "True");

  ctx.resume();

  // p_float = UnsafePointer[Float64] pointing to 3.125 — exercises scalar
  // path.
  SBValue p_float = ctx.frame.FindVariable("p_float");
  EXPECT_STREQ(p_float.GetSummary(), "3.125");
}
