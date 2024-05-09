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
