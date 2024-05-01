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
  std::optional<StopContext> ctx = buildAndLaunch("list.mojo");
  SBValue var = ctx->frame.FindVariable("point_vec");
  EXPECT_EQ(var.GetSummary(), std::string("(size 3)"));
  EXPECT_EQ(var.GetValueForExpressionPath("[0].x").GetValue(),
            std::string("1"));
  EXPECT_EQ(var.GetValueForExpressionPath("[1].y").GetValue(),
            std::string("-2"));
  EXPECT_EQ(var.GetValueForExpressionPath("[2].x").GetValue(),
            std::string("3"));

  ctx.emplace(ctx->resume());
  var = ctx->frame.FindVariable("int_vec");
  EXPECT_EQ(var.GetSummary(), std::string("(size 3)[1, 2, 3]"));

  ctx.emplace(ctx->resume());
  var = ctx->frame.FindVariable("int_vec");
  EXPECT_EQ(var.GetSummary(),
            std::string("(size 103)[1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, ...]"));
}
