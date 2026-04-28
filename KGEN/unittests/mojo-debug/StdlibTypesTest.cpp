//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(StdlibTypesTest, testVariant) {
  StopContext ctx = buildAndLaunch("variant.mojo");

  // v = Variant[Int, String](42) — active type is Int
  SBValue v = ctx.frame.FindVariable("v");
  EXPECT_STREQ(v.GetSummary(), "Int(42)");

  ctx.resume();

  // v.set[String]("hello, world") — heap-encoded String.
  v = ctx.frame.FindVariable("v");
  EXPECT_STREQ(v.GetSummary(), "String(\"hello, world\")");

  ctx.resume();

  // v.set[String]("hi") — inline/small-string form.
  v = ctx.frame.FindVariable("v");
  EXPECT_STREQ(v.GetSummary(), "String(\"hi\")");

  ctx.resume();

  // v.set[String](String("")) — heap path with size == 0.
  v = ctx.frame.FindVariable("v");
  EXPECT_STREQ(v.GetSummary(), "String(\"\")");

  ctx.resume();

  // w = Variant[Int, Bool, String](True) — discriminant > 1 in a 3-way
  // variant, confirming the union's `GetChildAtIndex(discr)` indexing.
  SBValue w = ctx.frame.FindVariable("w");
  EXPECT_STREQ(w.GetSummary(), "Bool(True)");

  ctx.resume();

  // w.set[String]("last arm") — boundary case: last arm of a 3-way variant.
  w = ctx.frame.FindVariable("w");
  EXPECT_STREQ(w.GetSummary(), "String(\"last arm\")");
}

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
