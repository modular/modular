//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(SourceNamesTest, testFunctionBeforeStructParsing) {
  // Tests that DWARF parsing is done correctly when LLDB parses a function
  // source name before its owning struct.
  // This happens when the debug session starts with a single breakpoint
  // within a struct method.

  StopContext ctx = buildAndLaunch("point.mojo");
  SBValue x = ctx.frame.FindVariable("x");
  EXPECT_EQ((int)x.GetValueAsSigned(), 1);

  ctx.stepOver();
  SBValue p1 = ctx.frame.FindVariable("p1");
  EXPECT_STREQ(p1.GetTypeName(), "!lit.struct<@point::@Point>");
}
