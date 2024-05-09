//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(StringsTest, testStrings) {
  // Ensures that String and StringLiteral can be parsed correctly from memory.
  StopContext ctx = buildAndLaunch("strings.mojo");

  SBValue st = ctx.frame.FindVariable("st");
  EXPECT_TRUE(StringRef(st.GetSummary()).contains("\"012345678910111213141"));

  ctx.resume();

  SBValue literal = ctx.frame.FindVariable("literal");
  SBValue s1 = ctx.frame.FindVariable("s1");
  SBValue s2 = ctx.frame.FindVariable("s2");
  SBValue s3 = ctx.frame.FindVariable("s3");
  SBValue s4 = ctx.frame.FindVariable("s4");

  // StringLiterals, being built-in, provide the underlying strings as value. On
  // the other hand, String, being parsed by a data formatter, provides the
  // underlying string as a Summary, following C++'s convention in LLDB.

  EXPECT_STREQ(literal.GetValue(), "\"string_literal\"");
  EXPECT_STREQ(s1.GetSummary(), "\"let_string\"");
  EXPECT_TRUE(StringRef(s2.GetSummary()).contains("\"012345678910111213141"));
  EXPECT_STREQ(s3.GetSummary(), "\"\"");
  EXPECT_TRUE(StringRef(s4.GetSummary()).contains("\"012345678910111213141"));
}
