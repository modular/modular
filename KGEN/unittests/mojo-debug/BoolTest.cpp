//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(BoolTest, testBool) {
  // There's no SB API for getting the bit size of a type, but the byte size
  // should be 1.

  StopContext ctx = buildAndLaunch("bool.mojo");
  SBValue trueVar = ctx.frame.FindVariable("true");
  EXPECT_EQ(trueVar.GetTypeName(), std::string("!pop.scalar<bool>"));
  EXPECT_EQ((int)trueVar.GetByteSize(), 1);
  EXPECT_EQ((int)trueVar.GetChildAtIndex(0).GetValueAsUnsigned(2), 1);
  EXPECT_EQ(trueVar.GetSummary(), std::string("True"));

  SBValue falseVar = ctx.frame.FindVariable("false");
  EXPECT_EQ(falseVar.GetTypeName(), std::string("!pop.scalar<bool>"));
  EXPECT_EQ((int)falseVar.GetByteSize(), 1);
  EXPECT_EQ((int)falseVar.GetChildAtIndex(0).GetValueAsUnsigned(2), 0);
  EXPECT_EQ(falseVar.GetSummary(), std::string("False"));

  SBValue otherVar = ctx.frame.FindVariable("other");
  EXPECT_EQ(otherVar.GetTypeName(), std::string("!pop.scalar<bool>"));
  EXPECT_EQ((int)otherVar.GetByteSize(), 1);
  EXPECT_EQ((int)otherVar.GetChildAtIndex(0).GetValueAsUnsigned(2), 1);
  EXPECT_EQ(otherVar.GetSummary(), std::string("True"));
}
