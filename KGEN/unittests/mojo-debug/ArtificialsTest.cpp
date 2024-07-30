//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

TEST(ArtificialsTest, testArtificialArguments) {
  StopContext ctx = buildAndLaunch("artificials.mojo");

  SBValueList visibleVariables = ctx.frame.GetVariables(/*arguments=*/true,
                                                        /*locals=*/true,
                                                        /*statics=*/false,
                                                        /*in_scope_only=*/true);

  EXPECT_EQ((int)visibleVariables.GetSize(), 2);
  EXPECT_TRUE(visibleVariables.GetFirstValueByName("a").IsValid());
  EXPECT_TRUE(visibleVariables.GetFirstValueByName("b").IsValid());

  EXPECT_TRUE(ctx.frame.FindVariable("__result__").IsValid());
  EXPECT_TRUE(ctx.frame.FindVariable("__error__").IsValid());
}
