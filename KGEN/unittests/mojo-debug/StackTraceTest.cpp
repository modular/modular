//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;
using namespace ::testing::internal;

TEST(StackTraceTest, testStackTraceFormat) {
  // Simple test that ensures frames can be printed out in a nice format.
  // It's covering simple parameter types, methods and nested functions, as
  // well as printing the values of arguments.

  // FIXME(25047): A current limitation when formatting frames is that the
  // source name of nested functions lose track of the parameters of the
  // parent, because of which `[...]` is printed instead, signaling that some
  // parameters are expected, but aren't available.

  // TODO(25048): Include argument types in functions.

  StopContext ctx = buildAndLaunch("stack-trace.mojo");

  std::vector<std::string> frameDescs;
  for (size_t i = 0; i < 4; ++i) {
    SBStream description;
    ctx.thread.GetFrameAtIndex(i).GetDescription(description);
    frameDescs.emplace_back(description.GetData());
  }

  EXPECT_TRUE(RE::PartialMatch(
      frameDescs[0], RE(R"(stack-trace.Foo\[...\].getParametrized\[...\])"
                        R"(.nested_function\(z=\(\[0\] = 105.25\)\) at)"
                        R"( stack-trace.mojo:15:13)")));
  EXPECT_TRUE(RE::PartialMatch(
      frameDescs[1],
      RE(R"(stack-trace.Foo\[index, index\])"
         R"(.getParametrized\[scalar<f32>\]\(self=.* @ 0x.*,)"
         R"( val=\(\[0] = 105.25\)\) at stack-trace.mojo:17:31)")));
  EXPECT_TRUE(RE::PartialMatch(
      frameDescs[2],
      RE(R"(stack-trace.Foo\[index, index\])"
         R"(.getFloat\(self=.* @ 0x.*, x=\(\[0\] = 1.125\), y=100\))"
         R"( at stack-trace.mojo:20:45)")));
  EXPECT_TRUE(StringRef(frameDescs[3])
                  .contains("stack-trace.main() at stack-trace.mojo:24:35"));
}
