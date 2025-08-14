//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/StringExtras.h"

#include "gtest/gtest.h"

using namespace M;

TEST(StringExtrasTest, ReplaceAll) {
  std::string str = "hello";
  replaceAll(str, "l", "L");
  EXPECT_EQ(str, "heLLo");

  str = "hello";
  replaceAll(str, "l", "");
  EXPECT_EQ(str, "heo");

  str = "hello";
  replaceAll(str, "ll", "L");
  EXPECT_EQ(str, "heLo");

  str = "hello";
  replaceAll(str, "", "L");
  EXPECT_EQ(str, "hello");
}
