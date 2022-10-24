//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/FunctionExtras.h"

#include "gtest/gtest.h"

using namespace M;

TEST(FunctionExtrasTest, bind_back) {
  EXPECT_EQ(3, bind_back(std::plus<int>{}, 1)(2));
  EXPECT_EQ(3, bind_back(std::plus<float>{}, 1.)(2.));
  EXPECT_EQ(3, bind_back(std::plus<float>{}, 2.)(1.));
  EXPECT_EQ(3, bind_back(std::plus<float>{}, 1., 2.)());
  EXPECT_EQ(3, bind_back(std::plus<float>{})(1., 2.));

  EXPECT_EQ(2, bind_back(std::divides<float>{}, 1.)(2.));
  EXPECT_EQ(0.5, bind_back(std::divides<float>{}, 2.)(1.));

  EXPECT_EQ(1, bind_back(std::minus<float>{}, 1.)(2.));
  EXPECT_EQ(-1, bind_back(std::minus<float>{}, 2.)(1.));

  auto addThreeValues = [](int a, int b, int c) { return a + b + c; };
  EXPECT_EQ(6, bind_back(addThreeValues, 2)(1, 3));
  EXPECT_EQ(6, bind_back(addThreeValues, 1, 2)(3));
}
