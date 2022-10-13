//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MathExtras.h"

#include "gtest/gtest.h"

#include <initializer_list>
#include <vector>

using namespace M;

TEST(MathExtrasTest, TrimmedMean) {
  auto trimmedMeanInitializerList = trimmedMean(
      std::initializer_list<float>{-10.0, 1.0, 1.0, 1.0, 1.0, 20.0});
  EXPECT_FLOAT_EQ(trimmedMeanInitializerList, 2.3333333);

  // You can call trimmedMean with any container. E.g. a vector.
  auto trimmedMean0 = trimmedMean(std::vector{-10.0, 1.0, 1.0, 1.0, 1.0, 20.0});
  EXPECT_FLOAT_EQ(trimmedMean0, 2.3333333);

  auto trimmedMean1 =
      trimmedMean(std::vector{-10.0, 1.0, 1.0, 1.0, 1.0, 20.0}, 0.2);
  EXPECT_FLOAT_EQ(trimmedMean1, 1.0);

  auto trimmedMean2 =
      trimmedMean(std::vector{-200.0f, 1.0f, 2.0f, 4.0f, 5.0f, 5.0f, 5.0f, 6.0f,
                              10.0f, 100000.0f},
                  0.1);
  EXPECT_FLOAT_EQ(trimmedMean2, 4.75);

  auto trimmedMean3 =
      trimmedMean(std::vector{-200.0f, 1.0f, 2.0f, 4.0f, 5.0f, 5.0f, 5.0f, 6.0f,
                              10.0f, 100000.0f},
                  0);
  EXPECT_FLOAT_EQ(trimmedMean3, 9983.8);
}
