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

TEST(MathExtrasTest, Mean) {
  auto meanValue0 = mean(std::vector<float>());
  EXPECT_FLOAT_EQ(meanValue0, 0.0);

  auto meanValue = mean(std::vector{-10.0, 1.0, 1.0, 1.0, 1.0, 20.0});
  EXPECT_FLOAT_EQ(meanValue, 2.3333333);

  // You can call mean with any container. E.g. a initializer_list.
  auto meanInitializerList =
      mean(std::initializer_list<float>{-10.0, 1.0, 1.0, 1.0, 1.0, 20.0});
  EXPECT_FLOAT_EQ(meanInitializerList, 2.3333333);
}

TEST(MathExtrasTest, TrimmedMean) {
  // You can call trimmedMean with any container. E.g. a initializer_list.
  auto trimmedMeanInitializerList = trimmedMean(
      std::initializer_list<float>{-10.0, 1.0, 1.0, 1.0, 1.0, 20.0});
  EXPECT_FLOAT_EQ(trimmedMeanInitializerList, 2.3333333);

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

TEST(MathExtrasTest, Median) {
  auto medianValue0 = median(std::vector<float>());
  EXPECT_FLOAT_EQ(medianValue0, 0.0);

  auto medianValueOdd = median(std::vector{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  EXPECT_FLOAT_EQ(medianValueOdd, 3.0);

  // You can call median with an even length list.
  auto medianValue = median(std::vector{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  EXPECT_FLOAT_EQ(medianValue, 3.5);
}

TEST(MathExtrasTest, Percentile) {
  {
    auto val = percentile(std::vector<float>(), 0.0);
    EXPECT_FLOAT_EQ(val, 0.0);
  }
  {
    auto val = percentile(std::vector{1.0, 2.0, 3.0}, 0.0);
    EXPECT_FLOAT_EQ(val, 1.0);
  }
  {
    auto val = percentile(
        std::vector{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
        0.8);
    EXPECT_FLOAT_EQ(val, 8.0);
  }
  {
    auto val = percentile(std::vector{1.0, 2.0, 3.0}, 0.9);
    EXPECT_FLOAT_EQ(val, 3.0);
  }
}
