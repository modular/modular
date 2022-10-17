//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/SIMD.h"

#include "gtest/gtest.h"

using namespace M;

TEST(SIMDMatrixTest, Constructor) {
  SIMDMatrix<float, 4, 4> m(99.0f);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto val = m[{i, j}];
      EXPECT_EQ(val, 99.0f);
    }
  }
}

TEST(SIMDMatrixTest, CopyConstructor) {
  SIMDMatrix<float, 4, 4> m(99.0f);
  auto m2 = m;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto val = m2[{i, j}];
      EXPECT_EQ(val, 99.0f);
    }
  }
}

TEST(SIMDMatrixTest, SetValue) {
  SIMDMatrix<float, 4, 4> m(99.0f);
  m[{0, 0}] = 1.0f;
  auto val = m[{0, 0}];
  EXPECT_EQ(val, 1.0f);
}

TEST(SIMDMatrixTest, AddToScalar) {
  SIMDMatrix<float, 4, 4> a(1.0f);
  auto c = a + 2.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto elem = c[{i, j}];
      EXPECT_EQ(elem, 3.0f);
    }
  }
}

TEST(SIMDMatrixTest, SubFromScalar) {
  SIMDMatrix<float, 4, 4> a(1.0f);
  auto c = a - 2.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto elem = c[{i, j}];
      EXPECT_EQ(elem, -1.0f);
    }
  }
}

TEST(SIMDMatrixTest, MultiplyWithScalar) {
  SIMDMatrix<float, 4, 4> a(1.0f);
  auto c = a * 2.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto elem = c[{i, j}];
      EXPECT_EQ(elem, 2.0f);
    }
  }
}

TEST(SIMDMatrixTest, DivideWithScalar) {
  SIMDMatrix<float, 4, 4> a(1.0f);
  auto c = a / 2.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto elem = c[{i, j}];
      EXPECT_EQ(elem, 0.5f);
    }
  }
}
