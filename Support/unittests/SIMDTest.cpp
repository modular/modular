//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/SIMD.h"

#include "gtest/gtest.h"

using namespace M;

TEST(SIMDMatrixTest, Constructor) {
  SIMDMatrix<float, 8, 8> m(99.0f);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto val = m[{i, j}];
      EXPECT_EQ(val, 99.0f);
    }
  }
}

TEST(SIMDMatrixTest, ConstructorArray) {
  float arry[8][8] = {
      {1, 2, 3, 4, 5, 6, 7, 8},         {9, 10, 11, 12, 13, 14, 15, 16},
      {17, 18, 19, 20, 21, 22, 23, 24}, {25, 26, 27, 28, 29, 30, 31, 32},
      {33, 34, 35, 36, 37, 38, 39, 40}, {41, 42, 43, 44, 45, 46, 47, 48},
      {49, 50, 51, 52, 53, 54, 55, 56}, {57, 58, 59, 60, 61, 62, 63, 64}};
  SIMDMatrix<float, 8, 8> mat(arry);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      auto val = mat[{i, j}];
      EXPECT_EQ(val, i * 8 + j + 1);
    }
  }
}

TEST(SIMDMatrixTest, CopyConstructor) {
  SIMDMatrix<float, 8, 8> m(99.0f);
  auto m2 = m;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto val = m2[{i, j}];
      EXPECT_EQ(val, 99.0f);
    }
  }
}

TEST(SIMDMatrixTest, SetValue) {
  SIMDMatrix<float, 8, 8> m(99.0f);
  m[{0, 0}] = 1.0f;
  auto val = m[{0, 0}];
  EXPECT_EQ(val, 1.0f);
}

TEST(SIMDMatrixTest, AddToScalar) {
  SIMDMatrix<float, 8, 8> a(1.0f);
  auto c = a + 2.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto elem = c[{i, j}];
      EXPECT_EQ(elem, 3.0f);
    }
  }
}

TEST(SIMDMatrixTest, SubFromScalar) {
  SIMDMatrix<float, 8, 8> a(1.0f);
  auto c = a - 2.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto elem = c[{i, j}];
      EXPECT_EQ(elem, -1.0f);
    }
  }
}

TEST(SIMDMatrixTest, MultiplyWithScalar) {
  SIMDMatrix<float, 8, 8> a(1.0f);
  auto c = a * 2.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto elem = c[{i, j}];
      EXPECT_EQ(elem, 2.0f);
    }
  }
}

TEST(SIMDMatrixTest, DivideWithScalar) {
  SIMDMatrix<float, 8, 8> a(1.0f);
  auto c = a / 2.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto elem = c[{i, j}];
      EXPECT_EQ(elem, 0.5f);
    }
  }
}

TEST(SIMDMatrixTest, Dot8x8_RowMajor) {
  float aData[8][8], bData[8][8], expectedCData[8][8];
  // Fill the data.
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      aData[i][j] = i * 8 + j + 1;
      bData[i][j] = -3.2 + 0.1 * (i * 8 + j);
    }
  }
  // Perform the matmul sequentially.
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      float sum = 0.0f;
      for (int k = 0; k < 8; k++)
        sum += aData[i][k] * bData[k][j];
      expectedCData[i][j] = sum;
    }
  }
  SIMDMatrix<float, 8, 8, SIMDMatrixLayout::kRowMajor> a(aData), b(bData);
  auto c = a.dot(b);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      auto elem = c[{i, j}];
      EXPECT_FLOAT_EQ(elem, expectedCData[i][j])
          << "at i= " << i << " j= " << j;
    }
  }
}

TEST(SIMDMatrixTest, Dot8x8_ColumnMajor) {
  float aData[8][8], bData[8][8], expectedCData[8][8];
  // Fill the data.
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      aData[i][j] = i * 8 + j + 1;
      bData[i][j] = -3.2 + 0.1 * (i * 8 + j);
    }
  }
  // Perform the matmul sequentially.
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      float sum = 0.0f;
      for (int k = 0; k < 8; k++)
        sum += aData[k][i] * bData[j][k];
      expectedCData[j][i] = sum;
    }
  }
  SIMDMatrix<float, 8, 8, SIMDMatrixLayout::kColumnMajor> a(aData), b(bData);
  auto c = a.dot(b);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      auto elem = c[{i, j}];
      EXPECT_FLOAT_EQ(elem, expectedCData[i][j])
          << "at i= " << i << " j= " << j;
    }
  }
}
