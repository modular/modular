//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/DType.h"

#include "gtest/gtest.h"

using namespace M;

TEST(DType, getSizeInBytesPacked) {
  EXPECT_EQ(DType(DType::si4).getSizeInBytes(2), 1);
  EXPECT_EQ(DType(DType::si4).getSizeInBytes(3), 2);
  EXPECT_EQ(DType(DType::si2).getSizeInBytes(5), 2);
  EXPECT_EQ(DType(DType::si2).getSizeInBytes(4), 1);
  EXPECT_EQ(DType(DType::si1).getSizeInBytes(3), 1);
  EXPECT_EQ(DType(DType::si1).getSizeInBytes(9), 2);
}
