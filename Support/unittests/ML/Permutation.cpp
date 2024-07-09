//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/Permutation.h"
#include "gtest/gtest.h"

using namespace M;

TEST(Permutation, permutationForwardStride1) {
  SmallVector<int64_t> data = {0, 1, 2, 3, 4, 5, 6};
  SmallVector<int64_t> identityPermutation = {0, 1, 2, 3, 4, 5, 6};
  SmallVector<int64_t> permutationTest = {4, 5, 1, 3, 0, 6, 2};

  SmallVector<int64_t> identityPermutationResult =
      permute(data, identityPermutation);
  SmallVector<int64_t> permutationTestResult = permute(data, permutationTest);

  // Because we are permuting the identity vector, the result should be
  // the same as the permutation we passed in
  EXPECT_EQ(identityPermutationResult, identityPermutation);
  EXPECT_EQ(permutationTestResult, permutationTest);
}

TEST(Permutation, permutationForwardStride2) {
  SmallVector<int64_t> data = {0, 1, 2, 3, 4, 5};
  SmallVector<int64_t> identityPermutation = {0, 1, 2};
  SmallVector<int64_t> permutationTest = {1, 0, 2};

  SmallVector<int64_t> identityPermutationResult =
      permute(data, identityPermutation, 2);
  SmallVector<int64_t> permutationTestResult =
      permute(data, permutationTest, 2);

  SmallVector<int64_t> permutationTestExpected = {2, 3, 0, 1, 4, 5};
  EXPECT_EQ(identityPermutationResult, data);
  EXPECT_EQ(permutationTestResult, permutationTestExpected);
}

TEST(Permutation, permuteReverse) {
  SmallVector<int64_t> data = {0, 1, 2, 3, 4, 5, 6};
  SmallVector<int64_t> identityPermutation = {0, 1, 2, 3, 4, 5, 6};
  SmallVector<int64_t> permutationTest = {4, 5, 1, 3, 0, 6, 2};

  SmallVector<int64_t> identityPermutationResult =
      permuteReverse(data, identityPermutation);
  SmallVector<int64_t> permutationTestResult =
      permuteReverse(data, permutationTest);

  EXPECT_EQ(identityPermutationResult, identityPermutation);

  SmallVector<int64_t> expected{4, 2, 6, 3, 0, 1, 5};
  EXPECT_EQ(permutationTestResult, expected);
}

TEST(Permutation, permuteSolve) {
  SmallVector<int64_t> data = {0, 1, 2, 3, 4, 5, 6};
  SmallVector<int64_t> identityPermutation = {0, 1, 2, 3, 4, 5, 6};
  SmallVector<int64_t> permutationTest = {4, 5, 1, 3, 0, 6, 2};

  SmallVector<int64_t> expectedIdentityPermutationResult =
      solvePermutation(data, data);
  EXPECT_EQ(expectedIdentityPermutationResult, identityPermutation);

  SmallVector<int64_t> permutationTestResult =
      solvePermutation(data, permutationTest);
  SmallVector<int64_t> expected{4, 2, 6, 3, 0, 1, 5};
  EXPECT_EQ(permutationTestResult, expected);
}
