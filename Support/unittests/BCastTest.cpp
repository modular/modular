//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/BCast.h"
#include "Support/ML/TensorShape.h"

#include "gtest/gtest.h"

using namespace M;

TEST(KGENKernels, broadcastedShape) {
  auto check = [](const TensorShape &a, const TensorShape &b,
                  const TensorShape &expected) {
    ErrorOr<TensorShape> computed = broadcastedShape(a, b);

    ASSERT_FALSE(computed.isError()) << computed.getError();

    EXPECT_TRUE(*computed == expected)
        << "computed " << computed->getAsString() << " but expecting "
        << expected.getAsString();
  };
  check({256, 256}, {256, 256}, {256, 256});
  check({256, 256, 1}, {1, 256, 256}, {256, 256, 256});
  check({8, 1, 6, 1}, {7, 1, 5}, {8, 7, 6, 5});
  check({256, 256, 1, 5, 8}, {1, 256, 5, 8}, {256, 256, 256, 5, 8});
  check({256, 256, 1, 5, 8}, {1}, {256, 256, 1, 5, 8});
  check({1}, {256, 256, 1, 5, 8}, {256, 256, 1, 5, 8});

  {
    TensorShape a({5, 8});
    TensorShape b({8, 4});
    ErrorOr<TensorShape> computed = broadcastedShape(a, b);

    EXPECT_TRUE(computed.isError());
  }
}
