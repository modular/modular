//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/SmartVariant.h"

#include "gtest/gtest.h"

using namespace M;

using SV1 = SmartVariant<int *, float *>;
using SV2 = SmartVariant<int, float>;

struct SmartVariantTest : public testing::Test {
  float f{3.14f};
  int i{42};
  double d{3.14};
  long long l{42};

  SV1 a1, b1, c1, n1;
  SV2 a2, b2, c2, n2;

  SmartVariantTest()
      : a1(&f), b1(&i), c1(&i), n1(), a2(f), b2(i), c2(i), n2() {}
};

TEST_F(SmartVariantTest, ComparisonOperators) {
  // Self comparison
  EXPECT_TRUE(a1 == a1);
  EXPECT_FALSE(a1 != a1);
  EXPECT_TRUE(a2 == a2);
  EXPECT_FALSE(a2 != a2);

  // Other comparison, but same underlying storage
  EXPECT_TRUE(a1 != b1);
  EXPECT_FALSE(a1 == b1);
  EXPECT_TRUE(a2 != b2);
  EXPECT_FALSE(a2 == b2);

  EXPECT_TRUE(b1 == c1);
  EXPECT_FALSE(b1 != c1);
  EXPECT_TRUE(b2 == c2);
  EXPECT_FALSE(b2 != c2);

  EXPECT_TRUE(b1 != n1);
  EXPECT_FALSE(b1 == n1);
  EXPECT_TRUE(b2 != n2);
  EXPECT_FALSE(b2 == n2);
}

TEST_F(SmartVariantTest, isa) {
  using namespace llvm;
  EXPECT_FALSE(isa<int *>(a1));
  EXPECT_FALSE(isa<int>(a2));
  EXPECT_TRUE(isa<float *>(a1));
  EXPECT_TRUE(isa<float>(a2));
  EXPECT_TRUE(isa<int *>(b1));
  EXPECT_TRUE(isa<int>(b2));
  EXPECT_FALSE(isa<float *>(b1));
  EXPECT_FALSE(isa<float>(b2));
  EXPECT_TRUE(isa<int *>(n1));
  EXPECT_TRUE(isa<int>(n2));
  EXPECT_FALSE(isa<float *>(n1));
  EXPECT_FALSE(isa<float>(n2));
}

TEST_F(SmartVariantTest, cast) {
  using namespace llvm;
  EXPECT_EQ(cast<float *>(a1), &f);
  EXPECT_EQ(cast<float>(a2), f);
  EXPECT_EQ(cast<int *>(b1), &i);
  EXPECT_EQ(cast<int>(b2), i);
  EXPECT_EQ(cast<int *>(n1), (int *)nullptr);
  EXPECT_EQ(cast<int>(n2), int{});
}

TEST_F(SmartVariantTest, dyn_cast) {
  using namespace llvm;
  EXPECT_EQ(dyn_cast<int *>(a1), nullptr);
  EXPECT_EQ(dyn_cast<int>(a2), int{});
  EXPECT_EQ(dyn_cast<float *>(a1), &f);
  EXPECT_EQ(dyn_cast<float>(a2), f);
  EXPECT_EQ(dyn_cast<int *>(b1), &i);
  EXPECT_EQ(dyn_cast<float>(b2), float{});
  EXPECT_EQ(dyn_cast<int *>(c1), &i);
  EXPECT_EQ(dyn_cast<int>(c2), i);
  EXPECT_EQ(dyn_cast<float *>(c1), nullptr);
  EXPECT_EQ(dyn_cast<float>(c2), float{});
  EXPECT_EQ(dyn_cast_if_present<int *>(n1), nullptr);
  EXPECT_EQ(dyn_cast_if_present<int>(n2), int{});
  EXPECT_EQ(dyn_cast_if_present<float *>(n1), nullptr);
  EXPECT_EQ(dyn_cast_if_present<float>(n2), float{});
}

TEST_F(SmartVariantTest, isNull) {
  EXPECT_FALSE(a1.isNull());
  EXPECT_FALSE(a2.isNull());
  EXPECT_FALSE(b1.isNull());
  EXPECT_FALSE(b2.isNull());
  EXPECT_FALSE(c1.isNull());
  EXPECT_FALSE(c2.isNull());
  EXPECT_TRUE(n1.isNull());
  EXPECT_FALSE(n2.isNull()); // a `std::variant` is never null
}
