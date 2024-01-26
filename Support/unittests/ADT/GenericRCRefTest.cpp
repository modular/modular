//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ADT/GenericRCRef.h"
#include "Support/ReferenceCounted.h"

#include "gtest/gtest.h"

using namespace M;

namespace {
struct TypeA : public ReferenceCounted<TypeA> {
  int i = 42;

  TypeA() = default;
  TypeA(int i) : i(i) {}
};

struct TypeB : public ReferenceCounted<TypeB> {
  int i = 50;
};

TEST(GenericRCRef, Lifecycle) {
  auto ref1 = GenericRCRef::create<TypeA>(5);
  auto ref2 = GenericRCRef::create<TypeA>();
  GenericRCRef ref3;

  ASSERT_TRUE(ref1);
  ASSERT_TRUE(ref2);
  ASSERT_FALSE(ref3);

  ASSERT_EQ(ref1.get<TypeA>()->i, 5);
  ASSERT_EQ(ref2.get<TypeA>()->i, 42);

  ref3 = GenericRCRef::create<TypeB>();
  ASSERT_TRUE(ref3);
  ASSERT_EQ(ref3.get<TypeB>()->i, 50);

  TypeB *released = ref3.release<TypeB>();
  ASSERT_NE(released, nullptr);
  RCRef<TypeB> rcref = RCRef<TypeB>::take(released);

  ref1 = std::move(ref2);
  ASSERT_FALSE(ref2);
  ASSERT_TRUE(ref1);
  ASSERT_EQ(ref1.get<TypeA>()->i, 42);
}

#ifndef NDEBUG
TEST(GenericRCRef, IllTyped_ExpectDeath) {
  auto ref = GenericRCRef::create<TypeA>(5);
  ASSERT_DEATH_IF_SUPPORTED(ref.get<TypeB>(),
                            "mismatch between actual and expected type ids");
}
#endif

} // namespace
