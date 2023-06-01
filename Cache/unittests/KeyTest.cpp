//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/Support/Keys.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace M::Cache;

namespace {

struct WrapOne {

  static std::string wrapKey(const std::string &key) { return key + "1"; }
};

struct WrapTwo {

  static std::string wrapKey(const std::string &key) { return key + "2"; }
};

using OneWrapped = Keys::WrappedKey<Keys::ReadOnlyKey, WrapOne>;

using OneTwoWrapped = Keys::WrappedKey<Keys::ReadOnlyKey, WrapOne, WrapTwo>;

using TwoOneWrapped = Keys::WrappedKey<Keys::ReadOnlyKey, WrapTwo, WrapOne>;

} // namespace

TEST(KeyTest, WrappedKeys) {

  std::string original = "test";
  EXPECT_EQ(original, Keys::ReadOnlyKey::hashKey(original));

  auto hasher = [](const std::string &input) -> std::string {
    llvm::BLAKE3 hashStateOne{};
    hashStateOne.update(input);
    auto hash = hashStateOne.final();
    return {hash.begin(), hash.end()};
  };

  // Result should be hash of "test1".
  std::string resultOne = hasher("test1");
  EXPECT_EQ(resultOne, OneWrapped::hashKey(original));

  // Result should be hash of "test12".
  std::string resultOneTwo = hasher("test12");
  EXPECT_EQ(resultOneTwo, OneTwoWrapped::hashKey(original));

  // Result should be hash of "test21".
  std::string resultTwoOne = hasher("test21");
  EXPECT_EQ(resultTwoOne, TwoOneWrapped::hashKey(original));
}

TEST(KeyTest, CPUFeatures) {
#if defined(__x86_64__) && defined(__AVX512F__)
  auto lhs1 = Keys::StringHashedKey::hashKey("testx86_64:avx512f");
  auto rhs1 = Keys::CPUFeatureWrappedKey<Keys::ReadOnlyKey>::hashKey("test");
  EXPECT_EQ(lhs1, rhs1);
#endif

#if defined(__x86_64__) && defined(__AVX2__) && !defined(__AVX512F__)
  auto lhs2 = Keys::StringHashedKey::hashKey("testx86_64:avx2");
  auto rhs2 = Keys::CPUFeatureWrappedKey<Keys::ReadOnlyKey>::hashKey("test");
  EXPECT_EQ(lhs2, rhs2);
#endif
}
