//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Version.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace M;

/// Check that we can parse some basic examples.
TEST(Version, Parsing) {
  auto v1Or = Version::parse("1.0.0");
  EXPECT_FALSE(v1Or.isError()) << v1Or.getError();
  EXPECT_TRUE(v1Or->getMajor() == 1 && v1Or->getMinor() == 0 &&
              v1Or->getPatch() == 0 && v1Or->getLabel() == "");

  auto v1PatchOr = Version::parse("1.0.0-abcd");
  EXPECT_FALSE(v1PatchOr.isError()) << v1PatchOr.getError();
  EXPECT_TRUE(v1PatchOr->getMajor() == 1 && v1PatchOr->getMinor() == 0 &&
              v1PatchOr->getPatch() == 0 && v1PatchOr->getLabel() == "abcd")
      << v1PatchOr->toString();

  v1PatchOr = Version::parse("1.0.0-abcd-123j.sdlkfj");
  EXPECT_FALSE(v1PatchOr.isError()) << v1PatchOr.getError();

  EXPECT_TRUE(v1PatchOr->getMajor() == 1 && v1PatchOr->getMinor() == 0 &&
              v1PatchOr->getPatch() == 0 &&
              v1PatchOr->getLabel() == "abcd-123j.sdlkfj")
      << v1PatchOr->toString();
}

/// Check that the precedence operator works the way we expect it to. This test
/// comes from the SemVer spec. We expect:
///   1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-alpha.beta < 1.0.0-beta < 1.0.0-beta.2
///   < 1.0.0-beta.11 < 1.0.0-rc.1 < 1.0.0
/// This also checks more basic precedence:
///   1.0.0 < 1.1.0 < 2.0.0, and 1.0.0-rc.1 < 1.0.1-rc.0
TEST(Version, Precedence) {
  auto v1Alpha = Version::parse("1.0.0-alpha");
  ASSERT_FALSE(v1Alpha.isError()) << v1Alpha.getError();

  auto v1Alpha1 = Version::parse("1.0.0-alpha.1");
  ASSERT_FALSE(v1Alpha1.isError()) << v1Alpha1.getError();

  auto v1AlphaBeta = Version::parse("1.0.0-alpha.beta");
  ASSERT_FALSE(v1AlphaBeta.isError()) << v1AlphaBeta.getError();

  auto v1Beta = Version::parse("1.0.0-beta");
  ASSERT_FALSE(v1Beta.isError()) << v1Beta.getError();

  auto v1Beta2 = Version::parse("1.0.0-beta.2");
  ASSERT_FALSE(v1Beta2.isError()) << v1Beta2.getError();

  auto v1Beta11 = Version::parse("1.0.0-beta.11");
  ASSERT_FALSE(v1Beta11.isError()) << v1Beta11.getError();

  auto v1RC1 = Version::parse("1.0.0-rc.1");
  ASSERT_FALSE(v1RC1.isError()) << v1RC1.getError();

  auto v101RC1 = Version::parse("1.0.1-rc.0");
  ASSERT_FALSE(v101RC1.isError()) << v101RC1.getError();

  auto v1 = Version::parse("1.0.0");
  ASSERT_FALSE(v1.isError()) << v1.getError();

  auto v1p1 = Version::parse("1.1.0");
  ASSERT_FALSE(v1p1.isError()) << v1p1.getError();

  auto v2 = Version::parse("2.0.0");
  ASSERT_FALSE(v2.isError()) << v2.getError();

  // Check the example given in the spec.
  EXPECT_TRUE((*v1Alpha < *v1Alpha1) && *v1Alpha1 > *v1Alpha);
  EXPECT_TRUE((*v1Alpha1 < *v1AlphaBeta) && (*v1AlphaBeta > *v1Alpha1));
  EXPECT_TRUE((*v1AlphaBeta < *v1Beta) && (*v1Beta > *v1AlphaBeta));
  EXPECT_TRUE((*v1Beta < *v1Beta2) && (*v1Beta2 > *v1Beta));
  EXPECT_TRUE((*v1Beta2 < *v1Beta11) && (*v1Beta11 > *v1Beta2));
  EXPECT_TRUE((*v1Beta11 < *v1RC1) && (*v1RC1 > *v1Beta11));
  EXPECT_TRUE((*v1RC1 < *v1) && (*v1 > *v1RC1));
  EXPECT_TRUE((*v1RC1 < *v101RC1) && (*v101RC1 > *v1RC1));
  EXPECT_TRUE((*v1 < *v1p1) && (*v1p1 > *v1));
  EXPECT_TRUE((*v1p1 < *v2) && (*v2 > *v1p1));

  // Check a specific problematic version.
  auto v020RC4 = Version::parse("0.2.0-rc4");
  ASSERT_FALSE(v020RC4.isError()) << v020RC4.getError();

  auto v021RC0 = Version::parse("0.2.1-rc0");
  ASSERT_FALSE(v021RC0.isError()) << v021RC0.getError();

  auto v021RC4 = Version::parse("0.2.1-rc4");
  ASSERT_FALSE(v021RC4.isError()) << v021RC4.getError();

  // 0.2.0-rc4 should be less than 0.2.1-rc0.
  EXPECT_TRUE((*v020RC4 < *v021RC0) && (*v021RC0 > *v020RC4));
  // 0.2.1-rc0 should be less than 0.2.1-rc4.
  EXPECT_TRUE((*v021RC0 < *v021RC4) && (*v021RC4 > *v021RC0));
  // 0.2.0-rc4 should be less than 0.2.1-rc4.
  EXPECT_TRUE((*v020RC4 < *v021RC4) && (*v021RC4 > *v020RC4));
}

TEST(Version, Equal) {
  auto v1Alpha = Version::parse("1.0.0-alpha");
  ASSERT_FALSE(v1Alpha.isError()) << v1Alpha.getError();

  auto v1Alpha2 = Version::parse("1.0.0-alpha");
  ASSERT_FALSE(v1Alpha2.isError()) << v1Alpha2.getError();

  EXPECT_TRUE(*v1Alpha == *v1Alpha2);
  EXPECT_FALSE(*v1Alpha != *v1Alpha2);
  EXPECT_FALSE(*v1Alpha < *v1Alpha2);
  EXPECT_TRUE(*v1Alpha <= *v1Alpha2);
  EXPECT_TRUE(*v1Alpha2 >= *v1Alpha);
  EXPECT_FALSE(*v1Alpha2 < *v1Alpha);
}

TEST(Version, NotEqual) {
  auto v1Alpha = Version::parse("1.0.0-alpha");
  ASSERT_FALSE(v1Alpha.isError()) << v1Alpha.getError();

  auto v1Alpha2 = Version::parse("1.0.0-beta");
  ASSERT_FALSE(v1Alpha2.isError()) << v1Alpha2.getError();

  EXPECT_TRUE(*v1Alpha != *v1Alpha2);
}
