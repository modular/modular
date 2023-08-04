//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Version.h"

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
      << *v1PatchOr;

  v1PatchOr = Version::parse("1.0.0-abcd-123j.sdlkfj");
  EXPECT_FALSE(v1PatchOr.isError()) << v1PatchOr.getError();

  EXPECT_TRUE(v1PatchOr->getMajor() == 1 && v1PatchOr->getMinor() == 0 &&
              v1PatchOr->getPatch() == 0 &&
              v1PatchOr->getLabel() == "abcd-123j.sdlkfj")
      << *v1PatchOr;
}

/// Check that the precedence operator works the way we expect it to. This test
/// comes from the SemVer spec. We expect:
///   1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-alpha.beta < 1.0.0-beta < 1.0.0-beta.2
///   < 1.0.0-beta.11 < 1.0.0-rc.1 < 1.0.0
/// This also checks more basic precedence:
///   1.0.0 < 1.1.0 < 2.0.0
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

  auto v1 = Version::parse("1.0.0");
  ASSERT_FALSE(v1.isError()) << v1.getError();

  auto v1p1 = Version::parse("1.1.0");
  ASSERT_FALSE(v1p1.isError()) << v1p1.getError();

  auto v2 = Version::parse("2.0.0");
  ASSERT_FALSE(v2.isError()) << v2.getError();

  // Check the example given in the spec.
  EXPECT_TRUE(*v1Alpha < *v1Alpha1);
  EXPECT_TRUE(*v1Alpha1 < *v1AlphaBeta);
  EXPECT_TRUE(*v1AlphaBeta < *v1Beta);
  EXPECT_TRUE(*v1Beta < *v1Beta2);
  EXPECT_TRUE(*v1Beta2 < *v1Beta11);
  EXPECT_TRUE(*v1Beta11 < *v1RC1);
  EXPECT_TRUE(*v1RC1 < *v1);
  EXPECT_TRUE(*v1 < *v1p1);
  EXPECT_TRUE(*v1p1 < *v2);
}

TEST(Version, Equal) {
  auto v1Alpha = Version::parse("1.0.0-alpha");
  ASSERT_FALSE(v1Alpha.isError()) << v1Alpha.getError();

  auto v1Alpha_2 = Version::parse("1.0.0-alpha");
  ASSERT_FALSE(v1Alpha_2.isError()) << v1Alpha_2.getError();

  EXPECT_TRUE(*v1Alpha == *v1Alpha_2);
  EXPECT_FALSE(*v1Alpha < *v1Alpha_2);
  EXPECT_FALSE(*v1Alpha_2 < *v1Alpha);
}
