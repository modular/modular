//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ASN1/ObjectID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace M;
using namespace ASN1;

TEST(ASN1, RoundtripOID) {
  ObjectID oid(/*withModularPrefix=*/true,
               {// A single octet number.
                32,
                // A just barely multi-octet number. This is also the bitmask
                // used for VLQ encoding, so it could be problematic.
                0x80,
                // A multi-octet large number.
                1172635});

  SmallVector<uint8_t> encoded = oid.getEncoded();
  auto decodedOr = ObjectID::fromEncoded(encoded);
  ASSERT_FALSE(decodedOr.isError()) << decodedOr.getError();

  std::array<uint64_t, 3> arc = {32, 0x80, 1172635};
  EXPECT_EQ(decodedOr->getArc(), ArrayRef(arc));

  EXPECT_TRUE(decodedOr->isModularOID());

  // Checks that we can get equality the way we intended.
  EXPECT_EQ(oid, *decodedOr);
}

TEST(ASN1, RoundtripNonModularOID) {
  // Check an OID that has most of the modular prefix: 1.3.6.1.4.1.311.21.20
  ObjectID oid(/*withModularPrefix=*/false, {1, 3, 6, 1, 4, 1, 311, 21, 20});

  SmallVector<uint8_t> encoded = oid.getEncoded();
  auto decodedOr = ObjectID::fromEncoded(encoded);
  ASSERT_FALSE(decodedOr.isError()) << decodedOr.getError();

  EXPECT_FALSE(decodedOr->isModularOID());

  std::array<uint64_t, 9> arc = {1, 3, 6, 1, 4, 1, 311, 21, 20};
  EXPECT_EQ(decodedOr->getArc(), ArrayRef(arc));

  auto fromStrOr = ObjectID::fromString("1.3.6.1.4.1.311.21.20");
  ASSERT_FALSE(fromStrOr.isError()) << fromStrOr.getError();
  EXPECT_EQ(*decodedOr, *fromStrOr);
}

TEST(ASN1, RoundtripStringOID) {
  constexpr StringLiteral oidStr = "1.2.3.4.5.1230984";
  auto oidOr = ObjectID::fromString(oidStr);
  ASSERT_FALSE(oidOr.isError()) << oidOr.getError();

  std::array<uint64_t, 6> arc = {1, 2, 3, 4, 5, 1230984};
  EXPECT_EQ(oidOr->getArc(), ArrayRef(arc));

  std::string s;
  llvm::raw_string_ostream stream(s);
  stream << *oidOr;
  EXPECT_EQ(s, oidStr);
}

TEST(ASN1, RoundtripStringModularOID) {
  constexpr StringLiteral oidStr = "1.3.6.1.4.1.61041.1.2.3.4.5.1230984";
  auto oidOr = ObjectID::fromString(oidStr);
  ASSERT_FALSE(oidOr.isError()) << oidOr.getError();

  std::array<uint64_t, 6> arc = {1, 2, 3, 4, 5, 1230984};
  EXPECT_TRUE(oidOr->isModularOID());
  EXPECT_EQ(oidOr->getArc(), ArrayRef(arc));

  std::string s;
  llvm::raw_string_ostream stream(s);
  stream << *oidOr;
  EXPECT_EQ(s, oidStr);
}

TEST(ASN1, TrickyOIDEquality) {
  constexpr StringLiteral oidStr = "1.3.6.1.4.1.61041.1.2.3.4.5.1230984";
  auto oidOr = ObjectID::fromString(oidStr);
  ASSERT_FALSE(oidOr.isError()) << oidOr.getError();

  constexpr StringLiteral otherOIDStr = "1.2.3.4.5.1230984";
  auto otherOIDOr = ObjectID::fromString(otherOIDStr);
  ASSERT_FALSE(otherOIDOr.isError()) << otherOIDOr.getError();

  std::array<uint64_t, 6> arc = {1, 2, 3, 4, 5, 1230984};
  EXPECT_TRUE(oidOr->isModularOID());
  EXPECT_EQ(oidOr->getArc(), ArrayRef(arc));

  EXPECT_FALSE(otherOIDOr->isModularOID());
  EXPECT_EQ(otherOIDOr->getArc(), ArrayRef(arc));

  // They should not be equal, even though their raw arcs are the same.
  EXPECT_FALSE(*oidOr == *otherOIDOr);

  std::string s;
  llvm::raw_string_ostream stream(s);
  stream << *oidOr;
  EXPECT_EQ(s, oidStr);
}

TEST(ASN1, RoundtripStringOIDErrors) {
  EXPECT_TRUE(ObjectID::fromString("1.2.3.a.5").isError());
  EXPECT_TRUE(ObjectID::fromString("1..234").isError());
}

TEST(ASN1, RoundtripOIDErrors) {
  // This should be impossible to decode since we'd hit a value much larger
  // than you could hold in a uint64_t.
  std::vector<uint8_t> badOID(32, 0xff);
  EXPECT_TRUE(ObjectID::fromEncoded(badOID).isError());
}

TEST(ASN1, TrickyOID) {
  constexpr StringLiteral trickyModularOID =
      "1.3.6.1.4.1.61041.1.3.6.1.4.1.61041.123";
  auto oidOr = ObjectID::fromString(trickyModularOID);
  ASSERT_FALSE(oidOr.isError()) << oidOr.getError();

  std::array<uint64_t, 8> arc = {1, 3, 6, 1, 4, 1, 61041, 123};
  EXPECT_TRUE(oidOr->isModularOID());
  EXPECT_EQ(oidOr->getArc(), ArrayRef(arc));
}
