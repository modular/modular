//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/JSON.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "llvm/Support/JSON.h"
#include "gtest/gtest.h"

using namespace M;

std::string convertToString(const llvm::Error &err) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << err;
  return canonical;
}

/// Check the test JSON from RFC8785. This checks that we have a reasonable sort
/// order. Note that the last two differ from the RFC because we sort in UTF-8.
TEST(JSONTest, CheckRFCTestVector) {
  llvm::StringRef testVector =
      "{\"\\u20ac\": \"Euro Sign\", \"\\r\": \"Carriage Return\", \"\\ufb33\": "
      "\"Hebrew Letter Dalet With Dagesh\", \"1\": \"One\", "
      "\"\\ud83d\\ude00\": \"Emoji: Grinning Face\", \"\\u0080\": \"Control\", "
      "\"\\u00f6\": \"Latin Small Letter O With Diaeresis\"}";
  llvm::StringRef correct =
      "{\"\r\":\"Carriage "
      "Return\",\"1\":\"One\",\"\xc2\x80\":\"Control\",\"\xc3\xb6\":\"Latin "
      "Small "
      "Letter O With Diaeresis\",\"\xe2\x82\xAC\":\"Euro "
      "Sign\",\"\xef\xac\xb3\":\"Hebrew "
      "Letter Dalet With Dagesh\",\"\xf0\x9f\x98\x80\":\"Emoji: Grinning "
      "Face\"}";
  llvm::Expected<llvm::json::Value> testJSON = llvm::json::parse(testVector);
  ASSERT_TRUE(bool(testJSON)) << convertToString(testJSON.takeError());

  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  serializeCanonicalJSON(&*testJSON, stream);
  ASSERT_EQ(canonical, correct);
}

/// This test steals an example from the TUF spec that is not sorted, and just
/// checks that it would be sorted (and escaped) correctly.
TEST(JSONTest, TestNested) {
  llvm::StringRef testVector = R"({
"signed": {
    "_type": "root",
    "spec_version": "1.0.0",
    "consistent_snapshot": false,
    "expires": "2030-01-01T00:00:00Z",
    "keys": {
      "1bf1c6e3cdd3d3a8420b19199e27511999850f4b376c4547b2f32fba7e80fca3": {
        "keytype": "ed25519",
        "scheme": "ed25519",
        "keyval": {
          "public": "72378e5bc588793e58f81c8533da64a2e8f1565c1fcc7f253496394ffc52542c"
        }
      },
      "135c2f50e57ff11e744d234a62cebad8c38daf399604a7655661cc9199c69164": {
        "keytype": "ed25519",
        "scheme": "ed25519",
        "keyval": {
          "public": "68ead6e54a43f8f36f9717b10669d1ef0ebb38cee6b05317669341309f1069cb"
        }
      }
    }
  }
})";
  // clang-format off
  llvm::StringRef correct =
      "{\"signed\":{"
      "\"_type\":\"root\","
      "\"consistent_snapshot\":false,"
      "\"expires\":\"2030-01-01T00:00:00Z\","
      "\"keys\":{"
        "\"135c2f50e57ff11e744d234a62cebad8c38daf399604a7655661cc9199c69164\":{"
          "\"keytype\":\"ed25519\","
          "\"keyval\":{"
            "\"public\":"
              "\"68ead6e54a43f8f36f9717b10669d1ef0ebb38cee6b05317669341309f1069cb\""
          "},"
          "\"scheme\":\"ed25519\""
        "},"
        "\"1bf1c6e3cdd3d3a8420b19199e27511999850f4b376c4547b2f32fba7e80fca3\":{"
          "\"keytype\":\"ed25519\","
          "\"keyval\":{"
            "\"public\":"
              "\"72378e5bc588793e58f81c8533da64a2e8f1565c1fcc7f253496394ffc52542c\""
          "},"
          "\"scheme\":\"ed25519\""
        "}"
      "},"
      "\"spec_version\":\"1.0.0\"}}";
  // clang-format on
  llvm::Expected<llvm::json::Value> testJSON = llvm::json::parse(testVector);
  ASSERT_TRUE(bool(testJSON)) << convertToString(testJSON.takeError());

  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  serializeCanonicalJSON(&*testJSON, stream);
  ASSERT_EQ(canonical, correct);
}

TEST(JSONTest, TestJSONControlChars) {
  llvm::StringRef testVector =
      "{\"hello\":\"newline\\n\", \"before\":\"hasbackspace\\b\"}";
  llvm::StringRef correct =
      "{\"before\":\"hasbackspace\b\",\"hello\":\"newline\n\"}";
  llvm::Expected<llvm::json::Value> testJSON = llvm::json::parse(testVector);
  ASSERT_TRUE(bool(testJSON)) << convertToString(testJSON.takeError());

  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  serializeCanonicalJSON(&*testJSON, stream);
  ASSERT_EQ(canonical, correct);
}
