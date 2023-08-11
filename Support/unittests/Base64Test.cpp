//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Base64.h"

#include "gtest/gtest.h"

using namespace M;

/// Just check that a weird string can roundtrip through URL-safe base64.
TEST(Base64, Roundtrip) {
  const llvm::StringLiteral str =
      "This is a string, it has \n and \t in it, maybe some \\ and some \x0A";

  auto encoded = encodeURLSafeBase64(str);
  auto decodedOr = decodeURLSafeBase64(encoded);
  EXPECT_FALSE(decodedOr.isError()) << decodedOr.takeError();
  EXPECT_EQ(str, *decodedOr);
}
