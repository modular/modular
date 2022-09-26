//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HMAC.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

using namespace M;

/// Compute the HMAC of some test vectors provided by the wikipedia article on
/// HMAC.
TEST(HMACTest, ComputeHmacSHA256ForShortKey) {
  auto output =
      hmacSHA256("The quick brown fox jumps over the lazy dog", "key");
  auto hexStr = llvm::toHex(output);
  EXPECT_STRCASEEQ(
      hexStr.data(),
      "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8");
}

TEST(HMACTest, ComputeHmacSHA256ForLongKey) {
  auto output =
      hmacSHA256("message", "The quick brown fox jumps over the lazy dogThe "
                            "quick brown fox jumps over the lazy dog");
  auto hexStr = llvm::toHex(output);
  EXPECT_STRCASEEQ(
      hexStr.data(),
      "5597b93a2843078cbb0c920ae41dfe20f1685e10c67e423c11ab91adfc319d12");
}
