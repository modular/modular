//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Cryptography/Keypair.h"
#include "llvm/ADT/ArrayRef.h"

#include "gtest/gtest.h"

using namespace M;

TEST(TestKeypair, RoundtripSignature) {
  constexpr llvm::StringLiteral dataToSign = "hello, world";
  auto keysOr = Keypair::generate();
  EXPECT_FALSE(keysOr.isError()) << keysOr.getError();

  auto sigOr = keysOr->sign(dataToSign);
  EXPECT_FALSE(sigOr.isError()) << sigOr.getError();

  auto err = keysOr->validateSignature(dataToSign, *sigOr);
  EXPECT_FALSE(err.isError()) << err.getError();
}
