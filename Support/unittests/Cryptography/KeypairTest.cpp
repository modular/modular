//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Cryptography/Keypair.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ADT/ArrayRef.h"

#include "gtest/gtest.h"

using namespace M;

TEST(TestKeypair, RoundtripSignature) {
  constexpr llvm::StringLiteral dataToSign = "hello, world";
  ErrorOr<TempFile> keyFileOr = TempFile::create("clientKey.XXXXXX");
  ASSERT_FALSE(keyFileOr.isError()) << keyFileOr.getError();
  keyFileOr->close();
  keyFileOr->remove();

  auto keysOr = Keypair::generate(keyFileOr->getPath());
  ASSERT_FALSE(keysOr.isError()) << keysOr.getError();

  auto sigOr = keysOr->sign(dataToSign);
  ASSERT_FALSE(sigOr.isError()) << sigOr.getError();

  auto err = keysOr->validateSignature(dataToSign, *sigOr);
  ASSERT_FALSE(err.isError()) << err.getError();
}
