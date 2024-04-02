//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Cryptography/Keypair.h"
#include "Support/Buffer.h"
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

TEST(TestKeypair, ReadEC256Buffer) {
  // PKCS8 formatted EC 256 key
  std::string pem =
      "-----BEGIN PRIVATE KEY-----\n"
      "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg1FQou3HTZ1SKk8Zx\n"
      "vdKO1Iv3COYy79CPIiBQl02sgFWhRANCAARQDTUmbajiKQ2ttWsPe0Arl84YZ6HD\n"
      "hqzdk3pLgTKSsWuUH4qs4wbX61P5SVVeKSW7vKPAhyp2dPx6mtgWCbXU\n"
      "-----END PRIVATE KEY-----\n";

  // Even though `pem` is null-terminated, BufferRef drops the null somewhere
  // causing Keypair::open() to fail.
  {
    BufferRef buf = Buffer::get(pem);
    auto keypairOr = Keypair::open(buf, false);
    ASSERT_TRUE(keypairOr.isError());
  }

  // Add a null to enable reading in-memory keys
  {
    BufferRef buf = Buffer::get(pem);
    auto keypairOr = Keypair::open(buf, true);
    ASSERT_FALSE(keypairOr.isError());
  }
}
