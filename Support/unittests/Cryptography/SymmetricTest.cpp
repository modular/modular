//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Cryptography/Symmetric.h"
#include "llvm/ADT/ArrayRef.h"

#include "gtest/gtest.h"

using namespace M;

/// Turn a StringRef into an ArrayRef<uint8_t> - useful for adapting between
/// strings and byte arrays.
static ArrayRef<uint8_t> toArrayRef(StringRef str) {
  return {(const uint8_t *)str.data(), str.size()};
}

TEST(TestSymmetricCrypto, RoundtripCrypto) {
  auto keyOr = AuthenticatedEncryptionKey::generate();
  ASSERT_FALSE(keyOr.isError()) << keyOr.getError();

  constexpr StringLiteral plaintext = "hello this is a very secret string";
  constexpr StringLiteral aad =
      "this is less secret, but still shouldn't be tampered with";

  auto encryptedOr = keyOr->encrypt(toArrayRef(plaintext), toArrayRef(aad));
  ASSERT_FALSE(encryptedOr.isError()) << encryptedOr.getError();

  // The ciphertext must not match the plaintext, obviously.
  EXPECT_NE(ArrayRef(encryptedOr->ciphertext), toArrayRef(plaintext));

  auto decryptedOr = keyOr->decrypt(*encryptedOr, toArrayRef(aad));
  ASSERT_FALSE(decryptedOr.isError()) << decryptedOr.getError();

  // Expect the plaintext to be exactly equal to the decrypted info.
  EXPECT_EQ(ArrayRef(*decryptedOr), toArrayRef(plaintext));

  // Encrypt again.
  auto anotherEncryptedOr =
      keyOr->encrypt(toArrayRef(plaintext), toArrayRef(aad));
  ASSERT_FALSE(anotherEncryptedOr.isError()) << anotherEncryptedOr.getError();

  // These must differ - they have different IVs (or they should)!
  EXPECT_NE(ArrayRef(encryptedOr->ciphertext),
            ArrayRef(anotherEncryptedOr->ciphertext));
}

TEST(TestSymmetricCrypto, RoundtripCryptoAndKey) {
  auto keyOr = AuthenticatedEncryptionKey::generate();
  ASSERT_FALSE(keyOr.isError()) << keyOr.getError();

  constexpr StringLiteral plaintext = "hello this is a very secret string";
  constexpr StringLiteral aad =
      "this is less secret, but still shouldn't be tampered with";

  auto encryptedOr = keyOr->encrypt(toArrayRef(plaintext), toArrayRef(aad));
  ASSERT_FALSE(encryptedOr.isError()) << encryptedOr.getError();

  // The ciphertext must not match the plaintext, obviously.
  EXPECT_NE(ArrayRef(encryptedOr->ciphertext), toArrayRef(plaintext));

  // Print and relinquish the current key.
  std::string keyStr;
  llvm::raw_string_ostream stream(keyStr);
  std::move(*keyOr).print(stream);

  // Parse a new key and ensure we can still decrypt the message.
  auto newKeyOr = AuthenticatedEncryptionKey::parse(toArrayRef(keyStr));
  ASSERT_FALSE(newKeyOr.isError()) << newKeyOr.getError();

  auto decryptedOr = newKeyOr->decrypt(*encryptedOr, toArrayRef(aad));
  ASSERT_FALSE(decryptedOr.isError()) << decryptedOr.getError();

  // Expect the plaintext to be exactly equal to the decrypted info.
  EXPECT_EQ(ArrayRef(*decryptedOr), toArrayRef(plaintext));

  // Encrypt again with the new key.
  auto anotherEncryptedOr =
      newKeyOr->encrypt(toArrayRef(plaintext), toArrayRef(aad));
  ASSERT_FALSE(anotherEncryptedOr.isError()) << anotherEncryptedOr.getError();

  // These must differ - they have different IVs (or they should)!
  EXPECT_NE(ArrayRef(encryptedOr->ciphertext),
            ArrayRef(anotherEncryptedOr->ciphertext));
}

TEST(TestSymmetricCrypto, RoundtripNoAAD) {
  auto keyOr = AuthenticatedEncryptionKey::generate();
  ASSERT_FALSE(keyOr.isError()) << keyOr.getError();

  constexpr StringLiteral plaintext = "hello this is a very secret string";

  auto encryptedOr = keyOr->encrypt(toArrayRef(plaintext), {});
  ASSERT_FALSE(encryptedOr.isError()) << encryptedOr.getError();

  // The ciphertext must not match the plaintext, obviously.
  EXPECT_NE(ArrayRef(encryptedOr->ciphertext), toArrayRef(plaintext));

  auto decryptedOr = keyOr->decrypt(*encryptedOr, {});
  ASSERT_FALSE(decryptedOr.isError()) << decryptedOr.getError();

  // Expect the plaintext to be exactly equal to the decrypted info.
  EXPECT_EQ(ArrayRef(*decryptedOr), toArrayRef(plaintext));

  // Encrypt again.
  auto anotherEncryptedOr = keyOr->encrypt(toArrayRef(plaintext), {});
  ASSERT_FALSE(anotherEncryptedOr.isError()) << anotherEncryptedOr.getError();

  // These must differ - they have different IVs (or they should)!
  EXPECT_NE(ArrayRef(encryptedOr->ciphertext),
            ArrayRef(anotherEncryptedOr->ciphertext));
}

TEST(TestSymmetricCrypto, KeyID) {
  auto keyOr = AuthenticatedEncryptionKey::generate();
  ASSERT_FALSE(keyOr.isError()) << keyOr.getError();

  auto kidOr = keyOr->getKeyID({});
  ASSERT_FALSE(kidOr.isError()) << kidOr.getError();

  auto kidWithSaltOr = keyOr->getKeyID({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
  ASSERT_FALSE(kidWithSaltOr.isError()) << kidWithSaltOr.getError();

  auto anotherKidWithSaltOr =
      keyOr->getKeyID({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
  ASSERT_FALSE(anotherKidWithSaltOr.isError())
      << anotherKidWithSaltOr.getError();

  // Just check that the key IDs we expect to be consistent are, and the others
  // are not.
  EXPECT_NE(*kidOr, *kidWithSaltOr);
  EXPECT_EQ(*kidWithSaltOr, *anotherKidWithSaltOr);
}

TEST(TestSymmetricCrypto, NoAADProvidedOnDecrypt) {
  auto keyOr = AuthenticatedEncryptionKey::generate();
  ASSERT_FALSE(keyOr.isError()) << keyOr.getError();

  constexpr StringLiteral plaintext = "hello this is a very secret string";
  constexpr StringLiteral aad =
      "this is less secret, but still shouldn't be tampered with";

  auto encryptedOr = keyOr->encrypt(toArrayRef(plaintext), toArrayRef(aad));
  ASSERT_FALSE(encryptedOr.isError()) << encryptedOr.getError();

  // The ciphertext must not match the plaintext, obviously.
  EXPECT_NE(ArrayRef(encryptedOr->ciphertext), toArrayRef(plaintext));

  // This should not work, the AAD has not been provided.
  auto decryptedOr = keyOr->decrypt(*encryptedOr, {});
  EXPECT_TRUE(decryptedOr.isError());
}

TEST(TestSymmetricCrypto, BadDER) {
  auto bad = AuthenticatedEncryptionKey::parse(toArrayRef("hello"));
  EXPECT_TRUE(bad.isError());
}
