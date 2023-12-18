//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Cryptography/Symmetric.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

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

/// This test uses a test vector from the RFC to ensure our implementation is
/// compliant.
TEST(TestSymmetricCrypto, RFCTestVector) {
  std::array<uint8_t, AuthenticatedEncryptionKey::keyLenBytes> keybytes = {
      0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a,
      0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95,
      0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f};
  auto keyOr = AuthenticatedEncryptionKey::unsafeFromRawBytes(keybytes);
  ASSERT_FALSE(keyOr.isError()) << keyOr.getError();

  AuthenticatedEncryptionKey::EncryptedMessage msg;
  msg.iv = {0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
            0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
            0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57};
  msg.tag = {0xc0, 0x87, 0x59, 0x24, 0xc1, 0xc7, 0x98, 0x79,
             0x47, 0xde, 0xaf, 0xd8, 0x78, 0x0a, 0xcf, 0x49};
  msg.ciphertext = {
      0xbd, 0x6d, 0x17, 0x9d, 0x3e, 0x83, 0xd4, 0x3b, 0x95, 0x76, 0x57, 0x94,
      0x93, 0xc0, 0xe9, 0x39, 0x57, 0x2a, 0x17, 0x00, 0x25, 0x2b, 0xfa, 0xcc,
      0xbe, 0xd2, 0x90, 0x2c, 0x21, 0x39, 0x6c, 0xbb, 0x73, 0x1c, 0x7f, 0x1b,
      0x0b, 0x4a, 0xa6, 0x44, 0x0b, 0xf3, 0xa8, 0x2f, 0x4e, 0xda, 0x7e, 0x39,
      0xae, 0x64, 0xc6, 0x70, 0x8c, 0x54, 0xc2, 0x16, 0xcb, 0x96, 0xb7, 0x2e,
      0x12, 0x13, 0xb4, 0x52, 0x2f, 0x8c, 0x9b, 0xa4, 0x0d, 0xb5, 0xd9, 0x45,
      0xb1, 0x1b, 0x69, 0xb9, 0x82, 0xc1, 0xbb, 0x9e, 0x3f, 0x3f, 0xac, 0x2b,
      0xc3, 0x69, 0x48, 0x8f, 0x76, 0xb2, 0x38, 0x35, 0x65, 0xd3, 0xff, 0xf9,
      0x21, 0xf9, 0x66, 0x4c, 0x97, 0x63, 0x7d, 0xa9, 0x76, 0x88, 0x12, 0xf6,
      0x15, 0xc6, 0x8b, 0x13, 0xb5, 0x2e};

  std::array<uint8_t, 12> aad = {0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1,
                                 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7};
  auto decryptedOr = keyOr->decrypt(msg, ArrayRef(aad));
  ASSERT_FALSE(decryptedOr.isError()) << decryptedOr.getError();

  std::vector<uint8_t> correct = {
      0x4c, 0x61, 0x64, 0x69, 0x65, 0x73, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x47,
      0x65, 0x6e, 0x74, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x20, 0x6f, 0x66, 0x20,
      0x74, 0x68, 0x65, 0x20, 0x63, 0x6c, 0x61, 0x73, 0x73, 0x20, 0x6f, 0x66,
      0x20, 0x27, 0x39, 0x39, 0x3a, 0x20, 0x49, 0x66, 0x20, 0x49, 0x20, 0x63,
      0x6f, 0x75, 0x6c, 0x64, 0x20, 0x6f, 0x66, 0x66, 0x65, 0x72, 0x20, 0x79,
      0x6f, 0x75, 0x20, 0x6f, 0x6e, 0x6c, 0x79, 0x20, 0x6f, 0x6e, 0x65, 0x20,
      0x74, 0x69, 0x70, 0x20, 0x66, 0x6f, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20,
      0x66, 0x75, 0x74, 0x75, 0x72, 0x65, 0x2c, 0x20, 0x73, 0x75, 0x6e, 0x73,
      0x63, 0x72, 0x65, 0x65, 0x6e, 0x20, 0x77, 0x6f, 0x75, 0x6c, 0x64, 0x20,
      0x62, 0x65, 0x20, 0x69, 0x74, 0x2e};
  EXPECT_EQ(ArrayRef(*decryptedOr), ArrayRef(correct));

  // We can't check the inverse direction because we don't provide a method to
  // pass in a static IV on encryption for safety and to avoid mis-use, but as
  // long as the decryption succeeds then due to the fact that ChaCha is
  // symmetric and the setup is exactly the same in both cases, this should
  // work.
}
