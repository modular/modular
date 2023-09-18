//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Cryptography/Symmetric.h"
#include "Support/Base64.h"
#include "Support/Random.h"
#include "mbedtls/asn1.h"
#include "mbedtls/hkdf.h"
#include "mbedtls/md.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

// For all these constants, we know we're using ChaCha20(Poly1305) so we take
// them from https://datatracker.ietf.org/doc/html/rfc7539#section-2.3. The
// mbedTLS implementation seems to be taken from this spec.

//===----------------------------------------------------------------------===//
// AuthenticatedEncryptionKey
//===----------------------------------------------------------------------===//

AuthenticatedEncryptionKey::~AuthenticatedEncryptionKey() {
  mbedtls_chachapoly_free(&ctx);
}

ErrorOr<AuthenticatedEncryptionKey> AuthenticatedEncryptionKey::generate() {
  SecureRandomBytesGenerator rng;

  AuthenticatedEncryptionKey key;
  if (auto err = key.setKeyBytes([&](uint8_t *buf, size_t n) {
        return rng.getRandomBytes(MutableArrayRef(buf, n));
      }))
    return err.takeError();

  return key;
}

ErrorOr<AuthenticatedEncryptionKey::EncryptedMessage>
AuthenticatedEncryptionKey::encrypt(ArrayRef<uint8_t> plaintext,
                                    ArrayRef<uint8_t> aad) {
  // Update the IV to a new value - increments the counter.
  if (auto err = updateIV())
    return err.takeError();

  // 256 GB is the max you can encrypt with an IETF-sized nonce.
  static constexpr size_t maxMessageSize = (size_t)256 * 1024 * 1024 * 1024;
  if (plaintext.size() + aad.size() > maxMessageSize) {
    return Error("too much data to be encrypted/tagged, must be less than "
                 "256 gb or broken up into multiple messages");
  }

  EncryptedMessage msg;
  // It's a stream cipher, so we only need `plaintext.size()` bytes in the
  // output.
  msg.ciphertext.resize(plaintext.size());
  // Do the encryption and generate the MAC. This will also add any AAD to the
  // MAC.
  int rc = mbedtls_chachapoly_encrypt_and_tag(
      &ctx, plaintext.size(), ivBytes, aad.data(), aad.size(), plaintext.data(),
      msg.ciphertext.data(), msg.tag);
  if (rc != 0)
    return Error("encryption failed");

  // Copy the current IV into the encrypted message.
  memcpy(msg.iv, ivBytes, ivLenBytes);

  return msg;
}

ErrorOr<std::vector<uint8_t>>
AuthenticatedEncryptionKey::decrypt(const EncryptedMessage &msg,
                                    ArrayRef<uint8_t> aad) {
  // Create a vector of bytes for the decrypted data.
  std::vector<uint8_t> data;
  // It's a stream cipher, so we only need `ciphertext.size()` bytes in the
  // output.
  data.resize(msg.ciphertext.size());
  // Do the decryption into the buffer we just allocated. This will also
  // authenticate the provided AAD bytes.
  int rc = mbedtls_chachapoly_auth_decrypt(&ctx, msg.ciphertext.size(), msg.iv,
                                           aad.data(), aad.size(), msg.tag,
                                           msg.ciphertext.data(), data.data());
  if (rc != 0)
    return Error("decryption failed");

  return data;
}

ErrorOrSuccess AuthenticatedEncryptionKey::updateIV() {
  // Update the IV to a new value by treating it as a counter. ChaCha20 and
  // friends expect new IVs for every message, so randomly generating them can
  // be extremely problematic. The IV is treated as 3 little-endian uint32_t
  // counters, so we can increment it that way. Technically, this means that the
  // first message will start with the IV 0x1, so we lose one out of (2^96 - 1)
  // messages you could theoretically encrypt with a single key, but we don't
  // expect to ever get there.
  auto *first = (uint32_t *)&ivBytes[0];
  auto *second = (uint32_t *)&ivBytes[4];
  auto *third = (uint32_t *)&ivBytes[8];

  // Increment each counter. If all the counters are maxed-out, then we have
  // a SERIOUS problem.
  if (*first < UINT32_MAX)
    *first = ++(*first);
  else if (*second < UINT32_MAX)
    *second = ++(*second);
  else if (*third < UINT32_MAX)
    *third = ++(*third);
  else
    return Error("we have somehow encrypted 2^96 - 2 messages with "
                 "a single key, and we cannot encrypt any more with this key");

  return success();
}

ErrorOrSuccess AuthenticatedEncryptionKey::setKeyBytes(
    llvm::function_ref<ErrorOrSuccess(uint8_t *, size_t)> fill) {
  // Do the fill.
  if (auto err = fill(bytes, keyLenBytes))
    return err.takeError();

  // Init the mbedTLS context.
  mbedtls_chachapoly_init(&ctx);

  // Set the key value.
  int rc = mbedtls_chachapoly_setkey(&ctx, bytes);
  if (rc != 0)
    return Error("could not set the key value");

  return success();
}

void AuthenticatedEncryptionKey::print(llvm::raw_ostream &os) && {
  // Since we know all the lengths in bytes before doing anything, this is a
  // simple thing to write out. All the lengths are a single octet, so the DER
  // form is extremely simple.

  // The tag for the top level sequence is SEQUENCE | CONSTRUCTED.
  os.write(MBEDTLS_ASN1_SEQUENCE | MBEDTLS_ASN1_CONSTRUCTED);
  os.write(serializedNumBytes);
  // Tag first (OCTET STRING).
  os.write(MBEDTLS_ASN1_OCTET_STRING);
  // Length (definite short form - keyLenBytes is less than 0x7f).
  static_assert(
      keyLenBytes < 0x7f,
      "keyLenBytes must be less than 0x7f to fit in a single length octet");
  os.write((uint8_t)keyLenBytes);
  // Bytes of the actual key.
  os.write((const char *)bytes, keyLenBytes);

  // Tag for the IV (OCTET STRING).
  os.write(MBEDTLS_ASN1_OCTET_STRING);
  // Length (definite short form - ivLenBytes is less than 0x7f).
  static_assert(
      ivLenBytes < 0x7f,
      "ivLenBytes must be less than 0x7f to fit in a single length octet");
  os.write((uint8_t)ivLenBytes);
  // Bytes of the IV.
  os.write((const char *)ivBytes, ivLenBytes);

  // All done, and the key should be dead once this returns too.
}

ErrorOr<AuthenticatedEncryptionKey>
AuthenticatedEncryptionKey::parse(ArrayRef<uint8_t> bytes) {
  // Too many bytes is OK, we just won't consume them all.
  if (bytes.size() < serializedNumBytes + 2) {
    return Error("too few bytes provided to parse an "
                 "AuthenticatedEncryptionKey, need at least " +
                 Twine(serializedNumBytes + 2));
  }

  // These are the prefixes we expect on the byte sequence.
  constexpr std::array keyPrefix = {
      (uint8_t)(MBEDTLS_ASN1_SEQUENCE | MBEDTLS_ASN1_CONSTRUCTED),
      (uint8_t)serializedNumBytes, (uint8_t)MBEDTLS_ASN1_OCTET_STRING,
      (uint8_t)keyLenBytes};
  if (bytes.take_front(keyPrefix.size()) != ArrayRef(keyPrefix))
    return Error("invalid ASN.1 encoding: mismatched prefix");

  bytes = bytes.drop_front(keyPrefix.size());

  // Pull out the key bytes and store them in the output object.
  AuthenticatedEncryptionKey key;
  if (auto err = key.setKeyBytes([&](uint8_t *buf, size_t n) {
        assert(n == keyLenBytes && "expected keyLenBytes in `buf`");
        memcpy(buf, bytes.begin(), keyLenBytes);
        return success();
      }))
    return err.takeError();

  bytes = bytes.drop_front(keyLenBytes);

  // Check for the prefix on the IV.
  constexpr std::array ivPrefix = {(uint8_t)MBEDTLS_ASN1_OCTET_STRING,
                                   (uint8_t)ivLenBytes};
  if (bytes.take_front(ivPrefix.size()) != ArrayRef(ivPrefix))
    return Error("invalid ASN.1 encoding: mismatched IV encoding");

  bytes = bytes.drop_front(ivPrefix.size());

  // Copy the bytes of the IV into the output object.
  memcpy(key.ivBytes, bytes.begin(), ivLenBytes);

  // All done, return the output object.
  return key;
}

ErrorOr<std::string>
AuthenticatedEncryptionKey::getKeyID(ArrayRef<uint8_t> salt) const {
  const auto *mdInfo = mbedtls_md_info_from_type(MBEDTLS_MD_SHA512);

  // Set up the output buffer.
  std::string out;
  out.resize(mbedtls_md_get_size(mdInfo));

  // Use the HKDF to generate new key bytes from the existing key. This
  // incorporates a salt as well as multiple rounds of hashing, while being
  // deterministic. NIST allows this as a PRF that can be used for generating
  // sub-keys from existing keys.
  int rc = mbedtls_hkdf(mdInfo, salt.data(), salt.size(), bytes, keyLenBytes,
                        nullptr, 0, (uint8_t *)out.data(), out.size());
  if (rc != 0)
    return Error("could not compute the key ID for the provided key");

  // Encode the output as URL-safe base64.
  return encodeURLSafeBase64(out);
}
