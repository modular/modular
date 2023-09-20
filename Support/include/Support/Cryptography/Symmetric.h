//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CRYPTOGRAPHY_SYMMETRIC_H
#define SUPPORT_CRYPTOGRAPHY_SYMMETRIC_H

#include "Support/ErrorOr.h"
#include "mbedtls/chachapoly.h"
#include "llvm/ADT/FunctionExtras.h"
#include <cstdint>

namespace M {
/// This class provides authenticated symmetric encryption. What that means is
/// that you can encrypt a message and provide a MAC as a single construction.
/// This is generally more secure than rolling a custom construction like
/// encrypting and then adding an HMAC. This also provides the ability to
/// include additional data in that MAC - you can provide additional
/// (unencrypted) data alongside the data to be encrypted whose integrity will
/// also be covered by the MAC.
///
/// The threat model this is designed to address is one where the protected data
/// is being sent between trusted peers. For that reason, there is little effort
/// paid to (for example) prevent dumping the key bytes in a core dump, or
/// preventing them from being paged to disk. Changing this is possible as the
/// threat model evolves.
///
/// Note that though the algorithm technically supports using this key as a
/// stream cipher, we disallow that from *this* API in favor of providing a
/// separate stream encryption API.
///
/// The recommended usage of an AuthenticatedEncryptionKey with a key store is
/// as follows:
///
/// ```c++
///  auto keyOr = AuthenticatedEncryptionKey::generate();
///  if (keyOr.isError())
///    // Handle the error
///
///  auto encryptedOr = keyOr->encrypt(data, aad);
///  // ... encrypt multiple (up to 2^96 - 2) messages with this key
///
///  // Passing a salt is optional - it is recommended to protect against
///  // precomputed rainbow tables, though.
///  std::string keyID = keyOr->getKeyID();
///
///  std::string str;
///  llvm::raw_string_ostream stream(str);
///  // Dump the key to the provided stream.
///  std::move(*keyOr).print(stream);
///
///  // Store the key in the map for future message decryption.
///  keyStore[keyID] = str;
///
///  // ... elsewhere ...
///
///  // Lookup the key string with the key ID
///  ArrayRef<uint8_t> keyData = toArrayRef(keyStore[keyID]);
///  auto parsedKeyOr = AuthenticatedEncryptionKey::parse(keyData);
///  if (parsedKeyOr.isError())
///    // Handle the error
///
///  // Now perform the decryption.
///  auto decryptedOr = parsedKeyOr->decrypt(*encryptedOr, aad);
/// ```
///
/// This example is careful to avoid key re-use in potentially unsafe contexts
/// by not using the key from keyStore for encrypting new messages.
class AuthenticatedEncryptionKey {
public:
  ~AuthenticatedEncryptionKey();

  /// You can't copy AuthenticatedEncryptionKeys, but you can move them.
  AuthenticatedEncryptionKey(const AuthenticatedEncryptionKey &other) = delete;
  AuthenticatedEncryptionKey(AuthenticatedEncryptionKey &&other) = default;

  /// Generate a new encryption key. This uses uniformly random bytes pulled
  /// from the system CSPRNG.
  static ErrorOr<AuthenticatedEncryptionKey> generate();

  /// Static constants that describe various important buffer lengths.
  static constexpr size_t ivLenBytes = 24;
  static constexpr size_t keyLenBytes = 32;
  static constexpr size_t tagLenBytes = 16;

  /// Create a new key from raw bytes. This is useful for testing our
  /// implementation, but is generally not user-friendly or necessarily even
  /// secure. Most users should use the parse/generate methods to obtain keys.
  static ErrorOr<AuthenticatedEncryptionKey>
  unsafeFromRawBytes(std::array<uint8_t, keyLenBytes> keyBytes);

  /// This layout of an encrypted message contains all the information needed to
  /// decrypt the message.
  struct EncryptedMessage {
    /// This is the ciphertext - i.e. the actually encrypted bytes.
    std::vector<uint8_t> ciphertext;

    /// This is the authentication code provided for the message.
    std::array<uint8_t, tagLenBytes> tag = {};

    /// This is the initialization vector (IV or nonce) used to encrypt this
    /// message. The IV is chosen uniquely per (key, message) combination, but
    /// it is not considered a secret.
    std::array<uint8_t, ivLenBytes> iv = {};
  };

  /// Encrypt a message. Optionally, add additional data to be authenticated
  /// along with the encrypted data. The tag stored in the returned
  /// EncryptedMessage will also cover any AAD provided here. Note that for
  /// decryption to succeed, the AAD must also be provided to that call.
  ErrorOr<EncryptedMessage> encrypt(ArrayRef<uint8_t> plaintext,
                                    ArrayRef<uint8_t> aad);

  /// Decrypt the message into a new buffer. Note that if any AAD was provided
  /// to the encrypt call, the same AAD must be provided to the decrypt call
  /// otherwise the tag will not match and the decryption will be rejected.
  ErrorOr<std::vector<uint8_t>> decrypt(const EncryptedMessage &msg,
                                        ArrayRef<uint8_t> aad);

  /// Write the key and associated state, and return the buffer. The key's full
  /// state can be reconstructed by parsing the bytes written to this stream.
  /// Using this method means the user must relinquish the key object because of
  /// the potential danger of re-using a nonce, which could compromise the key
  /// itself.
  ///
  /// The encoded format is a DER format:
  ///
  ///   AuthenticatedEncryptionKey ::= SEQUENCE {
  ///        keyBytes     OCTET STRING,
  ///   }
  ///
  /// This DER format can be PEM (Base64) encoded as desired, but take care to
  /// use a constant-time conversion such as one provided by mbedTLS.
  ///
  /// This encoding is *NOT* encryption, and so the encoded data *MUST* be kept
  /// secret.
  void print(llvm::raw_ostream &os) &&;

  /// Parse this key and associated state. The inverse of `print`. This parses
  /// the full state from the provided ArrayRef. The caller may provide extra
  /// bytes if desired, this function will simply consume the bytes from the
  /// front of the buffer. The caller should wipe the bytes provided to `parse`
  /// once parsing has completed.
  ///
  /// The user *MUST NOT* `parse` the same buffer multiple times, as such an
  /// action could result in IV re-use and has the potential for key compromise.
  static ErrorOr<AuthenticatedEncryptionKey> parse(ArrayRef<uint8_t> bytes);

  /// Get a canonical ID for this key that does not leak the bytes of the key,
  /// but does uniquely identify this key. The key ID does not take into account
  /// the IV, and so must only be used to identify an instance of the key for
  /// decryption, where the IV is stored on the EncryptedMessage itself. Keys
  /// looked-up and parsed using this key ID may have been parsed before, and
  /// should not be used for further encryption!
  ///
  /// The user can provide a salt, which we can use to further obscure the raw
  /// key bytes and protect against rainbow-table-type attacks.
  ///
  /// The output of this function is URL-safe Base64 encoded and safe to
  /// read/write as an identifier for this key.
  ErrorOr<std::string> getKeyID(ArrayRef<uint8_t> salt) const;

private:
  /// Private constructor so we have tight control over how these keys are
  /// generated.
  AuthenticatedEncryptionKey() = default;

  /// Set the actual bytes for this key. The `bytes` field of the particular
  /// instance this is the first argument, and the second is keyLenBytes (i.e.
  /// the length of the bytes field in bytes).
  ErrorOrSuccess
  setKeyBytes(llvm::function_ref<ErrorOrSuccess(uint8_t *, size_t)> fill);

  /// The bytes of the key itself. This is not an std::array because it needs to
  /// be passed into C APIs and this form is cleaner.
  uint8_t bytes[keyLenBytes] = {};

  /// The number of bytes inside the sequence is:
  ///   tag (1 octet) + length (1 octet; constant keyLenBytes) + keyLenBytes
  static constexpr size_t serializedNumBytes = 1 + 1 + keyLenBytes;
  static_assert(serializedNumBytes <= 0x7f,
                "total key length in bytes must be less than 0x7f to fit in a "
                "single length octet when encoded");

  /// This is the mbedTLS context object that provides ChaCha20-Poly1305
  /// support.
  mbedtls_chachapoly_context ctx = {};
};
} // namespace M

#endif // SUPPORT_CRYPTOGRAPHY_SYMMETRIC_H
