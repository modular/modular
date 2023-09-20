//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Cryptography/Symmetric.h"
#include "Support/Base64.h"
#include "Support/Random.h"
#include "mbedtls/asn1.h"
#include "mbedtls/chacha20.h"
#include "mbedtls/hkdf.h"
#include "mbedtls/md.h"
#include "mbedtls/platform_util.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

// For all these constants, we know we're using ChaCha20(Poly1305) so we take
// them from https://datatracker.ietf.org/doc/html/rfc7539#section-2.3. The
// mbedTLS implementation seems to be taken from this spec.

//===----------------------------------------------------------------------===//
// XChaCha20
//===----------------------------------------------------------------------===//

/// The code below is copied from mbedTLS source.

#if !defined(__BYTE_ORDER__)
static const uint16_t mbedtls_byte_order_detector = {0x100};
#define MBEDTLS_IS_BIG_ENDIAN                                                  \
  (*((unsigned char *)(&mbedtls_byte_order_detector)) == 0x01)
#else
#define MBEDTLS_IS_BIG_ENDIAN ((__BYTE_ORDER__) == (__ORDER_BIG_ENDIAN__))
#endif /* !defined(__BYTE_ORDER__) */

/**
 * Get the unsigned 32 bits integer corresponding to four bytes in
 * little-endian order (LSB first).
 *
 * \param   data    Base address of the memory to get the four bytes from.
 * \param   offset  Offset from \p data of the first and least significant
 *                  byte of the four bytes to build the 32 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT32_LE(data, offset)                                    \
  ((MBEDTLS_IS_BIG_ENDIAN)                                                     \
       ? MBEDTLS_BSWAP32(mbedtls_get_unaligned_uint32((data) + (offset)))      \
       : mbedtls_get_unaligned_uint32((data) + (offset)))

/**
 * Put in memory a 32 bits unsigned integer in little-endian order.
 *
 * \param   n       32 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 32
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the least significant
 *                  byte of the 32 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT32_LE(n, data, offset)                                 \
  {                                                                            \
    if (MBEDTLS_IS_BIG_ENDIAN) {                                               \
      mbedtls_put_unaligned_uint32((data) + (offset),                          \
                                   MBEDTLS_BSWAP32((uint32_t)(n)));            \
    } else {                                                                   \
      mbedtls_put_unaligned_uint32((data) + (offset), ((uint32_t)(n)));        \
    }                                                                          \
  }

#if !defined(MBEDTLS_BSWAP32)
static inline uint32_t mbedtls_bswap32(uint32_t x) {
  return (x & 0x000000ff) << 24 | (x & 0x0000ff00) << 8 |
         (x & 0x00ff0000) >> 8 | (x & 0xff000000) >> 24;
}
#define MBEDTLS_BSWAP32 mbedtls_bswap32
#endif /* !defined(MBEDTLS_BSWAP32) */

/**
 * Read the unsigned 32 bits integer from the given address, which need not
 * be aligned.
 *
 * \param   p pointer to 4 bytes of data
 * \return  Data at the given address
 */
static inline uint32_t mbedtls_get_unaligned_uint32(const void *p) {
  uint32_t r;
  memcpy(&r, p, sizeof(r));
  return r;
}

/**
 * Write the unsigned 32 bits integer to the given address, which need not
 * be aligned.
 *
 * \param   p pointer to 4 bytes of data
 * \param   x data to write
 */
static inline void mbedtls_put_unaligned_uint32(void *p, uint32_t x) {
  memcpy(p, &x, sizeof(x));
}

#define ROTL32(value, amount)                                                  \
  ((uint32_t)((value) << (amount)) | ((value) >> (32 - (amount))))

#define CHACHA20_CTR_INDEX (12U)

#define CHACHA20_BLOCK_SIZE_BYTES (4U * 16U)

/**
 * \brief           ChaCha20 quarter round operation.
 *
 *                  The quarter round is defined as follows (from RFC 7539):
 *                      1.  a += b; d ^= a; d <<<= 16;
 *                      2.  c += d; b ^= c; b <<<= 12;
 *                      3.  a += b; d ^= a; d <<<= 8;
 *                      4.  c += d; b ^= c; b <<<= 7;
 *
 * \param state     ChaCha20 state to modify.
 * \param a         The index of 'a' in the state.
 * \param b         The index of 'b' in the state.
 * \param c         The index of 'c' in the state.
 * \param d         The index of 'd' in the state.
 */
static void chacha20_quarter_round(uint32_t state[16], size_t a, size_t b,
                                   size_t c, size_t d) {
  /* a += b; d ^= a; d <<<= 16; */
  state[a] += state[b];
  state[d] ^= state[a];
  state[d] = ROTL32(state[d], 16);

  /* c += d; b ^= c; b <<<= 12 */
  state[c] += state[d];
  state[b] ^= state[c];
  state[b] = ROTL32(state[b], 12);

  /* a += b; d ^= a; d <<<= 8; */
  state[a] += state[b];
  state[d] ^= state[a];
  state[d] = ROTL32(state[d], 8);

  /* c += d; b ^= c; b <<<= 7; */
  state[c] += state[d];
  state[b] ^= state[c];
  state[b] = ROTL32(state[b], 7);
}

/**
 * \brief           Perform the ChaCha20 inner block operation.
 *
 *                  This function performs two rounds: the column round and the
 *                  diagonal round.
 *
 * \param state     The ChaCha20 state to update.
 */
static void chacha20_inner_block(uint32_t state[16]) {
  chacha20_quarter_round(state, 0, 4, 8, 12);
  chacha20_quarter_round(state, 1, 5, 9, 13);
  chacha20_quarter_round(state, 2, 6, 10, 14);
  chacha20_quarter_round(state, 3, 7, 11, 15);

  chacha20_quarter_round(state, 0, 5, 10, 15);
  chacha20_quarter_round(state, 1, 6, 11, 12);
  chacha20_quarter_round(state, 2, 7, 8, 13);
  chacha20_quarter_round(state, 3, 4, 9, 14);
}

//===----------------------------------------------------------------------===//
// doXChaCha20Setup
//===----------------------------------------------------------------------===//

/// This implements setup for XChaCha20 according to
/// https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha#section-2.2.
/// This call requires the key be set on the context via
/// mbedtls_chacha20_setkey() or mbedtls_chachapoly_setkey(). This will set up
/// the subkey in the ChaCha20 context and return the IETF-sized nonce.
using IETFNonceArray =
    std::array<uint8_t, AuthenticatedEncryptionKey::ivLenBytes>;
static ErrorOr<IETFNonceArray>
doXChaCha20Setup(mbedtls_chacha20_context *ctx,
                 const std::array<uint8_t, 24> &nonce) {

  using KeyArray = std::array<uint8_t, AuthenticatedEncryptionKey::keyLenBytes>;
  KeyArray subkey = {};
  IETFNonceArray ietfNonce = {};

  // HChaCha20 uses the first 32 bits of the nonce as the counter, and the next
  // 96 from the front of the provided nonce.
  int rc = mbedtls_chacha20_starts(ctx, nonce.data() + 4,
                                   MBEDTLS_GET_UINT32_LE(nonce.data(), 0));
  if (rc != 0)
    return Error("could not set up HChaCha20: " + Twine::utohexstr(rc));

  // Run 20 rounds (chacha20_inner_block runs one column and one diagonal round)
  // to obtain the ChaCha20 subkey and nonce.
  for (int i = 0; i < 10; i++)
    chacha20_inner_block(ctx->private_state);

  // The first 16 bytes and last 16 bytes are returned, concatenated,
  // little-endian, as the subkey.
  for (int i = 0; i < 4; i++)
    MBEDTLS_PUT_UINT32_LE(ctx->private_state[i], subkey.data(), i * 4);

  for (int i = 0; i < 4; i++)
    MBEDTLS_PUT_UINT32_LE(ctx->private_state[i + 12], subkey.data(),
                          i * 4 + 16);

  // We can then initialize the standard ChaCha20 context with the subkey and
  // the nonce. Call init to zero-out the context.
  mbedtls_chacha20_init(ctx);
  rc = mbedtls_chacha20_setkey(ctx, subkey.data());
  if (rc != 0)
    return Error("could not set XChaCha20 subkey: " + Twine::utohexstr(rc));

  // Wipe the subkey now that we don't need it anymore.
  mbedtls_platform_zeroize(subkey.data(), subkey.size());

  // The ietfNonce is already set to zero, so we can simply copy the final 8
  // bytes into it.
  memcpy(ietfNonce.data() + 4, nonce.data() + 16, 8);

  return ietfNonce;
}

//===----------------------------------------------------------------------===//
// AuthenticatedEncryptionKey
//===----------------------------------------------------------------------===//

AuthenticatedEncryptionKey::~AuthenticatedEncryptionKey() {
  // Zero-ize the bytes of the key securely.
  mbedtls_platform_zeroize(bytes, keyLenBytes);
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

ErrorOr<AuthenticatedEncryptionKey>
AuthenticatedEncryptionKey::unsafeFromRawBytes(
    std::array<uint8_t, keyLenBytes> keyBytes) {
  AuthenticatedEncryptionKey key;
  if (auto err = key.setKeyBytes([&](uint8_t *b, size_t n) {
        memcpy(b, keyBytes.data(), n);
        return success();
      }))
    return err.takeError();
  return key;
}

ErrorOr<AuthenticatedEncryptionKey::EncryptedMessage>
AuthenticatedEncryptionKey::encrypt(ArrayRef<uint8_t> plaintext,
                                    ArrayRef<uint8_t> aad) {
  std::array<uint8_t, ivLenBytes> ivBytes = {};

  // Generate a fully random IV - since we're using XChaCha we can do this.
  SecureRandomBytesGenerator rng;
  if (auto err = rng.getRandomBytes(MutableArrayRef(ivBytes)))
    return err.takeError();

  EncryptedMessage msg;
  // Copy the IV into the encrypted message.
  llvm::copy(ivBytes, msg.iv.begin());
  // It's a stream cipher, so we only need `plaintext.size()` bytes in the
  // output.
  msg.ciphertext.resize(plaintext.size());

  // We have to do our own setup for the key and nonce, since we want to use
  // XChaCha.
  auto nonceOr = doXChaCha20Setup(&ctx.private_chacha20_ctx, ivBytes);
  if (nonceOr.isError())
    return nonceOr.takeError();
  IETFNonceArray nonce = nonceOr.takeValue();

  // Do the encryption and generate the MAC. This will also add any AAD to the
  // MAC.
  int rc = mbedtls_chachapoly_encrypt_and_tag(
      &ctx, plaintext.size(), nonce.data(), aad.data(), aad.size(),
      plaintext.data(), msg.ciphertext.data(), msg.tag.data());
  if (rc != 0)
    return Error("encryption failed: " + Twine::utohexstr(rc));

  // Reset the key bytes to the bytes we've stored. Re-init the mbedTLS context
  // to zero everything out.
  mbedtls_chachapoly_init(&ctx);

  // Set the key value.
  rc = mbedtls_chachapoly_setkey(&ctx, bytes);
  if (rc != 0)
    return Error("could not set the key value: " + Twine::utohexstr(rc));

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

  // We have to do our own setup for the key and nonce, since we want to use
  // XChaCha.
  auto nonceOr = doXChaCha20Setup(&ctx.private_chacha20_ctx, msg.iv);
  if (nonceOr.isError())
    return nonceOr.takeError();
  IETFNonceArray nonce = nonceOr.takeValue();

  // Do the decryption into the buffer we just allocated. This will also
  // authenticate the provided AAD bytes.
  int rc = mbedtls_chachapoly_auth_decrypt(
      &ctx, msg.ciphertext.size(), nonce.data(), aad.data(), aad.size(),
      msg.tag.data(), msg.ciphertext.data(), data.data());
  if (rc != 0)
    return Error("decryption failed: " + Twine::utohexstr(rc));

  // Reset the key bytes to the bytes we've stored. Re-init the mbedTLS context
  // to zero everything out.
  mbedtls_chachapoly_init(&ctx);

  // Set the key value.
  rc = mbedtls_chachapoly_setkey(&ctx, bytes);
  if (rc != 0)
    return Error("could not set the key value: " + Twine::utohexstr(rc));

  return data;
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
