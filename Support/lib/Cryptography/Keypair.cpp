//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Cryptography/Keypair.h"
#include "Support/Buffer.h"
#include "Support/FileSystemExtras.h"
#include "Support/JSON.h"
#include "Support/Random.h"
#include "mbedtls/error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

Keypair::Keypair() { mbedtls_pk_init(&ctx); }

Keypair::~Keypair() { mbedtls_pk_free(&ctx); }

//===----------------------------------------------------------------------===//
// mbedTLSErrorToString
//===----------------------------------------------------------------------===//

/// Convert an mbedTLS error to a human-readable M::Error object. For non-debug
/// builds, this generates generic error codes.
static std::string mbedTLSErrorToString(int rc) {
#ifdef MODULAR_DEBUG
  std::string errStr(1024, '\0');
  mbedtls_strerror(rc, errStr.data(), errStr.size());
  // Use strlen because mbedtls_strerror guarantees a null-terminated string, so
  // we can use this to determine the actual length.
  errStr.resize(strlen(errStr.c_str()));
  errStr.shrink_to_fit();
  return errStr;
#else  // MODULAR_DEBUG
  return "mbedTLS encountered an error, aborting";
#endif // MODULAR_DEBUG
}

static int csprng(void *ctx, unsigned char *buf, size_t numBytes) {
  auto *rng = (SecureRandomBytesGenerator *)ctx;
  MutableArrayRef<uint8_t> randBuf(buf, numBytes);
  if (auto err = rng->getRandomBytes(randBuf))
    return 1;
  return 0;
}

ErrorOr<Keypair> Keypair::generate(const std::filesystem::path &priv,
                                   const std::filesystem::path &pub) {
  // Set up the keypair for generation.
  Keypair out;
  int rc = mbedtls_pk_setup(out.getRawKey(),
                            mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY));
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // Generate the keypair.
  SecureRandomBytesGenerator rng;
  rc = mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1,
                           mbedtls_pk_ec(*out.getRawKey()), &csprng, &rng);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // Create an array of scratch data we can use. This size works well because we
  // are using the elliptic curve P256 key - it should always be much less than
  // 512 bytes in size.
  std::array<uint8_t, 512> scratchBuf = {};
  rc = mbedtls_pk_write_key_pem(out.getRawKey(), scratchBuf.data(),
                                scratchBuf.size());
  if (rc != 0)
    return Error("could not write the keypair to PEM");

  // Write the private key first.
  auto err = writeFileUnderLock(priv, [&](llvm::raw_ostream &os) {
    const char *pemData = (const char *)scratchBuf.data();
    // Write the null terminator - we need it at parse time.
    os.write(pemData, strlen(pemData) + 1);
  });
  if (err)
    return err.takeError();

  // Then, write the public key.
  rc = mbedtls_pk_write_pubkey_pem(out.getRawKey(), scratchBuf.data(),
                                   scratchBuf.size());
  if (rc != 0)
    return Error("could not write the keypair to PEM");

  err = writeFileUnderLock(pub.string(), [&](llvm::raw_ostream &os) {
    const char *pemData = (const char *)scratchBuf.data();
    // Write the null terminator - we need it at parse time.
    os.write(pemData, strlen(pemData) + 1);
  });
  if (err)
    return err.takeError();

  // OK - now we're done, so return the keypair. We definitely have a private
  // key, we just generated it.
  out.havePrivateKey = true;
  return out;
}

ErrorOr<Keypair> Keypair::openPrivate(const std::filesystem::path &absolute) {
  Keypair out;

  // Local copy of the contents of the file we read. This is for hardening, so
  // we get a snapshot of the file contents.
  std::vector<uint8_t> fileContents;

  std::optional<Error> readErr;
  auto err =
      readFileUnderLock(absolute, [&](const std::filesystem::path &path) {
        auto fileBufOr = Buffer::getFile(path);
        if (fileBufOr.isError()) {
          readErr = fileBufOr.takeError();
          return;
        }

        // Copy the contents of the file into memory.
        fileContents.resize((*fileBufOr)->getBufferSize());
        llvm::copy((*fileBufOr)->getBuffer(), fileContents.begin());
      });
  if (err.isError())
    return err.takeError();

  if (readErr)
    return std::move(*readErr);

  // We need this for blinding if we're reading the private key.
  SecureRandomBytesGenerator rng;
  int rc = mbedtls_pk_parse_key(out.getRawKey(), fileContents.data(),
                                fileContents.size(), nullptr, 0, &csprng, &rng);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  out.havePrivateKey = true;

  return out;
}

ErrorOr<Keypair> Keypair::openPublic(const std::filesystem::path &absolute) {
  Keypair out;

  // Local copy of the contents of the file we read. This is for hardening, so
  // we get a snapshot of the file contents.
  std::vector<uint8_t> fileContents;

  std::optional<Error> readErr;
  auto err =
      readFileUnderLock(absolute, [&](const std::filesystem::path &path) {
        auto fileBufOr = Buffer::getFile(path);
        if (fileBufOr.isError()) {
          readErr = fileBufOr.takeError();
          return;
        }

        // Copy the contents of the file into memory.
        fileContents.resize((*fileBufOr)->getBufferSize());
        llvm::copy((*fileBufOr)->getBuffer(), fileContents.begin());
      });
  if (err.isError())
    return err.takeError();

  if (readErr)
    return std::move(*readErr);

  int rc = mbedtls_pk_parse_public_key(out.getRawKey(), fileContents.data(),
                                       fileContents.size());
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  return out;
}

ErrorOr<Keypair> Keypair::publicFromPEM(const std::string &pem) {
  Keypair out;
  // The length of the string must include the terminating null byte, and
  // std::string::size() doesn't include that.
  int rc = mbedtls_pk_parse_public_key(
      out.getRawKey(), (const uint8_t *)pem.c_str(), pem.size() + 1);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  return out;
}

ErrorOrSuccess Keypair::validateSignature(StringRef signedData,
                                          StringRef signature) {
  // For now, we always use SHA256 because that's what TUF uses. This could
  // easily be parametrized if we wanted to, though.
  const mbedtls_md_info_t *mdInfo =
      mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);

  // We can use a static hash size here because we statically know we're using
  // SHA256.
  std::array<uint8_t, 32> hashOutput = {};
  int rc = mbedtls_md(mdInfo, (const uint8_t *)signedData.data(),
                      signedData.size(), hashOutput.data());
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // Hash the input buffer.
  uint8_t numBytes = mbedtls_md_get_size(mdInfo);

  // Verify the signature. Again, we're using SHA-256 here.
  rc = mbedtls_pk_verify(getRawKey(), MBEDTLS_MD_SHA256, hashOutput.data(),
                         numBytes, (const uint8_t *)signature.data(),
                         signature.size());
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  return success();
}

ErrorOr<std::string> Keypair::sign(StringRef data) {
  // For now, we always use SHA256 because that's what TUF uses. This could
  // easily be parametrized if we wanted to, though.
  const mbedtls_md_info_t *mdInfo =
      mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);

  // We can use a static hash size here because we statically know we're using
  // SHA256.
  std::array<uint8_t, 32> hashOutput = {};
  int rc =
      mbedtls_md(mdInfo, data.bytes_begin(), data.size(), hashOutput.data());
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  uint8_t numBytes = mbedtls_md_get_size(mdInfo);

  // Allocate at least enough bytes for the signature.
  std::string signature(MBEDTLS_PK_SIGNATURE_MAX_SIZE, '\0');

  // Sign it - this will write the bytes directly to the string and then we'll
  // resize it to whatever the actual signature size is.
  size_t siglen = 0;
  SecureRandomBytesGenerator rng;
  rc = mbedtls_pk_sign(getRawKey(), MBEDTLS_MD_SHA256, hashOutput.data(),
                       numBytes, (uint8_t *)signature.data(), signature.size(),
                       &siglen, csprng, &rng);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // Done, return the resized signature string.
  signature.resize(siglen);
  return std::move(signature);
}

ErrorOr<std::string> Keypair::getTUFKeyID() const {
  // See securesystemslib/keys.py:374 (format_keyval_to_metadata) to see how the
  // key ID is expected to be formatted.

  llvm::json::Object obj;
  // These two are only constants because we currently only handle the NIST
  // P-256 curve.
  obj.try_emplace("keytype", "ecdsa");
  obj.try_emplace("scheme", "ecdsa-sha2-nistp256");
  obj.try_emplace("keyid_hash_algorithms",
                  llvm::json::Array({"sha256", "sha512"}));

  std::array<uint8_t, 512> scratchBuf = {};
  int rc =
      mbedtls_pk_write_pubkey_pem(&ctx, scratchBuf.data(), scratchBuf.size());
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // The scratch buffer will include a terminating null byte, so the StringRef
  // will take just the PEM bytes.
  StringRef pem((const char *)scratchBuf.data());

  // This will hold the PEM for the public key. The TUF writer doesn't include
  // the trailing '\n', so we have to drop it.
  llvm::json::Object keyval;
  keyval.try_emplace("public", pem.drop_back());

  // Put the pem into the top level object.
  obj.try_emplace("keyval", std::move(keyval));

  // Convert it to a value once we're done adding to it.
  llvm::json::Value val(std::move(obj));

  // Serialize the top level object as canonical JSON.
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  serializeCanonicalJSON(&val, stream);

  // The key ID is the SHA256 hash of this JSON.
  const mbedtls_md_info_t *mdInfo =
      mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);

  // Construct the string and allocate enough memory for the hash output.
  std::string out;
  out.resize(mbedtls_md_get_size(mdInfo));
  rc = mbedtls_md(mdInfo, (const uint8_t *)canonical.data(), canonical.size(),
                  (uint8_t *)out.data());
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // Return it as a hex-encoded string.
  return llvm::toHex(out, /*LowerCase=*/true);
}
