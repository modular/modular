//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CRYPTOGRAPHY_KEYPAIR_H
#define SUPPORT_CRYPTOGRAPHY_KEYPAIR_H

#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "mbedtls/pk.h"
#include <filesystem>
#include <utility>

namespace M {
/// This class is essentially an RAII wrapper around mbedtls_pk_context. It can
/// represent a public and private key pair, or just a public key.
class Keypair {
public:
  Keypair();
  ~Keypair();

  /// Keypairs are move-able, but not copy-able.
  Keypair(const Keypair &other) = delete;
  Keypair &operator=(const Keypair &other) = delete;
  Keypair(Keypair &&other) { std::swap(ctx, other.ctx); }

  /// Explicitly create the move assignment operator.
  Keypair &operator=(Keypair &&other) {
    if (this != &other) {
      std::swap(ctx, other.ctx);
    }

    return *this;
  }

  /// Generate a new keypair using the system CSPRNG. This will initialize a
  /// keypair that has a private key. If a directory is provided, the keys will
  /// be written in DER form as <dir>/client_priv.der and <dir>/client_pub.der.
  static ErrorOr<Keypair> generate(const std::filesystem::path &priv);

  /// Open the private key at `absolute`. This must be a full path to the
  /// private key.
  static ErrorOr<Keypair> open(const std::filesystem::path &absolute);

  /// Open the private key from a buffer.
  static ErrorOr<Keypair> open(const BufferRef buf, bool needsNull = true,
                               bool publicKey = false);

  /// Validate the provided signature over the signed data. Performs absolutely
  /// no modifications to signedData before verifying the signature.
  // TODO: This currently assumes that we want to use SHA256 for all
  //       validations, but we could pretty easily allow it to be configurable.
  //       Note that we'll need to use it when generating TUF key IDs.
  ErrorOrSuccess validateSignature(StringRef signedData, StringRef signature);

  /// Sign `data` and return the signature.
  // TODO: This currently assumes that we want to use SHA256 for all
  //       validations, but we could pretty easily allow it to be configurable.
  ErrorOr<std::string> sign(StringRef data);

  /// Return the key ID in the format that TUF would expect. This requires JSON
  /// formatting - if we end up deciding to change our TUF manifest
  /// representation we will need to change this as well. Return the string as a
  /// hex-encoded byte-string.
  ErrorOr<std::string> getTUFKeyID() const;

  /// Return the raw mbedtls_pk_context. Most users will not need this, only
  /// use this if you *know* this is the thing you need. Most users should
  /// use the higher-level APIs.
  mbedtls_pk_context *getRawKey() { return &ctx; }

  /// Return a buffer with the private key.
  BufferRef getBuffer();

private:
  mbedtls_pk_context ctx = {};
};
} // namespace M

#endif // SUPPORT_CRYPTOGRAPHY_KEYPAIR_H
