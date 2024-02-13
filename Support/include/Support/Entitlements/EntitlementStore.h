//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H
#define SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H

#include "Support/ASN1/ObjectID.h"
#include "Support/Configuration.h"
#include "Support/Cryptography/Keypair.h"
#include "Support/Entitlements/Entitlement.h"
#include "Support/ErrorOr.h"
#include "Support/HTTP/HTTPClient.h"
#include "llvm/ADT/DenseMap.h"
#include <filesystem>

namespace M {
/// Forward-declaration for a single entitlement.
class Entitlement;

/// Forward-declaration for a CertificateChain class that we will make extensive
/// use of in the EntitlementStore.
namespace Detail {
class CertificateChain;
}

/// This function specifies the default entitlement refresh policy. This is used
/// as the default argument for
/// `EntitlementStore::refreshIfNecessary::shouldRefresh` when no
/// application-specific policy is necessary.
bool defaultEntitlementRefreshPolicy(std::chrono::system_clock::time_point from,
                                     std::chrono::system_clock::time_point to);

/// This provides a way to look up and see if a given entitlement exists in the
/// current store.
class EntitlementStore {
public:
  EntitlementStore(const std::filesystem::path &clientKeyPrivPath,
                   const std::filesystem::path &clientKeyPubPath,
                   const std::filesystem::path &clientCertPath,
                   const std::filesystem::path &crlPath);
  ~EntitlementStore();

  /// EntitlementStore objects are non-copyable, but they are move-able.
  EntitlementStore(const EntitlementStore &other) = delete;
  EntitlementStore(EntitlementStore &&other);

  /// Move-assignment operator.
  EntitlementStore &operator=(EntitlementStore &&other) {
    if (&other != this)
      new (this) EntitlementStore(std::move(other));

    return *this;
  }

  /// Get the current user ID. This is effectively a view over the Subject
  /// string in the certificate.
  ErrorOr<StringRef> getUserID() const;

  /// Open the entitlements store. If the client certificate exists, this will
  /// return a valid and ready-to-use EntitlementStore. If the client
  /// certificate does not exist, then return std::nullopt because the enclosing
  /// application should decide what to do in that state. If an actual error
  /// occurs, return the error.
  static ErrorOr<std::optional<EntitlementStore>> open(Config &config,
                                                       HTTPClient *client);

  /// Remove an existing certificate from the user's system (if it exists) and
  /// fetch a new one. This always returns an EntitlementStore on success,
  /// because the only case where we cannot fetch a certificate is the one where
  /// an actual error occurred.
  static ErrorOr<EntitlementStore> generate(Config &config, HTTPClient &client);

  /// Always open an EntitlementStore, and simply default to an empty
  /// EntitlementStore if we don't have the required infrastructure, we don't
  /// have a certificate, or the opening fails. This will print a warning to
  /// `warnStream` if there was an actual error in opening the EntitlementStore.
  static EntitlementStore alwaysOpen(HTTPClient *client,
                                     llvm::raw_ostream &warnStream);

  /// Refresh the entitlement store by refreshing the client certificate. This
  /// will also invalidate any entitlements that currently exist, even if the
  /// user's entitlements have not changed.
  ErrorOrSuccess refresh(HTTPClient &client);

  /// Refresh the entitlement store if it's necessary to do so. The user can
  /// configure a policy on when a refresh is 'necessary', using the validFrom
  /// and validTo values of the certificate, converted to system clock time
  /// points.
  ErrorOrSuccess refreshIfNecessary(
      HTTPClient &client,
      llvm::function_ref<bool(std::chrono::system_clock::time_point from,
                              std::chrono::system_clock::time_point to)>
          shouldRefresh = defaultEntitlementRefreshPolicy);

  /// Get the instance of the entitlement with type `EntitlementT`, if it's been
  /// registered. If it hasn't been registered, we return `nullptr`. Consumers
  /// should NOT save entitlements returned by this method as they may be
  /// invalidated the next time `refresh` is called. Recommended usage is to get
  /// the entitlement in question, and parse its data immediately and act upon
  /// it.
  template <typename EntitlementT>
  EntitlementT *getEntitlement() const {
    auto found = entitlements.find(Entitlement::getObjectID<EntitlementT>());
    if (found == entitlements.end())
      return nullptr;

    return llvm::cast_if_present<EntitlementT>(found->second.get());
  }

  /// Lookup an entitlement by its name.
  Entitlement *getEntitlement(StringRef name) const {
    auto foundName = nameToOID.find(name);
    if (foundName == nameToOID.end())
      return nullptr;

    auto found = entitlements.find(foundName->second);
    if (found == entitlements.end())
      return nullptr;

    return found->second.get();
  }

private:
  /// This will verify the client's certificate chain and flush it to disk if
  /// and only if it's valid.
  ErrorOrSuccess verifyAndFlushClientCert(HTTPClient *client);

  /// Given a PEM buffer with one or more certificates, parse them and take any
  /// entitlements that might be encoded in the extensions and put them in the
  /// store.
  ErrorOrSuccess parseCertificateChain(BufferRef pem);

  /// Use the OAuth Device Authorization Flow to do initial authentication and
  /// bootstrap the client certificate. Note that this does not validate the
  /// client certificate, since that validation will happen when the cert is
  /// read in EntitlementStore::refresh. The client certificate will be stored
  /// in clientCertDER on successful completion.
  ErrorOrSuccess authAndFetchCertificate(HTTPClient &client);

  /// Takes a CSR and requests a certificate. The certificate is returned in PEM
  /// form and decoded. Once the certificate is received, it is stored to
  /// `clientCertDER`. No validation is performed at this stage to avoid parsing
  /// the certificate into the mbedtls_x509 structure. The previous key
  /// signature may be empty if we aren't rotating the client keypair.
  ErrorOrSuccess requestCertificate(HTTPClient &client, StringRef csr,
                                    StringRef prevKeySig, bool isRefresh);

  /// This is a map of all the entitlements we have, indexed by their OID. This
  /// means that we can only have a single instance of a given entitlement at a
  /// time.
  llvm::DenseMap<ASN1::ObjectID, std::unique_ptr<Entitlement>> entitlements;

  /// This maps from entitlement names to their OID. This allows us to look up
  /// an entitlement by name rather than forcing us to know the OID.
  llvm::StringMap<ASN1::ObjectID> nameToOID;

  /// This holds the client certificate chain. The implementation is hidden to
  /// avoid leaking details through this abstraction.
  std::unique_ptr<M::Detail::CertificateChain> clientCert;

  /// Store the client keys. If these cannot be found, they'll be generated.
  Keypair clientKeys;

  /// This is a local reference to the CRL. If we have it, it'll be PEM-encoded.
  BufferRef crlPEM;

public:
  /// Paths to use for on-disk keys.
  const std::filesystem::path clientKeyPrivPath;
  const std::filesystem::path clientKeyPubPath;
  const std::filesystem::path clientCertPath;
  const std::filesystem::path crlPath;
};
} // namespace M

#endif // SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H
