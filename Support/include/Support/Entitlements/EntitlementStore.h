//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H
#define SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H

#include "Support/ASN1/ObjectID.h"
#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/Cryptography/Keypair.h"
#include "Support/Entitlements/Entitlement.h"
#include "Support/Entitlements/EntitlementToken.h"
#include "Support/ErrorOr.h"
#include "Support/HTTP/HTTPClient.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"
#include <filesystem>
#include <thread>

namespace M {
/// Forward-declaration for a single entitlement.
class Entitlement;

namespace Detail {
/// Forward-declaration for a CertificateChain class that we will make extensive
/// use of in the EntitlementStore.
class CertificateChain;

class CertSubject {
public:
  CertSubject(const std::string &userId, const std::string &accessTokenId)
      : UserId(userId), AccessTokenId(accessTokenId) {}
  std::string format() const {
    // Until 04/2024, we set O=Modular Inc
    // Do not repurose this field until O is flushed from our system.
    return llvm::formatv("C=US,CN={0},OU={1}", UserId, AccessTokenId);
  }

  std::string UserId;
  std::string AccessTokenId;
};
} // namespace Detail

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
  ~EntitlementStore();

  /// EntitlementStore objects are non-copyable, but they are move-able.
  EntitlementStore(const EntitlementStore &other) = delete;
  EntitlementStore(EntitlementStore &&other);

  /// Get the current user ID. This is effectively a view over the Subject
  /// string in the certificate.
  ErrorOr<std::string> getUserID() const;

  /// Gets a copy of the current private key.
  BufferRef getPrivateKey() { return clientKeyPriv.copy(); }

  /// Gets a copy of the current certificate.
  BufferRef getCertificate() { return clientCert.copy(); }

  /// Open the entitlements store. If the client certificate exists, this will
  /// return a valid and ready-to-use EntitlementStore. If the client
  /// certificate does not exist, then return std::nullopt because the enclosing
  /// application should decide what to do in that state. If an actual error
  /// occurs, return the error.
  static ErrorOr<std::optional<EntitlementStore>>
  open(Config &config, std::optional<std::string> envVarOr = {});

  /// Remove an existing certificate from the user's system (if it exists) and
  /// fetch a new one. This always returns an EntitlementStore on success,
  /// because the only case where we cannot fetch a certificate is the one where
  /// an actual error occurred.
  static ErrorOr<EntitlementStore>
  generate(Config &config, const HTTPContextRef &httpCtx,
           std::optional<std::string> accessTokenOr, bool openBrowser);

  /// Always open an EntitlementStore, and simply default to an empty
  /// EntitlementStore if we don't have the required infrastructure, we don't
  /// have a certificate, or the opening fails. This will print a warning to
  /// `warnStream` if there was an actual error in opening the EntitlementStore.
  static EntitlementStore alwaysOpen(llvm::raw_ostream &warnStream);

  /// Refresh the entitlement store by refreshing the client certificate. This
  /// will also invalidate any entitlements that currently exist, even if the
  /// user's entitlements have not changed.
  ErrorOrSuccess refresh(Config &config, const HTTPContextRef &httpCtx);

  /// Refresh the entitlement store if it's necessary to do so. The user can
  /// configure a policy on when a refresh is 'necessary', using the validFrom
  /// and validTo values of the certificate, converted to system clock time
  /// points.
  ErrorOrSuccess refreshIfNecessary(
      Config &config, const HTTPContextRef &httpCtx,
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

  static std::optional<std::string> getAuthTokenFromEnv() {
    return llvm::sys::Process::GetEnv("MODULAR_AUTH_TOKEN");
  }

  static std::optional<std::string> getAccessTokenFromEnv() {
    return llvm::sys::Process::GetEnv("MODULAR_ACCESS_TOKEN");
  }

  static ErrorOr<EntitlementStore> fromToken(const EntitlementToken &token);

  static ErrorOr<EntitlementStore> fromConfig(Config &config);

private:
  /// Creates a new EntitlementStore. The given `clientKeys` must correspond
  /// to the private key in `clientKeyPriv` and the associated certificate in
  /// `clientClient`. The entitlement store is not generally complete, and
  /// creation should always be driven the the `create` method.
  EntitlementStore(Keypair &&clientKeys, BufferRef &&clientKeyPriv,
                   BufferRef &&clientCert);

  /// Create a store from the given key and certificate. They are validated.
  static ErrorOr<EntitlementStore> create(BufferRef &&clientKeyPriv,
                                          BufferRef &&clientCert);

  /// Given the local certificate, parse them and take any entitlements that
  /// might be encoded in the extensions and put them in the store. This is
  /// called by `create` exclusively.
  ErrorOrSuccess parseCertificateChain();

  /// Bootstrap the client certificate. Note that this does not validate the
  /// client certificate, since that validation will happen when the cert is
  /// read in EntitlementStore::refresh. The client certificate will be stored
  /// in clientCertDER on successful completion.
  static ErrorOr<BufferRef>
  fetchCertificate(HTTPClient &client, Keypair &clientKeys,
                   const M::Detail::CertSubject &subject,
                   std::optional<llvm::StringRef> accessToken);

  /// Takes a CSR and requests a certificate. The certificate is returned in PEM
  /// form and decoded. Once the certificate is received, it is stored to
  /// `clientCertDER`. No validation is performed at this stage to avoid parsing
  /// the certificate into the mbedtls_x509 structure. The previous key
  /// signature may be empty if we aren't rotating the client keypair.
  static ErrorOr<BufferRef>
  requestCertificate(HTTPClient &client, StringRef csr,
                     M::Detail::CertificateChain *chain, StringRef prevKeySig,
                     bool isRefresh, std::optional<StringRef> accessToken);

  /// This is a map of all the entitlements we have, indexed by their OID. This
  /// means that we can only have a single instance of a given entitlement at a
  /// time.
  llvm::DenseMap<ASN1::ObjectID, std::unique_ptr<Entitlement>> entitlements;

  /// This maps from entitlement names to their OID. This allows us to look up
  /// an entitlement by name rather than forcing us to know the OID.
  llvm::StringMap<ASN1::ObjectID> nameToOID;

  /// Store the client keys.
  Keypair clientKeys;

  /// This holds the client certificate chain. The implementation is hidden to
  /// avoid leaking details through this abstraction.
  std::unique_ptr<M::Detail::CertificateChain> clientCertChain;

  /// In-memory certificates.
  BufferRef clientKeyPriv;
  BufferRef clientCert;
};
} // namespace M

#endif // SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H
