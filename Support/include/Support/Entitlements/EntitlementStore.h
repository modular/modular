//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H
#define SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H

#include "Support/ASN1/ObjectID.h"
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

/// This provides a way to look up and see if a given entitlement exists in the
/// current store.
class EntitlementStore {
public:
  /// EntitlementStores are default-constructible. That just means they don't
  /// contain any entitlements.
  EntitlementStore();
  ~EntitlementStore();

  /// EntitlementStore objects are non-copyable, but they are move-able.
  EntitlementStore(const EntitlementStore &other) = delete;
  EntitlementStore(EntitlementStore &&other) = default;

  /// Open the entitlements store. If the client certificate exists, that one
  /// will be used. Otherwise, we drop into the OAuth Device Authorization Flow
  /// and prompt the user to perform actions in their browser to authorize this
  /// process. If the certificate found on the system is invalid or fails to
  /// parse, then we return an error. Users should prefer `openWithRetry` below
  /// unless there's a clear reason to error on an invalid certificate.
  static ErrorOr<EntitlementStore> open(HTTPClient &client);

  /// Open the entitlement store with a single retry. Performs the actions of
  /// `open` above. The main use-case for this function is when the certificate
  /// on-disk is invalid or does not parse; this function will remove that
  /// certificate and perform the OAuth Device Authorization Flow to issue a new
  /// certificate. This will also retry in case there is a separate error in the
  /// flow. To re-emphasize, this will perform a *single* retry if an error
  /// occurs in the `open` procedure.
  static ErrorOr<EntitlementStore> openWithRetry(HTTPClient &client);

  /// Refresh the entitlement store by refreshing the client certificate. This
  /// will also invalidate any entitlements that currently exist, even if the
  /// user's entitlements have not changed.
  ErrorOrSuccess refresh(HTTPClient &client);

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

private:
  /// This will verify the client's certificate chain and flush it to disk if
  /// and only if it's valid.
  ErrorOrSuccess verifyAndFlushClientCert();

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
  /// the certificate into the mbedtls_x509 structure.
  ErrorOrSuccess requestCertificate(HTTPClient &client, BufferRef csr);

  /// This is a map of all the entitlements we have, indexed by their OID. This
  /// means that we can only have a single instance of a given entitlement at a
  /// time.
  llvm::DenseMap<ASN1::ObjectID, std::unique_ptr<Entitlement>> entitlements;

  /// This holds the client certificate chain. The implementation is hidden to
  /// avoid leaking details through this abstraction.
  std::unique_ptr<Detail::CertificateChain> clientCert;

  /// Store the client keys. If these cannot be found, they'll be generated.
  Keypair clientKeys;
};
} // namespace M

#endif // SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H
