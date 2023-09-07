//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H
#define SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H

#include "Support/ASN1/ObjectID.h"
#include "Support/Entitlements/Entitlement.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/DenseMap.h"
#include <filesystem>

/// Forward declaration for the mbedtls_x509_crt struct.
struct mbedtls_x509_crt;

namespace M {
/// Forward-declaration for a single entitlement.
class Entitlement;

/// This provides a way to look up and see if a given entitlement exists in the
/// current store.
class EntitlementStore {
public:
  /// EntitlementStores are default-constructible. That just means they don't
  /// contain any entitlements.
  EntitlementStore() = default;

  /// EntitlementStore objects are non-copyable, but they are move-able.
  EntitlementStore(const EntitlementStore &other) = delete;
  EntitlementStore(EntitlementStore &&other) = default;

  /// Open the certificate that exists at a well-known location. This will
  /// parse the certificate and set up the list of entitlements.
  static ErrorOr<EntitlementStore> open();

  /// Open the certificate that exists at the provided location. This is
  /// identical to `EntitlementStore::open` above, but it takes the CA certs as
  /// a parameter. The caCerts option should only be used for local testing - we
  /// do *NOT* want users to be able to override the roots of trust, because if
  /// they can do that, they can forge certificates with arbitrary entitlements.
  static ErrorOr<EntitlementStore>
  open(const std::filesystem::path &clientCertPath,
       const std::filesystem::path &clientPrivKeyPath,
       mbedtls_x509_crt *caCerts);

  /// Get the instance of the entitlement with type `EntitlementT`, if it's been
  /// registered. If it hasn't been registered, we return `nullptr`.
  template <typename EntitlementT>
  EntitlementT *getEntitlement() const {
    auto found = entitlements.find(Entitlement::getObjectID<EntitlementT>());
    if (found == entitlements.end())
      return nullptr;

    return llvm::cast_if_present<EntitlementT>(found->second.get());
  }

  /// Add an instance of a given entitlement to the entitlement store.
  void addEntitlement(std::unique_ptr<Entitlement> entitlement);

private:
  /// This is a map of all the entitlements we have, indexed by their OID. This
  /// means that we can only have a single instance of a given entitlement at a
  /// time.
  llvm::DenseMap<ASN1::ObjectID, std::unique_ptr<Entitlement>> entitlements;
};
} // namespace M

#endif // SUPPORT_ENTITLEMENTS_ENTITLEMENTSTORE_H
