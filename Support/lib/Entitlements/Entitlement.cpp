//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Entitlements/Entitlement.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/ASN1/ObjectID.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

//===----------------------------------------------------------------------===//
// UnknownEntitlement
//===----------------------------------------------------------------------===//

StringRef UnknownEntitlement::getName() const { return "unknown"; }

ErrorOr<std::unique_ptr<Entitlement>>
UnknownEntitlement::create(bool critical, ArrayRef<uint8_t> data) {
  return std::make_unique<UnknownEntitlement>();
}

//===----------------------------------------------------------------------===//
// ModularDeveloperEntitlement
//===----------------------------------------------------------------------===//

StringRef ModularDeveloperEntitlement::getName() const {
  return "modular-developer";
}

ErrorOr<std::unique_ptr<Entitlement>>
ModularDeveloperEntitlement::create(bool critical, ArrayRef<uint8_t> data) {
  return std::make_unique<ModularDeveloperEntitlement>();
}

//===----------------------------------------------------------------------===//
// BetaEntitlement
//===----------------------------------------------------------------------===//

StringRef BetaEntitlement::getName() const { return "beta"; }

ErrorOr<std::unique_ptr<Entitlement>>
BetaEntitlement::create(bool critical, ArrayRef<uint8_t> data) {
  return std::make_unique<BetaEntitlement>();
}

//===----------------------------------------------------------------------===//
// GPUEntitlement
//===----------------------------------------------------------------------===//

StringRef GPUEntitlement::getName() const { return "gpu"; }

ErrorOr<std::unique_ptr<Entitlement>>
GPUEntitlement::create(bool critical, ArrayRef<uint8_t> data) {
  return std::make_unique<GPUEntitlement>();
}

//===----------------------------------------------------------------------===//
// MaxThreadsUnlimitedEntitlement
//===----------------------------------------------------------------------===//

StringRef MaxThreadsUnlimitedEntitlement::getName() const {
  return "max-threads-unlimited";
}

ErrorOr<std::unique_ptr<Entitlement>>
MaxThreadsUnlimitedEntitlement::create(bool critical, ArrayRef<uint8_t> data) {
  return std::make_unique<MaxThreadsUnlimitedEntitlement>();
}

//===----------------------------------------------------------------------===//
// EnterpriseEntitlement
//===----------------------------------------------------------------------===//

StringRef EnterpriseEntitlement::getName() const { return "enterprise"; }

ErrorOr<std::unique_ptr<Entitlement>>
EnterpriseEntitlement::create(bool critical, ArrayRef<uint8_t> data) {
  return std::make_unique<EnterpriseEntitlement>();
}

//===----------------------------------------------------------------------===//
// AdminEntitlement
//===----------------------------------------------------------------------===//

StringRef AdminEntitlement::getName() const { return "admin"; }

ErrorOr<std::unique_ptr<Entitlement>>
AdminEntitlement::create(bool critical, ArrayRef<uint8_t> data) {
  return std::make_unique<AdminEntitlement>();
}

//===----------------------------------------------------------------------===//
// BuilderRegistry
//===----------------------------------------------------------------------===//

namespace {
using BuilderTy = llvm::unique_function<ErrorOr<std::unique_ptr<Entitlement>>(
    bool, ArrayRef<uint8_t>)>;

/// Contains a list of builders, indexed by kind, along with a mutex. The mutex
/// guards reading and writing the registry.
struct BuilderRegistry {
  mutable std::mutex m;
  std::vector<BuilderTy> builders;
};
} // namespace

/// Get the static builder registry.
static BuilderRegistry *getRegistry() {
  static auto *registry = new BuilderRegistry{
      std::mutex(),
      // EK_UNKNOWN is a valid entitlement kind, so we'll end up indexing this
      // vector with `EK_UNKNOWN`, so size needs to be EK_UNKNOWN + 1.
      std::vector<BuilderTy>((size_t)Entitlement::EK_UNKNOWN + 1)};
  return registry;
}

//===----------------------------------------------------------------------===//
// Entitlement
//===----------------------------------------------------------------------===//

void Entitlement::setAsExtension(
    llvm::function_ref<void(ArrayRef<uint8_t>, bool, ArrayRef<uint8_t>)>
        acceptor) {
  // Get the OID. This is a Modular-prefixed OID with the kind of the
  // entitlement after it.
  ASN1::ObjectID oid = ASN1::ObjectID(/*withModularPrefix=*/true,
                                      {modularEntitlementArc, getKind()});
  SmallVector<uint8_t> oidData = oid.getEncoded();

  // Call the acceptor.
  acceptor(oidData, isCritical(), getDataBytes());
}

ErrorOr<std::unique_ptr<Entitlement>>
Entitlement::parse(const ASN1::ObjectID &oid, bool critical,
                   ArrayRef<uint8_t> data) {
  ArrayRef<uint64_t> arc = oid.getArc();
  // TODO: Should this be an error? Or should we be returning an optional?
  if (arc[0] != modularEntitlementArc)
    return Error("OID does not indicate an entitlement");

  // For OIDs under the entitlement arc, the number is exactly equal to the
  // kind.
  uint64_t kind = arc[1];

  // We don't know what kind of entitlement this is - store it as an unknown
  // entitlement. If it's non-critical, then it will only error if we try to
  // access it.
  if (kind >= EK_UNKNOWN) {
    if (critical)
      return Error("unknown entitlement kind " + Twine(kind));
    else
      return UnknownEntitlement::create(critical, data);
  }

  // Now we call create to actually create the entitlements.
  BuilderRegistry *registry = getRegistry();
  // The only thing we need to lock is the bit where we actually read the
  // registry, so wrap that in its own little scope.
  auto &ctor = [&]() -> BuilderTy & {
    std::lock_guard<std::mutex> lock(registry->m);
    return registry->builders[kind];
  }();
  if (!ctor) {
    return Error("could not find builder for entitlement with kind " +
                 Twine(kind));
  }

  return ctor(critical, data);
}

void Entitlement::registerBuilder(Entitlement::Kind k,
                                  Entitlement::BuilderTy builder) {
  BuilderRegistry *registry = getRegistry();
  std::lock_guard<std::mutex> lock(registry->m);
  (registry->builders)[k] = std::move(builder);
}

void M::registerAllEntitlements() {
  Entitlement::registerEntitlement<UnknownEntitlement>();
  Entitlement::registerEntitlement<ModularDeveloperEntitlement>();
  Entitlement::registerEntitlement<BetaEntitlement>();
  Entitlement::registerEntitlement<GPUEntitlement>();
  Entitlement::registerEntitlement<MaxThreadsUnlimitedEntitlement>();
  Entitlement::registerEntitlement<EnterpriseEntitlement>();
  Entitlement::registerEntitlement<AdminEntitlement>();
}
