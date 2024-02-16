//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ENTITLEMENTS_ENTITLEMENT_H
#define SUPPORT_ENTITLEMENTS_ENTITLEMENT_H

#include "Support/ASN1/ObjectID.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/FunctionExtras.h"
#include <cstdint>
#include <mutex>

namespace M {
/// This is the base class for any entitlement we might have. Since we want to
/// parse these generically, this must be a closed set, enumerated by kind.
/// Entitlements map one-to-one with X.509 extensions. The kind corresponds to
/// the final value of the OID arc for this entitlement. Entitlements can be
/// marked as critical and can have generic data. Read the virtual member
/// functions' descriptions below for more information on those two fields.
class Entitlement {
public:
  virtual ~Entitlement() = default;

  /// We're incredibly unlikely to need more than 2^32 entitlements. New
  /// entitlements must add a new EK_* entry and increment EK_UNKNOWN as
  /// appropriate.
  ///
  /// N.B. These entitlements should match:
  ///   CloudInfra/services/phoenix/util/entitlement/entitlement.go
  /// and have a canonical database at:
  ///   https://www.notion.so/modularai/1f0c3589df8b4571853cc4c26d138037
  enum Kind : uint32_t {
    EK_RESERVED = 0, ///< This number is reserved for testing and other internal
                     ///< purposes. Users must not rely on this entitlement.
    EK_MODULAR_DEVELOPER = 1,
    EK_BETA = 2,
    EK_GPU = 3,
    EK_MAX_THREADS_UNLIMITED = 4,
    EK_ENTERPRISE = 5,
    EK_UNKNOWN = 6, ///< This must be the max value of the currently known
                    ///< entitlements.
  };

  /// Return the entitlement's kind. Useful for `classof`.
  Kind getKind() const { return kind; }

  /// Return the ObjectID struct for an entitlement of type `EntitlementT`.
  template <typename EntitlementT>
  static ASN1::ObjectID getObjectID() {
    return ASN1::ObjectID(/*withModularPrefix=*/true,
                          {modularEntitlementArc, EntitlementT::getKind()});
  }

  /// Return the ObjectID struct for this entitlement.
  ASN1::ObjectID getObjectID() const {
    return ASN1::ObjectID(/*withModularPrefix=*/true,
                          {modularEntitlementArc, getKind()});
  }

  /// Set this Entitlement as an extension. Because there are multiple options
  /// for this, and we don't want to repeatedly copy the data for the
  /// entitlement, we have this set up with an acceptor function. The parameters
  /// for the acceptor are (in order): DER-encoded OID, critical, byte-encoded
  /// data for this extension.
  void setAsExtension(
      llvm::function_ref<void(ArrayRef<uint8_t>, bool, ArrayRef<uint8_t>)>
          acceptor);

  /// Parse the entitlement from an X.509 extension. This will construct the
  /// appropriate entitlement by inspection. The data passed to the individual
  /// entitlement class is passed by reference to the certificate, so each
  /// entitlement class should copy any data it requires, as the certificate
  /// itself may not stay alive longer than the entitlement object.
  static ErrorOr<std::unique_ptr<Entitlement>>
  parse(const ASN1::ObjectID &oid, bool critical, ArrayRef<uint8_t> data);

  /// Get the name of this entitlement. This should be user-readable.
  virtual StringRef getName() const = 0;

  /// Register an entitlement by providing the static builder.
  template <typename EntitlementT>
  static void registerEntitlement() {
    registerBuilder(EntitlementT::getKind(), EntitlementT::create);
  }

protected:
  /// Construct an Entitlement given a Kind.
  Entitlement(Kind k) : kind(k) {}

  /// Subclasses can use this to serialize any data they may contain. If they
  /// don't have any data, then they should simply return nothing.
  virtual std::vector<uint8_t> getDataBytes() { return {}; }

  /// Extensions should return true if they *must* be present for a certificate
  /// to be valid. Generally, defaulting to false is safest, as that simply
  /// means that the entitlement may not exist. Entitlements marked as critical
  /// must be recognized upon parsing. Entitlements that are not marked as
  /// critical can be parsed as UnknownEntitlement.
  virtual bool isCritical() const { return false; }

private:
  Kind kind;

  /// This provides a way to namespace entitlement nodes within the Modular arc.
  static constexpr uint64_t modularEntitlementArc = 1;

  /// This is the type required for an Entitlement builder.
  using BuilderTy = llvm::unique_function<ErrorOr<std::unique_ptr<Entitlement>>(
      bool, ArrayRef<uint8_t>)>;

  /// Register an entitlement builder in the static list. Users should interact
  /// with the templated version above.
  static void registerBuilder(Kind k, BuilderTy builder);
};

/// Each entitlement subclass must provide the following static methods:
///
///  class MyEntitlement {
///    static bool classof(const Entitlement *e); // For LLVM-style RTTI
///    static Kind getKind(); // Used during parsing
///    // Required to construct an entitlement of this type during parsing.
///    static ErrorOr<std::unique_ptr<Entitlement>> create(bool critical,
///                                                        ArrayRef<uint8_t>
///                                                        data);
///  }
///
/// Providing these static methods allows the generic parsing that the base
/// class is able to perform, which in turn, is required for setting up the
/// EntitlementStore. Any data provided to `create` is not owned by the
/// Entitlement, so simply taking a reference is unsafe. The Entitlement should
/// copy any data it plans to refer to after parsing.

/// When parsing an X.509 certificate, non-critical entitlements that are not
/// recognized (i.e. the current software doesn't know the mapping from OID to
/// C++ type), an UnknownEntitlement object is created. This drops `critical`
/// and `data` and does not carry any state.
class UnknownEntitlement : public Entitlement {
public:
  UnknownEntitlement() : Entitlement(EK_UNKNOWN) {}

  /// Provide LLVM-style RTTI support.
  static bool classof(const Entitlement *e) {
    return e->getKind() == EK_UNKNOWN;
  }

  /// Get the kind of all entitlements of this type.
  static Kind getKind() { return EK_UNKNOWN; }

  /// Returns the string "unknown" - that's the name of this entitlement.
  StringRef getName() const override;

  /// This is the static builder that needs to be registered so it can be called
  /// during parsing.
  static ErrorOr<std::unique_ptr<Entitlement>> create(bool critical,
                                                      ArrayRef<uint8_t> data);
};

/// This entitlement is only set for accounts belonging to (or roles controlled
/// by) Modular developers.
class ModularDeveloperEntitlement : public Entitlement {
public:
  ModularDeveloperEntitlement() : Entitlement(EK_MODULAR_DEVELOPER) {}

  /// Provide LLVM-style RTTI support.
  static bool classof(const Entitlement *e) {
    return e->getKind() == EK_MODULAR_DEVELOPER;
  }

  /// Get the kind of all entitlements of this type.
  static Kind getKind() { return EK_MODULAR_DEVELOPER; }

  /// Returns the string "modular-developer" - that's the name of this
  /// entitlement.
  StringRef getName() const override;

  /// This is the static builder that needs to be registered so it can be called
  /// during parsing.
  static ErrorOr<std::unique_ptr<Entitlement>> create(bool critical,
                                                      ArrayRef<uint8_t> data);
};

/// Entitles users access to generic beta software & features.
class BetaEntitlement : public Entitlement {
public:
  BetaEntitlement() : Entitlement(EK_BETA) {}
  static bool classof(const Entitlement *e) { return e->getKind() == EK_BETA; }
  static Kind getKind() { return EK_BETA; }

  StringRef getName() const override;
  static ErrorOr<std::unique_ptr<Entitlement>> create(bool critical,
                                                      ArrayRef<uint8_t> data);
};

/// Entitles users to access GPU-related features.
class GPUEntitlement : public Entitlement {
public:
  GPUEntitlement() : Entitlement(EK_GPU) {}
  static bool classof(const Entitlement *e) { return e->getKind() == EK_GPU; }
  static Kind getKind() { return EK_GPU; }

  StringRef getName() const override;
  static ErrorOr<std::unique_ptr<Entitlement>> create(bool critical,
                                                      ArrayRef<uint8_t> data);
};

/// Restricts users to a specific number of threads.
class MaxThreadsUnlimitedEntitlement : public Entitlement {
public:
  MaxThreadsUnlimitedEntitlement() : Entitlement(EK_MAX_THREADS_UNLIMITED) {}
  static bool classof(const Entitlement *e) {
    return e->getKind() == EK_MAX_THREADS_UNLIMITED;
  }
  static Kind getKind() { return EK_MAX_THREADS_UNLIMITED; }

  StringRef getName() const override;
  static ErrorOr<std::unique_ptr<Entitlement>> create(bool critical,
                                                      ArrayRef<uint8_t> data);
};

/// Restricts users to a specific number of threads.
class EnterpriseEntitlement : public Entitlement {
public:
  EnterpriseEntitlement() : Entitlement(EK_ENTERPRISE) {}
  static bool classof(const Entitlement *e) {
    return e->getKind() == EK_ENTERPRISE;
  }
  static Kind getKind() { return EK_ENTERPRISE; }

  StringRef getName() const override;
  static ErrorOr<std::unique_ptr<Entitlement>> create(bool critical,
                                                      ArrayRef<uint8_t> data);
};

/// Register all the entitlements declared here.
void registerAllEntitlements();
} // namespace M

#endif // SUPPORT_ENTITLEMENTS_ENTITLEMENT_H
