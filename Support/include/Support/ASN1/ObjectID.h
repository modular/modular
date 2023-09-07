//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ASN1_OBJECTID_H
#define SUPPORT_ASN1_OBJECTID_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include <array>

namespace M::ASN1 {
/// This class provides an ASN1-compatible Object Identifier. By default, we
/// assume that an OID is an arc under the Modular prefix (which is defined in
/// the class). This assumption allows us to use a more compact encoding in
/// those cases. That said, this class still allows the user to parse
/// non-Modular OID objects, in which case the arc should be considered an arc
/// under the global namespace.
class ObjectID {
public:
  /// An instance of this class by default has the modular prefix, but the user
  /// can choose to construct an instance without one. The user can also provide
  /// the full arc up front, if it's known at construction.
  explicit ObjectID(bool withModularPrefix = true, ArrayRef<uint64_t> arc = {});

  /// ObjectIDs are copyable and move-able (copyable because we want to use them
  /// as DenseMap keys).
  ObjectID(const ObjectID &other)
      : ObjectID(other.isModularOID(), other.getArc()) {}
  ObjectID &operator=(const ObjectID &other) = default;
  ObjectID(ObjectID &&other) : ObjectID(other.isModularOID(), other.getArc()) {}
  ObjectID &operator=(ObjectID &&other) = default;

  /// Represents a known invalid arc in the Modular OID namespace. If a Modular
  /// OID arc begins with this, the OID can be considered invalid/pointing to
  /// nothing.
  static constexpr uint64_t invalidArc = 0;

  /// Add one or more nodes onto the end of the OID.
  void append(ArrayRef<uint64_t> nodes);

  /// Returns the encoded form of the prefix that all Modular ObjectID will
  /// have.
  static ArrayRef<uint8_t> getEncodedModularPrefix() {
    return {(const uint8_t *)encodedModularPrefix.data(),
            encodedModularPrefix.size()};
  }

  /// Return the uint64_t form of the prefix that all Modular ObjectID will
  /// have.
  static ArrayRef<uint64_t> getModularPrefix() {
    return {(const uint64_t *)modularPrefix.data(), modularPrefix.size()};
  }

  /// Return the full encoded form of this ObjectID. This is the concatenation
  /// of the prefix (if any) and the arc.
  SmallVector<uint8_t> getEncoded() const;

  /// Return the arc of the ObjectID. If this is a Modular OID, all the prefixes
  /// are exactly the same so we can compare just the arc. If it's a non-Modular
  /// OID, the arc contains all the numbers in the sequence.
  ArrayRef<uint64_t> getArc() const { return arc; }

  /// Check if this OID is an arc under the Modular OID namespace.
  bool isModularOID() const { return withModularPrefix; }

  /// Return an ObjectID from its encoded state. This is capable of parsing
  /// Modular and non-Modular namespaced OIDs.
  static ErrorOr<ObjectID> fromEncoded(ArrayRef<uint8_t> buf);

  /// Parse an ObjectID from a string. The OID must be in dot notation with
  /// decimal numbers, e.g. 1.2.3.4.
  static ErrorOr<ObjectID> fromString(StringRef str);

private:
  // TODO (#20183): Apply for an enterprise number here:
  //   https://www.iana.org/assignments/enterprise-numbers/assignment/apply/,
  //   Eventually:
  //   iso.org.dod.internet.private.enterprise.modular 1.3.6.1.4.1.XXXXX

  /// For now, we're using an arc under the experimental
  /// arc 1.3.6.1.3.77.XXX. Why 77? 77 is the ASCII code for `M`.
  static constexpr std::array<uint8_t, 5> encodedModularPrefix = {40 * 1 + 3, 6,
                                                                  1, 3, 77};
  static constexpr std::array<uint64_t, 6> modularPrefix = {1, 3, 6, 1, 3, 77};

  /// Whether this OID is in the Modular namespace or not.
  bool withModularPrefix;
  /// We use 4 as the 'small' number of elements because we're extremely
  /// unlikely to have an arc with more than 2 or 3 elements past the prefix for
  /// the common (Modular-namespaced) case.
  SmallVector<uint64_t, 4> arc;
};

/// Check if two ObjectID objects are semantically equal.
bool operator==(const ObjectID &lhs, const ObjectID &rhs);
} // namespace M::ASN1

/// Print the ObjectID in dot notation. Place this in the M:: namespace so that
/// we need fewer `using` declarations if we decide to print an ObjectID.
namespace M {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ASN1::ObjectID &oid);
}

/// Provide a DenseMapInfo for ObjectID so we can use it as the key in
/// DenseMaps.
namespace llvm {
template <>
struct DenseMapInfo<M::ASN1::ObjectID> {
  static inline M::ASN1::ObjectID getEmptyKey() { return M::ASN1::ObjectID(); }

  static inline M::ASN1::ObjectID getTombstoneKey() {
    return M::ASN1::ObjectID(/*withModularPrefix=*/true,
                             {M::ASN1::ObjectID::invalidArc});
  }

  static unsigned getHashValue(const M::ASN1::ObjectID &val) {
    return llvm::hash_combine(val.getArc(), val.isModularOID());
  }

  static bool isEqual(const M::ASN1::ObjectID &lhs,
                      const M::ASN1::ObjectID &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // SUPPORT_ASN1_OBJECTID_H
