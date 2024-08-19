//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Runtime representation of TargetInfoAttr, DeviceRefAttr, DeviceSpecAttr and
// DeviceCollectionAttr. These can be used at runtime to both confirm the
// runtime environment matches that expected at compile time, and help
// models establish which actual runtime devices correspond to each
// 'abstract' device they assumed at compile time.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEVICE_SPECS_H
#define SUPPORT_DEVICE_SPECS_H

#include "Support/ErrorOr.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/TargetParser/Triple.h"

namespace M {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the given features in "+feature1,+feature2" form.
std::string encodeFeatures(ArrayRef<std::string> features);

/// Decodes the result of encodeFeatures.
ErrorOr<std::vector<std::string>> decodeFeatures(StringRef encodedFeatures);

//===----------------------------------------------------------------------===//
// TargetInfo
//===----------------------------------------------------------------------===//

/// Runtime analogue of part of TargetInfoAttr.
///
/// See also HostMachineInfo and PackageTarget which have similar
/// structure.
struct TargetInfo {
  llvm::Triple triple;
  std::string arch;
  std::vector<std::string> features;

  TargetInfo(llvm::Triple triple = llvm::Triple(""), std::string arch = {},
             std::vector<std::string> features = {})
      : triple(std::move(triple)), arch(std::move(arch)),
        features(std::move(features)) {}

  /// Serializes this target info to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;
  std::string serializeToJSON() const;

  /// Returns the target info deserialized from JSON.
  static ErrorOr<TargetInfo> deserializeFromJSON(const llvm::json::Value *json);
  static ErrorOr<TargetInfo> deserializeFromJSON(StringRef json);

  /// Returns error if this target info does not satisfy the assumptions
  /// in required.
  ErrorOrSuccess checkSatisfiesRequirements(const TargetInfo &required) const;
};

//===----------------------------------------------------------------------===//
// DeviceRef
//===----------------------------------------------------------------------===//

/// Compile time index of a particular abstract device amongst others with the
/// same or similar target info. The relationship between the device id
/// used in an #M.device_spec and an actual physical device (eg a CUDA device
/// id) is determined at runtime.
using DeviceId = uint64_t;

struct DeviceRef {
  std::string label;
  DeviceId id;

  explicit DeviceRef(std::string label = {}, DeviceId id = 0)
      : label(std::move(label)), id(id) {}

  /// Serialized this device ref to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;

  /// Returns the device ref deserialized from JSON.
  static ErrorOr<DeviceRef> deserializeFromJSON(const llvm::json::Value *json);

  /// Returns device reference in compact string form.
  std::string toString() const;

  bool operator==(const DeviceRef &that) const {
    return std::tie(label, id) == std::tie(that.label, that.id);
  }
};

/// Well known device labels.
constexpr const char *kCPULabel = "cpu";
constexpr const char *kCUDALabel = "cuda";

} // namespace M

namespace llvm {

template <>
struct DenseMapInfo<M::DeviceRef> {
  static inline M::DeviceRef getEmptyKey() { return M::DeviceRef(); }
  static inline M::DeviceRef getTombstoneKey() { return M::DeviceRef("", -1); }
  static unsigned getHashValue(const M::DeviceRef &ref) {
    return hash_value(std::make_pair(ref.label, ref.id));
  }
  static bool isEqual(const M::DeviceRef &lhs, const M::DeviceRef &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

namespace M {

//===----------------------------------------------------------------------===//
// DeviceSpec
//===----------------------------------------------------------------------===//

/// Runtime analogue of DeviceSpecAttr.
struct DeviceSpec {
  DeviceRef ref;
  TargetInfo target;

  /// Serialized this device spec to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;
  std::string serializeToJSON() const;

  /// Returns the device spec deserialized from JSON.
  static ErrorOr<DeviceSpec> deserializeFromJSON(const llvm::json::Value *json);
  static ErrorOr<DeviceSpec> deserializeFromJSON(StringRef json);
};

/// A map from device references (from the 'required' devices) to
/// the pair of device specs (required, provided), where provided is the
/// matching devices spec available in the runtime environment.
using DeviceSpecMap =
    llvm::DenseMap<DeviceRef, std::pair<DeviceSpec, DeviceSpec>>;

/// A reference counted version of DeviceSpecMap, suitable for use in the
/// runtime.
struct RuntimeDeviceSpecMap : public ReferenceCounted<RuntimeDeviceSpecMap> {
  DeviceSpecMap map;

  RuntimeDeviceSpecMap(DeviceSpecMap map) : map(std::move(map)) {}
};

//===----------------------------------------------------------------------===//
// DeviceSpecCollection
//===----------------------------------------------------------------------===//

/// Runtime analogue of DeviceSpecCollectionAttr.
struct DeviceSpecCollection {
  DeviceRef host;
  std::vector<DeviceSpec> devices;

  /// Serializes this devices spec collection to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;
  std::string serializeToJSON() const;

  /// Returns the device spec collection deserialized from JSON.
  static ErrorOr<DeviceSpecCollection>
  deserializeFromJSON(const llvm::json::Value *json);
  static ErrorOr<DeviceSpecCollection> deserializeFromJSON(StringRef json);

  /// Returns a map from each device reference in required to the pair of
  /// devices specs (requiredSpec, providedSpec) where:
  ///  - requiredSpec is the device spec in the required collection.
  ///  - providedSpec is the matching device spec in this collection which
  ///    meets the requiredSpec's requirements.
  ///
  /// Returns an error if there's a target in required which has no satisfying
  /// target in this collection. A device specification in this collection can
  /// appear in the result map at most once. It is ok for this collection to
  /// have more device specifications than required.
  ///
  /// This method is generally called on a collection representing all the
  /// available devices on the host. The result can be used by primitives within
  /// the model's 'init' block to guide which 'physical' devices (eg a CUDA
  /// device id) to use for each 'virtual' device used in the model.
  ErrorOr<DeviceSpecMap>
  reconcileDeviceSpecs(const DeviceSpecCollection &required) const;

  /// Returns the device spec which matches device reference, or an error if
  /// no match.
  ErrorOr<const DeviceSpec *> findDeviceSpec(const DeviceRef &ref) const;

  /// Returns the device spec corresponding to the 'host' in this collection.
  /// (Not to be confused with the actual host machine.)
  const DeviceSpec &getHostDeviceSpec() const;
};

//===----------------------------------------------------------------------===//
// SIMD Width
//===----------------------------------------------------------------------===//

/// Gets the SIMD width from the processor features. The features are comma
/// separated.
size_t simdWidthFromFeatures(StringRef features);
size_t simdWidthFromFeatures(ArrayRef<std::string> features);

} // namespace M

#endif // SUPPORT_DEVICE_SPECS_H
