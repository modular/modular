//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Runtime representation of TargetInfoAttr, DeviceAttr and
// ArrayRef<DeviceAttr>. This can be used at runtime to both confirm the
// runtime environment matches that expected at compile time, and help
// models establish which actual runtime devices correspond to each
// 'abstract' device they assumed at compile time.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEVICE_SPECS_H
#define SUPPORT_DEVICE_SPECS_H

#include "Support/Host.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/TargetParser/Triple.h"

namespace M {

//===----------------------------------------------------------------------===//
// TargetInfo
//===----------------------------------------------------------------------===//

/// Runtime analogue of part of TargetInfoAttr.
///
/// See also HostMachineInfo and PackageTarget which have similar
/// structure.
struct TargetInfo {
  llvm::Triple triple;
  std::string cpu;
  std::vector<std::string> features;

  /// Returns the target info partially describing the given HostMachineInfo.
  /// Only some fields are captured:
  ///  - triple (captured as triple)
  ///  - cpuArch (captured as cpu)
  ///  - cpuFeatures (captured as features)
  ///
  /// CAUTION: Temporary while we unravel the TargetInfoAttr/HostMachineInfo
  ///          confusion.
  static TargetInfo fromHostMachineInfo(const HostMachineInfo &hostMachineInfo);

  /// Returns the target info describing the current CPU host.
  static ErrorOr<TargetInfo> fromCPUHost();

  /// Returns a HostMachineInfo matching this target info. Only some fields
  /// are captured:
  ///  - triple (captured as triple and osName)
  ///  - cpu (captured as cpuArch)
  ///  - features (captured as cpuFeatures)
  ///
  /// CAUTION: Temporary while we unravel the TargetInfoAttr/HostMachineInfo
  ///          confusion.
  HostMachineInfo toHostMachineInfo() const;

  /// Serializes this target info to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;

  /// Returns the target info deserialized from JSON.
  static ErrorOr<TargetInfo> deserializeFromJSON(const llvm::json::Value *json);

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
using DeviceId = int64_t;

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

  /// Returns the device spec deserialized from JSON.
  static ErrorOr<DeviceSpec> deserializeFromJSON(const llvm::json::Value *json);
};

/// A map from device references to their corresponding device spec.
using DeviceSpecMap = llvm::DenseMap<DeviceRef, DeviceSpec>;

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

  /// Returns the device spec collection describing the current CPU host.
  /// It will contain only a single device. The caller is responsible for
  /// dealing with more exotic devices (eg all the available CUDA devices).
  static ErrorOr<DeviceSpecCollection> fromCPUHost();

  /// Returns a map from each device reference in required to the corresponding
  /// device specification in this collection which meets the required device's
  /// requirements. Returns an error if there's a target in required which has
  /// no satisfying target in this collection. A device specification in this
  /// collection can appear in the result map at most once. It is ok for this
  /// collection to have more device specifications than required.
  ///
  /// This method is generally called on the (augmented) result of
  /// fromCPUHost(), with required equal to the device specifications recovered
  /// from the model being setup for execution. The result can be used by
  /// primitives within the model's 'init' block to guide which 'physical'
  /// devices (eg a CUDA device id) to use for each 'virtual' device needed by
  /// the model.
  ErrorOr<DeviceSpecMap>
  reconcileDeviceSpecs(const DeviceSpecCollection &required) const;

  /// Returns the device spec which matches device reference, or an error if
  /// no match.
  ErrorOr<const DeviceSpec *> findDeviceSpec(const DeviceRef &ref) const;

  /// Returns the device spec for the host.
  const DeviceSpec &getHostDeviceSpec() const;
};

} // namespace M

#endif // SUPPORT_DEVICE_SPECS_H
