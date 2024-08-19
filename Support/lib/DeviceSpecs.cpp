//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DeviceSpecs.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "device-specs"

using namespace M;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {
/// An arch or feature name decoded into 'base' and 'version' components.
struct VersionedName {
  // eg "sm_80".
  StringRef fullName;
  // eg "sm".
  StringRef baseName;
  // eg 80.
  uint64_t version;

  VersionedName(StringRef fullName, StringRef baseName, uint64_t version)
      : fullName(fullName), baseName(baseName), version(version) {}

  bool operator<=(const VersionedName &that) const {
    return baseName == that.baseName && version <= that.version;
  }

  static VersionedName decode(StringRef name) {
    // Currently just hard coded and not target specific.
    static StringRef prefix("sm_");
    if (name.find(prefix) == 0) {
      StringRef suffix = name.drop_front(prefix.size());
      uint64_t version;
      if (!suffix.getAsInteger(10, version))
        return VersionedName(name, prefix.drop_back(), version);
    }
    return VersionedName(name, name, 0);
  }
};

} // namespace

std::string M::encodeFeatures(ArrayRef<std::string> features) {
  std::string featureStr;
  llvm::raw_string_ostream os(featureStr);
  llvm::interleave(features, os, [&](auto &f) { os << '+' << f; }, ",");
  return featureStr;
}

ErrorOr<std::vector<std::string>> M::decodeFeatures(StringRef encodedFeatures) {
  std::vector<std::string> features;
  SmallVector<StringRef> plusFeatureCommas;
  encodedFeatures.split(plusFeatureCommas, ',', /*MaxSplit=*/-1,
                        /*KeepEmpty=*/false);
  for (StringRef plusFeatureComma : plusFeatureCommas) {
    if (plusFeatureComma.empty() || plusFeatureComma.front() != '+')
      return Error(Twine("ill-formed features: '") + encodedFeatures + "'");
    StringRef feature = plusFeatureComma.trim("+,");
    if (feature.empty())
      return Error("ill-formed features: " + encodedFeatures + "'");
    features.emplace_back(feature);
  }
  return features;
}

//===----------------------------------------------------------------------===//
// TargetInfo
//===----------------------------------------------------------------------===//

void TargetInfo::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("triple", triple.str());
  json.attribute("arch", arch);
  json.attribute("features", features);
  json.objectEnd();
}

std::string TargetInfo::serializeToJSON() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::json::OStream json(os);
  serializeToJSON(json);
  return str;
}

ErrorOr<TargetInfo>
TargetInfo::deserializeFromJSON(const llvm::json::Value *json) {
  const llvm::json::Object *object = json->getAsObject();
  if (!object)
    return Error("ill-formed serialized TargetInfo: expecting object");

  std::optional<StringRef> optTriple = object->getString("triple");
  std::optional<StringRef> optArch = object->getString("arch");
  const llvm::json::Array *array = object->getArray("features");
  if (!optTriple || !optArch || !array)
    return Error("ill-formed serialized TargetInfo: missing attributes");

  TargetInfo result;
  result.triple = llvm::Triple(*optTriple);
  result.arch = *optArch;
  for (const llvm::json::Value &v : *array) {
    std::optional<StringRef> optFeature = v.getAsString();
    if (!optFeature)
      return Error(
          "ill-formed serialized TargetInfo: expecting string feature");
    result.features.emplace_back(*optFeature);
  }
  return result;
}

ErrorOr<TargetInfo> TargetInfo::deserializeFromJSON(StringRef json) {
  llvm::Expected<llvm::json::Value> errOrValue = llvm::json::parse(json);
  if (llvm::Error err = errOrValue.takeError()) {
    return Error(Twine("ill-formed serialized target info: ") +
                 toString(std::move(err)));
  }
  return deserializeFromJSON(&errOrValue.get());
}

/// Returns canonicalized architecture name.
static std::string canonArchName(StringRef archName) {
  // arm64 can also mean aarch64.
  if (archName == "aarch64")
    return "arm64";
  return std::string(archName);
}

/// Returns canonicalized OS type.
static llvm::Triple::OSType canonOSType(llvm::Triple::OSType type) {
  // Darwin and macosx are equivalent.
  if (type == llvm::Triple::OSType::Darwin)
    return llvm::Triple::OSType::MacOSX;
  return type;
}

/// Returns success if provided triple matches required triple, up to
/// heuristics to account for version numbers, vendor names, etc.
static ErrorOrSuccess satisfiesTriple(const llvm::Triple &provided,
                                      const llvm::Triple &required) {
  LLVM_DEBUG(llvm::dbgs() << "provided: " << provided.str() << "\n");
  LLVM_DEBUG(llvm::dbgs() << "required: " << required.str() << "\n");

  if (required.str().empty()) {
    // No constraint.
    return success();
  }

  std::string providedArch = canonArchName(provided.getArchName());
  std::string requiredArch = canonArchName(required.getArchName());
  if (required.getArch() != llvm::Triple::ArchType::UnknownArch &&
      providedArch != requiredArch) {
    return Error(Twine("Provided architecture '") + provided.getArchName() +
                 "' does not match required architecture '" +
                 required.getArchName() + "'.");
  }

  llvm::Triple::OSType providedOS = canonOSType(provided.getOS());
  llvm::Triple::OSType requiredOS = canonOSType(required.getOS());
  if (requiredOS != llvm::Triple::OSType::UnknownOS &&
      providedOS != requiredOS) {
    return Error(Twine("Provided OS '") + provided.getOSName() +
                 "' does not match required OS '" + required.getOSName() +
                 "'.");
  }

  return success();
}

/// Returns success if provided arch matches required arch.
static ErrorOrSuccess satisfiesArch(StringRef provided, StringRef required) {
  if (required.empty())
    // No constraint.
    return success();

  auto versionedProvided = VersionedName::decode(provided);
  auto versionedRequired = VersionedName::decode(required);

  if (!(versionedRequired <= versionedProvided)) {
    return Error(Twine("Provided arch '") + provided +
                 "' does not match required arch '" + required + "'.");
  }
  return success();
}

/// Returns features parsed into map form, where the key is the feature
/// base name.
static llvm::StringMap<VersionedName>
parseFeatures(const std::vector<std::string> &features) {
  llvm::StringMap<VersionedName> map;
  for (const auto &feature : features) {
    // HACK: Ignore features emited in older versions of system-info when
    // matching against new versions. These are pretty much a given anyways
    // on just about every x64 CPU for the last 20 years and shouldn't be
    // missed.
    // TODO: remove once we clean up old manfiests.
    if (feature == "64Bit" || feature == "shstk" || feature == "cmov")
      continue;
    auto versionedFeature = VersionedName::decode(feature);
    map.insert({versionedFeature.baseName, versionedFeature});
  }
  return map;
}

/// Returns success if provided features are a superset of required
/// features.
static ErrorOrSuccess
satisfiesFeatures(const std::vector<std::string> &provided,
                  const std::vector<std::string> &required) {
  if (required.empty())
    // No constraint (following code would also return success).
    return success();

  llvm::StringMap<VersionedName> providedMap = parseFeatures(provided);
  llvm::StringMap<VersionedName> requiredMap = parseFeatures(required);
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "The following features are required but not provided: ";
  bool anyMissing = false;
  for (const auto &[requiredBase, requiredVersioned] : requiredMap) {
    auto providedItr = providedMap.find(requiredBase);
    if (providedItr == providedMap.end()) {
      if (anyMissing)
        os << ", ";
      os << requiredVersioned.fullName;
      anyMissing = true;
    } else if (!(requiredVersioned <= providedItr->second)) {
      if (anyMissing)
        os << ", ";
      os << requiredVersioned.fullName;
      os << " (only have " << providedItr->second.fullName << ")";
      anyMissing = true;
    }
  }
  if (anyMissing)
    return Error(str);
  return success();
}

ErrorOrSuccess
TargetInfo::checkSatisfiesRequirements(const TargetInfo &required) const {
  LLVM_DEBUG(llvm::dbgs() << "provided:\n" << serializeToJSON() << "\n\n");
  LLVM_DEBUG(llvm::dbgs() << "required:\n"
                          << required.serializeToJSON() << "\n\n");
  if (auto errOr = satisfiesTriple(triple, required.triple))
    return errOr.takeError();
  if (auto errOr = satisfiesArch(arch, required.arch))
    return errOr.takeError();
  if (auto errOr = satisfiesFeatures(features, required.features))
    return errOr.takeError();
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceRef
//===----------------------------------------------------------------------===//

void DeviceRef::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("label", label);
  json.attribute("id", id);
  json.objectEnd();
}

ErrorOr<DeviceRef>
DeviceRef::deserializeFromJSON(const llvm::json::Value *json) {
  const llvm::json::Object *object = json->getAsObject();
  if (!object)
    return Error("ill-formed serialized DeviceRef: expecting object");

  std::optional<StringRef> optLabel = object->getString("label");
  std::optional<DeviceId> optId = object->getInteger("id");
  if (!optLabel || !optId)
    return Error("ill-formed serialized DeviceRef: missing attributes");

  DeviceRef result;
  result.label = *optLabel;
  result.id = *optId;
  return result;
}

std::string DeviceRef::toString() const {
  std::string str;
  str += label;
  str += ":";
  str += std::to_string(id);
  return str;
}

//===----------------------------------------------------------------------===//
// DeviceSpec
//===----------------------------------------------------------------------===//

void DeviceSpec::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attributeBegin("ref");
  ref.serializeToJSON(json);
  json.attributeEnd();
  json.attributeBegin("target");
  target.serializeToJSON(json);
  json.attributeEnd();
  json.objectEnd();
}

std::string DeviceSpec::serializeToJSON() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::json::OStream json(os);
  serializeToJSON(json);
  return str;
}

ErrorOr<DeviceSpec> DeviceSpec::deserializeFromJSON(StringRef json) {
  llvm::Expected<llvm::json::Value> errOrValue = llvm::json::parse(json);
  if (llvm::Error err = errOrValue.takeError()) {
    return Error(Twine("ill-formed serialized target info: ") +
                 toString(std::move(err)));
  }
  return deserializeFromJSON(&errOrValue.get());
}

ErrorOr<DeviceSpec>
DeviceSpec::deserializeFromJSON(const llvm::json::Value *json) {
  const llvm::json::Object *object = json->getAsObject();
  if (!object)
    return Error("ill-formed serialized DeviceSpec: expecting object");

  const llvm::json::Value *ref = object->get("ref");
  const llvm::json::Value *target = object->get("target");
  if (!ref || !target)
    return Error("ill-formed serialized DeviceSpec: missing attributes");

  auto refOr = DeviceRef::deserializeFromJSON(ref);
  if (refOr) {
    return Error(Twine("ill-formed serialized DeviceSpec: ") +
                 refOr.getError());
  }

  auto targetOr = TargetInfo::deserializeFromJSON(target);
  if (targetOr) {
    return Error(Twine("ill-formed serialized DeviceSpec: ") +
                 targetOr.getError());
  }

  DeviceSpec result;
  result.ref = std::move(*refOr);
  result.target = std::move(*targetOr);
  return result;
}

//===----------------------------------------------------------------------===//
// DeviceSpecCollection
//===----------------------------------------------------------------------===//

void DeviceSpecCollection::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attributeBegin("host");
  host.serializeToJSON(json);
  json.attributeEnd();
  json.attributeBegin("devices");
  json.arrayBegin();
  for (const auto &device : devices)
    device.serializeToJSON(json);
  json.arrayEnd();
  json.attributeEnd();
  json.objectEnd();
}

std::string DeviceSpecCollection::serializeToJSON() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::json::OStream json(os);
  serializeToJSON(json);
  return str;
}

ErrorOr<DeviceSpecCollection>
DeviceSpecCollection::deserializeFromJSON(const llvm::json::Value *json) {
  const llvm::json::Object *object = json->getAsObject();
  if (!object) {
    return Error(
        "ill-formed serialized DeviceSpecCollection: expecting object");
  }
  const llvm::json::Value *host = object->get("host");
  const llvm::json::Array *devices = object->getArray("devices");
  if (!host || !devices) {
    return Error(
        "ill-formed serialized DeviceSpecCollection: missing attributes");
  }
  auto hostOr = DeviceRef::deserializeFromJSON(host);
  if (hostOr) {
    return Error(Twine("ill-formed serialized DeviceSpecCollection: ") +
                 hostOr.getError());
  }

  DeviceSpecCollection result;
  result.host = std::move(*hostOr);

  for (const auto &v : *devices) {
    auto errOr = DeviceSpec::deserializeFromJSON(&v);
    if (errOr) {
      return Error(Twine("ill-formed serialized DeviceSpecCollection: ") +
                   errOr.getError());
    }
    result.devices.emplace_back(std::move(*errOr));
  }

  return result;
}

ErrorOr<DeviceSpecCollection>
DeviceSpecCollection::deserializeFromJSON(StringRef json) {
  llvm::Expected<llvm::json::Value> errOrValue = llvm::json::parse(json);
  if (llvm::Error err = errOrValue.takeError()) {
    return Error(Twine("ill-formed serialized target info: ") +
                 toString(std::move(err)));
  }
  return deserializeFromJSON(&errOrValue.get());
}

ErrorOr<DeviceSpecMap> DeviceSpecCollection::reconcileDeviceSpecs(
    const DeviceSpecCollection &required) const {
  LLVM_DEBUG(llvm::dbgs() << "provided:\n" << serializeToJSON() << "\n\n");
  LLVM_DEBUG(llvm::dbgs() << "required:\n"
                          << required.serializeToJSON() << "\n\n");

  DeviceSpecMap result;
  std::vector<DeviceSpec> unusedProvidedDevices(devices.begin(), devices.end());

  for (const auto &requiredDevice : required.devices) {
    if (result.contains(requiredDevice.ref)) {
      return Error(Twine("the device reference '") +
                   requiredDevice.ref.toString() +
                   "' is shared between required device specifications.");
    }

    auto providedItr = llvm::find_if(
        unusedProvidedDevices, [&](const DeviceSpec &providedDevice) {
          if (auto errOr = providedDevice.target.checkSatisfiesRequirements(
                  requiredDevice.target)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "no match: " << errOr.getError() << "\n");
            return false;
          }
          return true;
        });

    if (providedItr == unusedProvidedDevices.end()) {
      std::string str;
      llvm::raw_string_ostream os(str);
      os << "Cannot find an available device matching the required device '";
      os << requiredDevice.ref.toString();
      os << "'.";

      if (unusedProvidedDevices.empty()) {
        os << " Require " << required.devices.size() << " devices but only "
           << devices.size() << " are provided.";
      } else if (unusedProvidedDevices.size() == 1) {
        auto errOr =
            unusedProvidedDevices.front().target.checkSatisfiesRequirements(
                requiredDevice.target);
        assert(errOr);
        os << " " << errOr.getError();
      } else {
        os << " No match was found from the available provided devices: ";
        llvm::interleaveComma(unusedProvidedDevices, os,
                              [&](const DeviceSpec &unusedProvidedDevice) {
                                os << "'" << unusedProvidedDevice.ref.toString()
                                   << "'";
                              });
        os << ".";
      }

      return Error(str);
    }

    result.insert(std::make_pair(requiredDevice.ref,
                                 std::make_pair(requiredDevice, *providedItr)));
    unusedProvidedDevices.erase(providedItr);
  }

  return result;
}

ErrorOr<const DeviceSpec *>
DeviceSpecCollection::findDeviceSpec(const DeviceRef &ref) const {
  auto itr = llvm::find_if(
      devices, [&ref](const DeviceSpec &device) { return device.ref == ref; });
  if (itr == devices.end()) {
    return Error(
        Twine("no such device spec for reference '" + ref.toString() + "'"));
  }
  return &(*itr);
}

const DeviceSpec &DeviceSpecCollection::getHostDeviceSpec() const {
  auto specOr = findDeviceSpec(host);
  assert(!specOr.isError() && "no such host device spec");
  return **specOr;
}

//===----------------------------------------------------------------------===//
// SIMD Width
//===----------------------------------------------------------------------===//

size_t M::simdWidthFromFeatures(StringRef feature) {
  if (feature.contains("avx512"))
    return 512;
  if (feature.contains("avx2"))
    return 256;
  // The fallback is going to be 128.
  return 128;
}

size_t M::simdWidthFromFeatures(ArrayRef<std::string> features) {
  size_t maxWidth = 128;
  for (StringRef feature : features) {
    maxWidth = std::max(maxWidth, simdWidthFromFeatures(feature));
    if (maxWidth == 512) {
      // We can return now. It can't get bigger than this.
      return maxWidth;
    }
  }
  return maxWidth;
}
