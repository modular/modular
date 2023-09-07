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

std::string M::encodeFeatures(ArrayRef<std::string> features) {
  std::string featureStr;
  llvm::raw_string_ostream os(featureStr);
  llvm::interleave(
      features, os, [&](auto &f) { os << '+' << f; }, ",");
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
  json.attribute("cpu", cpu);
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
  std::optional<StringRef> optCpu = object->getString("cpu");
  const llvm::json::Array *array = object->getArray("features");
  if (!optTriple || !optCpu || !array)
    return Error("ill-formed serialized TargetInfo: missing attributes");

  TargetInfo result;
  result.triple = llvm::Triple(*optTriple);
  result.cpu = *optCpu;
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
  if (!requiredArch.empty() && providedArch != requiredArch) {
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

/// Returns success if provided cpu matches required cpu.
static ErrorOrSuccess satisfiesCPU(StringRef provided, StringRef required) {
  if (required.empty())
    // No constraint.
    return success();

  if (provided != required) {
    return Error(Twine("Provided CPU '") + provided +
                 "' does not match required CPU '" + required + "'.");
  }
  return success();
}

/// Adds feature to map. If feature supports multiple versions, map will
/// contain base feature as key and version as value. Otherwise the
/// entire feature will be the key and 0 the value.
static void parseFeature(StringRef feature, llvm::StringMap<int> &map) {
  static StringRef prefix("compute_");
  if (feature.find(prefix) == 0) {
    StringRef suffix = feature.drop_front(prefix.size());
    int version;
    if (!suffix.getAsInteger(10, version)) {
      map[prefix] = version;
      return;
    }
  }
  map[feature] = 0;
}

/// Returns features parsed into map form, where the map domain is the base
/// feature name and the map range is the feature's level (or just 0 if the
/// feature is boolean).
static llvm::StringMap<int>
parseFeatures(const std::vector<std::string> &features) {
  llvm::StringMap<int> map;
  for (const auto &feature : features)
    parseFeature(feature, map);
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

  llvm::StringMap<int> providedMap = parseFeatures(provided);
  llvm::StringMap<int> requiredMap = parseFeatures(required);
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "The following features are required but not provided: ";
  bool anyMissing = false;
  for (const auto &[requiredBase, requiredVersion] : requiredMap) {
    auto providedItr = providedMap.find(requiredBase);
    if (providedItr == providedMap.end()) {
      if (anyMissing)
        os << ", ";
      os << requiredBase;
      if (requiredVersion)
        os << requiredVersion;
      anyMissing = true;
    } else if (providedItr->second < requiredVersion) {
      if (anyMissing)
        os << ", ";
      os << requiredBase << requiredVersion;
      os << " (only have " << providedItr->getKey() << providedItr->second
         << ")";
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
  if (auto errOr = satisfiesCPU(cpu, required.cpu))
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
