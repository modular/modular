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
// TargetInfo
//===----------------------------------------------------------------------===//

TargetInfo
TargetInfo::fromHostMachineInfo(const HostMachineInfo &hostMachineInfo) {
  TargetInfo result;
  result.triple = llvm::Triple(hostMachineInfo.triple);
  result.cpu = hostMachineInfo.cpuArch;
  result.features = hostMachineInfo.cpuFeatures;
  return result;
}

ErrorOr<TargetInfo> TargetInfo::fromCPUHost() {
  ErrorOr<HostMachineInfo> hostMachineInfoOr = getHostMachineInfo();
  if (hostMachineInfoOr)
    return hostMachineInfoOr.takeError();
  return fromHostMachineInfo(*hostMachineInfoOr);
}

HostMachineInfo TargetInfo::toHostMachineInfo() const {
  HostMachineInfo result;
  result.triple = triple.str();
  result.osName = llvm::Triple::getOSTypeName(triple.getOS());
  result.cpuArch = cpu;
  result.cpuFeatures = features;
  return result;
}

void TargetInfo::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("triple", triple.str());
  json.attribute("cpu", cpu);
  json.attribute("features", features);
  json.objectEnd();
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
  if (required == llvm::Triple(""))
    // No constraint.
    return success();

  std::string providedArch = canonArchName(provided.getArchName());
  std::string requiredArch = canonArchName(required.getArchName());
  if (!requiredArch.empty() && providedArch != requiredArch) {
    return Error(Twine("host architecture '") + provided.getArchName() +
                 "' does not match required architecture '" +
                 required.getArchName() + "'");
  }

  llvm::Triple::OSType providedOS = canonOSType(provided.getOS());
  llvm::Triple::OSType requiredOS = canonOSType(required.getOS());
  if (requiredOS != llvm::Triple::OSType::UnknownOS &&
      providedOS != requiredOS) {
    return Error(Twine("host OS '") + provided.getOSName() +
                 "' does not match required OS '" + required.getOSName() + "'");
  }

  return success();
}

/// Returns success if provided cpu matches required cpu.
static ErrorOrSuccess satisfiesCPU(StringRef provided, StringRef required) {
  if (required.empty())
    // No constraint.
    return success();

  if (provided != required) {
    return Error(Twine("host CPU '") + provided +
                 "' does not match required CPU '" + required + "'");
  }
  return success();
}

/// Returns success if provided features are a superset of required features.
static ErrorOrSuccess
satisfiesFeatures(const std::vector<std::string> &provided,
                  const std::vector<std::string> &required) {
  if (required.empty())
    // No constraint (following code would also return false).
    return success();

  DenseSet<StringRef> providedSet(provided.begin(), provided.end());
  DenseSet<StringRef> requiredSet(required.begin(), required.end());
  bool superset = llvm::set_is_subset(requiredSet, providedSet);
  if (superset)
    return success();
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "host is missing the following feature(s) required by model: ";
  llvm::interleaveComma(llvm::set_difference(requiredSet, providedSet), os);
  return Error(str);
}

ErrorOrSuccess
TargetInfo::checkSatisfiesRequirements(const TargetInfo &required) const {
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

ErrorOr<DeviceSpecCollection> DeviceSpecCollection::fromCPUHost() {
  ErrorOr<HostMachineInfo> hostMachineInfoOr = getHostMachineInfo();
  if (hostMachineInfoOr)
    return hostMachineInfoOr.takeError();

  DeviceSpec deviceSpec;
  deviceSpec.ref.label = "cpu";
  deviceSpec.target = TargetInfo::fromHostMachineInfo(*hostMachineInfoOr);

  DeviceSpecCollection result;
  result.host.label = deviceSpec.ref.label;
  result.devices.emplace_back(std::move(deviceSpec));
  return result;
}

ErrorOr<DeviceSpecMap> DeviceSpecCollection::reconcileDeviceSpecs(
    const DeviceSpecCollection &required) const {
  LLVM_DEBUG(llvm::dbgs() << "provided:\n" << serializeToJSON() << "\n\n");
  LLVM_DEBUG(llvm::dbgs() << "required:\n"
                          << required.serializeToJSON() << "\n\n");

  if (required.devices.size() > devices.size()) {
    return Error(Twine("model requires ") + Twine(required.devices.size()) +
                 " devices but only " + Twine(devices.size()) +
                 " are provided.");
  }

  DeviceSpecMap result;
  std::vector<DeviceSpec> unusedProvidedDevices(devices.begin(), devices.end());

  for (const auto &requiredDevice : required.devices) {
    if (result.contains(requiredDevice.ref)) {
      return Error(Twine("the device reference '") +
                   requiredDevice.ref.toString() +
                   "' is shared between required device specifications.");
    }

    auto itr = llvm::find_if(
        unusedProvidedDevices, [&](const DeviceSpec &providedDevice) {
          if (auto errOr = providedDevice.target.checkSatisfiesRequirements(
                  requiredDevice.target)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "no match: " << errOr.getError() << "\n");
            return false;
          }
          return true;
        });

    if (itr == unusedProvidedDevices.end()) {
      std::string str;
      llvm::raw_string_ostream os(str);
      os << "unable to find a runtime device to match the requirements for '";
      os << requiredDevice.ref.toString();
      os << "' from amongst the yet-to-be matched devices ";
      llvm::interleaveComma(unusedProvidedDevices, os,
                            [&](const DeviceSpec &unusedProvidedDevice) {
                              os << "'" << unusedProvidedDevice.ref.toString()
                                 << "'";
                            });
      os << ".";
      return Error(str);
    }

    result.insert(std::make_pair(requiredDevice.ref, *itr));
    unusedProvidedDevices.erase(itr);
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
