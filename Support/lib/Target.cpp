//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Target.h"
#include "Support/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace M;

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

std::string M::getHostCPUFeatures() {
  ErrorOr<HostMachineInfo> hostOr = getHostMachineInfo();
  if (hostOr.isError())
    return "";
  return M::encodeFeatures(hostOr->cpuFeatures);
}

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
M::getTargetMachineForHost(bool isJIT, llvm::CodeGenOpt::Level optLevel) {
  ErrorOr<HostMachineInfo> hostOr = getHostMachineInfo();
  if (hostOr.isError())
    return hostOr.takeError();
  HostMachineInfo host = std::move(*hostOr);
  std::string targetFeatures = encodeFeatures(host.cpuFeatures);

  std::string errorMessage;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(host.triple, errorMessage);
  if (!target)
    return Error("no target exists for '" + host.triple + "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      host.triple, host.cpuArch, targetFeatures,
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/std::nullopt, /*OL=*/optLevel, /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}
