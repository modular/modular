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

std::string M::getCPUFeatures(HostMachineInfo &hostMachineInfo) {
  std::string featureStr;
  llvm::raw_string_ostream os(featureStr);
  llvm::interleave(
      hostMachineInfo.cpuFeatures, os, [&](auto &f) { os << '+' << f; }, ",");
  return featureStr;
}

std::string M::getHostCPUFeatures() {
  ErrorOr<HostMachineInfo> hostOr = getHostMachineInfo();
  if (hostOr.isError())
    return "";
  return M::getCPUFeatures(*hostOr);
}

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
M::getTargetMachineForHost(bool isJIT, llvm::CodeGenOpt::Level optLevel) {
  ErrorOr<HostMachineInfo> hostOr = getHostMachineInfo();
  if (hostOr.isError())
    return hostOr.takeError();
  HostMachineInfo host = std::move(*hostOr);
  std::string targetFeatures = getCPUFeatures(host);

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
