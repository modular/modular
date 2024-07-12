//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace M;

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
M::getTargetMachineForHost(bool isJIT, llvm::CodeGenOptLevel optLevel) {
  std::string hostTriple = llvm::sys::getDefaultTargetTriple();
  std::string hostCpu(llvm::sys::getHostCPUName());
  std::string targetFeatures = getHostCPUFeatures();

  std::string errorMessage;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(hostTriple, errorMessage);
  if (!target)
    return Error("no target exists for '" + hostTriple + "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      hostTriple, hostCpu, targetFeatures,
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/std::nullopt, /*OL=*/optLevel, /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

ErrorOr<TargetInfoAttr> M::getMArchFeatures(MLIRContext *ctx, StringRef march,
                                            StringRef mcpu, StringRef mtune) {
  auto runtimeTargetInfoOr = getMArchTargetInfo(march, mcpu, mtune);
  if (runtimeTargetInfoOr)
    return runtimeTargetInfoOr.takeError();

  return getTargetInfoFor(ctx, runtimeTargetInfoOr->triple.str(),
                          runtimeTargetInfoOr->arch,
                          encodeFeatures(runtimeTargetInfoOr->features), mtune);
}
