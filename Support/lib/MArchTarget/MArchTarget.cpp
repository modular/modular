//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MArchTarget/MArchTarget.h"
#include "Support/DeviceSpecs.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>
#include <optional>
#include <string>

using namespace M;

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
M::getTargetMachineForHost(bool isJIT, llvm::CodeGenOptLevel optLevel) {
  std::string hostTriple = llvm::sys::getDefaultTargetTriple();
  std::string hostCpu(llvm::sys::getHostCPUName());
  std::string targetFeatures = getHostCPUFeatures();

  std::string errorMessage;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(hostTriple), errorMessage);
  if (!target)
    return Error("no target exists for '" + hostTriple + "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(hostTriple), hostCpu, targetFeatures,
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/std::nullopt, /*OL=*/optLevel, /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

ErrorOr<TargetInfoAttr>
M::getMArchFeatures(MLIRContext *ctx, StringRef targetTriple, StringRef march,
                    StringRef mcpu, StringRef mtune, StringRef acceleratorArch,
                    llvm::Reloc::Model relocModel) {
  auto runtimeTargetInfoOr =
      getMArchTargetInfo(targetTriple, march, mcpu, mtune);
  if (runtimeTargetInfoOr)
    return runtimeTargetInfoOr.takeError();

  return getTargetInfoFor(
      ctx, runtimeTargetInfoOr->triple.str(), runtimeTargetInfoOr->arch,
      encodeFeatures(*runtimeTargetInfoOr), mtune, acceleratorArch, relocModel);
}
