//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/Target/TargetBackend.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"

namespace M::KGEN {

void addAddressSanitizerPass(llvm::ModulePassManager &mpm,
                             const CompilationOptions &options) {
  if (!options.sanitizers.has(M::Sanitizers::kAddress))
    return;
  llvm::AddressSanitizerOptions opts;
  bool moduleUseAfterScope = false;
  bool useOdrIndicator = false;
  mpm.addPass(
      llvm::AddressSanitizerPass(opts, moduleUseAfterScope, useOdrIndicator));
}

void addThreadSanitizerPass(llvm::ModulePassManager &mpm,
                            const CompilationOptions &options) {
  if (!options.sanitizers.has(M::Sanitizers::kThread))
    return;
  mpm.addPass(llvm::ModuleThreadSanitizerPass());
  mpm.addPass(
      llvm::createModuleToFunctionPassAdaptor(llvm::ThreadSanitizerPass()));
}

void TargetBackend::addSanitizers(llvm::ModulePassManager &mpm,
                                  const CompilationOptions &options) const {
  addAddressSanitizerPass(mpm, options);
  addThreadSanitizerPass(mpm, options);
}

void TargetBackend::emitBitcode(llvm::Module &module,
                                llvm::raw_pwrite_stream &os) const {
  llvm::WriteBitcodeToFile(module, os, /*ShouldPreserveUseListOrder=*/true);
}

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
defaultCreateTargetMachine(const CompilationOptions &options, bool isJIT) {
  std::string errorMessage;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(options.targetTriple), errorMessage);
  if (!target) {
    return Error("no target exists for '" + options.targetTriple +
                 "': " + errorMessage);
  }

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(options.targetTriple), options.targetCpu,
      options.targetFeatures,
      /*Options=*/{}, options.relocModel, /*CM=*/options.mcmodel,
      /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  if (options.largeDataThreshold)
    machine->setLargeDataThreshold(options.largeDataThreshold.value());

  return machine;
}

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
TargetBackend::createTargetMachine(const CompilationOptions &options,
                                   bool isJIT) const {
  return defaultCreateTargetMachine(
      adjustOptionsForTargetMachine(options, options.targetTriple), isJIT);
}

static llvm::ManagedStatic<TargetBackendRegistry> TheRegistry;

TargetBackendRegistry &TargetBackendRegistry::get() { return *TheRegistry; }

void TargetBackendRegistry::add(std::unique_ptr<TargetBackend> backend) {
  Backends.push_back(std::move(backend));
}

const TargetBackend *
TargetBackendRegistry::lookup(const llvm::Triple &triple) const {
  for (const std::unique_ptr<TargetBackend> &backend : Backends)
    if (backend->matches(triple))
      return backend.get();
  return nullptr;
}

} // namespace M::KGEN
