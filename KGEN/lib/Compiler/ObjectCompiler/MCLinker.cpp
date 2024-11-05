//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MCLinker.h"
#include "LLVMAccessorHelper.h"
#include "LLVMPassesPipeline.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace KGEN;

void SymbolAndMCInfo::clear() {
  symbolLinkageTypes.clear();
  mcInfos.clear();
}

MCLinker::MCLinker(
    SmallVectorImpl<SymbolAndMCInfo *> &symbolAndMCInfos,
    llvm::TargetMachine &targetMachine, CompilationOptions options,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes,
    llvm::StringMap<unsigned> originalFnOrdering)
    : symbolAndMCInfos(symbolAndMCInfos), targetMachine(targetMachine),
      options(std::move(options)),
      symbolLinkageTypes(std::move(symbolLinkageTypes)),
      originalFnOrdering(std::move(originalFnOrdering)) {

  llvm::LLVMTargetMachine &llvmTargetMachine =
      static_cast<llvm::LLVMTargetMachine &>(targetMachine);

  machineModInfoPass =
      new llvm::MachineModuleInfoWrapperPass(&llvmTargetMachine);
}

ErrorOrSuccess MCLinker::linkLLVMModules(StringRef moduleName) {
  ErrorOrSuccess createModuleResult =
      linkedModule.create([&](llvm::LLVMContext &ctx) {
        return std::make_unique<llvm::Module>(moduleName, ctx);
      });

  if (createModuleResult.isError())
    return Error("failed to create an empty LLVMModule for MCLinker");

  llvm::Linker linker(*linkedModule);

  for (auto [i, smcInfos] : llvm::enumerate(symbolAndMCInfos)) {
    for (auto &[key, value] : smcInfos->symbolLinkageTypes)
      symbolLinkageTypes.insert({key, value});

    for (auto [j, mcInfo] : llvm::enumerate(smcInfos->mcInfos)) {
      mcInfos.push_back(mcInfo.get());

      // Modules have to be in the same LLVMContext to be linked.
      llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
          llvm::parseBitcodeFile(
              llvm::MemoryBufferRef(
                  StringRef(mcInfo->moduleBuf->getBufferStart(),
                            mcInfo->moduleBuf->getBufferSize()),
                  ""),
              linkedModule->getContext());
      if (!moduleOr)
        return Error("failed to serialize post-llc modules");

      std::unique_ptr<llvm::Module> module = std::move(moduleOr.get());
      if (linker.linkInModule(std::move(module)))
        return Error("failed to link post-llc modules");
      mcInfo->mcContext->setUseNamesOnTempLabels(true);
    }
  }

  // Restore linkage type.
  for (llvm::GlobalValue &global : linkedModule->globals()) {
    if (!global.hasWeakLinkage())
      continue;
    auto iter = symbolLinkageTypes.find(global.getName().str());
    if (iter == symbolLinkageTypes.end())
      continue;

    global.setLinkage(iter->second);
    global.setDSOLocal(true);
  }

  for (llvm::Function &fn : linkedModule->functions()) {
    if (!fn.hasWeakLinkage())
      continue;

    auto iter = symbolLinkageTypes.find(fn.getName().str());
    if (iter == symbolLinkageTypes.end())
      continue;

    fn.setLinkage(iter->second);
    fn.setDSOLocal(true);
  }

  return {};
}

void MCLinker::prepareMachineModuleInfo() {
  llvm::LLVMTargetMachine &llvmTargetMachine =
      static_cast<llvm::LLVMTargetMachine &>(targetMachine);

  for (auto [i, smcInfos] : llvm::enumerate(symbolAndMCInfos)) {
    for (auto [j, mcInfo] : llvm::enumerate(smcInfos->mcInfos)) {
      // Move MachineFunctions from each split's codegen result
      // into machineModInfoPass to print out together in one .o
      llvm::DenseMap<const llvm::Function *,
                     std::unique_ptr<llvm::MachineFunction>> &machineFunctions =
          getMachineFunctionsFromMachineModuleInfo(*mcInfo->machineModuleInfo);

      llvm::StringMap<const llvm::Function *> &fnNameToFnPtr =
          mcInfo->fnNameToFnPtr;

      mcInfo->machineModuleInfo->getContext().setObjectFileInfo(
          llvmTargetMachine.getObjFileLowering());

      for (auto &fn : linkedModule->functions()) {
        if (fn.isDeclaration())
          continue;
        if (machineModInfoPass->getMMI().getMachineFunction(fn))
          continue;

        auto fnPtrIter = fnNameToFnPtr.find(fn.getName().str());
        if (fnPtrIter == fnNameToFnPtr.end())
          continue;
        auto mfPtrIter = machineFunctions.find(fnPtrIter->second);
        if (mfPtrIter == machineFunctions.end())
          continue;

        llvm::Function &origFn = mfPtrIter->second->getFunction();

        machineModInfoPass->getMMI().insertFunction(
            fn, std::move(mfPtrIter->second));

        // Restore function linkage types.
        if (!origFn.hasWeakLinkage())
          continue;

        auto iter = symbolLinkageTypes.find(fn.getName().str());
        if (iter == symbolLinkageTypes.end())
          continue;

        origFn.setLinkage(iter->second);
        origFn.setDSOLocal(true);
      }

      // Restore global variable linkage types.
      for (auto &global : mcInfo->moduleAndContext->globals()) {
        if (!global.hasWeakLinkage())
          continue;
        auto iter = symbolLinkageTypes.find(global.getName().str());
        if (iter == symbolLinkageTypes.end())
          continue;

        global.setLinkage(iter->second);
        global.setDSOLocal(true);
      }

      // Release memory as soon as possible to reduce peak memory footprint.
      mcInfo->machineModuleInfo.reset();
      mcInfo->fnNameToFnPtr.clear();
      mcInfo->moduleBuf.reset();
    }
  }
}

ErrorOr<WriteableBufferRef> MCLinker::linkAndPrint(StringRef moduleName,
                                                   bool emitAssembly) {
  // link at llvm::Module level.
  ErrorOrSuccess lmResult = linkLLVMModules(moduleName);
  if (lmResult.isError())
    return Error(lmResult.getError());

  prepareMachineModuleInfo();

  WriteableBufferRef linkedObj = WriteableBuffer::get();

  llvm::legacy::PassManager passMgr;
  // Add an appropriate TargetLibraryInfo pass for the module's
  // triple.
  llvm::TargetLibraryInfoImpl targetLibInfo(
      llvm::Triple(linkedModule->getTargetTriple()));

  llvm::LLVMTargetMachine &llvmTargetMachine =
      static_cast<llvm::LLVMTargetMachine &>(targetMachine);

  llvmTargetMachine.Options.MCOptions.AsmVerbose = options.verboseOutput;
  llvmTargetMachine.Options.MCOptions.PreserveAsmComments =
      options.verboseOutput;

  // Add AsmPrint pass and run the pass manager.
  passMgr.add(new llvm::TargetLibraryInfoWrapperPass(targetLibInfo));

  // Function ordering may be changed in the linkedModule due to Linker,
  // but the original order matters for NVPTX backend to generate function
  // declaration properly to avoid use before def/decl illegal instructions.
  // Sort the linkedModule's functions back to to its original order
  // (only definition matter, declaration doesn't).
  if (llvm::Triple(options.targetTriple).isNVPTX()) {
    linkedModule->getFunctionList().sort([&](const auto &lhs, const auto &rhs) {
      if (lhs.isDeclaration() && rhs.isDeclaration())
        return true;

      if (lhs.isDeclaration())
        return false;

      if (rhs.isDeclaration())
        return true;

      auto iter1 = originalFnOrdering.find(lhs.getName());
      if (iter1 == originalFnOrdering.end())
        return true;
      auto iter2 = originalFnOrdering.find(rhs.getName());
      if (iter2 == originalFnOrdering.end())
        return true;

      return iter1->second < iter2->second;
    });
  }

  if (KGEN::addPassesToAsmPrint(options, llvmTargetMachine, passMgr, *linkedObj,
                                emitAssembly
                                    ? llvm::CodeGenFileType::AssemblyFile
                                    : llvm::CodeGenFileType::ObjectFile,
                                true, machineModInfoPass, mcInfos)) {
    // Release some of the AsyncValue memory to avoid
    // wrong version of LLVMContext destructor being called due to
    // multiple LLVM being statically linked in dylibs that have
    // access to this code path.
    for (SymbolAndMCInfo *smcInfo : symbolAndMCInfos)
      smcInfo->clear();

    return Error("failed to add to ObjectFile Print pass");
  }

  const_cast<llvm::TargetLoweringObjectFile *>(
      llvmTargetMachine.getObjFileLowering())
      ->Initialize(machineModInfoPass->getMMI().getContext(), targetMachine);

  passMgr.run(*linkedModule);

  // Release some of the AsyncValue memory to avoid
  // wrong version of LLVMContext destructor being called due to
  // multiple LLVM being statically linked in dylibs that have
  // access to this code path.
  for (SymbolAndMCInfo *smcInfo : symbolAndMCInfos)
    smcInfo->clear();

  return linkedObj;
}
