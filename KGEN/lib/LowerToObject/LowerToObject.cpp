//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "BytecodeUtils.h"
#include "LowerToObjectImpl.h"
#include "Support/ErrorOr.h"
#include "Support/TempFile.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "lower-to-object"

//===----------------------------------------------------------------------===//
// compileLLVMToObject
//===----------------------------------------------------------------------===//

LogicalResult KGEN::compileLLVMToObject(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        SmallVectorImpl<char> &buf) {
  TimeTraceScope<> traceScope("compile-llvm-to-object", module.getName());
  module.setDataLayout(targetMachine.createDataLayout());

  llvm::legacy::PassManager passManager;
  llvm::PassManagerBuilder passManagerBuilder;

  // Set up the pass manager builder to populate the passes we want.
  passManagerBuilder.OptLevel = 3;

  // Set up the pass manager and populate it.
  targetMachine.adjustPassManager(passManagerBuilder);
  passManagerBuilder.populateModulePassManager(
      llvm::cast<llvm::PassManagerBase>(passManager));

  // Add passes to emit an object file.
  llvm::raw_svector_ostream objStream(buf);
  targetMachine.addPassesToEmitFile(passManager, objStream, nullptr,
                                    llvm::CGFT_ObjectFile);
  // Run the pass manager to compile the module.
  passManager.run(module);

  return success();
}

//===----------------------------------------------------------------------===//
// createTargetMachine
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
KGEN::createTargetMachine(TargetInfoAttr targetInfo, bool isJIT) {
  { // TODO: remove this once we have more cross-compilation capability.
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    assert(targetInfo.getTriple() == targetTriple &&
           "TODO: target info must match host for now");
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  std::string errorMessage;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      targetInfo.getTriple().str(), errorMessage);
  if (!target)
    return Error("no target exists for '" + targetInfo.getTriple() +
                 "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetInfo.getTriple(), targetInfo.getCpu(), targetInfo.getFeatures(),
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/None, /*OL=*/llvm::CodeGenOpt::Aggressive, /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}
