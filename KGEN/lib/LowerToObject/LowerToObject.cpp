//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "KGEN/CompilationOptions.h"
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
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "lower-to-object"

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

/// Given a module operation, return its exported symbols.
static DenseSet<StringAttr> getExportedSymbols(ModuleOp module) {
  DenseSet<StringAttr> exportedSymbols;
  for (auto e : module.getOps<ExportOp>())
    for (auto sym : e.getExports().getAsRange<FlatSymbolRefAttr>())
      exportedSymbols.insert(sym.getAttr());
  return exportedSymbols;
}

ErrorOr<ObjectCompiler>
ObjectCompiler::create(LLCL::Runtime &runtime, StringRef basePath,
                       SymbolTable &symtab, const CompilationOptions &options) {
  return create(runtime, basePath, symtab,
                getExportedSymbols(cast<ModuleOp>(symtab.getOp())), options);
}

ErrorOr<ObjectCompiler>
ObjectCompiler::create(LLCL::Runtime &runtime, StringRef basePath,
                       SymbolTable &symtab, DenseSet<StringAttr> exports,
                       const CompilationOptions &options) {
  auto transformCache = Cache::getDefaultBackendChain(
      runtime, (std::filesystem::path(basePath.str()) / "transform").string());
  if (failed(transformCache))
    return transformCache.takeError();
  return ObjectCompiler(runtime, symtab, std::move(exports),
                        std::move(*transformCache), options);
}

ObjectCompiler::ObjectCompiler(
    LLCL::Runtime &runtime, SymbolTable &symtab, DenseSet<StringAttr> exports,
    std::unique_ptr<Cache::BlobCacheBackend> transformCache,
    const CompilationOptions &options)
    : transformCache(std::move(transformCache)), runtime(runtime),
      module(cast<ModuleOp>(symtab.getOp())), symtab(symtab),
      exportedSymbols(std::move(exports)), options(options) {
  // Register types used during async compilation.
  LLCL::AsyncValue::registerTypes<Cache::BufferRef>();
}

//===----------------------------------------------------------------------===//
// compileLLVMToObject
//===----------------------------------------------------------------------===//

LogicalResult KGEN::compileLLVMToObject(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        llvm::raw_pwrite_stream &objStream,
                                        bool emitAssembly) {
  TimeTraceScope<> traceScope("compile-llvm-to-object", module.getName());
  module.setDataLayout(targetMachine.createDataLayout());

  llvm::legacy::PassManager passManager;
  llvm::legacy::FunctionPassManager functionPassManager(&module);
  llvm::PassManagerBuilder passManagerBuilder;

  // Set up the pass manager builder to populate the passes we want.
  passManagerBuilder.OptLevel = targetMachine.getOptLevel();

  if (targetMachine.getOptLevel())
    passManagerBuilder.Inliner =
        llvm::createFunctionInliningPass(targetMachine.getOptLevel(), 0, false);

  // Set up the pass manager and populate it.
  passManagerBuilder.populateFunctionPassManager(functionPassManager);
  passManagerBuilder.populateModulePassManager(passManager);

  functionPassManager.doInitialization();
  functionPassManager.doFinalization();

  // Add passes to emit an object file.
  targetMachine.addPassesToEmitFile(passManager, objStream, nullptr,
                                    emitAssembly ? llvm::CGFT_AssemblyFile
                                                 : llvm::CGFT_ObjectFile);

  // Run the pass manager to compile the module.
  for (auto &fun : module)
    functionPassManager.run(fun);

  passManager.run(module);

  return success();
}

//===----------------------------------------------------------------------===//
// createTargetMachine
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
KGEN::createTargetMachine(TargetInfoAttr targetInfo,
                          const CompilationOptions &options, bool isJIT) {
  { // TODO: remove this once we have more cross-compilation capability.
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    assert(targetInfo.getTripleStr() == targetTriple &&
           "TODO: target info must match host for now");
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser(); // needed for inline_asm

  std::string errorMessage;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      targetInfo.getTripleStr(), errorMessage);
  if (!target)
    return Error("no target exists for '" + targetInfo.getTripleStr() +
                 "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetInfo.getTripleStr(), targetInfo.getCpu(), targetInfo.getFeatures(),
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/None, /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}
