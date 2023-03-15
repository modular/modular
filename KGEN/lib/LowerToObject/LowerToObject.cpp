//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LLVMPassesPipeline.h"
#include "LowerToObjectImpl.h"
#include "Support/SIMD.h"
#include "Support/TempFile.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "lower-to-object"

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

ErrorOr<ObjectCompiler>
ObjectCompiler::create(LLCL::Runtime &runtime, mlir::PassManager &mgr,
                       StringRef basePath, SymbolTable &symtab,
                       const CompilationOptions &options) {
  llvm::MapVector<StringAttr, ExportedSymbol> exports =
      getExportedSymbols(cast<ModuleOp>(symtab.getOp()));
  return create(runtime, mgr, basePath, symtab, std::move(exports), options);
}

ErrorOr<ObjectCompiler>
ObjectCompiler::create(LLCL::Runtime &runtime, mlir::PassManager &mgr,
                       StringRef basePath, SymbolTable &symtab,
                       llvm::MapVector<StringAttr, ExportedSymbol> &&exports,
                       const CompilationOptions &options) {
  auto transformCache = Cache::getDefaultBackendChain(
      runtime, (std::filesystem::path(basePath.str()) / "transform").string());
  if (failed(transformCache))
    return transformCache.takeError();
  return ObjectCompiler(runtime, mgr, symtab, std::move(exports),
                        std::move(*transformCache), options);
}

ObjectCompiler::ObjectCompiler(
    LLCL::Runtime &runtime, mlir::PassManager &mgr, SymbolTable &symtab,
    llvm::MapVector<StringAttr, ExportedSymbol> &&exports,
    LLCL::RCRef<Cache::BlobCacheBackend> transformCache,
    const CompilationOptions &options)
    : transformCache(
          decltype(this->transformCache)::create(std::move(transformCache))),
      runtime(runtime), mgr(mgr), module(cast<ModuleOp>(symtab.getOp())),
      symtab(symtab), exportedSymbols(std::move(exports)), options(options) {}

//===----------------------------------------------------------------------===//
// compileLLVMToObject
//===----------------------------------------------------------------------===//

/// Run the default LLVM optimization pipeline based on the select optimization
/// level.
static LogicalResult runOptPasses(llvm::Module &module,
                                  llvm::TargetMachine &targetMachine) {
  TimeTraceScope<> traceScope("llvm-optimize", module.getName());
  using namespace llvm;

  LoopAnalysisManager loopAnalysisMgr;
  FunctionAnalysisManager funcAnalysisMgr;
  CGSCCAnalysisManager sccAnalysisMgr;
  ModuleAnalysisManager moduleAnalysisMgr;

  TargetLibraryInfoImpl targetLibInfo(Triple(module.getTargetTriple()));
  PassBuilder passBuilder(&targetMachine);

  // Specially handle the alias analysis manager so that we can register
  // a custom pipeline of AA passes with it.
  AAManager analysisAnalysisMgr;
  if (llvm::Error err =
          passBuilder.parseAAPipeline(analysisAnalysisMgr, "default")) {
    errs() << toString(std::move(err)) << "\n";
    return failure();
  }

  // Register the alias analysis manager first so that our version is the one
  // used.
  funcAnalysisMgr.registerPass([&] { return std::move(analysisAnalysisMgr); });
  // Register our TargetLibraryInfoImpl.
  funcAnalysisMgr.registerPass(
      [&] { return TargetLibraryAnalysis(targetLibInfo); });

  // Register all the basic analyses with the managers.
  passBuilder.registerModuleAnalyses(moduleAnalysisMgr);
  passBuilder.registerCGSCCAnalyses(sccAnalysisMgr);
  passBuilder.registerFunctionAnalyses(funcAnalysisMgr);
  passBuilder.registerLoopAnalyses(loopAnalysisMgr);
  passBuilder.crossRegisterProxies(loopAnalysisMgr, funcAnalysisMgr,
                                   sccAnalysisMgr, moduleAnalysisMgr);

  ModulePassManager modulePassMgr = buildPipeline(targetMachine.getOptLevel());

  // Now that we have all of the passes ready, run them.
  modulePassMgr.run(module, moduleAnalysisMgr);
  return success();
}

/// Run the default llc passes required to generate object code.
static LogicalResult runLlcPasses(llvm::Module &module,
                                  llvm::TargetMachine &targetMachine,
                                  llvm::raw_pwrite_stream &os,
                                  llvm::CodeGenFileType fileType) {
  TimeTraceScope<> traceScope("llvm-codegen", module.getName());
  using namespace llvm;

  // Build up all of the passes that we want to do to the module.
  legacy::PassManager passMgr;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl targetLibInfo(Triple(module.getTargetTriple()));
  passMgr.add(new TargetLibraryInfoWrapperPass(targetLibInfo));

  // Verify module immediately to catch problems before doInitialization() is
  // called on any passes.
  if (verifyModule(module, &errs()))
    return failure();

  LLVMTargetMachine &llvmTargetMachine =
      static_cast<LLVMTargetMachine &>(targetMachine);
  auto *machineModInfoPass =
      new MachineModuleInfoWrapperPass(&llvmTargetMachine);

  // Construct a custom pass pipeline that starts after instruction
  // selection.
  if (targetMachine.addPassesToEmitFile(passMgr, os, nullptr, fileType, true,
                                        machineModInfoPass))
    return failure();

  const_cast<TargetLoweringObjectFile *>(llvmTargetMachine.getObjFileLowering())
      ->Initialize(machineModInfoPass->getMMI().getContext(), targetMachine);

  passMgr.run(module);
  return success();
}

LogicalResult KGEN::compileLLVMToObject(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        llvm::raw_pwrite_stream &objStream,
                                        bool emitAssembly) {
  TimeTraceScope<> traceScope("compile-llvm-to-object", module.getName());
  module.setDataLayout(targetMachine.createDataLayout());

  if (failed(runOptPasses(module, targetMachine)))
    return failure();

  if (failed(runLlcPasses(module, targetMachine, objStream,
                          emitAssembly ? llvm::CGFT_AssemblyFile
                                       : llvm::CGFT_ObjectFile)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// createTargetMachine
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
KGEN::createTargetMachine(const CompilationOptions &options, bool isJIT) {
  std::string errorMessage;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(options.targetTriple, errorMessage);
  if (!target)
    return Error("no target exists for '" + options.targetTriple +
                 "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      options.targetTriple, options.targetCpu, options.targetFeatures,
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/std::nullopt, /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

//===----------------------------------------------------------------------===//
// getTargetInfoFor
//===----------------------------------------------------------------------===//

ErrorOr<TargetInfoAttr> KGEN::getTargetInfoFor(MLIRContext *ctx,
                                               StringRef targetTriple,
                                               StringRef cpu,
                                               StringRef features) {
  std::string errorMessage;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple.str(), errorMessage);
  if (!target)
    return Error("could not construct host target info: " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(targetTriple, cpu, features, /*Options=*/{},
                                  /*RM=*/llvm::Reloc::Model::PIC_, /*CM=*/{}));
  if (!machine)
    return Error("failed to create target machine for data layout lookup");

  ErrorOr<DataLayout> dl =
      DataLayout::parse(machine->createDataLayout().getStringRepresentation());
  assert(!dl.isError() && "failed to parse LLVM data layout?");

  // Return a TargetInfoAttr built for the host.
  return TargetInfoAttr::get(ctx, llvm::Triple(targetTriple), cpu, features,
                             std::move(*dl), kPreferredSIMDBitWidth);
}
