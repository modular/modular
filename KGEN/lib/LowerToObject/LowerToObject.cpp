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
#include "llvm/Support/TargetSelect.h"
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
ObjectCompiler::create(LLCL::Runtime &runtime, StringRef basePath,
                       SymbolTable &symtab, const CompilationOptions &options) {
  DenseMap<StringAttr, StringAttr> exports =
      getExportedSymbols(cast<ModuleOp>(symtab.getOp()));
  return create(runtime, basePath, symtab, exports, options);
}

ErrorOr<ObjectCompiler>
ObjectCompiler::create(LLCL::Runtime &runtime, StringRef basePath,
                       SymbolTable &symtab,
                       const DenseMap<StringAttr, StringAttr> &exports,
                       const CompilationOptions &options) {
  auto transformCache = Cache::getDefaultBackendChain(
      runtime, (std::filesystem::path(basePath.str()) / "transform").string());
  if (failed(transformCache))
    return transformCache.takeError();
  return ObjectCompiler(runtime, symtab, exports, std::move(*transformCache),
                        options);
}

ObjectCompiler::ObjectCompiler(
    LLCL::Runtime &runtime, SymbolTable &symtab,
    const DenseMap<StringAttr, StringAttr> &exports,
    LLCL::RCRef<Cache::BlobCacheBackend> transformCache,
    const CompilationOptions &options)
    : transformCache(
          decltype(this->transformCache)::create(std::move(transformCache))),
      runtime(runtime), module(cast<ModuleOp>(symtab.getOp())), symtab(symtab),
      exportedSymbols(std::move(exports)), options(options) {
  // Register types used during async compilation.
  LLCL::AsyncValue::registerTypes<Cache::BufferRef>();
}

//===----------------------------------------------------------------------===//
// compileLLVMToObject
//===----------------------------------------------------------------------===//

/// Run the default LLVM optimization pipeline based on the select optimization
/// level.
static LogicalResult runOptPasses(llvm::Module &module,
                                  llvm::TargetMachine &targetMachine) {
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
      /*CM=*/std::nullopt, /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

//===----------------------------------------------------------------------===//
// getHostTargetInfo
//===----------------------------------------------------------------------===//

ErrorOr<TargetInfoAttr> KGEN::getHostTargetInfo(MLIRContext *ctx) {
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();

  // Get the host CPU and set up to get the features.
  std::string cpu(llvm::sys::getHostCPUName());
  llvm::StringMap<bool> hostFeatures;

  // Get the host features.
  std::string featureStr;
  llvm::raw_string_ostream os(featureStr);
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    llvm::interleave(
        llvm::make_filter_range(hostFeatures, [](auto &f) { return f.second; }),
        os, [&](auto &f) { os << '+' << f.first(); }, ",");
  }

  // Initialize the host target so that we can find it in the lookup.
  llvm::InitializeNativeTarget();

  std::string errorMessage;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target)
    return Error("could not construct host target info: " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(targetTriple, cpu, featureStr, /*Options=*/{},
                                  /*RM=*/llvm::Reloc::Model::PIC_, /*CM=*/{}));
  if (!machine)
    return Error("failed to create target machine for data layout lookup");

  ErrorOr<DataLayout> dl =
      DataLayout::parse(machine->createDataLayout().getStringRepresentation());
  assert(!dl.isError() && "failed to parse LLVM data layout?");

  // Return a TargetInfoAttr built for the host.
  return TargetInfoAttr::get(ctx, llvm::Triple(targetTriple), cpu, os.str(),
                             std::move(*dl), kPreferredSIMDBitWidth);
}
