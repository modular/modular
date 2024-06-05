//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LowerToObject.h"
#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/LLVMIRUtils.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/NameMangling.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLVMPassesPipeline.h"
#include "Support/FileSystemExtras.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcError.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

#include <utility>

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "lower-to-object"

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<ObjectCompiler>>
ObjectCompiler::create(StringRef basePath, CompilationOptions options,
                       bool isJIT, MLIRContext &context,
                       PassManagerConfigOptions pmOptions, bool isSearch) {
  auto transformCache = Cache::getLocalDefaultBackendChain(
      std::filesystem::path(basePath.str()) / "transform", KGEN_VERSION_STRING);
  if (failed(transformCache))
    return transformCache.takeError();
  return std::unique_ptr<ObjectCompiler>(
      new ObjectCompiler(std::move(*transformCache), std::move(options), isJIT,
                         isSearch, context, std::move(pmOptions)));
}

ObjectCompiler::ObjectCompiler(RCRef<Cache::BlobCacheBackend> transformCache,
                               CompilationOptions options, bool isJIT,
                               bool isSearch, MLIRContext &context,
                               PassManagerConfigOptions pmOptions)
    : transformCache(
          decltype(this->transformCache)::create(std::move(transformCache))),
      options(std::move(options)), isJIT(isJIT), isSearch(isSearch),
      pmOptions(std::move(pmOptions)), context(context) {}

//===----------------------------------------------------------------------===//
// Time Trace Instrumentation
//===----------------------------------------------------------------------===//

/// Given an Any containing an LLVM IR unit, return a string representation of
/// the name of the unit.
static std::string getLLVMIRName(llvm::Any &ir) {
  if (llvm::any_cast<const llvm::Module *>(&ir)) {
    return ("[module](" +
            (*llvm::any_cast<const llvm::Module *>(&ir))->getName() + ")")
        .str();
  }
  if (const auto **fn = llvm::any_cast<const llvm::Function *>(&ir))
    return (*fn)->getName().str();
  if (const auto **scc = llvm::any_cast<const llvm::LazyCallGraph::SCC *>(&ir))
    return (*scc)->getName();
  if (const auto **loop = llvm::any_cast<const llvm::Loop *>(&ir))
    return (*loop)->getName().str();
  llvm_unreachable("unknown wrapped IR type");
}

namespace {
class LLVMTimeTraceInstrumentation {
public:
  LLVMTimeTraceInstrumentation(llvm::PassInstrumentationCallbacks &pic) {
    pic.registerBeforeNonSkippedPassCallback(
        [=](StringRef passID, llvm::Any ir) { runBeforePass(passID, ir); });
    pic.registerAfterPassCallback(
        [=](StringRef, llvm::Any, const llvm::PreservedAnalyses &) {
          runAfterPass();
        },
        /*ToFront=*/true);
    pic.registerAfterPassInvalidatedCallback(
        [=](StringRef, const llvm::PreservedAnalyses &) { runAfterPass(); },
        true);
    pic.registerBeforeAnalysisCallback(
        [=](StringRef passID, llvm::Any ir) { runBeforePass(passID, ir); });
    pic.registerAfterAnalysisCallback(
        [=](StringRef, llvm::Any) { runAfterPass(); }, /*ToFront=*/true);
  }

private:
  static void runBeforePass(StringRef passID, llvm::Any &ir) {
    CompilerProfilerEntry::createAndPush(passID, getLLVMIRName(ir));
  }

  static void runAfterPass() { CompilerProfilerEntry::endAndPop(); }
};
} // namespace

/// Run the default LLVM optimization pipeline based on the select optimization
/// level.
LogicalResult KGEN::runLLVMOptPasses(llvm::Module &module,
                                     llvm::TargetMachine &targetMachine,
                                     const CompilationOptions &options,
                                     LLCL::Runtime &runtime) {
  CompilerTimeTraceScope traceScope("llvm-optimize", module.getName());
  [[maybe_unused]] auto timeScope =
      runtime.context->get<M::Telemetry::TelemetryContext>()
          ->createUInt64Timer<std::chrono::milliseconds>(
              "mojo.llvm.optimize.time", M::Telemetry::Level::L2);
  using namespace llvm;

  LoopAnalysisManager loopAnalysisMgr;
  FunctionAnalysisManager funcAnalysisMgr;
  CGSCCAnalysisManager sccAnalysisMgr;
  ModuleAnalysisManager moduleAnalysisMgr;

  llvm::PassInstrumentationCallbacks pic;
  LLVMTimeTraceInstrumentation timeTraceInstrumentation(pic);

  StandardInstrumentations standardInstrumentations(module.getContext(),
                                                    /*DebugLogging=*/false);
  standardInstrumentations.registerCallbacks(pic, &moduleAnalysisMgr);

  TargetLibraryInfoImpl targetLibInfo(Triple(module.getTargetTriple()));
  PassBuilder passBuilder(&targetMachine, PipelineTuningOptions(),
                          /*PGOOpt=*/std::nullopt, &pic);

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

  ModulePassManager modulePassMgr = buildLLVMOptimizationPipeline(options);

  // Now that we have all of the passes ready, run them.
  modulePassMgr.run(module, moduleAnalysisMgr);
  return success();
}

//===----------------------------------------------------------------------===//
// compileLLVMToObject
//===----------------------------------------------------------------------===//

/// Run the default llc passes required to generate object code.
static LogicalResult
runLlcPasses(llvm::Module &module, llvm::TargetMachine &targetMachine,
             llvm::raw_pwrite_stream &os, llvm::CodeGenFileType fileType,
             M::Telemetry::TelemetryContext *telemetryCtx = nullptr) {
  CompilerTimeTraceScope traceScope("llvm-codegen", module.getName());
  std::optional<M::Telemetry::Timer<uint64_t, std::chrono::milliseconds>>
      timeScope;
  if (telemetryCtx) {
    llvm::StringMap<Telemetry::MetricAttributeValue> attrs = {
        {"filename", module.getSourceFileName()}};
    timeScope = telemetryCtx->createUInt64Timer<std::chrono::milliseconds>(
        "mojo.llvm.optimize.time", M::Telemetry::Level::L2, attrs);
  }
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

/// Compile optimized llvm::Module module to object through the llc pipeline
/// asynchronously and cache the transformation.
static LLCL::AnyAsyncValueRef compileOptimizedLLVMModuleToObject(
    llvm::Module &module, Location loc, LLCL::Runtime &runtime, bool isJIT,
    bool emitAssembly, CompilationOptions options,
    RCRef<Cache::TransformCache> transformCache,
    std::optional<size_t> moduleIdx, std::optional<size_t> splitIdx) {
  WriteableBufferRef keyBuf = WriteableBuffer::get();
  options.print(*keyBuf << "compileOptimizedLLVMModuleToObject(");
  *keyBuf << ")";
  size_t nonBitcodeKeySize = keyBuf->getBufferSize();

  llvm::WriteBitcodeToFile(module, *keyBuf);

  auto runTransformation = [nonBitcodeKeySize, loc, &runtime, emitAssembly,
                            keyBuf = keyBuf.copy(), options, isJIT, moduleIdx,
                            splitIdx](WriteableBufferRef buf,
                                      LLCL::AnyAsyncValueRef chain) mutable {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
#ifdef MODULAR_ENABLE_TELEMETRY
    Cache::CacheTelemetryContext::getCacheTelemetryContext(runtime.context)
        .recordCacheMiss("compileOptimizedLLVMModuleToObject");
#endif

    chain.andThenAsync([nonBitcodeKeySize, loc, &runtime, emitAssembly,
                        output = output.copy(), buf = buf.copy(),
                        keyBuf = std::move(keyBuf), options, isJIT, moduleIdx,
                        splitIdx]() mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
      [[maybe_unused]] auto timeScope =
          runtime.context->get<M::Telemetry::TelemetryContext>()
              ->createUInt64Timer<std::chrono::milliseconds>(
                  "mojo.compile.cache.miss.time", M::Telemetry::Level::L2,
                  {{"pipeline", "compileOptimizedLLVMModuleToObject"}});

#endif

      BufferRef keyBufRef(std::move(keyBuf));
      StringRef bitcodeBuffer = keyBufRef->getBuffer();
      bitcodeBuffer = bitcodeBuffer.drop_front(nonBitcodeKeySize);

      // Load the cached bytecode into a new context. This is necessary to
      // avoid data races during multi-threading.
      llvm::LLVMContext ctx;
      llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
          llvm::parseBitcodeFile(
              llvm::MemoryBufferRef(bitcodeBuffer, "<split-module-llc>"), ctx);
      if (!moduleOr) {
        return std::move(output).setToError(
            LLCL::getMLIRDiagnostic("failed to load LLVM IR bitcode", loc));
      }
      std::unique_ptr<llvm::Module> module = std::move(*moduleOr);

      // Create TargetMachine for this module. This is also necessary to
      // avoid data races during multi-threading.
      ErrorOr<std::unique_ptr<llvm::TargetMachine>> machineOr =
          createTargetMachine(options, isJIT);
      if (failed(machineOr)) {
        return std::move(output).setToError(
            LLCL::getMLIRDiagnostic("failed to create TargetMachine", loc));
      }

      // Run llc passes.
      if (failed(runLlcPasses(
              *module, **machineOr, *buf,
              emitAssembly ? llvm::CodeGenFileType::AssemblyFile
                           : llvm::CodeGenFileType::ObjectFile,
              runtime.context->get<M::Telemetry::TelemetryContext>()))) {
        return std::move(output).setToError(LLCL::getMLIRDiagnostic(
            "llc failed to codegen LLVM IR to object code", loc));
      }

      if (!options.saveTempsPrefix.empty()) {
        std::string saveTempsPrefix = options.saveTempsPrefix;
        if (moduleIdx)
          saveTempsPrefix += "_" + std::to_string(*moduleIdx);
        if (splitIdx)
          saveTempsPrefix += "__" + std::to_string(*splitIdx);

        std::string outPath = saveTempsPrefix + ".asm";
        auto outFile = mlir::openOutputFile(outPath);
        if (!outFile) {
          return std::move(output).setToError(
              LLCL::getMLIRDiagnostic("failed open output asm file", loc));
        }

        if (failed(runLlcPasses(*module, **machineOr, outFile->os(),
                                llvm::CodeGenFileType::AssemblyFile)))
          return std::move(output).setToError(LLCL::getMLIRDiagnostic(
              "llc failed to codegen LLVM IR to object code", loc));
        outFile->keep();
      }

      std::move(output).emplace(buf.copy());
    });
    return output;
  };

  auto onCacheHit = [&runtime](BufferRef buf) {
#ifdef MODULAR_ENABLE_TELEMETRY
    Cache::CacheTelemetryContext::getCacheTelemetryContext(runtime.context)
        .recordCacheHit("compileOptimizedLLVMModuleToObject");
#endif
    return buf.copy();
  };

  return Cache::cachedTransform(
      LLCL::MLIRLocationDecoder::getEncodedLocation(loc), transformCache.copy(),
      LLCL::AsyncValueRef<Chain>::createReady(runtime), keyBuf.copy(),
      std::move(runTransformation), onCacheHit);
}

static LogicalResult writeTempModule(const std::string &phase,
                                     const std::string &saveTempsPrefix,
                                     llvm::Module &module) {
  if (saveTempsPrefix.empty())
    return success();
  std::string outPath = saveTempsPrefix + "." + phase + ".ll";
  auto outFile = mlir::openOutputFile(outPath);
  if (!outFile)
    return failure();
  outFile->os() << module;
  outFile->keep();
  return success();
};

LogicalResult KGEN::optimizeLLVMModule(llvm::Module &module,
                                       llvm::TargetMachine &targetMachine,
                                       CompilationOptions &options,
                                       LLCL::Runtime &runtime,
                                       std::optional<size_t> moduleIdx) {
  CompilerTimeTraceScope traceScope("optimize-llvm", module.getName());
  module.setDataLayout(targetMachine.createDataLayout());

  std::string saveTempsPrefix = options.saveTempsPrefix;
  if (moduleIdx && !options.saveTempsPrefix.empty())
    saveTempsPrefix += "." + std::to_string(moduleIdx.value());

  if (failed(writeTempModule("pre-opt", saveTempsPrefix, module)))
    return failure();

  if (failed(runLLVMOptPasses(module, targetMachine, options, runtime)))
    return failure();

  if (failed(writeTempModule("post-opt", saveTempsPrefix, module)))
    return failure();

  return success();
}

SmallVector<LLCL::AnyAsyncValueRef> KGEN::compileOptimizedLLVMToObjects(
    llvm::Module &module, mlir::Location loc, CompilationOptions &options,
    LLCL::Runtime &runtime, RCRef<Cache::TransformCache> transformCache,
    bool isParLLC, bool isJIT, bool emitAssembly,
    std::optional<size_t> moduleIdx) {
  CompilerTimeTraceScope traceScope("compile-optimized-llvm-to-object",
                                    module.getName());

  SmallVector<LLCL::AnyAsyncValueRef> cacheResults;
  auto compileToObject = [&](llvm::Module *inputModule, int64_t idx,
                             bool sync) {
    if (!inputModule)
      return;

    std::string saveTempsPrefix = options.saveTempsPrefix;
    if (!saveTempsPrefix.empty())
      saveTempsPrefix += "." + std::to_string(idx);
    (void)writeTempModule("llc-split", saveTempsPrefix, *inputModule);

    cacheResults.push_back(compileOptimizedLLVMModuleToObject(
        *inputModule, loc, runtime, isJIT, emitAssembly, options,
        transformCache, moduleIdx, idx));
  };

  if (isParLLC) {
    splitPerFunction(module, runtime.getWorkQueue()->getParallelismLevel(),
                     compileToObject);
  } else {
    cacheResults.push_back(compileOptimizedLLVMModuleToObject(
        module, loc, runtime, isJIT, emitAssembly, options, transformCache,
        moduleIdx, std::nullopt));
  }
  return cacheResults;
}

LogicalResult KGEN::compileLLVMToObject(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        llvm::raw_pwrite_stream &objStream,
                                        CompilationOptions &options,
                                        LLCL::Runtime &runtime,
                                        bool emitAssembly,
                                        std::optional<size_t> moduleIdx) {
  CompilerTimeTraceScope traceScope("compile-llvm-to-object", module.getName());
  module.setDataLayout(targetMachine.createDataLayout());

  std::string saveTempsPrefix = options.saveTempsPrefix;
  if (moduleIdx && !options.saveTempsPrefix.empty())
    saveTempsPrefix += "." + std::to_string(moduleIdx.value());

  if (failed(writeTempModule("pre-opt", saveTempsPrefix, module)))
    return failure();

  if (failed(runLLVMOptPasses(module, targetMachine, options, runtime)))
    return failure();

  if (failed(writeTempModule("post-opt", saveTempsPrefix, module)))
    return failure();

  if (failed(
          runLlcPasses(module, targetMachine, objStream,
                       emitAssembly ? llvm::CodeGenFileType::AssemblyFile
                                    : llvm::CodeGenFileType::ObjectFile,
                       runtime.context->get<M::Telemetry::TelemetryContext>())))
    return failure();

  if (!options.saveTempsPrefix.empty()) {
    std::string outPath = saveTempsPrefix + ".asm";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile)
      return failure();

    if (failed(runLlcPasses(module, targetMachine, outFile->os(),
                            llvm::CodeGenFileType::AssemblyFile)))
      return failure();
    outFile->keep();
  }

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
      /*Options=*/{}, options.relocModel, /*CM=*/{},
      /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}
