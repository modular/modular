//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGENToLLVMPipeline.h"
#include "LLVMPassesPipeline.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/CompilerSupport/MLIRLocationDecoder.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/LLVMIRUtils.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/NameMangling.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGENToLLVMPipeline.h"
#include "LLVMPassesPipeline.h"
#include "Support/Context.h"
#include "Support/FileSystemExtras.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace M;
using namespace KGEN;
using namespace Cache;

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<ObjectCompiler>>
ObjectCompiler::create(StringRef basePath, CompilationOptions options,
                       bool isJIT, MLIRContext &context,
                       PassManagerConfigOptions pmOptions) {
  auto transformCache = Cache::getLocalDefaultBackendChain(
      std::filesystem::path(basePath.str()) / "transform", KGEN_VERSION_STRING);
  if (failed(transformCache))
    return transformCache.takeError();
  return std::unique_ptr<ObjectCompiler>(
      new ObjectCompiler(std::move(*transformCache), std::move(options), isJIT,
                         context, std::move(pmOptions)));
}

ObjectCompiler::ObjectCompiler(RCRef<Cache::BlobCacheBackend> transformCache,
                               CompilationOptions options, bool isJIT,
                               MLIRContext &context,
                               PassManagerConfigOptions pmOptions)
    : transformCache(
          decltype(this->transformCache)::create(std::move(transformCache))),
      options(std::move(options)), isJIT(isJIT),
      pmOptions(std::move(pmOptions)), context(context),
      runtime(*loadContext(&context)->get<AsyncRT::Runtime>()) {}

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
    VerboseCompilerProfilerEntry::createAndPush(passID, getLLVMIRName(ir));
  }

  static void runAfterPass() { VerboseCompilerProfilerEntry::endAndPop(); }
};
} // namespace

/// Run the default LLVM optimization pipeline based on the select optimization
/// level.
static LogicalResult runLLVMOptPasses(llvm::Module &module,
                                      llvm::TargetMachine &targetMachine,
                                      const CompilationOptions &options,
                                      AsyncRT::Runtime &runtime) {
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
  return mlir::success();
}

/// Run the default llc passes required to generate object code.
static LogicalResult
runLlcPasses(llvm::Module &module, CompilationOptions &options,
             llvm::TargetMachine &targetMachine, llvm::raw_pwrite_stream &os,
             llvm::CodeGenFileType fileType,
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

#ifndef MODULAR_PRODUCTION
  // Verify module immediately to catch problems before doInitialization() is
  // called on any passes.
  if (verifyModule(module, &errs()))
    return failure();
#endif

  LLVMTargetMachine &llvmTargetMachine =
      static_cast<LLVMTargetMachine &>(targetMachine);
  auto *machineModInfoPass =
      new MachineModuleInfoWrapperPass(&llvmTargetMachine);

  // Construct a custom pass pipeline that starts after instruction
  // selection.
  if (KGEN::addPassesToEmitFile(options, llvmTargetMachine, passMgr, os,
                                nullptr, fileType, true, machineModInfoPass))
    return failure();

  const_cast<TargetLoweringObjectFile *>(llvmTargetMachine.getObjFileLowering())
      ->Initialize(machineModInfoPass->getMMI().getContext(), targetMachine);

  passMgr.run(module);
  return mlir::success();
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
}

/// Compile optimized llvm::Module module to object through the llc pipeline
/// asynchronously and cache the transformation.
static AsyncRT::AnyAsyncValueRef compileOptimizedLLVMModuleToObject(
    LLVMModuleAndContext module, Location loc, AsyncRT::Runtime &runtime,
    bool isJIT, bool emitAssembly, CompilationOptions options,
    RCRef<Cache::TransformCache> transformCache,
    std::optional<size_t> moduleIdx, std::optional<size_t> splitIdx) {
  WriteableBufferRef keyBuf = WriteableBuffer::get();
  options.print(*keyBuf << "compileOptimizedLLVMModuleToObject(");
  *keyBuf << ")";
  size_t nonBitcodeKeySize = keyBuf->getBufferSize();

  llvm::WriteBitcodeToFile(*module, *keyBuf);

  auto output = AsyncRT::AsyncValueRef<BufferRef>::allocate(runtime);

  runtime.getWorkQueue()->addTask([nonBitcodeKeySize, loc, &runtime,
                                   emitAssembly, keyBuf = keyBuf.copy(),
                                   output = output.copy(), options, isJIT,
                                   moduleIdx, splitIdx]() mutable {
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
          AsyncRT::getMLIRDiagnostic("failed to load LLVM IR bitcode", loc));
    }
    std::unique_ptr<llvm::Module> module = std::move(*moduleOr);

    // Create TargetMachine for this module. This is also necessary to
    // avoid data races during multi-threading.
    ErrorOr<std::unique_ptr<llvm::TargetMachine>> machineOr =
        createTargetMachine(options, isJIT);
    if (failed(machineOr)) {
      return std::move(output).setToError(
          AsyncRT::getMLIRDiagnostic("failed to create TargetMachine", loc));
    }

    std::string saveTempsPrefix = options.saveTempsPrefix;
    if (!options.saveTempsPrefix.empty()) {
      if (moduleIdx)
        saveTempsPrefix += "_" + std::to_string(*moduleIdx);
      if (splitIdx)
        saveTempsPrefix += "__" + std::to_string(*splitIdx);
    }

    if (failed(writeTempModule("pre-llc", saveTempsPrefix, *module))) {
      return std::move(output).setToError(
          AsyncRT::getMLIRDiagnostic("failed save pre-llc llvm IR", loc));
    }

    auto buf = WriteableBuffer::get();

    // Run llc passes.
    if (failed(runLlcPasses(
            *module, options, **machineOr, *buf,
            emitAssembly ? llvm::CodeGenFileType::AssemblyFile
                         : llvm::CodeGenFileType::ObjectFile,
            runtime.context->get<M::Telemetry::TelemetryContext>()))) {
      return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
          "llc failed to codegen LLVM IR to object code", loc));
    }

    if (!options.saveTempsPrefix.empty()) {
      std::string outPath = saveTempsPrefix + ".asm";
      auto outFile = mlir::openOutputFile(outPath);
      if (!outFile) {
        return std::move(output).setToError(
            AsyncRT::getMLIRDiagnostic("failed open output asm file", loc));
      }

      if (failed(runLlcPasses(*module, options, **machineOr, outFile->os(),
                              llvm::CodeGenFileType::AssemblyFile)))
        return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
            "llc failed to codegen LLVM IR to object code", loc));
      outFile->keep();

      if (failed(writeTempModule("post-llc", saveTempsPrefix, *module))) {
        return std::move(output).setToError(
            AsyncRT::getMLIRDiagnostic("failed save post-llc llvm IR", loc));
      }
    }
    std::move(output).emplace(buf.copy());
  });

  return output;
}

/// Optimize the llvm module to prepare for codegen object file.
static LogicalResult optimizeLLVMModule(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        CompilationOptions &options,
                                        AsyncRT::Runtime &runtime,
                                        std::optional<size_t> moduleIdx) {
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

/// Compile the given LLVM module to object files and return the async values
/// that contains the compiled object file.
/// isParLLC is true: split module into per function for parallel llc lowering
///                   and return multiple object files.
/// isParLLC is false: compile module without splitting into one object file.
static SmallVector<AsyncRT::AnyAsyncValueRef> compileOptimizedLLVMToObjects(
    LLVMModuleAndContext module, mlir::Location loc,
    CompilationOptions &options, AsyncRT::Runtime &runtime,
    RCRef<Cache::TransformCache> transformCache, bool isParLLC, bool isJIT,
    bool emitAssembly, std::optional<size_t> moduleIdx) {
  CompilerTimeTraceScope traceScope("compile-optimized-llvm-to-object",
                                    module->getName());

  // Perform module materialization in another task.
  auto launchCompilation =
      [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
          std::optional<int64_t> idx) {
        auto result = AsyncRT::AsyncValueRef<BufferRef>::allocate(runtime);
        runtime.getWorkQueue()->addTask(
            [produceModule = std::move(produceModule), loc, &runtime, isJIT,
             emitAssembly, &options, cache = transformCache.copy(), moduleIdx,
             idx, result = result.copy()]() mutable {
              AsyncRT::AnyAsyncValueRef output =
                  compileOptimizedLLVMModuleToObject(
                      produceModule(), loc, runtime, isJIT, emitAssembly,
                      options, cache, moduleIdx, idx);
              andThenSyncMoving(
                  output,
                  [result = std::move(result)](
                      MutableArrayRef<AnyAsyncValueRef> output) mutable {
                    std::move(result).emplace(output.front().get<BufferRef>());
                  });
            });
        return result;
      };

  SmallVector<AsyncRT::AnyAsyncValueRef> cacheResults;
  if (!isParLLC) {
    cacheResults.push_back(
        launchCompilation(forwardModule(std::move(module)), std::nullopt));
  } else {
    splitPerFunction(
        std::move(module),
        [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
            std::optional<int64_t> idx) {
          cacheResults.push_back(
              launchCompilation(std::move(produceModule), idx));
        });
  }
  return cacheResults;
}

//===----------------------------------------------------------------------===//
// compileLLVMToAssembly
//===----------------------------------------------------------------------===//

LogicalResult KGEN::compileLLVMToAssembly(LLVMModuleAndContext module,
                                          llvm::TargetMachine &targetMachine,
                                          llvm::raw_pwrite_stream &objStream,
                                          CompilationOptions &options,
                                          AsyncRT::Runtime &runtime) {
  CompilerTimeTraceScope traceScope("compileLLVMToAssembly", module->getName());
  module->setDataLayout(targetMachine.createDataLayout());

  if (failed(writeTempModule("pre-opt", options.saveTempsPrefix, *module)))
    return failure();

  if (failed(runLLVMOptPasses(*module, targetMachine, options, runtime)))
    return failure();

  if (failed(writeTempModule("post-opt", options.saveTempsPrefix, *module)))
    return failure();

  if (failed(
          runLlcPasses(*module, options, targetMachine, objStream,
                       llvm::CodeGenFileType::AssemblyFile,
                       runtime.context->get<M::Telemetry::TelemetryContext>())))
    return failure();

  if (!options.saveTempsPrefix.empty()) {
    std::string outPath = options.saveTempsPrefix + ".asm";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile)
      return failure();
    if (failed(runLlcPasses(*module, options, targetMachine, outFile->os(),
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

//===----------------------------------------------------------------------===//
// lowerAllFuncsToLLVM
//===----------------------------------------------------------------------===//

/// If requested, attach sanitizer/XRay/etc. instrumentations to the given
/// module.
/// TODO: Eventually we should explore attaching this information at a higher
/// level of the stack.
static void attachInstrumentationAttributes(llvm::Module &module,
                                            const CompilationOptions &options) {
  if (!options.enableXRayInstrumentation && !options.sanitizers)
    return;

  for (llvm::Function &f : module.functions()) {
    if (f.isDeclaration())
      continue;
    if (options.enableXRayInstrumentation)
      f.addFnAttr("function-instrument", "xray-always");
    if (options.sanitizers.has(Sanitizers::kAddress))
      f.addFnAttr(llvm::Attribute::SanitizeAddress);
    if (options.sanitizers.has(Sanitizers::kThread))
      f.addFnAttr(llvm::Attribute::SanitizeThread);
  }
}

/// HACK HACK HACK https://github.com/modularml/modular/issues/27478
/// Using LineTables for NVPTX backend disables optimizations in cuda JIT. Use
/// DebugDirectives instead for equivalent performance to no-debug.
static void adaptDebugEmissionKind(ModuleOp module, StringRef targetTriple,
                                   DebugInfo::EmissionKind debugLevel) {
  bool generatingPtx = targetTriple.contains("nvptx");
  if (generatingPtx && debugLevel == DebugInfo::EmissionKind::LineTablesOnly) {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [](DebugInfo::DICompileUnitAttr CU) -> std::optional<Attribute> {
          if (CU.getEmissionKind() == DebugInfo::EmissionKind::LineTablesOnly) {
            return DebugInfo::DICompileUnitAttr::get(
                CU.getSourceLanguage(), CU.getFile(), CU.getProducer(),
                CU.getIsOptimized(),
                DebugInfo::EmissionKind::DebugDirectivesOnly,
                CU.getNameTableKind());
          }
          return std::nullopt;
        });
    replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/false,
                                          /*replaceLocs=*/true);
  }
}

static mlir::PassManager
createPassManager(const std::optional<std::string> &operationName,
                  mlir::MLIRContext *context) {
  if (operationName)
    return {context, *operationName};
  return {context};
}

ErrorOr<std::unique_ptr<llvm::Module>>
ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module) {
  CompilerTimeTraceScope traceScope("lower-to-llvm");

  mlir::PassManager mgr = createPassManager(pmOptions.operationName, &context);

  ErrorOrSuccess configPM = pmOptions.configurePassManager(mgr);
  if (configPM)
    return configPM.takeError();

  adaptDebugEmissionKind(module, options.targetTriple,
                         options.getDIEmissionKind());

  LowerToLLVMOptions llvmOptions(
      options.getDIEmissionKind(), options.debugAtLevel,
      static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage));
  llvmOptions.globalCtorFnName = ExecutionEngine::getGlobalCtorFnName();
  llvmOptions.globalDtorFnName = ExecutionEngine::getGlobalDtorFnName();
  // Use KGENCompilerRT allocators.
  llvmOptions.alignedAllocFnName = "KGEN_CompilerRT_AlignedAlloc";
  llvmOptions.alignedFreeFnName = "KGEN_CompilerRT_AlignedFree";

  buildLowerToLLVMPipeline(mgr, llvmOptions);

  if (failed(mgr.run(module)))
    return Error("run LowerToLLVMPipeline failed");

  // Use the input filename for the module name if possible.
  StringRef moduleName = "LLVMDialectModule";
  if (auto moduleLoc = module.getLoc()->findInstanceOf<FileLineColLoc>())
    moduleName = llvm::sys::path::filename(moduleLoc.getFilename());

  // Translate the operation into an LLVM module.
  CompilerTimeTraceScope mlirScope("mlir-to-llvmir");
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, ctx, moduleName);
  if (!llvmModule)
    return Error("translate module to LLVMIR failed");

  // Attach any necessary instrumentation to the module.
  attachInstrumentationAttributes(*llvmModule, options);

  return llvmModule;
}

//===----------------------------------------------------------------------===//
// emitArchive
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef> ObjectCompiler::emitArchive(ModuleOp module) {
  CompilerTimeTraceScope traceScope("produce-archive");

  // Perform a cache aware transformation to translate the module to an archive
  // file.
  auto runTransformation = [&](Operation *op, WriteableBufferRef buf,
                               AsyncRT::AnyAsyncValueRef chain) {
    auto output = AsyncRT::AsyncValueRef<BufferRef>::allocate(runtime);
#ifdef MODULAR_ENABLE_TELEMETRY
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheMiss("ObjectCompiler::emitArchive");
#endif
    chain.andThenSync([this, op, output = output.copy(),
                       buf = buf.copy()]() mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
      [[maybe_unused]] auto timeScope =
          loadContext(op->getContext())
              ->get<M::Telemetry::TelemetryContext>()
              ->createUInt64Timer<std::chrono::milliseconds>(
                  "mojo.compile.cache.miss.time", M::Telemetry::Level::L2,
                  {{"pipeline", "ObjectCompiler::emitArchive"}});
#endif

      // Lower the module to LLVM.
      LLVMModuleAndContext llvmModule;
      if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
            return lowerAllFuncsToLLVM(ctx, cast<ModuleOp>(op));
          })) {
        return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
            Twine(
                "failed to lower module to LLVM IR for archive compilation, ") +
                err.getError(),
            op->getLoc()));
      }

      CompilerTimeTraceScope traceScope("split-input-module");

      std::string moduleName = llvmModule->getName().str();

      // Split the module into multiple slices and compile each in parallel.
      // HACK HACK HACK https://github.com/modularml/modular/issues/22959
      // HACK: If we are generating PTX we don't want to split.
      bool generatingPtx =
          options.targetTriple.find("nvptx") != std::string::npos;
      bool noSplitting =
          runtime.getWorkQueue()->getParallelismLevel() < 2 || generatingPtx;
      bool parLLC = runtime.getWorkQueue()->getParallelismLevel() >= 2 &&
                    !generatingPtx && options.enableParallelLLC;

      SmallVector<AsyncRT::AnyAsyncValueRef> cacheResults;
      if (noSplitting) {
        cacheResults.push_back(
            lowerLLVMModuleToObjects(forwardModule(std::move(llvmModule)),
                                     op->getLoc(), parLLC, std::nullopt));
      } else {
        (void)writeTempModule("pre-split", options.saveTempsPrefix,
                              *llvmModule);

        auto handleSplit =
            [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
                std::optional<int64_t> idx) {
              cacheResults.push_back(lowerLLVMModuleToObjects(
                  std::move(produceModule), op->getLoc(),
                  !options.enableLLVMPerFunctionSplitting && parLLC, idx));
            };
        if (options.enableLLVMPerFunctionSplitting)
          splitPerFunction(std::move(llvmModule), handleSplit);
        else
          splitPerExported(std::move(llvmModule), handleSplit);
      }

      andThenSyncMoving(
          cacheResults,
          [moduleName = std::move(moduleName), op, buf = buf.copy(),
           output = output.copy(),
           generatingPtx](MutableArrayRef<AnyAsyncValueRef> values) mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
            [[maybe_unused]] auto timeScope =
                loadContext(op->getContext())
                    ->get<M::Telemetry::TelemetryContext>()
                    ->createUInt64Timer<std::chrono::milliseconds>(
                        "mojo.compile.cache.miss.time", M::Telemetry::Level::L2,
                        {{"pipeline", "ObjectCompiler::emitArchive"}});
#endif

            // If any of the cache results failed, propagate the error.
            for (auto &result : values) {
              if (result.isError())
                return std::move(output).setToError(result.takeDiagnostic());
            }
            CompilerTimeTraceScope traceScope("concatenate-object-files");

            if (generatingPtx) {
              // If we're not splitting just copy directly to the output
              // buffer.
              assert(values.size() == 1 && "should have one result");
              auto bufs =
                  std::move(values.front().get<SmallVector<BufferRef>>());
              assert(bufs.size() == 1 && "should have one result");

              *buf << bufs.front()->getBuffer();
              std::move(output).emplace(buf.copy());
              return;
            }

            // Now that all the object files have been compiled, merge them
            // all into a single archive.
            SmallVector<std::string> archiveMemberNames;
            SmallVector<llvm::NewArchiveMember> archiveMembers;
            unsigned idx = 0;
            // Make a pass first to allocate the names
            for (AnyAsyncValueRef &result : values) {
              for (BufferRef &buf : result.get<SmallVector<BufferRef>>()) {
                (void)buf;
                archiveMemberNames.push_back(
                    (moduleName + "." + Twine(idx++) + ".o").str());
              }
            }
            idx = 0;
            for (AnyAsyncValueRef &result : values) {
              for (BufferRef &buf : result.get<SmallVector<BufferRef>>()) {
                archiveMembers.emplace_back(llvm::MemoryBufferRef(
                    buf->getBuffer(), archiveMemberNames[idx++]));
              }
            }

            auto result = llvm::writeArchiveToBuffer(
                archiveMembers,
                /*WriteSymtab=*/llvm::SymtabWritingMode::NormalSymtab,
                archiveMembers.front().detectKindFromObject(),
                /*Deterministic=*/true, /*Thin=*/false);
            if (!result) {
              return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                  "failed to concatenate object files into archive",
                  op->getLoc()));
            }

            // Copy the result into the output buffer.
            *buf << (*result)->getBuffer();
            std::move(output).emplace(buf.copy());
          });
    });
    return output;
  };
  auto onCacheHit = [](Operation *op, BufferRef buf) {
#ifdef MODULAR_ENABLE_TELEMETRY
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheHit("ObjectCompiler::emitArchive");
#endif
    return buf.copy();
  };

  WriteableBufferRef produceArchiveKey = WriteableBuffer::get();
  options.print(*produceArchiveKey << "emitArchive(");
  *produceArchiveKey << ", isJIT=" << isJIT
                     << ", enableLLVMPerFunctionSplitting="
                     << options.enableLLVMPerFunctionSplitting << ')';

  auto output = cachedTransform(
      module, transformCache.copy(),
      AsyncRT::AsyncValueRef<Chain>::createReady(runtime),
      std::move(produceArchiveKey), runTransformation, onCacheHit);
  await(output);

  if (output.isError())
    return {std::move(output.takeDiagnostic().getMessage())};
  return {std::move(output.get<BufferRef>())};
}

//===----------------------------------------------------------------------===//
// lowerLLVMModuleToObjects
//===----------------------------------------------------------------------===//

AsyncRT::AsyncValueRef<SmallVector<BufferRef>>
ObjectCompiler::lowerLLVMModuleToObjects(
    llvm::unique_function<LLVMModuleAndContext()> produceModule, Location loc,
    bool parLLC, std::optional<size_t> moduleIdx) {
  auto result =
      AsyncRT::AsyncValueRef<SmallVector<BufferRef>>::allocate(runtime);

  runtime.getWorkQueue()->addTask([this, result = result.copy(),
                                   produceModule = std::move(produceModule),
                                   loc, moduleIdx, parLLC]() mutable {
    CompilerTimeTraceScope traceScope("optimizeLLVMTask");

    // Create the target machine.
    auto tmOr = createTargetMachine(options, isJIT);
    if (failed(tmOr)) {
      return std::move(result).setToError(
          AsyncRT::getMLIRDiagnostic(tmOr.takeError(), loc));
    }
    llvm::TargetMachine &tm = **tmOr;

    // Materialize the module.
    LLVMModuleAndContext module = produceModule();

    // Optimize the llvm Module.
    if (failed(optimizeLLVMModule(*module, tm, options, runtime, moduleIdx))) {
      return std::move(result).setToError(
          AsyncRT::getMLIRDiagnostic("failed to optimize LLVM IR.", loc));
    }

    // HACK HACK HACK https://github.com/modularml/modular/issues/22959
    // HACK: Some targets like PTX don't support object files so can only
    // emit assembly.
    bool emitAssembly =
        tm.getTargetTriple().str().find("nvptx") != std::string::npos;
    SmallVector<AnyAsyncValueRef> buffers = compileOptimizedLLVMToObjects(
        std::move(module), loc, options, runtime, transformCache, parLLC, isJIT,
        emitAssembly, moduleIdx);

    andThenAsyncMoving(
        buffers, [result = std::move(result)](
                     MutableArrayRef<AnyAsyncValueRef> values) mutable {
          SmallVector<BufferRef> results;
          results.reserve(values.size());
          for (AnyAsyncValueRef &result : values)
            results.push_back(std::move(result.get<BufferRef>()));
          std::move(result).emplace(std::move(results));
        });
  });

  return result;
}

ErrorOr<ElementsAttr> ObjectCompiler::emitArchiveAttr(ModuleOp module) {
  auto bufferOr = emitArchive(module);
  if (bufferOr.isError())
    return bufferOr.takeError();
  BufferRef buffer = bufferOr.takeValue();

  // Get the standalone archive key to use as the archive name.
  WriteableBufferRef produceStandaloneArchiveKey = WriteableBuffer::get();
  options.print(*produceStandaloneArchiveKey << "emitArchiveAttr(");
  *produceStandaloneArchiveKey << ")";
  if (failed(mlir::writeBytecodeToFile(module.getOperation(),
                                       *produceStandaloneArchiveKey)))
    return Error("failed to write bytecode file");
  // Hash it so the name isn't enormous.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)produceStandaloneArchiveKey->getBufferStart(),
               produceStandaloneArchiveKey->getBufferSize()));

  // Produce a DenseResourceElementsAttr from the file.
  auto resourceManager =
      DenseResourceElementsHandle::getManagerInterface(module.getContext());

  // Pretend this is a "tensor" of data.
  // TODO (#6986) It would be much nicer if we didn't have to clone this data
  //   and we could just reference the data already in the CAS. That would also
  //   prevent us from having to hash the module above.
  auto attrType = RankedTensorType::get(
      {(int64_t)buffer->getBufferSize()},
      IntegerType::get(module.getContext(), 8, IntegerType::Unsigned));
  auto attrName = "archive_" + llvm::toHex(hash, /*LowerCase=*/true);
  ArrayRef<char> blobData(buffer->getBufferStart(), buffer->getBufferSize());
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(blobData,
                                                                  /*align=*/8);
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.insert(attrName, std::move(blob)));
}

//===----------------------------------------------------------------------===//
// emitAssembly
//===----------------------------------------------------------------------===//

ErrorOrSuccess ObjectCompiler::emitAssembly(ModuleOp module,
                                            llvm::raw_pwrite_stream &os) {
  CompilerTimeTraceScope traceScope("emitAssembly");

  LLVMModuleAndContext llvmModule;
  if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
        return lowerAllFuncsToLLVM(ctx, module);
      }))
    return err.takeError();

  auto machineOr = createTargetMachine(options, /*isJIT=*/false);
  if (failed(machineOr))
    return machineOr.takeError();

  // Set the data layout on the module.
  llvmModule->setDataLayout((*machineOr)->createDataLayout());

  // Emit the assembly.
  if (failed(KGEN::compileLLVMToAssembly(std::move(llvmModule), **machineOr, os,
                                         options, runtime)))
    return Error("failed to lower LLVM IR to assembly");

  return success();
}
