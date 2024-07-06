//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGENToLLVMPipeline.h"
#include "LLVMPassesPipeline.h"

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
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLVMPassesPipeline.h"
#include "Support/Context.h"
#include "Support/FileSystemExtras.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/IRMapping.h"
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
static LogicalResult runLLVMOptPasses(llvm::Module &module,
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
  if (KGEN::addPassesToEmitFile(options, llvmTargetMachine, passMgr, os,
                                nullptr, fileType, true, machineModInfoPass))
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
              *module, options, **machineOr, *buf,
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

        if (failed(runLlcPasses(*module, options, **machineOr, outFile->os(),
                                llvm::CodeGenFileType::AssemblyFile)))
          return std::move(output).setToError(LLCL::getMLIRDiagnostic(
              "llc failed to codegen LLVM IR to object code", loc));
        outFile->keep();
      }

      std::move(output).emplace(buf.copy());
    });
    return output;
  };

  auto onCacheHit = [&](BufferRef buf) {
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
}

/// Optimize the llvm module to prepare for codegen object file.
static LogicalResult optimizeLLVMModule(llvm::Module &module,
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

/// Compile the given LLVM module to object files and return the async values
/// that contains the compiled object file.
/// isParLLC is true: split module into per function for parallel llc lowering
///                   and return multiple object files.
/// isParLLC is false: compile module without splitting into one object file.
static SmallVector<LLCL::AnyAsyncValueRef> compileOptimizedLLVMToObjects(
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
          runLlcPasses(module, options, targetMachine, objStream,
                       emitAssembly ? llvm::CodeGenFileType::AssemblyFile
                                    : llvm::CodeGenFileType::ObjectFile,
                       runtime.context->get<M::Telemetry::TelemetryContext>())))
    return failure();

  if (!options.saveTempsPrefix.empty()) {
    std::string outPath = saveTempsPrefix + ".asm";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile)
      return failure();

    if (failed(runLlcPasses(module, options, targetMachine, outFile->os(),
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
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, ctx, moduleName);
  if (!llvmModule)
    return Error("translate module to LLVMIR failed");

  // Attach any necessary instrumentation to the module.
  attachInstrumentationAttributes(*llvmModule, options);

  return llvmModule;
}

//===----------------------------------------------------------------------===//
// produceStandaloneModule
//===----------------------------------------------------------------------===//

/// Slice the dependencies of an operation out of the existing module into the
/// self-contained slice module.
static void sliceDependencies(Operation *op, SymbolTable &sliceSymtab,
                              const SymbolTable &symtab) {
  // Extract a dependency from the IR parent module and place it into the slice
  // module if it does not already exist. If a symbol was copied, return it.
  auto extractDependency = [&](StringAttr name) -> Operation * {
    // Don't copy the symbol if it is already copied.
    if (sliceSymtab.lookup(name))
      return nullptr;

    Operation *symbol = symtab.lookup(name);
    // If the symbol reference attribute doesn't reference a symbol, somehow
    // invalid IR made it to the ObjectCompiler.
    assert(symbol && "invalid IR?");

    // Clone the symbol into the new symbol table.
    Operation *copy = symbol->clone();
    sliceSymtab.insert(copy);
    return copy;
  };

  std::vector<Operation *> worklist;
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](FlatSymbolRefAttr ref) {
    if (Operation *decl = extractDependency(ref.getAttr()))
      worklist.push_back(decl);
  });
  auto extractDependencies = [&](Operation *op) {
    // Extract references to type declarations.
    walker.walk(op->getAttrDictionary());
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions())
      for (Type type : region.getArgumentTypes())
        walker.walk(type);
  };

  worklist.push_back(op);
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();
    op->walk(extractDependencies);
  }
}

OwningOpRef<ModuleOp>
ObjectCompiler::produceStandaloneModule(const SymbolTable &symtab,
                                        const ExportMap &exportedSymbols) {
  IRMapping unused;
  return produceStandaloneModule(symtab, exportedSymbols, unused);
}

OwningOpRef<ModuleOp>
ObjectCompiler::produceStandaloneModule(const SymbolTable &symtab,
                                        const ExportMap &exportedSymbols,
                                        IRMapping &mapping) {
  CompilerTimeTraceScope traceScope("produceStandaloneModule");
  auto module = cast<ModuleOp>(symtab.getOp());
  // Create a new module for these funcs. This will go away at the end
  // of this function.
  OwningOpRef<ModuleOp> singleModule = ModuleOp::create(module->getLoc());
  singleModule.get()->setAttrs(module->getAttrDictionary());

  // Create a new symbol table for the sliced module.
  SymbolTable sliceSymtab(*singleModule);

  for (auto [sym, exportVal] : exportedSymbols) {
    auto func = symtab.lookup<ExportInterface>(sym);
    assert(func && "Unknown exported symbol");

    // Traverse the call graph and clone all the callees into this module.
    sliceDependencies(func, sliceSymtab, symtab);

    // Clone the func into this new module. We don't want to remove it from
    // the current module. Make sure the function is also exported in the slice.
    auto sliceFn = sliceSymtab.lookup<ExportInterface>(sym);
    if (!sliceFn) {
      sliceFn = cast<ExportInterface>(func->clone(mapping));
      sliceSymtab.insert(sliceFn);
    }
    ExportKind kind = func.getExportKind();
    sliceFn.setExportKind(kind == ExportKind::NotExported ? exportVal.kind
                                                          : kind);
  }

  return singleModule;
}

//===----------------------------------------------------------------------===//
// emitArchive
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef> ObjectCompiler::emitArchive(ModuleOp module) {
  CompilerTimeTraceScope traceScope("produce-archive");

  // Perform a cache aware transformation to translate the module to an archive
  // file.
  LLCL::Runtime &runtime = *loadContext(&context)->get<LLCL::Runtime>();
  auto runTransformation = [&](Operation *op, WriteableBufferRef buf,
                               LLCL::AnyAsyncValueRef chain) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
#ifdef MODULAR_ENABLE_TELEMETRY
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheMiss("ObjectCompiler::emitArchive");
#endif
    chain.andThenSync([this, &runtime, op, output = output.copy(),
                       buf = buf.copy()]() mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
      [[maybe_unused]] auto timeScope =
          loadContext(op->getContext())
              ->get<M::Telemetry::TelemetryContext>()
              ->createUInt64Timer<std::chrono::milliseconds>(
                  "mojo.compile.cache.miss.time", M::Telemetry::Level::L2,
                  {{"pipeline", "ObjectCompiler::emitArchive"}});

#endif

      SmallVector<BufferRef> archiveBuffers;
      // Lower the module to LLVM.
      llvm::LLVMContext ctx;
      ErrorOr<std::unique_ptr<llvm::Module>> llvmModuleOr =
          lowerAllFuncsToLLVM(ctx, cast<ModuleOp>(op));

      if (llvmModuleOr) {
        return std::move(output).setToError(LLCL::getMLIRDiagnostic(
            Twine(
                "failed to lower module to LLVM IR for archive compilation, ") +
                llvmModuleOr.getError(),
            op->getLoc()));
      }
      CompilerTimeTraceScope traceScope("split-input-module");

      std::unique_ptr<llvm::Module> llvmModule = llvmModuleOr.takeValue();
      StringRef moduleName = llvmModule->getName();

      // HACK HACK HACK https://github.com/modularml/modular/issues/22959
      // HACK: If we are generating PTX we don't want to split.
      bool generatingPtx =
          options.targetTriple.find("nvptx") != std::string::npos;

      // Split the module into multiple slices and compile each in parallel.
      // FIXME(#25622): Disable module splitting for non-standalone archives.
      SmallVector<LLCL::AnyAsyncValueRef> cacheResults;
      bool noSplitting =
          runtime.getWorkQueue()->getParallelismLevel() < 2 || generatingPtx;

      auto processSync = [&]() {
        // If sync, await cacheResults so that the cloned sub-module
        // can be released before launching the next batch to reduce
        // memory pressure.
        await(cacheResults);

        // If any of the cache results failed, propagate the error.
        for (auto &result : cacheResults) {
          if (result.isError())
            return std::move(output).setToError(result.takeDiagnostic());
        }
        for (LLCL::AnyAsyncValueRef &result : cacheResults) {
          // Move the result buffer to archiveBuffers so that we can
          // concatenate them later together.
          archiveBuffers.emplace_back(std::move(result.get<BufferRef>()));
        }
        // Clear cacheResults for next batch.
        cacheResults.clear();
      };

      bool parLLC = runtime.getWorkQueue()->getParallelismLevel() >= 2 &&
                    !generatingPtx && options.enableParallelLLC;
      if (noSplitting) {
        SmallVector<AnyAsyncValueRef> results = lowerLLVMModuleToObjects(
            *llvmModule, op->getLoc(), op->getContext(), parLLC);

        for (AnyAsyncValueRef &result : results)
          cacheResults.emplace_back(std::move(result));

      } else {
        if (!options.saveTempsPrefix.empty()) {
          std::string outPath = options.saveTempsPrefix + ".pre-split.ll";
          std::unique_ptr<llvm::ToolOutputFile> outFile =
              mlir::openOutputFile(outPath);
          if (outFile) {
            outFile->os() << *llvmModule;
            outFile->keep();
          }
        }

        if (options.enableLLVMPerFunctionSplitting) {
          splitPerFunction(
              *llvmModule, runtime.getWorkQueue()->getParallelismLevel(),
              [&](llvm::Module *inputModule, int64_t idx, bool sync) {
                if (inputModule) {
                  SmallVector<AnyAsyncValueRef> results =
                      lowerLLVMModuleToObjects(*inputModule, op->getLoc(),
                                               op->getContext(),
                                               /*parLLC=*/false, idx);
                  for (AnyAsyncValueRef &result : results)
                    cacheResults.emplace_back(std::move(result));
                }
                if (sync)
                  processSync();
              });
        } else {
          // TODO: Keep this less aggressive splitting for:
          // - REPL which has different object layout requirements layouts
          // (#35345).
          // - Other cases where aggressive splitting actually slow down
          // compilation and needs better heuristics to improve.
          splitPerExported(*llvmModule, [&](llvm::Module &inputModule,
                                            int64_t idx) {
            SmallVector<AnyAsyncValueRef> results = lowerLLVMModuleToObjects(
                inputModule, op->getLoc(), op->getContext(), parLLC, idx);
            for (AnyAsyncValueRef &result : results)
              cacheResults.emplace_back(std::move(result));
          });
        }
      }

      if (noSplitting || !options.enableLLVMPerFunctionSplitting) {
        andThenSyncMoving(
            cacheResults,
            [moduleName = moduleName.str(), op, buf = buf.copy(),
             output = output.copy(),
             generatingPtx](MutableArrayRef<AnyAsyncValueRef> values) mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
              [[maybe_unused]] auto timeScope =
                  loadContext(op->getContext())
                      ->get<M::Telemetry::TelemetryContext>()
                      ->createUInt64Timer<std::chrono::milliseconds>(
                          "mojo.compile.cache.miss.time",
                          M::Telemetry::Level::L2,
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
                assert(values.size() == 1 &&
                       "should have one result if generating PTX");
                *buf << values[0].get<BufferRef>()->getBuffer();
                std::move(output).emplace(buf.copy());
                return;
              }

              // Now that all the object files have been compiled, merge them
              // all into a single archive.
              SmallVector<std::string> archiveMemberNames(values.size());
              SmallVector<llvm::NewArchiveMember> archiveMembers;
              for (auto [idx, result] : llvm::enumerate(values)) {
                auto &resultBuf = result.get<BufferRef>();
                archiveMemberNames[idx] =
                    (moduleName + "." + Twine(idx) + ".o").str();

                archiveMembers.emplace_back(llvm::MemoryBufferRef(
                    resultBuf->getBuffer(), archiveMemberNames[idx]));
              }

              auto result = llvm::writeArchiveToBuffer(
                  archiveMembers,
                  /*WriteSymtab=*/llvm::SymtabWritingMode::NormalSymtab,
                  archiveMembers.front().detectKindFromObject(),
                  /*Deterministic=*/true, /*Thin=*/false);
              if (!result) {
                return std::move(output).setToError(LLCL::getMLIRDiagnostic(
                    "failed to concatenate object files into archive",
                    op->getLoc()));
              }

              // Copy the result into the output buffer.
              *buf << (*result)->getBuffer();
              std::move(output).emplace(buf.copy());
            });
      } else {
        CompilerTimeTraceScope traceScope("concatenate-object-files");
        // Now that all the object files have been compiled,
        // merge them all into a single archive.
        SmallVector<std::string> archiveMemberNames(archiveBuffers.size());
        SmallVector<llvm::NewArchiveMember> archiveMembers;

        for (auto [index, resultBuf] : llvm::enumerate(archiveBuffers)) {
          archiveMemberNames[index] =
              (moduleName + "." + Twine(index) + ".o").str();
          archiveMembers.emplace_back(llvm::MemoryBufferRef(
              resultBuf->getBuffer(), archiveMemberNames[index]));
        }
        auto result = llvm::writeArchiveToBuffer(
            archiveMembers,
            /*WriteSymtab=*/llvm::SymtabWritingMode::NormalSymtab,
            archiveMembers.front().detectKindFromObject(),
            /*Deterministic=*/true, /*Thin=*/false);
        if (!result) {
          return std::move(output).setToError(LLCL::getMLIRDiagnostic(
              "failed to concatenate object files into archive", op->getLoc()));
        }
        // Copy the result into the output buffer.
        *buf << (*result)->getBuffer();
        std::move(output).emplace(buf.copy());
      }
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
      LLCL::AsyncValueRef<Chain>::createReady(runtime),
      std::move(produceArchiveKey), runTransformation, onCacheHit);
  await(output);

  if (output.isError())
    return {std::move(output.takeDiagnostic().getMessage())};
  return {std::move(output.get<BufferRef>())};
}

//===----------------------------------------------------------------------===//
// lowerLLVMModuleToObjects
//===----------------------------------------------------------------------===//

SmallVector<LLCL::AnyAsyncValueRef>
ObjectCompiler::lowerLLVMModuleToObjects(llvm::Module &module, Location loc,
                                         MLIRContext *mlirContext, bool parLLC,
                                         std::optional<size_t> moduleIdx) {
  LLCL::Runtime &runtime = *loadContext(&context)->get<LLCL::Runtime>();
  SmallVector<LLCL::AnyAsyncValueRef> results;

  // Create the target machine.
  auto machineOr = createTargetMachine(options, isJIT);
  if (failed(machineOr)) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
    std::move(output).setToError(
        LLCL::getMLIRDiagnostic(machineOr.takeError(), loc));
    results.emplace_back(std::move(output));
    return results;
  }

  // Set the data layout on the module.
  module.setDataLayout((*machineOr)->createDataLayout());

  // Optimize the llvm Module.
  if (failed(optimizeLLVMModule(module, **machineOr, options, runtime,
                                moduleIdx))) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
    std::move(output).setToError(
        LLCL::getMLIRDiagnostic("failed to optimize LLVM IR.", loc));
    results.emplace_back(std::move(output));
    return results;
  }

  // HACK HACK HACK https://github.com/modularml/modular/issues/22959
  // HACK: Some targets like PTX don't support object files so can only
  // emit assembly.
  bool emitAssembly =
      (*machineOr)->getTargetTriple().str().find("nvptx") != std::string::npos;

  // Codegen optimized llvm module to object files.
  return compileOptimizedLLVMToObjects(module, loc, options, runtime,
                                       transformCache, parLLC, isJIT,
                                       emitAssembly, moduleIdx);
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
  CompilerTimeTraceScope traceScope("produce-standalone-assembly");

  llvm::LLVMContext ctx;
  ErrorOr<std::unique_ptr<llvm::Module>> llvmModuleOr =
      lowerAllFuncsToLLVM(ctx, module);

  if (llvmModuleOr)
    return Error(llvmModuleOr.getError());

  auto machineOr = createTargetMachine(options, /*isJIT=*/false);
  if (failed(machineOr))
    return machineOr.takeError();

  std::unique_ptr<llvm::Module> llvmModule = llvmModuleOr.takeValue();

  // Set the data layout on the module.
  llvmModule->setDataLayout((*machineOr)->createDataLayout());

  // Emit the assembly.
  LLCL::Runtime &runtime = *loadContext(&context)->get<LLCL::Runtime>();
  if (failed(KGEN::compileLLVMToObject(*llvmModule, **machineOr, os, options,
                                       runtime, /*emitAssembly=*/true)))
    return Error("failed to lower LLVM IR to assembly");

  return success();
}
