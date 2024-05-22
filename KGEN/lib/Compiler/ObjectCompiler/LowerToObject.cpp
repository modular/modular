//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LowerToObject.h"
#include "Cache/CacheTelemetryContext.h"
#include "JITSupport.h"
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
#include "llvm/TargetParser/Host.h"

#include <utility>

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "lower-to-object"

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

ErrorOr<ObjectCompiler> ObjectCompiler::create(mlir::PassManager &mgr,
                                               StringRef basePath,
                                               CompilationOptions options,
                                               bool isJIT, bool isSearch) {
  auto transformCache = Cache::getLocalDefaultBackendChain(
      std::filesystem::path(basePath.str()) / "transform", KGEN_VERSION_STRING);
  if (failed(transformCache))
    return transformCache.takeError();
  return ObjectCompiler(mgr, std::move(*transformCache), std::move(options),
                        isJIT, isSearch);
}

ObjectCompiler::ObjectCompiler(mlir::PassManager &mgr,
                               RCRef<Cache::BlobCacheBackend> transformCache,
                               CompilationOptions options, bool isJIT,
                               bool isSearch)
    : transformCache(
          decltype(this->transformCache)::create(std::move(transformCache))),
      mgr(&mgr), options(std::move(options)), isJIT(isJIT), isSearch(isSearch) {
}

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
    RCRef<Cache::TransformCache> transformCache) {
  WriteableBufferRef keyBuf = WriteableBuffer::get();
  options.print(*keyBuf << "compileOptimizedLLVMModuleToObject(");
  *keyBuf << ")";
  size_t nonBitcodeKeySize = keyBuf->getBufferSize();

  llvm::WriteBitcodeToFile(module, *keyBuf);

  auto runTransformation = [nonBitcodeKeySize, loc, &runtime, emitAssembly,
                            keyBuf = keyBuf.copy(), options,
                            isJIT](WriteableBufferRef buf,
                                   LLCL::AnyAsyncValueRef chain) mutable {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
#ifdef MODULAR_ENABLE_TELEMETRY
    Cache::CacheTelemetryContext::getCacheTelemetryContext(runtime.context)
        .recordCacheMiss("compileOptimizedLLVMModuleToObject");
#endif

    chain.andThenAsync([nonBitcodeKeySize, loc, &runtime, emitAssembly,
                        output = output.copy(), buf = buf.copy(),
                        keyBuf = std::move(keyBuf), options, isJIT]() mutable {

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
        transformCache));
  };

  if (isParLLC) {
    splitPerFunction(module, runtime.getWorkQueue()->getParallelismLevel(),
                     compileToObject);
  } else {
    cacheResults.push_back(compileOptimizedLLVMModuleToObject(
        module, loc, runtime, isJIT, emitAssembly, options, transformCache));
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

//===----------------------------------------------------------------------===//
// ObjectCompilerMaterializationUnit
//===----------------------------------------------------------------------===//

class ObjectCompilerLayer::ObjectCompilerMaterializationUnit
    : public llvm::orc::MaterializationUnit {
public:
  ObjectCompilerMaterializationUnit(ObjectCompilerLayer &layer,
                                    SymbolTable symtab,
                                    const ExportMap &exports)
      : MaterializationUnit(layer.getInterface(symtab, exports)),
        genLayer(layer), symtab(symtab), exports(exports) {}

  /// Provide a name for this MU that will show up in ORC debug logs.
  StringRef getName() const override {
    return "KGEN::ObjectCompilerMaterializationUnit";
  }

  /// Given a MaterializationResponsibility, materialize the code for those
  /// symbols and forward them to the next layer.
  void materialize(
      std::unique_ptr<llvm::orc::MaterializationResponsibility> mr) override {
    genLayer.emit(std::move(mr), symtab, exports);
  }

  /// Notify that the symbol `name` has been overridden and this MU should
  /// remove it from the source. This removes the symbol from `symtab`.
  void discard(const llvm::orc::JITDylib &jd,
               const llvm::orc::SymbolStringPtr &name) override {
    // TODO: Figure out what to do here for the REPL. Symbols cannot be erased
    // during elaboration.
  }

  ObjectCompilerLayer &genLayer;
  SymbolTable symtab;
  ExportMap exports;
};

//===----------------------------------------------------------------------===//
// ObjectCompilerLayer
//===----------------------------------------------------------------------===//

ObjectCompilerLayer::ObjectCompilerLayer(ObjectCompiler &&objCompiler,
                                         llvm::orc::ObjectLayer &base,
                                         llvm::orc::ExecutionSession &sess,
                                         const llvm::DataLayout &dl,
                                         AddToSearchOrderFn add)
    : MaterializationLayer(LayerKind::kObjectCompilerLayer, sess, dl,
                           std::move(add)),
      objectCompiler(std::move(objCompiler)), baseLayer(base) {}

/// Produce an ExportMap with every symbol in the module.
static ExportMap getAllSymbols(ModuleOp theModule) {
  ExportMap exports;
  for (auto sym : theModule.getOps<mlir::SymbolOpInterface>())
    exports.insert({sym.getNameAttr(), {ExportKind::Exported}});
  return exports;
}

ErrorOrSuccess ObjectCompilerLayer::add(StringRef libName, ModuleOp theModule) {
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();

  llvm::orc::JITDylib *dylib = *dylibOr;
  llvm::orc::ResourceTrackerSP resourceTracker =
      dylib->getDefaultResourceTracker();

  // Add the materialization unit by computing the exports and the symbol
  // table, and passing those off.
  SymbolTable symtab(theModule);
  ExportMap exports = getExportedSymbols(theModule);
  if (exports.empty())
    exports = getAllSymbols(theModule);

  // Add the materialization unit.
  return toModularErrorOr(
      dylib->define(std::make_unique<ObjectCompilerMaterializationUnit>(
                        *this, symtab, exports),
                    resourceTracker));
}

void ObjectCompilerLayer::emit(
    std::unique_ptr<llvm::orc::MaterializationResponsibility> mr,
    const SymbolTable &symtab, const ExportMap &exports) {
  if (auto err = emitImpl(*mr, symtab, exports)) {
    error = err.takeError();
    mr->failMaterialization();
  }
}

ErrorOrSuccess
ObjectCompilerLayer::emitImpl(llvm::orc::MaterializationResponsibility &mr,
                              const SymbolTable &symtab,
                              const ExportMap &exports) {
  auto theModule = cast<ModuleOp>(symtab.getOp());

  ErrorOr<BufferRef> bufOr = Error(" ");
  if (exports.empty()) {
    bufOr = objectCompiler.produceStandaloneArchive(symtab,
                                                    getAllSymbols(theModule));
  } else {
    bufOr = objectCompiler.produceStandaloneArchive(symtab, exports);
  }

  // No buffer - materialization fails.
  if (bufOr.isError())
    return bufOr.takeError();
  BufferRef archiveBuf = std::move(*bufOr);
  // Create an Archive object.
  auto archiveOr = toModularErrorOr(llvm::object::Archive::create(
      llvm::MemoryBufferRef(archiveBuf->getBuffer(),
                            /*BufferName=*/"")));
  if (archiveOr.isError())
    return archiveOr.takeError();
  std::unique_ptr<llvm::object::Archive> archive = std::move(*archiveOr);

  // Create a set of necessary objects from the requested symbols so we don't
  // double-add anything.
  StringSet<> necessaryObjects;

  // Set up a worklist that starts from the set of requested symbols.
  SmallVector<llvm::orc::SymbolStringPtr> worklist;
  llvm::append_range(worklist, llvm::make_first_range(mr.getSymbols()));
  LLVM_DEBUG(llvm::dbgs() << "Initial lookup worklist: [";
             llvm::interleaveComma(worklist, llvm::dbgs(),
                                   [](auto ptr) { llvm::dbgs() << *ptr; });
             llvm::dbgs() << "]\n");

  // We do this iteration/DFS search of dependencies here rather than in
  // StandaloneObject because here we have a set of symbols we care about, so we
  // are more likely to (a) get a benefit and (b) it's simpler because there are
  // fewer symbols to deal with.
  // TODO: This is really not great - the definition generator infrastructure
  //   should 'just handle' this. Investigate and see if we could avoid this by
  //   using the infra better.

  // Vector of binaries we will delegate to JITLink.
  SmallVector<std::tuple<std::unique_ptr<llvm::object::Binary>,
                         std::unique_ptr<llvm::MemoryBuffer>, size_t>>
      toDelegate;
  while (!worklist.empty()) {
    llvm::orc::SymbolStringPtr sym = worklist.pop_back_val();
    ErrorOr<std::optional<llvm::object::Archive::Child>> childOr =
        toModularErrorOr(archive->findSym(*sym));
    if (childOr.isError())
      return childOr.takeError();

    // We don't have the symbol in this archive, move on.
    if (!*childOr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Could not find '" << *sym << "' in this archive\n");
      continue;
    }

    // Grab the memory buffer for that object.
    llvm::object::Archive::Child &child = **childOr;
    auto bufferRef = toModularErrorOr(child.getMemoryBufferRef());
    if (bufferRef.isError())
      return bufferRef.takeError();

    // Already handled this exact binary, continue. We don't use the file name
    // because that has the potential to conflict.
    if (!necessaryObjects.insert(bufferRef->getBuffer()).second)
      continue;

    std::unique_ptr<llvm::MemoryBuffer> objectBuf =
        llvm::MemoryBuffer::getMemBuffer(*bufferRef,
                                         /*RequiresNullTerminator=*/false);

    // Get the child as a binary - that will allow us to look up the symbols
    // contained in the object.
    auto binaryOr = toModularErrorOr(child.getAsBinary());
    if (binaryOr.isError())
      return binaryOr.takeError();
    std::unique_ptr<llvm::object::Binary> objectBin = std::move(*binaryOr);

    auto *objectFile = dyn_cast<llvm::object::ObjectFile>(objectBin.get());
    if (!objectFile)
      return Error("archive member " + objectBuf->getBufferIdentifier() +
                   " was not an object file");

    // If the object file has undefined symbols in it, add those to the
    // worklist. Don't discriminate here, we don't need to do the full
    // providence checking. We early-exit if a symbol is defined in a file we're
    // already going to handle, so we don't have to worry about being
    // overly-inclusive.
    for (auto symbol : objectFile->symbols()) {
      auto flagsOr = toModularErrorOr(symbol.getFlags());
      if (flagsOr.isError())
        return flagsOr.takeError();
      uint32_t flags = *flagsOr;

      if (!(flags & llvm::object::BasicSymbolRef::SF_Undefined))
        continue;

      auto nameOr = toModularErrorOr(symbol.getName());
      if (nameOr.isError())
        return nameOr.takeError();

      // Add this undefined symbol to the worklist.
      LLVM_DEBUG(llvm::dbgs()
                 << "Undefined symbol found: '" << *nameOr << "'\n");
      worklist.push_back(session.intern(*nameOr));
    }

    // Add the object file and its memory buffer to the list of objects to be
    // delegated to JITLink.
    toDelegate.emplace_back(std::move(objectBin), std::move(objectBuf),
                            child.getDataOffset());
  }

  // Sort the objects by their offset in the archive. This makes sure that we
  // maintain the order of the objects in the archive.
  llvm::sort(toDelegate, [](const auto &lhs, const auto &rhs) {
    return std::get<2>(lhs) < std::get<2>(rhs);
  });

  // Delegate each object file through its own materialization unit.
  for (auto &[bin, buf, offset] : toDelegate) {
    // Get all the symbols defined by the object file.
    auto itf = toModularErrorOr(
        llvm::orc::getObjectFileInterface(session, buf->getMemBufferRef()));
    if (itf.isError())
      return itf.takeError();

    // Ask the MR to define the new symbols as materializing. The MR may reject
    // some of the symbols.
    for (auto &[symbol, flags] : itf->SymbolFlags) {
      if (llvm::Error err = mr.defineMaterializing({{symbol, flags}})) {
        // We don't error out for duplicate symbols, just ignore them.
        if (err.isA<llvm::orc::DuplicateDefinition>()) {
          llvm::consumeError(std::move(err));
          continue;
        }
        return toModularErrorOr(std::move(err));
      }
    }

    // Construct a set of all symbols in the object file that were not rejected
    // by the MR, and then delegate them from the MR.
    llvm::orc::SymbolNameSet delegatedSymbols;
    llvm::orc::SymbolFlagsMap symbolFlags;
    for (auto &[symbol, flags] : itf->SymbolFlags) {
      if (!mr.getSymbols().contains(symbol))
        continue;
      delegatedSymbols.insert(symbol);
      symbolFlags.try_emplace(symbol, flags);
    }

    // If for whatever reason all delegated symbols were rejected, then there is
    // nothing to do.
    if (delegatedSymbols.empty())
      continue;

    auto delMr = toModularErrorOr(mr.delegate(delegatedSymbols));
    if (delMr.isError())
      return delMr.takeError();
    itf->SymbolFlags = std::move(symbolFlags);

    // Replace the materialization responsibility with a new materialization
    // unit consisting of the accepted object file symbols.
    auto delMu =
        std::make_unique<llvm::orc::BasicObjectLayerMaterializationUnit>(
            baseLayer, std::move(buf), std::move(*itf));
    if (auto err = toModularErrorOr((*delMr)->replace(std::move(delMu))))
      return err.takeError();
  }

  // If all symbols have been materialized, then return success.
  if (mr.getSymbols().empty())
    return success();

  // Otherwise, complain about not being able to find the leftover symbols.
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Failed to materialize all symbols, leftover "
        "MaterializationResponsibility symbols: [\n";
  for (llvm::orc::SymbolStringPtr ptr : llvm::make_first_range(mr.getSymbols()))
    os << "  " << *ptr << "\n";
  os << "]\n";
  return Error(std::move(msg));
}

llvm::orc::MaterializationUnit::Interface
ObjectCompilerLayer::getInterface(const SymbolTable &symtab,
                                  const ExportMap &exports) {
  llvm::orc::MangleAndInterner mangler(session, dataLayout);
  llvm::orc::SymbolFlagsMap symbols;

  auto addSymbols = [&](const ExportMap &exportedSymbols) {
    for (auto &[name, symbol] : exports)
      symbols[mangler(name)] = getFlagsForExportedSymbol(symbol);
  };

  // If we don't have any exports, infer them from the module.
  if (!exports.empty())
    addSymbols(exports);
  else
    addSymbols(getAllSymbols(cast<ModuleOp>(symtab.getOp())));

  if (objectCompiler.isJIT) {
    symbols[mangler(ExecutionEngine::getGlobalCtorFnName())] =
        getGlobalFnSymbolFlags();
    symbols[mangler(ExecutionEngine::getGlobalDtorFnName())] =
        getGlobalFnSymbolFlags();
  }

  return {std::move(symbols), /*InitSymbol=*/nullptr};
}
