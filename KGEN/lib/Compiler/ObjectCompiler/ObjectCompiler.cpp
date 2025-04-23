//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/ObjectCompiler.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/CompilerSupport/MLIRLocationDecoder.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/LLVMIRUtils.h"
#include "KGEN/Support/BuildInfo.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/Debug.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGENToLLVMPipeline.h"
#include "LLVMAccessorHelper.h"
#include "LLVMPassesPipeline.h"
#include "MCLinker.h"
#include "Support/Context.h"
#include "Support/FileSystemExtras.h"
#include "Support/Telemetry/Telemetry.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace M;
using namespace KGEN;
using namespace Cache;

#define DEBUG_TYPE "object-compiler"
#define KGEN_DEBUG_TYPE "object-compiler"

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<ObjectCompiler>>
ObjectCompiler::create(StringRef basePath, CompilationOptions options,
                       bool isJIT, MLIRContext &context,
                       PassManagerConfigOptions pmOptions) {
  auto transformCache = Cache::getLocalDefaultBackendChain(
      std::filesystem::path(basePath.str()) / "transform", getVersionString());
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

  ModulePassManager modulePassMgr =
      buildLLVMOptimizationPipeline(passBuilder, options);

  // Now that we have all of the passes ready, run them.
  modulePassMgr.run(module, moduleAnalysisMgr);
  return mlir::success();
}

/// Run the default llc passes required to generate object code.
static LogicalResult
runLlcPasses(llvm::Module &module, CompilationOptions &options,
             llvm::TargetMachine &targetMachine, llvm::raw_pwrite_stream &os,
             std::unique_ptr<llvm::MachineModuleInfo> &machineModuleInfo,
             std::unique_ptr<llvm::MCContext> &mcContext,
             llvm::CodeGenFileType fileType, bool stopBeforeAsmPrint,
             unsigned numFunctionsBase = 0,
             llvm::TargetMachine *sharedTargetMachine = nullptr,
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

  TargetMachine &llvmTargetMachine =
      static_cast<TargetMachine &>(targetMachine);

  MachineModuleInfoWrapperPass *machineModInfoPass;

  if (stopBeforeAsmPrint) {
    if (sharedTargetMachine) {
      mcContext = std::make_unique<llvm::MCContext>(
          sharedTargetMachine->getTargetTriple(),
          sharedTargetMachine->getMCAsmInfo(),
          sharedTargetMachine->getMCRegisterInfo(),
          sharedTargetMachine->getMCSubtargetInfo(), nullptr,
          &sharedTargetMachine->Options.MCOptions, false);

    } else {
      mcContext = std::make_unique<llvm::MCContext>(
          llvmTargetMachine.getTargetTriple(), llvmTargetMachine.getMCAsmInfo(),
          llvmTargetMachine.getMCRegisterInfo(),
          llvmTargetMachine.getMCSubtargetInfo(), nullptr,
          &llvmTargetMachine.Options.MCOptions, false);
    }

    machineModInfoPass =
        new MachineModuleInfoWrapperPass(&llvmTargetMachine, &(*mcContext));

    mcContext->setObjectFileInfo(llvmTargetMachine.getObjFileLowering());

    if (KGEN::addPassesToEmitMC(options, llvmTargetMachine, passMgr, os, true,
                                machineModInfoPass, numFunctionsBase))

      return failure();
  } else {
    machineModInfoPass = new MachineModuleInfoWrapperPass(&llvmTargetMachine);
    if (KGEN::addPassesToEmitFile(options, llvmTargetMachine, passMgr, os,
                                  nullptr, fileType, true, machineModInfoPass))
      return failure();
  }

  const_cast<TargetLoweringObjectFile *>(llvmTargetMachine.getObjFileLowering())
      ->Initialize(machineModInfoPass->getMMI().getContext(), targetMachine);

  passMgr.run(module);

  if (stopBeforeAsmPrint) {
    machineModuleInfo = std::make_unique<llvm::MachineModuleInfo>(
        std::move(machineModInfoPass->getMMI()));
  }

  return mlir::success();
}

// Write the given `buf` to a file with the given prefix and postfix.
// Appends a hash based on `buf` contents to emitted file name.
static LogicalResult
writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                         const std::string &postfix, StringRef buf) {
  if (saveTempsPrefix.empty())
    return success();

  // Include unique hash as part of name.
  assert(sizeof(uint8_t) == sizeof(char) && "Assume char is 8 bits");
  auto hash =
      llvm::xxh3_128bits(ArrayRef((const uint8_t *)buf.data(), buf.size()));
  std::string outPath =
      saveTempsPrefix + "." + llvm::utohexstr(hash.high64, /*LowerCase=*/true) +
      llvm::utohexstr(hash.low64, /*LowerCase=*/true) + postfix;

  auto outFile = mlir::openOutputFile(outPath);
  if (!outFile)
    return failure();
  outFile->os() << buf;
  outFile->keep();
  return success();
}

template <typename ModuleT>
static LogicalResult writeTempModule(const std::string &saveTempsPrefix,
                                     const std::string &phase, ModuleT &module,
                                     const std::string &fileExt = ".ll") {
  if (saveTempsPrefix.empty())
    return success();

  const std::string finalSavePrefix = saveTempsPrefix + phase;
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << module;
  return writeBytesToTempWithHash(finalSavePrefix, fileExt, str);
}

/// Compile optimized llvm::Module module to object through the llc pipeline
/// asynchronously and cache the transformation.
static AsyncRT::AnyAsyncValueRef compileOptimizedLLVMModuleToObject(
    LLVMModuleAndContext module, Location loc,
    llvm::TargetMachine &targetMachine, std::mutex &tmMutex,
    AsyncRT::Runtime &runtime, bool isJIT, bool isParLLC,
    CompilationOptions options, RCRef<Cache::TransformCache> transformCache,
    std::optional<size_t> moduleIdx, std::optional<size_t> splitIdx,
    unsigned numFunctionBase) {
  WriteableBufferRef keyBuf;
  size_t nonBitcodeKeySize = 0;

  // No need to reload the module to a different context if we are not
  // going to further parallelizing compilation.
  // This is essential for NVPTX backend to avoid false hit
  // with stale AnnotationCache which is populated during both
  // llvm-opt and llc pipeline passes but is only cleared at the end of
  // codegen in AsmPrint. We need to make sure that llvm-opt and llc
  // are using the same llvm::Module so that the cache can be properly
  // cleaned.
  if (isParLLC) {
    keyBuf = WriteableBuffer::get();
    options.print(*keyBuf << "compileOptimizedLLVMModuleToObject(");
    *keyBuf << ")";
    nonBitcodeKeySize = keyBuf->getBufferSize();
    llvm::WriteBitcodeToFile(*module, *keyBuf);
    // Release memory.
    module.reset();
  }

  auto output = AsyncRT::AsyncValueRef<MCInfo>::allocate(runtime);

  runtime.getWorkQueue()->addTask(
      [nonBitcodeKeySize, loc, &runtime, keyBuf = keyBuf.copy(),
       output = output.copy(), options, isJIT, isParLLC, moduleIdx, splitIdx,
       numFunctionBase, inputModule = std::move(module), &targetMachine,
       &tmMutex]() mutable {
        if (isNVPTXBackend(options) && isParLLC) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "cannot do per function codegen for NVPTX backedn.", loc));
        }
        LLVMModuleAndContext moduleAndContext;
        if (isParLLC) {
          BufferRef keyBufRef(std::move(keyBuf));
          StringRef bitcodeBuffer = keyBufRef->getBuffer();
          bitcodeBuffer = bitcodeBuffer.drop_front(nonBitcodeKeySize);

          // Load the cached bytecode into a new context.
          // This is necessary to avoid data races during multi-threading with
          // per function parallelization.
          ErrorOrSuccess createModuleResult = moduleAndContext.create(
              [&](llvm::LLVMContext &ctx)
                  -> ErrorOr<std::unique_ptr<llvm::Module>> {
                llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
                    llvm::parseBitcodeFile(
                        llvm::MemoryBufferRef(bitcodeBuffer, ""), ctx);
                if (!moduleOr)
                  return Error("failed to create LLVMModuleAndContext");
                return std::move(*moduleOr);
              });
          if (createModuleResult) {
            return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                "failed to load LLVM IR bitcode", loc));
          }
        } else {
          moduleAndContext = std::move(inputModule);
        }

        // Create TargetMachine for this module. This is also necessary to
        // avoid data races during multi-threading.
        ErrorOr<std::unique_ptr<llvm::TargetMachine>> machineOr =
            createTargetMachine(options, isJIT);
        if (failed(machineOr)) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "failed to create TargetMachine", loc));
        }

        llvm::TargetMachine &llvmTargetMachine =
            static_cast<llvm::TargetMachine &>(**machineOr);
        llvmTargetMachine.Options.MCOptions.AsmVerbose = options.verboseOutput;
        llvmTargetMachine.Options.MCOptions.PreserveAsmComments =
            options.verboseOutput;

        std::string saveTempsPrefix = options.saveTempsPrefix;
        if (!options.saveTempsPrefix.empty()) {
          if (moduleIdx)
            saveTempsPrefix += "_" + std::to_string(*moduleIdx);
          if (splitIdx)
            saveTempsPrefix += "__" + std::to_string(*splitIdx);
        }

        if (failed(writeTempModule(saveTempsPrefix, ".pre-llc",
                                   *moduleAndContext))) {
          return std::move(output).setToError(
              AsyncRT::getMLIRDiagnostic("failed save pre-llc llvm IR", loc));
        }

        std::unique_ptr<llvm::MachineModuleInfo> machineModuleInfo;
        std::unique_ptr<llvm::MCContext> mcContext;
        auto buf = WriteableBuffer::get();

        // Run llc passes.
        if (failed(runLlcPasses(
                *moduleAndContext, options, **machineOr, *buf,
                machineModuleInfo, mcContext, llvm::CodeGenFileType::ObjectFile,
                /*stopBeforeAsmPrint=*/true, numFunctionBase, &targetMachine,
                runtime.context->get<M::Telemetry::TelemetryContext>()))) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "llc failed to codegen LLVM IR to object code", loc));
        }

        if (!options.saveTempsPrefix.empty()) {
          if (failed(writeTempModule(saveTempsPrefix, ".post-llc",
                                     *moduleAndContext))) {
            return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                "failed save post-llc llvm IR", loc));
          }
        }

        llvm::StringMap<const llvm::Function *> fnNameToFnPtr;
        auto wbuf = WriteableBuffer::get();

        for (llvm::Function &fn : moduleAndContext->functions())
          fnNameToFnPtr.insert({fn.getName().str(), &fn});

        llvm::WriteBitcodeToFile(*moduleAndContext, *wbuf);

        // Move and reset SubtargetInfo for MachineFunctions to
        // shared TargetMachine so as to reduce memory footprint
        // as soon as possible before reaching the mclinking barrier.
        {
          // Need to use a mutex here while modifying the shared targetMachine.
          std::lock_guard<std::mutex> lock(tmMutex);
          resetSubtargetInfo(targetMachine, *machineModuleInfo);
        }

        // Release more memory before reaching the mclinking barrier.
        releaseTargetMachineConstants(**machineOr);

        std::move(output).emplace(
            wbuf, std::move(machineModuleInfo),
            // Keep the original llvm::Module alive so that the MachineFunction
            // reference to llvm::Function is still valid.
            std::move(moduleAndContext), fnNameToFnPtr, std::move(*machineOr),
            std::move(mcContext), splitIdx);
      });

  return output;
}

/// Optimize the llvm module to prepare for codegen object file.
static LogicalResult optimizeLLVMModule(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        CompilationOptions &options,
                                        AsyncRT::Runtime &runtime,
                                        std::optional<size_t> moduleIdx) {
  llvm::DataLayout targetDataLayout =
      options.targetDataLayout.empty()
          ? targetMachine.createDataLayout()
          : llvm::DataLayout(options.targetDataLayout);
  module.setDataLayout(targetDataLayout);

  std::string saveTempsPrefix = options.saveTempsPrefix;
  if (moduleIdx && !options.saveTempsPrefix.empty())
    saveTempsPrefix += "." + std::to_string(moduleIdx.value());

  if (failed(writeTempModule(saveTempsPrefix, ".pre-opt", module)))
    return failure();

  if (failed(runLLVMOptPasses(module, targetMachine, options, runtime)))
    return failure();

  if (failed(writeTempModule(saveTempsPrefix, ".post-opt", module)))
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
    llvm::TargetMachine &targetMachine, std::mutex &tmMutex,
    CompilationOptions &options, AsyncRT::Runtime &runtime,
    RCRef<Cache::TransformCache> transformCache, bool isParLLC, bool isJIT,
    std::optional<size_t> moduleIdx, SymbolAndMCInfo &symbolAndMirInfo,
    unsigned numFunctionBase) {
  CompilerTimeTraceScope traceScope("compile-optimized-llvm-to-object",
                                    module->getName());

  // Perform module materialization in another task.
  auto launchCompilation = [&](llvm::unique_function<LLVMModuleAndContext()>
                                   produceModule,
                               std::optional<int64_t> idx,
                               unsigned numFunctions, bool isParLLC) {
    auto result = AsyncRT::AsyncValueRef<MCInfo>::allocate(runtime);

    runtime.getWorkQueue()->addTask([produceModule = std::move(produceModule),
                                     loc, &runtime, isJIT, isParLLC, &options,
                                     cache = transformCache.copy(), moduleIdx,
                                     idx, result = result.copy(), numFunctions,
                                     &targetMachine, &tmMutex]() mutable {
      AsyncRT::AnyAsyncValueRef output = compileOptimizedLLVMModuleToObject(
          produceModule(), loc, targetMachine, tmMutex, runtime, isJIT,
          isParLLC, options, cache, moduleIdx, idx, numFunctions);
      andThenSyncMoving(
          output, [result = std::move(result)](
                      MutableArrayRef<AnyAsyncValueRef> outputs) mutable {
            for (auto &out : outputs) {
              if (out.isError())
                return std::move(result).setToError(out.takeDiagnostic());
            }
            std::move(result).emplace(std::move(outputs.front().get<MCInfo>()));
          });
    });
    return result;
  };

  SmallVector<AsyncRT::AnyAsyncValueRef> cacheResults;
  if (!isParLLC) {
    cacheResults.push_back(launchCompilation(forwardModule(std::move(module)),
                                             std::nullopt, numFunctionBase,
                                             isParLLC));
  } else {
    if (failed(writeTempModule(options.saveTempsPrefix, ".pre-llc-split",
                               *module))) {
      auto error = AsyncRT::AnyAsyncValueRef::createError(
          runtime, AsyncRT::getMLIRDiagnostic(
                       "writing module to file before llc split failed", loc));
      cacheResults.push_back(std::move(error));
      return cacheResults;
    }
    splitPerFunction(
        std::move(module),
        [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
            std::optional<int64_t> idx, unsigned numFunctions) {
          cacheResults.push_back(launchCompilation(
              std::move(produceModule), idx, numFunctions, isParLLC));
        },
        symbolAndMirInfo.symbolLinkageTypes, numFunctionBase);
  }
  return cacheResults;
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
      llvm::Triple(options.targetTriple), options.targetCpu,
      options.targetFeatures,
      /*Options=*/{}, options.relocModel, /*CM=*/options.mcmodel,
      /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/isJIT));
  if (options.largeDataThreshold)
    machine->setLargeDataThreshold(options.largeDataThreshold.value());
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

//===----------------------------------------------------------------------===//
// lowerAllFuncsToLLVM
//===----------------------------------------------------------------------===//

/// If requested, attach sanitizer, etc. instrumentations to the given
/// module.
/// TODO: Eventually we should explore attaching this information at a higher
/// level of the stack.
static void attachInstrumentationAttributes(llvm::Module &module,
                                            const CompilationOptions &options) {
  if (!options.sanitizers)
    return;

  for (llvm::Function &f : module.functions()) {
    if (f.isDeclaration())
      continue;
    if (options.sanitizers.has(Sanitizers::kAddress))
      f.addFnAttr(llvm::Attribute::SanitizeAddress);
    if (options.sanitizers.has(Sanitizers::kThread))
      f.addFnAttr(llvm::Attribute::SanitizeThread);
  }
}

/// HACK HACK HACK https://github.com/modularml/modular/issues/27478
/// Using LineTables for NVPTX backend disables optimizations in cuda JIT. Use
/// DebugDirectives instead for equivalent performance to no-debug.
static void adaptDebugEmissionKind(ModuleOp module,
                                   CompilationOptions &options) {
  if (isNVPTXBackend(options) &&
      options.getDIEmissionKind() == DebugInfo::EmissionKind::LineTablesOnly) {
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
                  MLIRContext *context) {
  if (operationName)
    return {context, *operationName};
  return {context};
}

static std::unique_ptr<llvm::Module>
translateModuleToLLVMIR(llvm::LLVMContext &ctx, ModuleOp module,
                        const CompilationOptions &options) {
  // Use the input filename for the module name if possible.
  StringRef moduleName = "LLVMDialectModule";
  if (auto moduleLoc = module.getLoc()->findInstanceOf<FileLineColLoc>())
    moduleName = llvm::sys::path::filename(moduleLoc.getFilename());

  // Translate the operation into an LLVM module.
  CompilerTimeTraceScope mlirScope("mlir-to-llvmir");
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, ctx, moduleName);
  if (!llvmModule)
    return nullptr;

  // Attach any necessary instrumentation to the module.
  attachInstrumentationAttributes(*llvmModule, options);
  return llvmModule;
}

ErrorOr<std::unique_ptr<llvm::Module>>
ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module) {
  CompilerTimeTraceScope traceScope("lower-to-llvm");

  mlir::PassManager mgr = createPassManager(pmOptions.operationName, &context);

  ErrorOrSuccess configPM = pmOptions.configurePassManager(mgr);
  if (configPM)
    return configPM.takeError();

  adaptDebugEmissionKind(module, options);

  LowerToLLVMOptions llvmOptions(
      options.optimizationLevel, options.getDIEmissionKind(),
      options.debugAtLevel,
      static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage));
  llvmOptions.globalCtorFnName = ExecutionEngine::getGlobalCtorFnName();
  llvmOptions.globalDtorFnName = ExecutionEngine::getGlobalDtorFnName();

  buildLowerToLLVMPipeline(mgr, llvmOptions);

  if (failed(writeTempModule(options.saveTempsPrefix, ".pre-llvm-dialect",
                             module, ".mlir")))
    return Error(
        "writing module to file before converting to LLVM Dialect failed");

  if (failed(mgr.run(module)))
    return Error("run LowerToLLVMPipeline failed");

  if (failed(writeTempModule(options.saveTempsPrefix, ".pre-llvm-ir", module,
                             ".mlir")))
    return Error("writing module to file before converting to LLVM IR failed");

  // Translate the operation into an LLVM module.
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(ctx, module, options);
  if (!llvmModule)
    return Error("translate module to LLVMIR failed");

  return llvmModule;
}

SmallVector<AsyncRT::AnyAsyncValueRef>
ObjectCompiler::emitArchiveParallelCompilation(
    LLVMModuleAndContext llvmModule, Location opLoc,
    llvm::TargetMachine &targetMachine,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes) {
  CompilerTimeTraceScope traceScope("split-input-module");

  std::string moduleName = llvmModule->getName().str();

  bool noSplitting = runtime.getWorkQueue()->getParallelismLevel() < 2;

  // Disable parLLC for NVPTX because NVPTX codegen is inter-procedural for
  // arguments' alignment when calling a function.
  // TODO: MOCO-1407 investigate how to workaround NVPTX backend for
  // per function codegen.
  bool parLLC = runtime.getWorkQueue()->getParallelismLevel() >= 2 &&
                options.enableParallelLLC && !isGPUBackend(options);

  SmallVector<AsyncRT::AnyAsyncValueRef> cacheResults;

  if (noSplitting) {
    cacheResults.push_back(lowerLLVMModuleToObjects(
        forwardModule(std::move(llvmModule)), opLoc, targetMachine, parLLC,
        std::nullopt, /*numFunctionsBase=*/0));
  } else {
    (void)writeTempModule(options.saveTempsPrefix, ".pre-split", *llvmModule);

    auto handleSplit =
        [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
            std::optional<int64_t> idx, unsigned numFunctionsBase) {
          cacheResults.push_back(lowerLLVMModuleToObjects(
              std::move(produceModule), opLoc, targetMachine,
              !options.enableLLVMPerFunctionSplitting && parLLC, idx,
              numFunctionsBase));
        };
    if (options.enableLLVMPerFunctionSplitting)
      splitPerFunction(std::move(llvmModule), handleSplit, symbolLinkageTypes);
    else
      splitPerExported(std::move(llvmModule), handleSplit);
  }
  return cacheResults;
}

ErrorOr<WriteableBufferRef> ObjectCompiler::emitArchiveMCLinking(
    MutableArrayRef<AnyAsyncValueRef> values, StringRef moduleName,
    bool emitAssembly,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    const llvm::StringMap<unsigned> &originalFnOrdering) {
  // If any of the cache results failed, propagate the error.
  for (auto &result : values) {
    if (result.isError())
      return Error(result.takeDiagnostic().getMessage().get());
  }
  CompilerTimeTraceScope traceScope("concatenate-object-files");

  // Link MC before printing.
  auto machineOr = createTargetMachine(options, /*isJIT=*/isJIT);
  if (failed(machineOr)) {
    return Error("failed to create TargetMachine");
  }

  SmallVector<SymbolAndMCInfo *> symbolAndMCInfos;
  symbolAndMCInfos.reserve(values.size());

  for (auto [i, result] : llvm::enumerate(values)) {
    auto &symbolAndMCInfo = result.get<SymbolAndMCInfo>();
    symbolAndMCInfos.emplace_back(&symbolAndMCInfo);
  }

  MCLinker mcLinker(symbolAndMCInfos, **machineOr, options, symbolLinkageTypes,
                    originalFnOrdering);
  ErrorOr<WriteableBufferRef> mcLinkResult =
      mcLinker.linkAndPrint(moduleName, emitAssembly);
  if (mcLinkResult.isError()) {
    return Error(mcLinkResult.getError());
  }

  return *mcLinkResult;
}

ErrorOrSuccess ObjectCompiler::emitArchiveSaveTemps(ModuleOp module,
                                                    StringRef moduleName) {
  // Generate saveTempsPrefix file for the assembly result of compilation.
  // This is expensive to do because  we need to go through llvm compilation
  // from the top so that AsmPrint can codegen properly for assembly output.
  // We can't use the compilation for AsmPrint with binary here because
  // AsmPrint writes back to the MC results such as SymbolTables etc. which
  // is not reusable for a second run of AsmPrint.
  auto output = AsyncRT::AsyncValueRef<BufferRef>::allocate(runtime);
  LLVMModuleAndContext llvmModule;
  if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
        return translateModuleToLLVMIR(ctx, module, options);
      })) {
    return Error(
        Twine("failed to lower module to LLVM IR for archive compilation, ") +
        err.getError());
  }
  auto tmOr = createTargetMachine(options, isJIT);
  if (failed(tmOr))
    return tmOr.takeError();

  llvm::TargetMachine &tm = **tmOr;

  llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes;

  SmallVector<AsyncRT::AnyAsyncValueRef> cachedResults =
      emitArchiveParallelCompilation(std::move(llvmModule), module->getLoc(),
                                     tm, symbolLinkageTypes);

  andThenSyncMoving(
      cachedResults,
      [this, moduleName, module, output = output.copy(), options = options,
       symbolLinkageTypes = std::move(symbolLinkageTypes)](
          MutableArrayRef<AnyAsyncValueRef> values) mutable {
        // If any of the cache results failed, propagate the error.
        for (auto &result : values) {
          if (result.isError())
            return std::move(output).setToError(result.takeDiagnostic());
        }

        ErrorOr<WriteableBufferRef> mcLinkResult =
            emitArchiveMCLinking(values, moduleName, /*emitAssembly=*/true,
                                 symbolLinkageTypes, /*originalFnOrdering=*/{});

        if (mcLinkResult.isError()) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              Error(mcLinkResult.getError()), module->getLoc()));
        }

        WriteableBufferRef linkedObj = *mcLinkResult;
        StringRef toEmit(linkedObj->getBufferStart(),
                         linkedObj->getBufferSize());
        if (failed(writeBytesToTempWithHash(options.saveTempsPrefix, ".s",
                                            toEmit))) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "failed to save asm to saveTempsPrefix", module->getLoc()));
        }
        std::move(output).emplace(linkedObj.copy());
      });
  await(output);
  if (output.isError())
    return Error(output.takeDiagnostic().getMessage().get());

  return {};
}

// Compute the original order of Function in an llvm::Module.
// This is needed to help sort the linkedModule's functions list
// for correct codegen with NVPTX backend.
static void computeFnOrdering(llvm::Module &module,
                              llvm::StringMap<unsigned> &result) {
  unsigned idx = 0;
  for (auto &func : module.functions()) {
    if (func.isDeclaration())
      continue;
    result.insert({func.getName(), idx++});
  }
}

static std::string getAsmFilePostfix(const CompilationOptions &options) {
  if (isNVPTXBackend(options))
    return ".ptx";

  if (isAMDBackend(options))
    return ".amdgcn";

  return ".s";
}

//===----------------------------------------------------------------------===//
// emitArchive
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef> ObjectCompiler::emitArchive(OwningOpRef<ModuleOp> module,
                                               bool emitAssembly,
                                               std::string *outKeyHash) {
  CompilerTimeTraceScope traceScope("produce-archive");

  auto tmOr = createTargetMachine(options, isJIT);
  if (failed(tmOr))
    return tmOr.takeError();

  llvm::TargetMachine &tm = **tmOr;

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
    chain.andThenSync([this, op, output = output.copy(), buf = buf.copy(), &tm,
                       emitAssembly]() mutable {

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
      Location moduleLoc = op->getLoc();
      MLIRContext *moduleCtx = op->getContext();

      if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
            return lowerAllFuncsToLLVM(ctx, cast<ModuleOp>(op));
          })) {
        op->erase();
        return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
            Twine(
                "failed to lower module to LLVM IR for archive compilation, ") +
                err.getError(),
            moduleLoc));
      }

      // Split the module into multiple slices and compile each in parallel.
      [[maybe_unused]] bool isNVPTX = isNVPTXBackend(options);
      assert((!isNVPTX || (isNVPTX && emitAssembly)) &&
             "should only emit assembly with NVPTX backend");

      // Release mlir::ModuleOp before codegen happens to reduce memory
      // pressure.
      if (options.saveTempsPrefix.empty() || emitAssembly)
        op->erase();

      CompilerTimeTraceScope traceScope("split-input-module");

      std::string moduleName = llvmModule->getName().str();

      llvm::StringMap<unsigned> originalFnOrdering;

      // MCLinker changes function ordering in the linkedModule,
      // but the original order matters for NVPTX backend to generate function
      // declaration properly to avoid use before def/decl illegal instructions.
      // Keep record of the ordering here so that we can sort the linkedModule
      // to its original order.
      if (isNVPTXBackend(options))
        computeFnOrdering(*llvmModule, originalFnOrdering);

      // Split the module into multiple slices and compile each in parallel.
      llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes;
      SmallVector<AsyncRT::AnyAsyncValueRef> cachedResults =
          emitArchiveParallelCompilation(std::move(llvmModule), moduleLoc, tm,
                                         symbolLinkageTypes);

      andThenSyncMoving(
          cachedResults,
          [this, moduleName = std::move(moduleName), op, moduleLoc, moduleCtx,
           buf = buf.copy(), output = output.copy(), emitAssembly,
           options = options, symbolLinkageTypes,
           originalFnOrdering = std::move(originalFnOrdering)](
              MutableArrayRef<AnyAsyncValueRef> values) mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
            [[maybe_unused]] auto timeScope =
                loadContext(moduleCtx)
                    ->get<M::Telemetry::TelemetryContext>()
                    ->createUInt64Timer<std::chrono::milliseconds>(
                        "mojo.compile.cache.miss.time", M::Telemetry::Level::L2,
                        {{"pipeline", "ObjectCompiler::emitArchive"}});
#endif

            ErrorOr<WriteableBufferRef> mcLinkResult =
                emitArchiveMCLinking(values, moduleName, emitAssembly,
                                     symbolLinkageTypes, originalFnOrdering);
            if (mcLinkResult.isError()) {

              return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                  Error(mcLinkResult.getError()), moduleLoc));
            }

            WriteableBufferRef linkedObj = *mcLinkResult;
            if (emitAssembly) {
              std::string postfix = getAsmFilePostfix(options);
              StringRef toEmit(linkedObj->getBufferStart(),
                               linkedObj->getBufferSize());
              if (failed(writeBytesToTempWithHash(options.saveTempsPrefix,
                                                  postfix, toEmit))) {
                return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                    "failed to save asm to saveTempsPrefix", moduleLoc));
              }
              *buf << linkedObj->Buffer::getBuffer();
              std::move(output).emplace(buf.copy());
              return;
            }

            // Print assembly for saveTemps if needed.
            if (!options.saveTempsPrefix.empty()) {
              // Clear seenCodeGenFns since we are going to do the whole codegen
              // step all over again for printing saveTemps.
              seenCodeGenFns.clear();
              ErrorOrSuccess saveTempsResult =
                  emitArchiveSaveTemps(cast<ModuleOp>(op), moduleName);
              op->erase();
              if (saveTempsResult.isError()) {
                return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                    saveTempsResult.takeError(), moduleLoc));
              }
            }

            // Copy the result into the output buffer.
            *buf << linkedObj->Buffer::getBuffer();
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
    op->erase();
    return buf.copy();
  };

  WriteableBufferRef produceArchiveKey = WriteableBuffer::get();
  options.print(*produceArchiveKey << "emitArchive(");
  *produceArchiveKey << ", isJIT=" << isJIT
                     << ", enableLLVMPerFunctionSplitting="
                     << options.enableLLVMPerFunctionSplitting
                     << ", emitAssembly=" << emitAssembly
                     << ", verboseOutput=" << options.verboseOutput << ')';

  AsyncRT::AnyAsyncValueRef output = cachedTransform(
      module.release(), transformCache.copy(),
      AsyncRT::AsyncValueRef<Chain>::createReady(runtime),
      std::move(produceArchiveKey), runTransformation, onCacheHit, outKeyHash);
  await(output);

  if (output.isError())
    return {std::move(output.takeDiagnostic().getMessage())};
  return {std::move(output.get<BufferRef>())};
}

//===----------------------------------------------------------------------===//
// lowerLLVMModuleToObjects
//===----------------------------------------------------------------------===//

AsyncRT::AsyncValueRef<SymbolAndMCInfo>
ObjectCompiler::lowerLLVMModuleToObjects(
    llvm::unique_function<LLVMModuleAndContext()> produceModule, Location loc,
    llvm::TargetMachine &targetMachine, bool parLLC,
    std::optional<size_t> moduleIdx, unsigned numFunctionsBase) {

  auto result = AsyncRT::AsyncValueRef<SymbolAndMCInfo>::allocate(runtime);

  runtime.getWorkQueue()->addTask([this, result = result.copy(),
                                   produceModule = std::move(produceModule),
                                   loc, moduleIdx, parLLC, numFunctionsBase,
                                   &targetMachine]() mutable {
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

    {
      // Deduplicate functions between splits.
      // A mutex is needed here to make access to seenCodeGenFns thread-safe.
      std::lock_guard<std::mutex> lock(dedupMutex);
      for (auto &fn : module->functions()) {
        if (fn.isDeclaration())
          continue;
        if (!seenCodeGenFns.insert(fn.getName()).second)
          module.duplicatedFns.insert(fn.getName());
      }
    }

    SymbolAndMCInfo symbolAndMirInfo;
    SmallVector<AnyAsyncValueRef> buffers = compileOptimizedLLVMToObjects(
        std::move(module), loc, targetMachine, tmMutex, options, runtime,
        transformCache, parLLC, isJIT, moduleIdx, symbolAndMirInfo,
        numFunctionsBase);

    andThenAsyncMoving(
        buffers, [result = std::move(result),
                  symbolAndMirInfo = std::move(symbolAndMirInfo)](
                     MutableArrayRef<AnyAsyncValueRef> values) mutable {
          for (AnyAsyncValueRef &result : values)
            symbolAndMirInfo.mcInfos.emplace_back(
                std::make_unique<MCInfo>(std::move(result.get<MCInfo>())));
          std::move(result).emplace(std::move(symbolAndMirInfo));
        });
  });

  return result;
}

//===----------------------------------------------------------------------===//
// emitLLVMIR
//===----------------------------------------------------------------------===//

ErrorOrSuccess ObjectCompiler::emitLLVMIR(ModuleOp module,
                                          llvm::raw_pwrite_stream &os) {
  CompilerTimeTraceScope traceScope("emitLLVMIR");

  LLVMModuleAndContext llvmModule;
  if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
        return lowerAllFuncsToLLVM(ctx, module);
      }))
    return err.takeError();

  auto machineOr = createTargetMachine(options, /*isJIT=*/false);
  if (failed(machineOr))
    return machineOr.takeError();

  if (failed(runLLVMOptPasses(*llvmModule, **machineOr, options, runtime)))
    return Error("failed to run LLVM opt passes");

  llvmModule->print(os, /*AAW=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// emitAssembly
//===----------------------------------------------------------------------===//

ErrorOrSuccess ObjectCompiler::emitAssembly(OwningOpRef<ModuleOp> module,
                                            llvm::raw_pwrite_stream &os) {
  CompilerTimeTraceScope traceScope("emitAssembly");
  ErrorOr<BufferRef> buf =
      ObjectCompiler::emitArchive(std::move(module), /*emitAssembly=*/true);
  if (buf.isError())
    return Error("failed to lower LLVM IR to assembly");
  os << buf->getPointer()->getBuffer();
  return success();
}

//===----------------------------------------------------------------------===//
// emitSharedObject
//===----------------------------------------------------------------------===//

/// Utility function for creating shared object from buf
/// (mostly for AMD GPU kernels)
static ErrorOr<BufferRef> createSharedObject(BufferRef buf,
                                             CompilationOptions options,
                                             StringRef moduleName) {
  llvm::StringRef libInExt = ".o";
  llvm::StringRef libOutExt = ".so";
  std::string objName = moduleName.str() + "-%%%%%%%" + libInExt.str();

  // Write .o to a file.
  auto objFileOr = writeTempFile(objName, buf->getBuffer());

  if (objFileOr.isError())
    return Error("failed to write object binary into a file");

  std::string objFilePath = objFileOr->getPath().string();
  std::string sharedObjName =
      objFileOr->getPath().stem().string() + libOutExt.str();
  std::error_code ec;
  std::filesystem::path sharedObjPath =
      std::filesystem::temp_directory_path(ec);
  sharedObjPath = sharedObjPath / sharedObjName;

  auto triple = llvm::Triple(options.targetTriple);
  std::string version = triple.getOSVersion().getAsString();
  std::string arch = "unknown";
  if (triple.getArch() == llvm::Triple::ArchType::aarch64)
    arch = "arm64";
  else if (triple.getArch() == llvm::Triple::ArchType::x86_64)
    arch = "x86_64";

  // Call lld to generate a dynamic library.
  // For ELF:
  //  ld.lld -shared tmp.o -o tmp.so
  // For MACHO (on MacOS)
  //  ld64.lld -platform_version macos 16.0 16.0 -arch arm64
  //           -dylib tmp.o -o tmp.so -undefined dynamic_lookup
  StringRef linkerFileName = "ld.lld";
  if (llvm::Triple(options.targetTriple).getObjectFormat() ==
      llvm::Triple::MachO) {
    linkerFileName = "ld64.lld";
  }
  llvm::ErrorOr<std::string> linker =
      llvm::sys::findProgramByName(linkerFileName);
  if (!linker) {
    return Error("unable to find linker for linking");
  }

  SmallVector<StringRef> lldArgs = [&]() -> SmallVector<StringRef> {
    if (llvm::Triple(options.targetTriple).getObjectFormat() ==
        llvm::Triple::MachO) {
      return {*linker,
              "-platform_version",
              "macos",
              version.c_str(),
              version.c_str(),
              "-arch",
              arch.c_str(),
              "-undefined",
              "dynamic_lookup",
              "-dylib",
              objFilePath.c_str(),
              "-o",
              sharedObjPath.c_str()};
    }
    return {*linker, "-shared", objFilePath.c_str(), "-o",
            sharedObjPath.c_str()};
  }();

  std::string errorMsg;
  int linkExitCode = llvm::sys::ExecuteAndWait(
      lldArgs[0], lldArgs, /*Env=*/std::nullopt, /*Redirects=*/{},
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, /*ErrMsg=*/&errorMsg);

  if (linkExitCode) {
    if (!errorMsg.empty())
      errorMsg.insert(0, ": ");
    return Error(Twine("failed to generate shared object binary: ") + errorMsg);
  }

  // Read linked dynamic library in to memory.
  ErrorOr<BufferRef> sharedObjBufOr =
      M::Buffer::getFile(sharedObjPath, std::nullopt, 0);
  if (sharedObjBufOr.isError())
    return Error("failed to open shared object binary");

  // Save to temp file if needed.
  if (failed(writeBytesToTempWithHash(options.saveTempsPrefix,
                                      std::string(".") +
                                          sharedObjPath.stem().c_str() + ".so",
                                      (*sharedObjBufOr)->getBuffer())))
    return Error("failed to write shared object binary to saveTemps");

  return sharedObjBufOr;
}

ErrorOrSuccess ObjectCompiler::emitSharedObject(OwningOpRef<ModuleOp> module,
                                                llvm::raw_pwrite_stream &os) {
  llvm::Triple triple(options.targetTriple);

  // This function is added to support AMD GPU compilation to hsaco binary.
  // Generalize to all platforms+formats when needed.
  if (!llvm::is_contained({llvm::Triple::ELF, llvm::Triple::MachO},
                          triple.getObjectFormat()))
    return Error("cannot create shared object binary from target triple that "
                 "is not ELF or MachO");

  CompilerTimeTraceScope traceScope("emitSharedObj");

  StringRef moduleName = "mojo-object";
  if (auto moduleLoc = module->getLoc()->findInstanceOf<FileLineColLoc>())
    moduleName = llvm::sys::path::filename(moduleLoc.getFilename());

  // Generate .o in memory.
  ErrorOr<BufferRef> bufOr =
      ObjectCompiler::emitArchive(std::move(module), /*emitAssembly=*/false);

  if (bufOr.isError())
    return Error("failed to lower LLVM IR to object binary");

  // Create shared object in buffer.
  ErrorOr<BufferRef> sharedObjBufOr =
      createSharedObject(*bufOr, options, moduleName);

  if (sharedObjBufOr.isError())
    return sharedObjBufOr.takeError();

  // Send dynamic library to output stream.
  os << sharedObjBufOr->getPointer()->getBuffer();

  return success();
}

static ErrorOr<std::pair<uint64_t, llvm::Function *>>
getKernelIDFromLLVMModule(llvm::Module &module) {
  for (llvm::Function &func : module) {
    if (func.isDeclaration())
      continue;

    const llvm::AttributeList &funcAttrs = func.getAttributes();

    for (auto &attr :
         funcAttrs.getAttributes(llvm::AttributeList::FunctionIndex)) {
      if (!attr.isStringAttribute())
        continue;
      if (attr.getKindAsString() == "kgen.offload.kernelid") {
        uint64_t kernelId;
        if (llvm::to_integer(attr.getValueAsString(), kernelId)) {
          // Remove the ID attribute so that caching won't take this into
          // consideration.
          llvm::AttributeList newList = funcAttrs.removeAttribute(
              func.getContext(), llvm::AttributeList::FunctionIndex,
              attr.getKindAsString());

          func.setAttributes(newList);
          return std::make_pair(kernelId, &func);
        }
      }
    }
  }

  return Error("Can't find kgen.offload.kernelid from the llvm split.");
}

static void attachGPUCodeGenAttributes(llvm::Function *kernelEntry) {
  // Recursion is not supported for kernel entry functions.
  // This is that same as what clang does:
  // https://github.com/llvm/llvm-project/blob/c1ec5beb4ab36c2c4d99ed6d735d217e74364771/clang/lib/CodeGen/CodeGenFunction.cpp#L1086
  kernelEntry->addFnAttr(llvm::Attribute::NoRecurse);
}

namespace {
/// Override ROCDL module serializer to access pre-lowering bitcode linking
/// logic.
class AMDGPUModuleLinker : public mlir::ROCDL::SerializeGPUModuleBase {
public:
  /// The module passed in does not matter as we do not use this serializer to
  /// perform MLIR to LLVM translation.
  AMDGPUModuleLinker(Operation &module, mlir::ROCDL::ROCDLTargetAttr target,
                     const mlir::gpu::TargetOptions &targetOptions)
      : SerializeGPUModuleBase(module, target, targetOptions) {}

  /// Link the set of `amdLibs` into the LLVM module, along with any other libs
  /// explicitly specified in `targetOptions` when creating this object.
  LogicalResult link(llvm::Module &llvmModule,
                     mlir::ROCDL::AMDGCNLibraries amdLibs) {
    // This is required to set control variables during prelink.
    deviceLibs = amdLibs;
    // This is required to actually find the .bc files.
    if (failed(appendStandardLibs(amdLibs)))
      return failure();

    handleModulePreLink(llvmModule);
    auto libs = loadBitcodeFiles(llvmModule);
    if (!libs)
      return failure();
    if (!libs->empty())
      if (failed(linkFiles(llvmModule, std::move(*libs))))
        return failure();
    handleModulePostLink(llvmModule);
    return success();
  }
};
} // namespace

/// TODO(billyz): Move this upstream into header for `AMDGCNLibraries`.
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Link vendor-provided LLVM bitcode libraries into the LLVM module when
/// necessary.
static LogicalResult linkBitcodeLibraries(Location loc,
                                          llvm::Module &llvmModule,
                                          const CompilationOptions &options) {
  // AMD GPU only needs additional linking if address sanitizer is needed.
  if (isAMDBackend(options) && options.sanitizers.has(Sanitizers::kAddress)) {
    mlir::MLIRContext *ctx = loc.getContext();
    mlir::OpBuilder b(ctx);
    OwningOpRef<ModuleOp> mlirModule(b.create<ModuleOp>(loc, "dummy"));
    mlir::ROCDL::ROCDLTargetAttr target = mlir::ROCDL::ROCDLTargetAttr::get(
        ctx, options.optimizationLevel, options.targetTriple, options.targetCpu,
        options.targetFeatures);
    // TODO(billyz): Add "asanrtl.bc" to the upstream standard set of libraries.
    mlir::gpu::TargetOptions targetOptions(
        /*toolkitPath=*/"/opt/rocm",
        /*librariesToLink=*/{
            b.getStringAttr("/opt/rocm/amdgcn/bitcode/asanrtl.bc")});
    AMDGPUModuleLinker moduleLinker(**mlirModule, target, targetOptions);
    // Both ocml & ockl libs are requried for asan.
    mlir::ROCDL::AMDGCNLibraries libs =
        mlir::ROCDL::AMDGCNLibraries::Ockl | mlir::ROCDL::AMDGCNLibraries::Ocml;
    if (failed(moduleLinker.link(llvmModule, libs)))
      return mlirModule->emitError(
          "failed to link bitcode libraries for asan on AMD GPUs");
    return success();
  }
  return success();
}

static std::string getNVGPUName(StringRef targetAccelerator) {
  std::pair<StringRef, StringRef> s = targetAccelerator.rsplit(":");
  if (s.second.starts_with("sm_"))
    return s.second.str();
  StringRef versionString =
      s.second.starts_with("sm") ? s.second.substr(2) : s.second;
  int version;
  if (llvm::to_integer(versionString, version)) {
    std::string result = "sm_" + std::to_string(version);
    if (version >= 90)
      return result + "a";
    return result;
  }
  // Set default to sm_52 as what NVPTX lib does.
  // https://docs.nvidia.com/cuda/ptx-compiler-api/index.html
  return "sm_52";
}

static ErrorOr<BufferRef> compilePTXToCUBIN(AsyncRT::DeviceContextRef &ctx,
                                            llvm::Module &inputModule,
                                            StringRef ptx,
                                            CompilationOptions options) {
  // If the environment variable MODULAR_USE_DRIVER_CUBIN_COMPILER is set, we
  // use the driver cubin compiler. This allows for temporary experimentation.
  // If this pathway is always benificial, then we will stop checking the env
  // var.
  if (llvm::sys::Process::GetEnv("MODULAR_USE_DRIVER_CUBIN_COMPILER")) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Falling back to using the driver to compile PTX to CUBIN.\n");
    KGEN_DEBUG(0, {
      llvm::dbgs()
          << "Falling back to using the driver to compile PTX to CUBIN.\n";
    });
    return ctx->compileFunction_v2(ptx, options.getDebugLevelString(),
                                   options.optimizationLevel);
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Using the NVPTXCompiler API to compile PTX to CUBIN.\n");
  KGEN_DEBUG(0, {
    llvm::dbgs() << "Using the NVPTXCompiler API to compile PTX to CUBIN.\n";
  });
  // FIXME: Will clean this _v2 up once we decide which compiler to use to get
  // to cubin.
  return ctx->compileFunction(ptx, options.getDebugLevelString(),
                              options.optimizationLevel,
                              getNVGPUName(options.targetAccelerator));
}

static AnyAsyncValueRef
lowerLLVMModuleToObject(llvm::Module &inputModule, Location loc,
                        RCRef<Cache::TransformCache> transformCache,
                        size_t moduleIdx, AsyncRT::Runtime &runtime,
                        CompilationOptions options, bool isJIT,
                        bool shouldDeserialize, EmitAs emissionKind) {
  WriteableBufferRef keyBuf = WriteableBuffer::get();
  options.print(*keyBuf << "compileLLVMModuleToObject(");
  *keyBuf << ")";
  *keyBuf << " emitAs = " << emissionKind;
  *keyBuf << " isJIT = " << isJIT;
  if (!options.emissionOptions.empty())
    *keyBuf << " emissionOptions = " << options.emissionOptions;

  size_t nonBitcodeKeySize = keyBuf->getBufferSize();

  llvm::WriteBitcodeToFile(inputModule, *keyBuf);

  auto runTransformation = [loc, moduleIdx, isJIT, options, &runtime,
                            emissionKind, keyBuf = keyBuf.copy(), &inputModule,
                            nonBitcodeKeySize, shouldDeserialize](
                               WriteableBufferRef buf,
                               AsyncRT::AnyAsyncValueRef chain) mutable {
    auto output = AsyncRT::AsyncValueRef<BufferRef>::allocate(runtime);
#ifdef MODULAR_ENABLE_TELEMETRY
    Cache::CacheTelemetryContext::getCacheTelemetryContext(runtime.context)
        .recordCacheMiss("lowerLLVMModuleToObjectGPU");
#endif

    chain.andThenAsync([loc, &runtime, emissionKind, output = output.copy(),
                        buf = buf.copy(), keyBuf = std::move(keyBuf), options,
                        isJIT, moduleIdx, &inputModule, nonBitcodeKeySize,
                        shouldDeserialize]() mutable {
      CompilerTimeTraceScope traceScope("lowerLLVMModuleToObjectGPU");

      LLVMModuleAndContext deserializedModule;
      // We need to deserialize the llvm::Module into a separate copy here
      // when the same module split needs different emission kinds to avoid
      // data race on running optimization son the same module for different
      // parallel emission kind tasks. If there is only one emission kind,
      // we don't nee to do the extra deserialize since there will not be
      // any data race.
      if (shouldDeserialize) {
        if (auto err = deserializedModule.create(
                [&](llvm::LLVMContext &ctx)
                    -> ErrorOr<std::unique_ptr<llvm::Module>> {
                  BufferRef keyBufRef(std::move(keyBuf));
                  StringRef bitcodeBuffer = keyBufRef->getBuffer();
                  bitcodeBuffer = bitcodeBuffer.drop_front(nonBitcodeKeySize);

                  // Load the cached bytecode into a new context. This is
                  // necessary to avoid data races during multi-threading.
                  llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
                      llvm::parseBitcodeFile(
                          llvm::MemoryBufferRef(bitcodeBuffer,
                                                inputModule.getName()),
                          ctx);
                  if (!moduleOr) {
                    return Error("failed to load LLVM IR bitcode");
                  }
                  return std::move(*moduleOr);
                })) {
          return std::move(output).setToError(
              AsyncRT::getMLIRDiagnostic(err.takeError(), loc));
        }
      }

      llvm::Module &module =
          shouldDeserialize ? *deserializedModule : inputModule;

      if (emissionKind == EmitAs::LLVM) {
        *buf << module;
        std::move(output).emplace(buf.copy());
        return;
      }

      // Create the target machine.
      auto tmOr = createTargetMachine(options, isJIT);
      if (failed(tmOr)) {
        return std::move(output).setToError(
            AsyncRT::getMLIRDiagnostic(tmOr.takeError(), loc));
      }
      llvm::TargetMachine &tm = **tmOr;

      // Optimize the llvm Module.
      if (failed(optimizeLLVMModule(module, tm, options, runtime, moduleIdx))) {
        return std::move(output).setToError(
            AsyncRT::getMLIRDiagnostic("failed to optimize LLVM IR.", loc));
      }

      if (emissionKind == EmitAs::LLVM_OPT) {
        *buf << module;
        std::move(output).emplace(buf.copy());
        return;
      }

      std::unique_ptr<llvm::MachineModuleInfo> machineModuleInfo;
      std::unique_ptr<llvm::MCContext> mcContext;

      if (emissionKind == EmitAs::ASM) {
        if (failed(runLlcPasses(
                module, options, tm, *buf, machineModuleInfo, mcContext,
                llvm::CodeGenFileType::AssemblyFile,
                /*stopBeforeAsmPrint=*/false, 0, nullptr,
                runtime.context->get<M::Telemetry::TelemetryContext>()))) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "llc failed to codegen LLVM IR to object code", loc));
        }

        std::string postfix = getAsmFilePostfix(options);
        StringRef toEmit(buf->getBufferStart(), buf->getBufferSize());
        if (failed(writeBytesToTempWithHash(options.saveTempsPrefix, postfix,
                                            toEmit))) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "failed to save asm to saveTempsPrefix", loc));
        }

        std::move(output).emplace(buf.copy());
      } else {
        auto codeBuf = WriteableBuffer::get();
        if (isNVPTXBackend(options)) {
          // Compile to PTX first.
          if (failed(runLlcPasses(
                  module, options, tm, *codeBuf, machineModuleInfo, mcContext,
                  llvm::CodeGenFileType::AssemblyFile,
                  /*stopBeforeAsmPrint=*/false, 0, nullptr,
                  runtime.context->get<M::Telemetry::TelemetryContext>()))) {
            return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                "llc failed to codegen LLVM IR to object code", loc));
          }

          StringRef ptx(codeBuf->getBufferStart(), codeBuf->getBufferSize());

          ErrorOr<AsyncRT::DeviceContextRef> errCtx =
              AsyncRT::DeviceContext::create(AsyncRT::Device::cudaAPI);
          if (errCtx.isError()) {
            return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                "failed to create cuda device context to compile to cubin",
                loc));
          }

          ErrorOr<BufferRef> cubinOr =
              compilePTXToCUBIN(*errCtx, inputModule, ptx, options);
          if (cubinOr.isError()) {
            return std::move(output).setToError(
                AsyncRT::getMLIRDiagnostic(cubinOr.takeError(), loc));
          }
          (*buf) << (*cubinOr)->getBuffer();
        } else {
          // This is mostly for AMD GPU codegen, but works for CPU as well
          // (mostly for testing).
          if (failed(runLlcPasses(
                  module, options, tm, *codeBuf, machineModuleInfo, mcContext,
                  llvm::CodeGenFileType::ObjectFile,
                  /*stopBeforeAsmPrint=*/false, 0, nullptr,
                  runtime.context->get<M::Telemetry::TelemetryContext>()))) {
            return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                "llc failed to codegen LLVM IR to object code", loc));
          }
          StringRef name = "mojo-object";
          if (auto moduleLoc = loc->findInstanceOf<FileLineColLoc>())
            name = llvm::sys::path::filename(moduleLoc.getFilename());
          std::string moduleName = (name + Twine(moduleIdx)).str();

          // Emitting as a shared object
          ErrorOr<BufferRef> bufOr = createSharedObject(
              BufferRef::create(codeBuf->Buffer::getBuffer()), options,
              moduleName);
          if (bufOr.isError()) {
            return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                "failed to create shared object file", loc));
          }
          (*buf) << (*bufOr)->getBuffer();
        }

        std::move(output).emplace(buf.copy());
      }
    });
    return output;
  };

  auto onCacheHit = [&](BufferRef buf) {
#ifdef MODULAR_ENABLE_TELEMETRY
    Cache::CacheTelemetryContext::getCacheTelemetryContext(runtime.context)
        .recordCacheHit("lowerLLVMModuleToObjectGPU");
#endif
    return buf.copy();
  };

  return Cache::cachedTransform(
      AsyncRT::MLIRLocationDecoder::getEncodedLocation(loc),
      transformCache.copy(),
      AsyncRT::AsyncValueRef<Chain>::createReady(runtime), keyBuf.copy(),
      std::move(runTransformation), onCacheHit);
}

static std::pair<AnyAsyncValueRef, AnyAsyncValueRef> lowerLLVMModuleToObject(
    llvm::unique_function<LLVMModuleAndContext()> produceModule, Location loc,
    RCRef<Cache::TransformCache> transformCache,
    std::optional<size_t> moduleIdx, AsyncRT::Runtime &runtime,
    CompilationOptions options, bool isJIT,
    DenseMap<uint64_t, llvm::SmallSet<EmitAs, 4>> &kernelEmissionKinds) {
  auto resultBufs =
      AsyncRT::AsyncValueRef<DenseMap<EmitAs, BufferRef>>::allocate(runtime);
  auto resultKernelId = AsyncRT::AsyncValueRef<uint64_t>::allocate(runtime);

  runtime.getWorkQueue()->addTask([resultBufs = resultBufs.copy(),
                                   resultKernelId = resultKernelId.copy(),
                                   produceModule = std::move(produceModule),
                                   loc, isJIT, options, &runtime,
                                   transformCache = transformCache.copy(),
                                   &kernelEmissionKinds]() mutable {
    CompilerTimeTraceScope traceScope("lowerLLVMModuleToObjectGPU");

    // Materialize the module.
    LLVMModuleAndContext module = produceModule();

    ErrorOr<std::pair<uint64_t, llvm::Function *>> kernelIdFuncOr =
        getKernelIDFromLLVMModule(*module);
    if (kernelIdFuncOr) {
      std::move(resultBufs)
          .setToError(AsyncRT::getMLIRDiagnostic("Can't find kernelId", loc));
      std::move(resultKernelId)
          .setToError(
              AsyncRT::getMLIRDiagnostic(kernelIdFuncOr.takeError(), loc));
      return;
    }

    uint64_t kernelId = (*kernelIdFuncOr).first;
    llvm::Function *kernelEntry = (*kernelIdFuncOr).second;
    attachGPUCodeGenAttributes(kernelEntry);
    if (failed(linkBitcodeLibraries(loc, *module, options)))
      return;

    SmallVector<EmitAs> emissionKinds;
    SmallVector<AsyncRT::AnyAsyncValueRef> emissionResults;
    llvm::SmallSet<EmitAs, 4> &kinds = kernelEmissionKinds[kernelId];
    bool shouldDeserialize = kinds.size() > 1;
    bool shouldRunExtraAsm = !options.saveTempsPrefix.empty() &&
                             kinds.contains(EmitAs::OBJECT) &&
                             !kinds.contains(EmitAs::ASM);
    shouldDeserialize |= shouldRunExtraAsm;

    for (EmitAs kind : kinds) {
      emissionKinds.push_back(kind);
      emissionResults.push_back(lowerLLVMModuleToObject(
          *module, loc, transformCache, kernelId, runtime, options, isJIT,
          shouldDeserialize, kind));
    }

    if (shouldRunExtraAsm) {
      // We need to run the llvm lowering again to saveTempsPrefix for
      // assembly if we are generating object (for GPUs). Since codegen has
      // side effect, we cannot reuse the same llvm module for assembly and
      // object file, we have to run the llvm lowering separately for each
      // codegen result.
      emissionResults.push_back(lowerLLVMModuleToObject(
          *module, loc, transformCache, kernelId, runtime, options, isJIT,
          shouldDeserialize, EmitAs::ASM));
    }

    auto kernelBufs =
        AsyncRT::AsyncValueRef<DenseMap<EmitAs, BufferRef>>::allocate(runtime);

    andThenSyncMoving(
        emissionResults,
        [emissionKinds, shouldRunExtraAsm, resultBufs = kernelBufs.copy()](
            MutableArrayRef<AnyAsyncValueRef> values) mutable {
          DenseMap<EmitAs, BufferRef> kernelResults;

          if (shouldRunExtraAsm) {
            // No need to process the last result as it's just for printing.
            values = values.drop_back(1);
          }

          for (auto [idx, result] : llvm::enumerate(values)) {
            if (result.isError())
              return std::move(resultBufs).setToError(result.takeDiagnostic());
            kernelResults.insert({emissionKinds[idx], result.get<BufferRef>()});
          }
          std::move(resultBufs).emplace(kernelResults);
        });
    await(kernelBufs);

    if (kernelBufs.isError())
      std::move(resultBufs).setToError(kernelBufs.takeDiagnostic());
    else
      std::move(resultBufs).emplace(kernelBufs.get());

    std::move(resultKernelId).emplace(kernelId);
  });

  return std::make_pair(std::move(resultBufs), std::move(resultKernelId));
}

// Emit GPU kernels.
// The input module is a bundle of multiple GPU kernels.
// Th output is a vector of compiled GPU kernels with their corresponding
// kernel ids.
// This function does the following steps:
// - Split the input module into submodules for each kernel.
//   We don't do per function splitting for GPU kernels since
//   the backends are inter-procedural.
// - Extract kernel ID for each split.
// - Run LLVM pipleine (opt + asmprint) to generate code for each kernel:
//   PTX for Nvidia, an so lib for AMD.
ErrorOr<DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>>>
ObjectCompiler::emitGPUKernels(
    OwningOpRef<ModuleOp> module,
    llvm::DenseMap<uint64_t, llvm::SmallSet<EmitAs, 4>> kernelEmissionKinds) {
  CompilerTimeTraceScope traceScope("emitGPUKernels");

  // Perform a cache aware transformation to translate the module to an
  // archive file.

#ifdef MODULAR_ENABLE_TELEMETRY
  [[maybe_unused]] auto timeScope =
      loadContext(module->getContext())
          ->get<M::Telemetry::TelemetryContext>()
          ->createUInt64Timer<std::chrono::milliseconds>(
              "mojo.compile.cache.miss.time", M::Telemetry::Level::L2,
              {{"pipeline", "ObjectCompiler::emitArchive"}});
#endif

  // Lower the module to LLVM.
  LLVMModuleAndContext llvmModule;
  Location moduleLoc = module->getLoc();

  // Save elaborated MLIR module to saveTempsPrefix.
  if (!options.saveTempsPrefix.empty()) {
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << *module;
    if (failed(writeBytesToTempWithHash(options.saveTempsPrefix, ".mlir", str)))
      return Error("failed to save mlir to saveTempPrefix");
  }

  if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
        return lowerAllFuncsToLLVM(ctx, *module);
      })) {
    module->erase();
    return Error(
        Twine("failed to lower module to LLVM IR for archive compilation, ") +
        err.getError());
  }

  std::string moduleName = llvmModule->getName().str();

  (void)writeTempModule(options.saveTempsPrefix, ".pre-split", *llvmModule);

  SmallVector<AsyncRT::AnyAsyncValueRef> cachedResults;
  auto handleSplit =
      [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
          std::optional<int64_t> idx, unsigned numFunctionsBase) {
        auto result = lowerLLVMModuleToObject(
            std::move(produceModule), moduleLoc, transformCache, idx, runtime,
            options, isJIT, kernelEmissionKinds);
        cachedResults.push_back(std::move(result.first));
        cachedResults.push_back(std::move(result.second));
      };

  splitPerExported(std::move(llvmModule), handleSplit);

  auto result = AsyncRT::AsyncValueRef<
      DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>>>::allocate(runtime);

  andThenSyncMoving(
      cachedResults, [result = result.copy(), options = options](
                         MutableArrayRef<AnyAsyncValueRef> values) mutable {
        DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>> results;

        for (size_t i = 0; i < values.size(); i += 2) {
          AnyAsyncValueRef &bufs = values[i];
          AnyAsyncValueRef &kernelId = values[i + 1];

          if (kernelId.isError()) {
            if (isGPUBackend(options))
              return std::move(result).setToError(kernelId.takeDiagnostic());
            else
              continue;
          }

          if (bufs.isError())
            return std::move(result).setToError(bufs.takeDiagnostic());

          results.insert({kernelId.get<uint64_t>(),
                          std::move(bufs.get<DenseMap<EmitAs, BufferRef>>())});
        }
        std::move(result).emplace(std::move(results));
      });

  await(result);

  if (result.isError())
    return {std::move(result.takeDiagnostic().getMessage())};

  return std::move(result.get());
}
