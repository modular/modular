//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "kgen-compiler"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// evaluateSpecializations
//===----------------------------------------------------------------------===//

/// A default specialization evaluator that JITs and invokes the specialized
/// functions with the provided evaluator.
static ErrorOr<ElaboratorSearchFn>
evaluateSpecializations(FuncOp evaluator, const SymbolTable &symtab,
                        LLCL::Runtime &runtime, TargetInfoAttr target,
                        const CompilationOptions &options,
                        ArrayRef<FuncOp> specializations) {
  // TODO(#2717): Cross-compilation and execution for search!
  if (target.getArch() != llvm::sys::getHostCPUName())
    return Error("cross-compilation execution in search is not yet supported");

  mlir::PassManager mgr(target.getContext());
  ExecutionEngineOptions eeOptions;
  eeOptions.sanitizers = options.sanitizers;
  if (options.debugLevel != CompilationOptions::kNoDebug)
    eeOptions.registerDebugPlugins = true;
  auto engineOr =
      initializeExecutionEngine(runtime, mgr, options, std::move(eeOptions),
                                /*isJIT=*/true, target, /*isSearch=*/true);
  if (engineOr.isError())
    return engineOr.takeError();
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  // We only want the funcs passed-in and the evaluator to be code-generated.
  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);

  // Create the set of symbols to export.
  ExportMap exportedSymbols;
  for (FuncOp func : funcsToCompile) {
    StringAttr symName = func.getSymNameAttr();
    exportedSymbols.insert({symName, ExportedSymbol(ExportKind::Weak)});
  }

  // Add the exported symbols to the ObjectCompilerLayer. This will not actually
  // compile anything - that happens at lookup time.
  if (auto err = engine->add<ObjectCompilerLayer>("evaluateSpecializations",
                                                  symtab, exportedSymbols))
    return err.takeError();

  SmallVector<void *> candidatePtrs;
  {
    TimeTraceScope<> traceScope("compile-specializations");
    // Get pointers to all the candidates.
    for (FuncOp candidate : specializations) {
      auto funcOr = engine->lookup(candidate.getNameAttr());
      if (funcOr.isError())
        return funcOr.takeError();
      candidatePtrs.push_back(funcOr->getFunctionPointer());
    }
  }

  // Lookup the evaluator function
  auto evaluatorFuncOr = engine->lookup(evaluator.getNameAttr());
  if (evaluatorFuncOr.isError())
    return evaluatorFuncOr.takeError();
  auto evaluatorFunc = std::move(*evaluatorFuncOr);

  return
      [engine = std::move(engine), evaluatorFunc = std::move(evaluatorFunc),
       candidatePtrs = std::move(candidatePtrs)]() mutable -> ErrorOr<ssize_t> {
        TimeTraceScope<> traceScope("execute-specializations");
        return evaluatorFunc.invoke<ssize_t, void **, ssize_t>(
            candidatePtrs.data(), candidatePtrs.size());
      };
}

//===----------------------------------------------------------------------===//
// compileElaboratorAsm
//===----------------------------------------------------------------------===//

/// Given the pre-elaboration function `func` belonging to a module with the
/// symbol table `symtab`, slice out a standalone module rooted at `func` and
/// elaborate it and compile to assembly for the provided `target.
static ErrorOrSuccess
compileElaboratorAsm(GeneratorOp func, const SymbolTable &symtab,
                     LLCL::Runtime &runtime, TargetInfoAttr target,
                     CompilationOptions options, llvm::raw_pwrite_stream &os) {
  // Configure the compilation options given the new target.
  options.targetTriple = target.getTripleStr();
  options.targetCpu = target.getArch();
  options.targetFeatures = target.getFeatures();
  options.relocModel = target.getRelocationModel();

  // Initialize the object compiler.
  mlir::PassManager compilerPm(target.getContext());
  ErrorOr<ObjectCompiler> compilerOr = ObjectCompiler::create(
      runtime, compilerPm, ".mojo_cache", options, /*isJIT=*/false);
  if (compilerOr.isError())
    return compilerOr.takeError();
  ObjectCompiler compiler = compilerOr.takeValue();

  // Initialize the target machine.
  auto tmOr = createTargetMachine(options, /*isJIT=*/false);
  if (tmOr.isError())
    return tmOr.takeError();
  std::unique_ptr<llvm::TargetMachine> targetMachine = tmOr.takeValue();

  // Slice out a pre-elaboration module for the new target to compile for.
  ExportMap exportedSymbols;
  exportedSymbols.insert({func.getSymNameAttr(), ExportKind::Exported});
  OwningOpRef<ModuleOp> module =
      compiler.produceStandaloneModule(symtab, exportedSymbols);
  // Override the target.
  eraseTargetInfo(*module);
  setTargetInfo(*module, target);

  // Run elaboration through to the end of the optimization pipeline.
  ElaborateGeneratorsOptions elaboratorOptions;
  elaboratorOptions.enableSearch = options.enableSearch;
  elaboratorOptions.elaborateLocations =
      options.debugLevel == CompilationOptions::kLineTablesOnly ||
      options.debugLevel == CompilationOptions::kFullDebugInfo;
  mlir::PassManager pm(target.getContext());
  pm.addPass(createElaborateGenerators(
      runtime, target, BuildInfoAttr::getForCurrentBuild(target.getContext()),
      elaboratorOptions,
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      },
      [=, &runtime](GeneratorOp func, const SymbolTable &symtab,
                    TargetInfoAttr target, llvm::raw_pwrite_stream &os) {
        // Recursion...!
        return compileElaboratorAsm(func, symtab, runtime, target, options, os);
      }));
  buildPostElaborationPipeline(pm, runtime, options);

  // TODO: cachedTransform
  if (failed(pm.run(*module)))
    return Error("failed to run the pass manager");
  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule =
      compiler.lowerAllFuncsToLLVM(llvmCtx, *module);

  if (failed(compileLLVMToObject(*llvmModule, *targetMachine, os, options,
                                 runtime, /*emitAssembly=*/true)))
    return Error("failed to emit assembly");

  return success();
}

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

void KGEN::populateElaborateModulePasses(mlir::PassManager &pm,
                                         LLCL::Runtime &runtime,
                                         TargetInfoAttr target,
                                         BuildInfoAttr build,
                                         const CompilationOptions &options) {
  buildElaborateModulePipeline(
      pm, runtime, target, build,
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      },
      [=, &runtime](GeneratorOp func, const SymbolTable &symtab,
                    TargetInfoAttr target, llvm::raw_pwrite_stream &os) {
        return compileElaboratorAsm(func, symtab, runtime, target, options, os);
      },
      options);
  buildPostElaborationPipeline(pm, runtime, options);
}

//===----------------------------------------------------------------------===//
// Caching
//===----------------------------------------------------------------------===//

ErrorOr<
    std::pair<RCRef<Cache::BlobCacheBackend>, RCRef<Cache::BlobCacheBackend>>>
KGEN::getMojoCacheBackends(LLCL::Runtime &runtime) {
  auto transformCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".mojo_cache") / "transform").string(),
      KGEN_VERSION_STRING);
  if (transformCacheBackend.isError())
    return transformCacheBackend.takeError();

  auto regionCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".mojo_cache") / "region").string(),
      KGEN_VERSION_STRING);
  if (regionCacheBackend.isError())
    return regionCacheBackend.takeError();

  return std::make_pair(transformCacheBackend.takeValue(),
                        regionCacheBackend.takeValue());
}

//===----------------------------------------------------------------------===//
// KGENCompilerMaterializationUnit
//===----------------------------------------------------------------------===//

/// Produce an ExportMap with every symbol in the module.
static ExportMap getAllSymbols(ModuleOp theModule) {
  ExportMap exports;
  for (auto sym : theModule.getOps<mlir::SymbolOpInterface>())
    exports.insert({sym.getNameAttr(), {ExportKind::Exported}});
  return exports;
}

class KGENCompilerLayer::KGENCompilerMaterializationUnit
    : public llvm::orc::MaterializationUnit {
public:
  KGENCompilerMaterializationUnit(KGENCompilerLayer &layer, SymbolTable s,
                                  ExportMap e)
      : MaterializationUnit(layer.getInterface(e)), genLayer(layer),
        symtab(std::move(s)), exports(std::move(e)) {}

  /// Provide a name for this MU that will show up in ORC debug logs.
  StringRef getName() const override {
    return "KGEN::KGENCompilerMaterializationUnit";
  }

  /// Given a MaterializationResponsibility, materialize the code for those
  /// symbols and forward them to the next layer.
  void materialize(
      std::unique_ptr<llvm::orc::MaterializationResponsibility> mr) override {
    genLayer.emit(std::move(mr), symtab, exports);
  }

  /// Notify that the symbol `name` has been overridden and this MU should
  /// remove it from the source. This removes the symbol from the module.
  void discard(const llvm::orc::JITDylib &jd,
               const llvm::orc::SymbolStringPtr &name) override {
    // If the operation exists, erase it. Otherwise, do nothing.
    if (auto sym = symtab.lookup<mlir::SymbolOpInterface>(*name))
      symtab.erase(sym);
  }

private:
  KGENCompilerLayer &genLayer;
  SymbolTable symtab;
  ExportMap exports;
};

//===----------------------------------------------------------------------===//
// KGENCompilerLayer
//===----------------------------------------------------------------------===//

char KGENCompilerLayer::ID;

KGENCompilerLayer::KGENCompilerLayer(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    BuildInfoAttr build, const CompilationOptions &options,
    ObjectCompilerLayer &base,
    RCRef<Cache::BlobCacheBackend> transformCacheBackend,
    RCRef<Cache::BlobCacheBackend> regionCacheBackend,
    llvm::orc::ExecutionSession &sess, const llvm::DataLayout &dl,
    MaterializationLayer::AddToSearchOrderFn add)
    : llvm::RTTIExtends<KGENCompilerLayer, MaterializationLayer>(
          sess, dl, std::move(add)),
      pm(pm), runtime(runtime), target(target), build(build), options(options),
      baseLayer(base) {
  // Construct the caches.
  transformCache =
      RCRef<Cache::TransformCache>::create(std::move(transformCacheBackend));
  regionCache =
      RCRef<Cache::RegionCache>::create(std::move(regionCacheBackend));
}

ErrorOrSuccess KGENCompilerLayer::add(StringRef libName, ModuleOp theModule) {
  TimeTraceScope<> traceScope("KGENCompilerLayer::add(" + libName.str() + ")");
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();

  llvm::orc::JITDylib *dylib = *dylibOr;
  llvm::orc::ResourceTrackerSP resourceTracker =
      dylib->getDefaultResourceTracker();

  // Set the target and build info now, so it's included in the cache key.
  setTargetInfo(theModule, target);
  setBuildInfo(theModule, build);
  // Populate the passes.
  buildGenerateLibraryPipeline(pm, runtime, options);
  populateElaborateModulePasses(pm, runtime, target, build, options);

  // Run the passes as a cached transform. Don't deflate the op as part of this
  // - we don't want that cost right now.
  {
    [[maybe_unused]] auto timeScope =
        runtime.emplaceContextIfMissing<M::Telemetry::TelemetryContext>()
            .createUInt64Timer<std::chrono::milliseconds>(
                "mojo.kgen.compile.time", M::Telemetry::Level::L2);

    LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
        theModule, regionCache.copy(), transformCache.copy(),
        runtime.getReadyChain().copy(), pm, /*deflateTarget=*/false);
    LLCL::await(ready);
    if (ready.isError())
      return ready.takeDiagnostic().getMessage().copy();
  }

  // Add the materialization unit by computing the exports and the symbol
  // table, and passing those off.
  SymbolTable st(theModule);
  ExportMap ex = getExportedSymbols(theModule);
  if (ex.empty())
    ex = getAllSymbols(theModule);

  return toModularErrorOr(
      dylib->define(std::make_unique<KGENCompilerMaterializationUnit>(
                        *this, std::move(st), std::move(ex)),
                    resourceTracker));
}

void KGENCompilerLayer::emit(
    std::unique_ptr<llvm::orc::MaterializationResponsibility> mr,
    SymbolTable &symtab, const ExportMap &exports) {
  // Delegate all requested symbols to the base layer.
  baseLayer.emit(std::move(mr), symtab, exports);
}

llvm::orc::MaterializationUnit::Interface
KGENCompilerLayer::getInterface(const ExportMap &exports) {
  llvm::orc::MangleAndInterner mangler(session, dataLayout);
  llvm::orc::SymbolFlagsMap symbols;

  for (auto &[name, symbol] : exports) {
    symbols[mangler(name)] =
        llvm::JITSymbolFlags::Callable | llvm::JITSymbolFlags::Exported |
        (symbol.kind == ExportKind::Weak ? llvm::JITSymbolFlags::Weak
                                         : llvm::JITSymbolFlags::None);
  }

  return {std::move(symbols), /*InitSymbol=*/nullptr};
}

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
KGEN::createElaborateGeneratorsWithDefaultJIT(LLCL::Runtime &runtime) {
  return createElaborateGenerators(
      runtime, /*target=*/{}, /*build=*/{}, /*options=*/{},
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       /*options=*/{}, specializations);
      },
      [=, &runtime](GeneratorOp func, const SymbolTable &symtab,
                    TargetInfoAttr target, llvm::raw_pwrite_stream &os) {
        return compileElaboratorAsm(func, symtab, runtime, target,
                                    /*options=*/{}, os);
      });
}

ErrorOr<std::unique_ptr<ExecutionEngine>>
KGEN::initializeExecutionEngine(LLCL::Runtime &runtime, mlir::PassManager &pm,
                                const CompilationOptions &compilationOptions,
                                ExecutionEngineOptions executionEngineOptions,
                                bool isJIT, TargetInfoAttr target,
                                bool isSearch) {
  MLIRContext *ctx = pm.getContext();

  // Now create the execution engine so we can JIT.
  auto tmOr = createTargetMachine(compilationOptions, isJIT);
  if (tmOr.isError())
    return tmOr.takeError();

  // Forward the sanitizers into the execution engine if we are JITing.
  if (isJIT)
    executionEngineOptions.sanitizers = compilationOptions.sanitizers;

  auto engineOr = ExecutionEngine::createWithStandardLayers(
      std::move(executionEngineOptions), **tmOr);
  if (failed(engineOr))
    return engineOr.takeError();
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  // Add the object compiler layer.
  auto compiler = ObjectCompiler::create(runtime, pm, ".mojo_cache",
                                         compilationOptions, isJIT, isSearch);
  if (failed(compiler))
    return compiler.takeError();

  auto &objLayer = engine->addLayer<ObjectCompilerLayer>(
      std::move(*compiler), engine->getLinkingLayer());

  // Add the KGEN compiler layer. First though, get the backend chains to pass
  // into the compile layer.
  auto cacheBackends = getMojoCacheBackends(runtime);
  if (cacheBackends.isError())
    return cacheBackends.takeError();

  // Get the build info from the current build.
  BuildInfoAttr build = BuildInfoAttr::getForCurrentBuild(ctx);

  engine->addLayer<KGENCompilerLayer>(
      pm, runtime, target, build, compilationOptions, objLayer,
      std::move(cacheBackends->first), std::move(cacheBackends->second));
  return std::move(engine);
}
