//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LowerToObject.h"
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

void KGEN::populateGenerateLibraryFilePasses(
    mlir::PassManager &pm, LLCL::Runtime &runtime,
    const CompilationOptions &options) {
  pm.addPass(createVerifyParameters());

  // These passes doesn't touch parameters, no need to re-verify them after it.

  // Lower semantic control flow operations like lit.return to terminators and
  // diagnose unreachable code.
  pm.addPass(createLowerSemanticCF());

  // Check if a struct contains recursive nested struct fields and emit error if
  // found.
  pm.addPass(createCheckRecursiveStructs());

  // Insert calls to destructors, reject use before free, and borrow check.
  pm.addPass(createCheckLifetimes());

  pm.addPass(createLowerLIT());
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerStructs());
  pm.addPass(createVerifyParameters());
  // Eliminate dead symbols. If we don't use the symbol *somewhere* it doesn't
  // need to be in the IR.
  pm.addPass(createEliminateDeadSymbols());

  // Only inline `always_inline_no_debug` functions during parametric inlining.
  // Too much inlining pre-elaboration increases pressure on the elaborator and
  // reduces cache granularity. By restricting inlining to `nodebug` functions,
  // we still maintain the zero-cost abstraction.
  AlwaysInlineParametricOptions inlinerOpts;
  inlinerOpts.nodebugOnly = true;
  pm.addPass(createAlwaysInlineParametric(runtime, inlinerOpts));
  if (options.optimizationLevel >= 1) {
    pm.addPass(createVerifyParameters(
        VerifyParametersOptions{/*simplifyParameters=*/true}));
  }

  // These passes don't influence parameters, so we don't need to verify them.

  // We use the canonicalizer, but disable region simplifications, since it is
  // very CFG centric and we have region trees with a single block per region.
  if (options.optimizationLevel >= 1) {
    mlir::GreedyRewriteConfig cannConfig;
    cannConfig.enableRegionSimplification = false;
    pm.addNestedPass<GeneratorOp>(createSROA());
    pm.addNestedPass<GeneratorOp>(createMem2Reg());
    pm.addNestedPass<GeneratorOp>(mlir::createCanonicalizerPass(cannConfig));
    pm.addNestedPass<GeneratorOp>(createConstraintReduction());
  }
}

/// A default specialization evaluator that JITs and invokes the specialized
/// functions with the provided evaluator.
static ErrorOr<ElaboratorSearchFn>
evaluateSpecializations(FuncOp evaluator, const SymbolTable &symtab,
                        LLCL::Runtime &runtime, TargetInfoAttr target,
                        const CompilationOptions &options,
                        ArrayRef<FuncOp> specializations) {
  auto tmOr = createTargetMachine(options, true);
  if (tmOr.isError())
    return tmOr.takeError();

  // Create the execution engine.
  UNWRAP_ERROR(engine,
               ExecutionEngine::createWithStandardLayers(
                   ExecutionEngineOptions{/*registerDebugPlugins=*/false,
                                          /*sanitizers=*/options.sanitizers},
                   **tmOr));

  // Create the object compiler so we can add its layer to the execution engine.
  mlir::PassManager mgr(target.getContext());
  auto compilerOr =
      ObjectCompiler::create(runtime, mgr, ".mojo_cache", options);
  if (failed(compilerOr))
    return compilerOr.takeError();
  engine->addLayer<ObjectCompilerLayer>(std::move(*compilerOr),
                                        engine->getLinkingLayer());

  // We only want the funcs passed-in and the evaluator to be code-generated.
  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);

  // Create the set of symbols to export.
  llvm::MapVector<StringAttr, ExportedSymbol> exportedSymbols;
  for (auto e : funcsToCompile) {
    StringAttr symName = e.getSymNameAttr();
    exportedSymbols.insert({symName, ExportedSymbol()});
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
      UNWRAP_ERROR(func, engine->lookup(candidate.getNameAttr()));
      candidatePtrs.push_back(func.getFunctionPointer());
    }
  }

  // Lookup the evaluator function
  UNWRAP_ERROR(evaluatorFunc, engine->lookup(evaluator.getNameAttr()));

  return
      [engine = std::move(engine), evaluatorFunc = std::move(evaluatorFunc),
       candidatePtrs = std::move(candidatePtrs)]() mutable -> ErrorOr<ssize_t> {
        TimeTraceScope<> traceScope("execute-specializations");
        return evaluatorFunc.invoke<ssize_t, void **, ssize_t>(
            candidatePtrs.data(), candidatePtrs.size());
      };
}

/// Return whether locations should be elaborated based on the debug level.
static bool
shouldElaborateLocations(CompilationOptions::DebugInfoLevel debugLevel) {
  return debugLevel == CompilationOptions::kFullDebugInfo ||
         debugLevel == CompilationOptions::kLineTablesOnly;
}

std::unique_ptr<Pass> KGEN::createElaborateGeneratorsWithDefaultJIT(
    LLCL::Runtime &runtime, TargetInfoAttr target, BuildInfoAttr build,
    const CompilationOptions &options) {
  ElaborateGeneratorsOptions elaboratorOptions;
  elaboratorOptions.enableSearch = options.enableSearch;
  elaboratorOptions.elaborateLocations =
      shouldElaborateLocations(options.debugLevel);
  return createElaborateGenerators(
      runtime, target, build, elaboratorOptions,
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      });
}

void KGEN::populateElaborateModulePasses(mlir::PassManager &pm,
                                         LLCL::Runtime &runtime,
                                         TargetInfoAttr target,
                                         BuildInfoAttr build,
                                         const CompilationOptions &options) {
  return populateElaborateModulePasses(
      pm, runtime, target, build,
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      },
      options);
}

void KGEN::populateElaborateModulePasses(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    BuildInfoAttr build, EvaluatorExecutorFn evaluatorExecutorFn,
    const CompilationOptions &options) {
  // At the end of the LIT lowering pipeline, pull in the bodies of constructs
  // that were already elaborated.
  pm.addPass(createLowerPreElaboratedLIT());

  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures());
  pm.addPass(createVerifyParameters());
  pm.addPass(createLiftAndFoldApply());

  // After elaboration, we have no use for the parameter verifier anymore.
  ElaborateGeneratorsOptions elaboratorOptions;
  elaboratorOptions.enableSearch = options.enableSearch;
  elaboratorOptions.elaborateLocations =
      shouldElaborateLocations(options.debugLevel);
  pm.addPass(createElaborateGenerators(runtime, target, build,
                                       elaboratorOptions, evaluatorExecutorFn));

  populatePostElaborationPasses(pm, runtime, options);
}

void KGEN::populatePostElaborationPasses(mlir::PassManager &pm,
                                         LLCL::Runtime &runtime,
                                         const CompilationOptions &options) {
  // Run DCE first coming out of the elaborator.
  pm.addPass(createEliminateDeadSymbols());

  // Run the inliner with an inner function pass pipeline.
  auto buildInlinerFuncPasses = [options](mlir::OpPassManager &pm) {
    pm.addPass(createCleanupCompilerGlobals());
    if (options.optimizationLevel < 1)
      return;
    pm.addPass(createSimplifyCF());
    pm.addPass(createSROA());
    pm.addPass(createMem2Reg());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createHoistTrivialInvariants());
    pm.addPass(createCanonicalizer());
    pm.addPass(createSROA());
    pm.addPass(createMem2Reg());
    pm.addPass(createStackReuse());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createCanonicalizer());
  };
  pm.addPass(createForceInline(
      runtime,
      {options.debugLevel != CompilationOptions::DebugInfoLevel::kNoDebug},
      std::move(buildInlinerFuncPasses)));

  // Process debuginfo based on the selected debugging level.
  if (options.debugLevel == CompilationOptions::DebugInfoLevel::kSynthetic)
    pm.addPass(createSynthesizeDebugInfo());
  else if (options.debugLevel == CompilationOptions::kNoDebug)
    pm.addNestedPass<FuncOp>(DebugInfo::createDebugInfoStrip());

  // Long-tail optimization passes.
  // FIXME: This section needs to be trimmed down.
  if (options.optimizationLevel >= 1) {
    pm.addNestedPass<FuncOp>(createFoldGlobalConstLoads());
    pm.addNestedPass<FuncOp>(createSROA());
    pm.addNestedPass<FuncOp>(createMem2Reg());
    pm.addNestedPass<FuncOp>(createCanonicalizer());
  }
  if (options.optimizationLevel >= 2) {
    pm.addNestedPass<FuncOp>(createSROA());
    pm.addNestedPass<FuncOp>(createMem2Reg());
    pm.addNestedPass<FuncOp>(createCanonicalizer());
  }

  // Lower async functions and closures as late as possible.
  pm.addPass(createLowerClosures());

  // Loop raising must happen after `hoist-trivial-invariants`.
  // FIXME: Move this earlier in the pipeline.
  pm.addNestedPass<FuncOp>(createRaiseForLoops());
  // FIXME: Despite being a "must run" optimization, loop unrolling requires
  // other optimization passes to run because it does not use SCEV.
  if (options.optimizationLevel >= 1)
    pm.addNestedPass<FuncOp>(createLoopUnrolling({options.optimizationLevel}));
  pm.addNestedPass<FuncOp>(createLowerLoops());

  // At the end of the pipeline, externalize any functions that have been
  // precompiled so that they aren't sent to LLVM again.
  pm.addPass(createExternalizePrecompiledFunctions());
}

//===----------------------------------------------------------------------===//
// KGENCompilerMaterializationUnit
//===----------------------------------------------------------------------===//

/// Produce an ExportMap with every symbol in the module.
static ExportMap getAllSymbols(ModuleOp theModule) {
  ExportMap exports;
  for (auto sym : theModule.getOps<mlir::SymbolOpInterface>())
    exports.insert({sym.getNameAttr(), {false}});
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
    LLCL::RCRef<Cache::BlobCacheBackend> transformCacheBackend,
    LLCL::RCRef<Cache::BlobCacheBackend> regionCacheBackend,
    llvm::orc::ExecutionSession &sess, const llvm::DataLayout &dl,
    MaterializationLayer::AddToSearchOrderFn add)
    : llvm::RTTIExtends<KGENCompilerLayer, MaterializationLayer>(
          sess, dl, std::move(add)),
      pm(pm), runtime(runtime), target(target), build(build), options(options),
      baseLayer(base) {
  // Construct the caches.
  transformCache = LLCL::RCRef<Cache::TransformCache>::create(
      std::move(transformCacheBackend));
  regionCache =
      LLCL::RCRef<Cache::RegionCache>::create(std::move(regionCacheBackend));
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
  populateGenerateLibraryFilePasses(pm, runtime, options);
  populateElaborateModulePasses(pm, runtime, target, build, options);

  // Run the passes as a cached transform. Don't deflate the op as part of this
  // - we don't want that cost right now.
  {
    [[maybe_unused]] auto timeScope =
        runtime.emplaceContextIfMissing<M::Telemetry::TelemetryContext>()
            .createUInt64Timer<std::chrono::milliseconds>(
                "mojo.kgen.compile.time");

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

  auto err = dylib->define(std::make_unique<KGENCompilerMaterializationUnit>(
                               *this, std::move(st), std::move(ex)),
                           resourceTracker);
  if (err)
    return Error(llvm::toString(std::move(err)));
  return success();
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

  for (auto &[name, symbol] : exports)
    symbols[mangler(name)] =
        llvm::JITSymbolFlags::Callable | llvm::JITSymbolFlags::Exported;

  return {std::move(symbols), /*InitSymbol=*/nullptr};
}
