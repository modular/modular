//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENCompiler.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
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

void KGEN::populateGenerateLibraryFilePasses(mlir::PassManager &pm,
                                             LLCL::Runtime &runtime) {
  pm.addPass(createVerifyParameters());

  // These passes doesn't touch parameters, no need to re-verify them after it.

  // Lower semantic control flow operations like lit.return to terminators and
  // diagnose unreachable code.
  pm.addPass(createLowerSemanticCF());

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
  AlwaysInlineParametricOptions options;
  options.nodebugOnly = true;
  pm.addPass(createAlwaysInlineParametric(runtime, options));
  pm.addPass(createVerifyParameters(
      VerifyParametersOptions{/*simplifyParameters=*/true}));

  // These passes don't influence parameters, so we don't need to verify them.

  // We use the canonicalizer, but disable region simplifications, since it is
  // very CFG centric and we have region trees with a single block per region.
  mlir::GreedyRewriteConfig cannConfig;
  cannConfig.enableRegionSimplification = false;
  pm.addNestedPass<GeneratorOp>(createSROA());
  pm.addNestedPass<GeneratorOp>(createMem2Reg());
  pm.addNestedPass<GeneratorOp>(mlir::createCanonicalizerPass(cannConfig));
  pm.addNestedPass<GeneratorOp>(createConstraintReduction());
}

/// A default specialization evaluator that JITs and invokes the specialized
/// functions with the provided evaluator.
static ErrorOr<size_t>
evaluateSpecializations(FuncOp evaluator, SymbolTable &symtab,
                        LLCL::Runtime &runtime, TargetInfoAttr target,
                        const CompilationOptions &options,
                        ArrayRef<FuncOp> specializations) {
  auto tmOr = createTargetMachine(options, true);
  if (tmOr.isError())
    return tmOr.takeError();

  // Create the execution engine.
  UNWRAP_ERROR(
      engine,
      ExecutionEngine::createWithStandardLayers(
          ExecutionEngineOptions{/*registerDebugPlugins=*/false}, **tmOr));

  // Create the object compiler so we can add its layer to the execution engine.
  mlir::PassManager mgr(target.getContext());
  auto compilerOr =
      ObjectCompiler::create(runtime, mgr, ".kgen_cache", options);
  if (failed(compilerOr))
    return compilerOr.takeError();
  engine->addLayer<ObjectCompilerLayer>(std::move(*compilerOr),
                                        engine->getLinkingLayer());

  if (!options.explicitLinking) {
    // TODO (8082): This should not be necessary.
    std::vector<std::pair<StringLiteral, void *>> compilerRTFunctions;
    registerIntelAMX(compilerRTFunctions);
    registerLLCL(compilerRTFunctions);
    registerPython(compilerRTFunctions);
    registerMemory(compilerRTFunctions);
    registerPrint(compilerRTFunctions);
    registerRandom(compilerRTFunctions);
    registerSystem(compilerRTFunctions);
    registerTracing(compilerRTFunctions);
    for (auto [name, ptr] : compilerRTFunctions)
      if (auto err = engine->add<StaticSymbolLayer>("evaluateSpecializations",
                                                    name, ptr))
        return err.takeError();
  }

  // We only want the funcs passed-in and the evaluator to be code-generated.
  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);

  // Create the set of symbols to export.
  llvm::MapVector<StringAttr, ExportedSymbol> exportedSymbols;
  for (auto e : funcsToCompile) {
    StringAttr symName = e.getSymNameAttr();
    exportedSymbols.insert({symName, ExportedSymbol(symName)});
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

  // Invoke the evaluator.
  ssize_t bestIdx;
  {
    TimeTraceScope<> traceScope("execute-specializations");
    bestIdx = evaluatorFunc.invoke<ssize_t, void **, ssize_t>(
        candidatePtrs.data(), candidatePtrs.size());
  }
  if (bestIdx == -1)
    return Error("user-provided evaluator returned failure (-1)");
  if (bestIdx < 0)
    return Error("user-provided evaluator returned an erroneous result: " +
                 Twine(bestIdx));
  if (static_cast<size_t>(bestIdx) >= candidatePtrs.size())
    return Error("user-provided evaluator returned an out-of-bounds result: " +
                 Twine(bestIdx));

  LLVM_DEBUG({
    llvm::dbgs() << "Fastest implementation:\n";
    specializations[bestIdx]->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Return the best kernel.
  return bestIdx;
}

std::unique_ptr<Pass> KGEN::createElaborateGeneratorsWithDefaultJIT(
    LLCL::Runtime &runtime, TargetInfoAttr target, BuildInfoAttr build,
    const CompilationOptions &options) {
  return createElaborateGenerators(
      runtime, target, build, {options.enableSearch},
      [=, &runtime](KGEN::FuncOp evaluator, SymbolTable &symtab,
                    TargetInfoAttr target,
                    ArrayRef<KGEN::FuncOp> specializations) {
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
      [=, &runtime](KGEN::FuncOp evaluator, SymbolTable &symtab,
                    TargetInfoAttr target,
                    ArrayRef<KGEN::FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      },
      options);
}

void KGEN::populateElaborateModulePasses(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    BuildInfoAttr build, EvaluatorExecutorFn evaluatorExecutorFn,
    const CompilationOptions &options) {
  populateGenerateLibraryFilePasses(pm, runtime);

  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures());
  pm.addPass(createVerifyParameters());
  pm.addPass(createLiftAndFoldApply());

  // After elaboration, we have no use for the parameter verifier anymore.
  pm.addPass(createElaborateGenerators(
      runtime, target, build, {options.enableSearch}, evaluatorExecutorFn));

  populatePostElaborationPasses(pm, runtime, options);
}

void KGEN::populatePostElaborationPasses(mlir::PassManager &pm,
                                         LLCL::Runtime &runtime,
                                         const CompilationOptions &options) {
  // Run the inliner, DCE, and cleanup the compiler globals.
  pm.addPass(createEliminateDeadSymbols());
  pm.addPass(createForceInline(
      runtime,
      {options.debugLevel != CompilationOptions::DebugInfoLevel::kNoDebug}));

  if (options.debugLevel == CompilationOptions::DebugInfoLevel::kSynthetic)
    pm.addPass(createSynthesizeDebugInfo());
  else if (options.debugLevel == CompilationOptions::kNoDebug)
    pm.addNestedPass<FuncOp>(DebugInfo::createDebugInfoStrip());

  pm.addNestedPass<FuncOp>(createSimplifyCF());
  pm.addNestedPass<FuncOp>(createCleanupCompilerGlobals());
  pm.addNestedPass<FuncOp>(createSROA());
  pm.addNestedPass<FuncOp>(createMem2Reg());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());

  // We use the canonicalizer, but disable region simplifications, since it is
  // very CFG centric and we have region trees with a single block per region.
  mlir::GreedyRewriteConfig cannConfig;
  cannConfig.enableRegionSimplification = false;
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass(cannConfig));

#if 0
  // TODO(Issue #7158): This pass is causing a compile time explosion and needs
  // to be investigated.  It is "just" a performance optimization for raised
  // exceptions, so disable it until we can investigate it more.
  // See: https://github.com/modularml/modular/issues/7158
  pm.addPass(createPruneImpossibleVariants());
#endif

  pm.addNestedPass<FuncOp>(createSROA());
  pm.addNestedPass<FuncOp>(createMem2Reg());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass(cannConfig));
  pm.addNestedPass<FuncOp>(createFoldGlobalConstLoads());
  pm.addNestedPass<FuncOp>(createSROA());
  pm.addNestedPass<FuncOp>(createMem2Reg());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass(cannConfig));
  pm.addNestedPass<FuncOp>(createSROA());
  pm.addNestedPass<FuncOp>(createMem2Reg());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass(cannConfig));

  // Lower async functions and closures as late as possible.
  pm.addPass(createLowerClosures());

  pm.addNestedPass<FuncOp>(createHoistTrivialInvariants());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
}

//===----------------------------------------------------------------------===//
// KGENCompilerMaterializationUnit
//===----------------------------------------------------------------------===//

/// Produce an ExportMap with every symbol in the module.
static ExportMap getAllSymbols(ModuleOp theModule) {
  ExportMap exports;
  for (auto sym : theModule.getOps<mlir::SymbolOpInterface>())
    exports.insert({sym.getNameAttr(), {sym.getNameAttr(), false}});
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
  populateElaborateModulePasses(pm, runtime, target, build, options);

  // TODO(11051): This is how it *should* be done, but because of the stack
  //   overflow issues, we have to do this manually for now.

  // Run the passes as a cached transform. Don't deflate the op as part of this
  // - we don't want that cost right now.
  //  LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
  //      theModule, regionCache.copy(), transformCache.copy(),
  //      runtime.getReadyChain().copy(), pm, /*deflateTarget=*/false);
  //  LLCL::await(ready);
  //  if (ready.isError())
  //    return ready.takeDiagnostic().getMessage().copy();

  { // This should *all* be handled by the snippet above, but because it ends up
    // being done on a separate thread, we have a smaller stack, and so we hit
    // the stack overflow bug much more often.

    // Construct the input key to the transform.
    auto transformKey = Cache::WriteableBuffer::get();
    pm.printAsTextualPipeline(*transformKey);
    if (failed(mlir::writeBytecodeToFile(theModule, *transformKey)))
      return Error("failed to write bytecode file");
    *transformKey << KGEN_VERSION_STRING;

    // Attempt to find the buffer in the cache, and if it's not found then run
    // the transform and insert it.
    auto found = transformCache->find(
        transformKey.copy(),
        LLCL::MLIRLocationDecoder::getEncodedLocation(theModule->getLoc()));
    LLCL::await(found);

    if (found.isError())
      return found.takeDiagnostic().getMessage().copy();

    // Didn't find anything, run the transform and put it into the cache.
    if (!found->has_value()) {
      if (failed(pm.run(theModule))) {
        return Error("compilation failed");
      }
      // Put the thing into the cache.
      auto transformed = Cache::WriteableBuffer::get();
      if (failed(mlir::writeBytecodeToFile(theModule, *transformed)))
        return Error("failed to write bytecode file");
      transformCache->insert(std::move(transformKey), std::move(transformed));
      // And we're done. Can't return yet though, have to pass through the rest
      // of the function.
    } else {
      // We have something stored, pull it out of the cache now.
      std::unique_ptr<llvm::MemoryBuffer> bytecode =
          llvm::MemoryBuffer::getMemBuffer((**found)->getBuffer(),
                                           /*BufferName=*/"",
                                           /*RequiresNullTerminator=*/false);

      // Create a dummy block that we can use to inflate the cached module into.
      Block b;
      if (failed(mlir::readBytecodeFile(
              *bytecode, &b,
              mlir::ParserConfig(theModule->getContext(),
                                 /*verifyAfterParse=*/false)))) {
        return Error("reading bytecode file failed");
      }
      // Take the body from the module we just parsed.
      theModule.getBodyRegion().takeBody(b.front().getRegion(0));
    }
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
    symbols[mangler(symbol.alias.getValue())] = llvm::JITSymbolFlags::Callable;

  return {std::move(symbols), /*InitSymbol=*/nullptr};
}
