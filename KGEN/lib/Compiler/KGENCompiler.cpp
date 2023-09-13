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
  auto tmOr = createTargetMachine(options, true);
  if (tmOr.isError())
    return tmOr.takeError();

  // Create the execution engine.
  auto engineOr = ExecutionEngine::createWithStandardLayers(
      ExecutionEngineOptions{/*registerDebugPlugins=*/false,
                             /*sanitizers=*/options.sanitizers},
      **tmOr);
  if (engineOr.isError())
    return engineOr.takeError();
  auto engine = std::move(*engineOr);

  // Create the object compiler so we can add its layer to the execution engine.
  mlir::PassManager mgr(target.getContext());
  auto compilerOr = ObjectCompiler::create(runtime, mgr, ".mojo_cache", options,
                                           /*isJIT=*/true, /*isSearch=*/true);
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
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
KGEN::createElaborateGeneratorsWithDefaultJIT(LLCL::Runtime &runtime) {
  return createElaborateGenerators(
      runtime, /*target=*/{}, /*build=*/{}, /*options=*/{},
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       /*options=*/{}, specializations);
      });
}

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
    symbols[mangler(name)] = llvm::JITSymbolFlags::Callable |
                             llvm::JITSymbolFlags::Exported |
                             (symbol.isCExport ? llvm::JITSymbolFlags::None
                                               : llvm::JITSymbolFlags::Weak);
  }

  return {std::move(symbols), /*InitSymbol=*/nullptr};
}
