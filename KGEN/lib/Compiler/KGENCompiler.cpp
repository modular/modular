//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LowerToObject.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

static void populatePreElaborationPipeline(mlir::PassManager &pm) {
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerSemanticCF());
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerLIT());
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerStructs());
  pm.addPass(createVerifyParameters());

  pm.addPass(createAlwaysInlineParametric());
  pm.addPass(createVerifyParameters());

  // These passes don't influence parameters, so we don't need to verify them.

  // We use the canonicalizer, but disable region simplifications, since it is
  // very CFG centric and we have region trees with a single block per region.
  mlir::GreedyRewriteConfig cannConfig;
  cannConfig.enableRegionSimplification = false;
  pm.addNestedPass<GeneratorOp>(mlir::createCanonicalizerPass(cannConfig));
  pm.addNestedPass<GeneratorOp>(createConstraintReduction());
  pm.addNestedPass<GeneratorOp>(createMem2Reg());
}

void KGEN::populateGenerateLibraryFilePasses(mlir::PassManager &pm) {
  // Set up the pass pipeline.
  populatePreElaborationPipeline(pm);
}

void KGEN::populateElaborateModulePasses(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    const ElaborateGeneratorsOptions &elaborateOptions) {
  populatePreElaborationPipeline(pm);
  // Eliminate dead symbols. If we don't use the symbol *somewhere* it doesn't
  // need to be in the IR.
  pm.addPass(createEliminateDeadSymbols());

  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures());
  pm.addPass(createVerifyParameters());

  // After elaboration, we have no use for the parameter verifier anymore.
  pm.addPass(createElaborateGenerators(runtime, target, elaborateOptions));

  // Run the inliner, DCE, and cleanup the compiler globals.
  pm.addPass(createForceInline());
  pm.addPass(createEliminateDeadSymbols());
  pm.addNestedPass<KGEN::FuncOp>(createCleanupCompilerGlobals());

  // We use the canonicalizer, but disable region simplifications, since it is
  // very CFG centric and we have region trees with a single block per region.
  mlir::GreedyRewriteConfig cannConfig;
  cannConfig.enableRegionSimplification = false;
  pm.addNestedPass<KGEN::FuncOp>(mlir::createCanonicalizerPass(cannConfig));

#if 0
  // TODO(Issue #7158): This pass is causing a compile time explosion and needs
  // to be investigated.  It is "just" a performance optimization for raised
  // exceptions, so disable it until we can investigate it more.
  // See: https://github.com/modularml/modular/issues/7158
  pm.addPass(createPruneImpossibleVariants());
#endif

  // Lower async functions as late as possible.
  pm.addPass(createLowerAsyncFunctions());
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
    ElaborateGeneratorsOptions elaborateOptions, ObjectCompilerLayer &base,
    llvm::orc::ExecutionSession &sess, const llvm::DataLayout &dl,
    MaterializationLayer::AddToSearchOrderFn add)
    : llvm::RTTIExtends<KGENCompilerLayer, MaterializationLayer>(
          sess, dl, std::move(add)),
      pm(pm), runtime(runtime), target(target),
      elaborateOptions(std::move(elaborateOptions)), baseLayer(base) {}

/// This elaborates all the generators in `theModule` and takes the module from
/// a just-parsed state to a state we can use to produce an object file. This
/// modifies the module in place. The granularity of this operation is tentative
/// and should be re-evaluated, we may end up in a place where we want to split
/// pre-elaboration, elaboration, and post-elaboration into explicit phases.
///
/// The purpose of this function is largely for cases where we don't want to add
/// additional options to the pass manager, such as when we're evaluating a
/// module in a JIT context.
static LogicalResult
concretizeModule(mlir::PassManager &pm, ModuleOp theModule,
                 LLCL::Runtime &runtime, TargetInfoAttr target,
                 const ElaborateGeneratorsOptions &elaborateOptions) {
  pm.clear();
  populateElaborateModulePasses(pm, runtime, target, elaborateOptions);
  return pm.run(theModule);
}

ErrorOrSuccess KGENCompilerLayer::add(StringRef libName, ModuleOp theModule) {
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();

  llvm::orc::JITDylib *dylib = *dylibOr;
  llvm::orc::ResourceTrackerSP resourceTracker =
      dylib->getDefaultResourceTracker();

  // TODO(#10920): We need to do this here to get the mangled names.
  if (failed(
          concretizeModule(pm, theModule, runtime, target, elaborateOptions)))
    return Error("compilation failed");

  // Add the materialization unit by computing the exports and the symbol table,
  // and passing those off.
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
