//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains core logic to parameterized generators into concrete
// function implementations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Elaborator.h"
#include "Elaborator.h"
#include "KGEN/KGENPasses.h"
#include "Support/STLExtras.h"

using namespace M;
using namespace KGEN;

/// Elaborate generators in the specified module, incorporating implementation
/// logic from the specified library.
LogicalResult M::elaborateGenerators(mlir::SymbolTableAnalysis &analysis,
                                     LLCL::Runtime &runtime,
                                     TargetInfoAttr target,
                                     ArrayRef<GeneratorOp> primaryGenerators,
                                     bool useOldImpl, bool enableSearch) {
  // If we want to use the new impl, use it and return immediately.
  return elaborateGeneratorsV2(analysis, runtime, target, primaryGenerators,
                               enableSearch);
}

//===----------------------------------------------------------------------===//
// ElaborateGeneratorsPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_ELABORATEGENERATORS
#define GEN_PASS_DEF_RESOLVEINCLUDES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Run the elaborator as a pass. The elaborator requires imports to be
/// resolved, so first resolve imports and then elaborate.
struct ElaborateGeneratorsPass
    : public KGEN::impl::ElaborateGeneratorsBase<ElaborateGeneratorsPass> {
  ElaborateGeneratorsPass(LLCL::Runtime &runtime, bool oldImpl,
                          SmallVectorImpl<std::string> &includedFiles,
                          const ElaborateGeneratorsOptions &options)
      : ElaborateGeneratorsBase(options), runtime(&runtime), oldImpl(oldImpl),
        includedFiles(&includedFiles) {}
  using ElaborateGeneratorsBase::ElaborateGeneratorsBase;

  void runOnOperation() override {
    auto rt = ConditionallyOwnedPointer<LLCL::Runtime>::allocateIfNeeded(
        runtime, LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
        LLCL::createSingleThreadWorkQueue());

    ModuleOp theModule = getOperation();

    SmallVector<std::filesystem::path> paths;
    for (const auto &p : searchPaths)
      paths.push_back(p);

    paths.push_back(std::filesystem::path("."));

    // Extract the top-level, parameterless generators from the main module.
    // These are the only generators that will be elaborated.
    SmallVector<GeneratorOp> primaryGenerators;
    for (auto gen : theModule.getOps<GeneratorOp>())
      if (gen.getInputParamDecls().empty() && !gen.getImplementsAttr())
        primaryGenerators.push_back(gen);

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    // TODO: This should not be happening inside the elaborate-generators pass.
    if (failed(resolveIncludes(analysis.getTopLevelSymbolTable(), paths,
                               includedFiles)))
      return signalPassFailure();

    if (failed(elaborateGenerators(analysis, *rt,
                                   TargetInfoAttr::getForHost(&getContext()),
                                   primaryGenerators, oldImpl, shouldDoSearch)))
      return signalPassFailure();
  }

  /// An optional LLCL runtime pointer.
  LLCL::Runtime *runtime = nullptr;
  /// Whether to use the new or the old elaborator implementation.
  bool oldImpl = false;
  /// Vector of files we included.
  SmallVectorImpl<std::string> *includedFiles = nullptr;
};

/// Resolve includes in a pass. This pass only does include resolution.
struct ResolveIncludesPass
    : public KGEN::impl::ResolveIncludesBase<ResolveIncludesPass> {
  using ResolveIncludesBase::ResolveIncludesBase;
  ResolveIncludesPass(SmallVectorImpl<std::string> &includedFiles,
                      const ResolveIncludesOptions &options)
      : ResolveIncludesBase(options), includedFiles(&includedFiles) {}

  void runOnOperation() override {
    SmallVector<std::filesystem::path> paths;
    for (const auto &p : searchPaths)
      paths.push_back(p);
    paths.push_back(std::filesystem::path("."));

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    if (failed(resolveIncludes(analysis.getTopLevelSymbolTable(), paths,
                               includedFiles)))
      return signalPassFailure();
  }

  SmallVectorImpl<std::string> *includedFiles = nullptr;
};
} // namespace

std::unique_ptr<mlir::Pass>
KGEN::createElaborateGenerators(LLCL::Runtime &runtime, bool oldImpl,
                                SmallVectorImpl<std::string> &includedFiles,
                                const ElaborateGeneratorsOptions &options) {
  return std::make_unique<ElaborateGeneratorsPass>(runtime, oldImpl,
                                                   includedFiles, options);
}

std::unique_ptr<mlir::Pass>
KGEN::createResolveIncludes(SmallVectorImpl<std::string> &includedFiles,
                            const ResolveIncludesOptions &options) {
  return std::make_unique<ResolveIncludesPass>(includedFiles, options);
}
