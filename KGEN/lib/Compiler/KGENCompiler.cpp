//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

static void populatePreElaborationPipeline(mlir::PassManager &pm) {
  pm.addPass(createLowerLIT());
  pm.addPass(createLiftMLIROperations());
  pm.addPass(createLowerStructs());
  pm.addNestedPass<GeneratorOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GeneratorOp>(createMem2Reg());
}

void KGEN::generateLibraryFile(mlir::PassManager &pm) {
  // Set up the pass pipeline.
  populatePreElaborationPipeline(pm);
}

void KGEN::elaborateModule(mlir::PassManager &pm, LLCL::Runtime &runtime,
                           const ElaborateGeneratorsOptions &elaborateOptions,
                           SmallVectorImpl<std::string> &includedFiles) {
  populatePreElaborationPipeline(pm);
  // Resolve includes before we outline closures.
  pm.addPass(createResolveIncludes(
      includedFiles, ResolveIncludesOptions{elaborateOptions.searchPaths}));
  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures());
  pm.addPass(createElaborateGenerators(runtime, /*oldImpl=*/false,
                                       includedFiles, elaborateOptions));
  // Run the inliner and cleanup the compiler globals.
  pm.addPass(createForceInline());
  pm.addNestedPass<KGEN::FuncOp>(createCleanupCompilerGlobals());
  pm.addNestedPass<KGEN::FuncOp>(mlir::createCanonicalizerPass());
  // Finally, DCE the symbols we don't want.
  pm.addPass(createEliminateDeadSymbols());

#if 0
  // TODO(Issue #7158): This pass is causing a compile time explosion and needs
  // to be investigated.  It is "just" a performance optimization for raised
  // exceptions, so disable it until we can investigate it more.
  // See: https://github.com/modularml/modular/issues/7158
  pm.addPass(createPruneImpossibleVariants());
#endif
}
