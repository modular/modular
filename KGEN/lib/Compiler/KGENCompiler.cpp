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
  pm.addPass(createLowerStructs());
  pm.addNestedPass<GeneratorOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GeneratorOp>(createMem2Reg());
  pm.addPass(createAlwaysInlineParametric());
}

void KGEN::generateLibraryFile(mlir::PassManager &pm) {
  // Set up the pass pipeline.
  populatePreElaborationPipeline(pm);
}

void KGEN::elaborateModule(mlir::PassManager &pm, LLCL::Runtime &runtime,
                           const ElaborateGeneratorsOptions &elaborateOptions) {
  populatePreElaborationPipeline(pm);
  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures());
  pm.addPass(createElaborateGenerators(runtime, elaborateOptions));
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
