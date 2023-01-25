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
  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures());
  pm.addPass(
      createElaborateGenerators(includedFiles, runtime, elaborateOptions));
  // Run the inliner and cleanup the compiler globals.
  pm.addPass(createForceInline());
  pm.addNestedPass<KGEN::FuncOp>(createCleanupCompilerGlobals());
  pm.addNestedPass<KGEN::FuncOp>(mlir::createCanonicalizerPass());
  // Finally, DCE the symbols we don't want.
  pm.addPass(createEliminateDeadSymbols());
  pm.addPass(createPruneImpossibleVariants());
}
