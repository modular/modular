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
  pm.addPass(createLowerLITTerminators());
  pm.addPass(createLowerLIT());
  pm.addPass(createLowerStructs());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<GeneratorOp>(createMem2Reg());
}

LogicalResult KGEN::generateLibraryFile(ModuleOp theModule) {
  // Set up the pass pipeline.
  mlir::PassManager pm(theModule->getContext());
  populatePreElaborationPipeline(pm);
  return pm.run(theModule);
}

LogicalResult
KGEN::elaborateModule(ModuleOp theModule, LLCL::Runtime &runtime,
                      const ElaborateGeneratorsOptions &elaborateOptions,
                      SmallVectorImpl<std::string> &includedFiles) {
  mlir::PassManager pm(theModule->getContext());
  populatePreElaborationPipeline(pm);
  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures());
  pm.addPass(
      createElaborateGenerators(includedFiles, runtime, elaborateOptions));
  // Run the inliner and cleanup the compiler globals.
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<KGEN::FuncOp>(createCleanupCompilerGlobals());
  pm.addPass(mlir::createCanonicalizerPass());
  // Finally, DCE the symbols we don't want.
  pm.addPass(createEliminateDeadSymbols());
  pm.addPass(createPruneImpossibleVariants());

  return pm.run(theModule);
}
