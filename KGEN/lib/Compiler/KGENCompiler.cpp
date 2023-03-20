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
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerLITTerminators());
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerLIT());
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerStructs());
  pm.addPass(createVerifyParameters());

  pm.addPass(createAlwaysInlineParametric());
  pm.addPass(createVerifyParameters());

  // These passes don't influence parameters, so we don't need to verify them.
  pm.addNestedPass<GeneratorOp>(mlir::createCanonicalizerPass());
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
  pm.addNestedPass<KGEN::FuncOp>(mlir::createCanonicalizerPass());

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

LogicalResult
KGEN::concretizeModule(mlir::PassManager &pm, ModuleOp theModule,
                       LLCL::Runtime &runtime, TargetInfoAttr target,
                       const ElaborateGeneratorsOptions &elaborateOptions) {
  pm.clear();
  populateElaborateModulePasses(pm, runtime, target, elaborateOptions);
  return pm.run(theModule);
}
