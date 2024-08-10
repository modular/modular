//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

void KGEN::registerDefaultKGENPasses() {
  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerConvertIndexToLLVMPass();
  mlir::registerReconcileUnrealizedCasts();
  mlir::registerPrintOpStats();

  // Register opt passes.
  KGEN::registerApplyInliner();
  KGEN::registerCheckLifetimes();
  KGEN::registerEliminateDeadSymbols();
  KGEN::registerFunctionStats();
  KGEN::registerHoistTrivialInvariants();
  KGEN::registerLiftAndFoldApply();
  KGEN::registerLoopUnrolling();
  KGEN::registerLowerAsyncFunctions();
  KGEN::registerLowerCallingConventions();
  KGEN::registerLowerClosures();
  KGEN::registerLowerControlFlow();
  KGEN::registerLowerCustomOpsPreElab();
  KGEN::registerLowerGlobalPOPToLLVM();
  KGEN::registerLowerArgConventions();
  KGEN::registerLowerLoops();
  KGEN::registerLowerKGENToLLVM();
  KGEN::registerLowerLIT();
  KGEN::registerLowerPOPToLLVM();
  KGEN::registerLowerRuntimeClosures();
  KGEN::registerLowerSemanticCF();
  KGEN::registerLowerLITTypes();
  KGEN::registerMem2Reg();
  KGEN::registerOutlineClosures();
  KGEN::registerRaiseForLoops();
  KGEN::registerRegisterCustomOps();
  KGEN::registerRemoveUnusedParams();
  KGEN::registerSROA();
  KGEN::registerSimplifyCF();
  KGEN::registerStackReuse();
  KGEN::registerSynthesizeDebugInfo();
  KGEN::registerVerifyParameters();
  KGEN::registerLowerSuspensionPoints();
  KGEN::registerLowerToLLVMPipeline();
  KGEN::registerIPDF();
  KGEN::registerSCCP();
  KGEN::registerStripParserMetadata();
  DebugInfo::registerDebugInfoToLLVM();
  DebugInfo::registerDebugInfoStrip();

  KGEN::MOGGPreElab::registerMOGGAnnotate();
  KGEN::MOGGPreElab::registerMOGGAutoparameterize();
  KGEN::MOGGPreElab::registerMOGGPreElabPipeline();
  KGEN::MOGGPreElab::registerOutlineMOGGFuncs();
  KGEN::MOGGPreElab::registerSliceMOGGFuncs();

  // Passes that require a runtime.
  mlir::registerPass(
      [&] { return KGEN::createElaborateGeneratorsWithDefaultJIT(); });
  mlir::registerPass(
      [&] { return KGEN::createLowerCustomOpsWithDefaultJIT(); });
  mlir::registerPass([&] { return KGEN::createCanonicalizerWithDefaultJIT(); });
  KGEN::registerInlineParametric();
  KGEN::registerAutomaticInline();
  KGEN::registerDeadArgumentElimination();
  KGEN::registerResolveCompilerPromises();

  // Register passes that require other arguments.
  KGEN::CompilationOptions options;
}
