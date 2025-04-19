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
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerPrintOpStats();
  mlir::registerStripDebugInfo();

  // Register opt passes.
  KGEN::registerApplyInliner();
  KGEN::registerArgPromotion();
  KGEN::registerCanonicalizer();
  KGEN::registerCheckLifetimes();
  KGEN::registerEliminateDeadSymbols();
  KGEN::registerEliminateDuplicateFunctions();
  KGEN::registerEnsureNoParameters();
  KGEN::registerExpandStructs();
  KGEN::registerFunctionStats();
  KGEN::registerHoistTrivialInvariants();
  KGEN::registerLiftAndFoldApply();
  KGEN::registerLoopUnrolling();
  KGEN::registerLowerAsyncFunctions();
  KGEN::registerLowerCallingConventions();
  KGEN::registerLowerClosures();
  KGEN::registerLowerControlFlow();
  KGEN::registerLowerGlobalPOPToLLVM();
  KGEN::registerLowerArgConventions();
  KGEN::registerLowerLoops();
  KGEN::registerLowerKGENToLLVM();
  KGEN::registerLowerLIT();
  KGEN::registerLegalizePOPOperations();
  KGEN::registerLowerPOPToLLVM();
  KGEN::registerLowerRuntimeClosures();
  KGEN::registerLowerSemanticCF();
  KGEN::registerMem2Reg();
  KGEN::registerOutlineClosures();
  KGEN::registerOutlineClosuresNew();
  KGEN::registerRaiseForLoops();
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

  KGEN::MOGGPreElab::registerAnnotateKernels();
  KGEN::MOGGPreElab::registerMOGGPreElabPipeline();
  KGEN::MOGGPreElab::registerOutlineKernels();
  KGEN::MOGGPreElab::registerVerifyKernels();
  KGEN::MOGGPreElab::registerDumpKernels();

  // Passes that require a runtime.
  mlir::registerPass(
      [&] { return KGEN::createElaborateGeneratorsWithDefaultJIT(); });
  KGEN::registerInlineParametric();
  KGEN::registerAutomaticInline();
  KGEN::registerDeadArgumentElimination();
  KGEN::registerResolveCompilerPromises();
}
