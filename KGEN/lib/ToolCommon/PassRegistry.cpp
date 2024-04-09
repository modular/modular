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

void KGEN::registerDefaultKGENPasses(LLCL::Runtime &runtime) {
  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerConvertIndexToLLVMPass();

  // Register opt passes.
  KGEN::registerCanonicalizer();
  KGEN::registerCheckLifetimes();
  KGEN::registerEliminateDeadSymbols();
  KGEN::registerFoldGlobalConstLoads();
  KGEN::registerHoistTrivialInvariants();
  KGEN::registerLiftAndFoldApply();
  KGEN::registerLoopUnrolling();
  KGEN::registerLowerCallingConventions();
  KGEN::registerLowerClosures();
  KGEN::registerLowerControlFlow();
  KGEN::registerLowerGlobalPOPToLLVM();
  KGEN::registerLowerArgConventions();
  KGEN::registerLowerLoops();
  KGEN::registerLowerKGENCoroutinesAsync();
  KGEN::registerLowerKGENToLLVM();
  KGEN::registerLowerLIT();
  KGEN::registerLowerPOPToLLVM();
  KGEN::registerLowerRuntimeClosures();
  KGEN::registerLowerSemanticCF();
  KGEN::registerLowerLITTypes();
  KGEN::registerMem2Reg();
  KGEN::registerOutlineClosures();
  KGEN::registerPruneImpossibleVariants();
  KGEN::registerRaiseForLoops();
  KGEN::registerSROA();
  KGEN::registerSimplifyCF();
  KGEN::registerStackReuse();
  KGEN::registerSynthesizeDebugInfo();
  KGEN::registerTweakSpilledAllocas();
  KGEN::registerVerifyParameters();
  KGEN::registerLowerToLLVMPipeline();
  KGEN::registerSCCP();
  KGEN::registerStripParserMetadata();
  DebugInfo::registerDebugInfoToLLVM();
  DebugInfo::registerDebugInfoStrip();

  KGEN::MOGGPreElab::registerSliceMOGGFuncs();

  // Register passes that require a runtime.
  mlir::registerPass(
      [&] { return KGEN::createElaborateGeneratorsWithDefaultJIT(runtime); });
  mlir::registerPass([&] { return KGEN::createForceInline(runtime); });
  mlir::registerPass([&] { return KGEN::createInlineParametric(runtime); });
  mlir::registerPass([&] { return KGEN::createAutomaticInline(runtime); });
  mlir::registerPass(
      [&] { return KGEN::createDeadArgumentElimination(runtime); });
  mlir::registerPass(
      [&] { return KGEN::createResolveCompilerPromises(runtime); });

  // Register passes that require other arguments.
  KGEN::CompilationOptions options;
  mlir::registerPass([=, &runtime] {
    return KGEN::createMaterializePackagesWithDefaultGen(runtime, options);
  });
}
