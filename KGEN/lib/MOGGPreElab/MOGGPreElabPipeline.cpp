//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "Helpers.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_MOGGPREELABPIPELINE
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

namespace {

class MOGGPreElabPipeline
    : public M::KGEN::MOGGPreElab::impl::MOGGPreElabPipelineBase<
          MOGGPreElabPipeline> {
  using MOGGPreElabPipelineBase::MOGGPreElabPipelineBase;

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool hasKernels = false;
    for (auto func : mod.getOps<GeneratorOp>())
      hasKernels |= MOGGPreElab::isKernel(func);

    if (hasKernels && !debugBuild) {
      mlir::OpPassManager pm(ModuleOp::getOperationName());
      pm.addPass(MOGGPreElab::createSliceMOGGFuncs());
      pm.addPass(MOGGPreElab::createOutlineMOGGFuncs());

      if (failed(runPipeline(pm, mod)))
        return signalPassFailure();
    }
  }
};

} // namespace
