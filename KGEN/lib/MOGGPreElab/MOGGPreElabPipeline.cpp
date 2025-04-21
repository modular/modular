//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

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

    bool hasKernels =
        llvm::any_of(mod.getOps<GeneratorOp>(), [](GeneratorOp func) {
          return MOGGPreElab::isKernel(func) ||
                 MOGGPreElab::isExtensibilityFunc(func);
        });

    if (hasKernels && !debugBuild) {
      mlir::OpPassManager pm(ModuleOp::getOperationName());
      pm.addPass(MOGGPreElab::createOutlineKernels());

      if (failed(runPipeline(pm, mod)))
        return signalPassFailure();
    }
  }
};

} // namespace
