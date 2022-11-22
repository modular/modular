//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPASSES_H
#define KGEN_KGENPASSES_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
class ModuleOp;
class OpPassManager;
namespace LLVM {
class LLVMDialect;
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace M::LLCL {
class Runtime;
}

namespace M::KGEN {
class KGENDialect;
class FuncOp;
namespace POP {
class POPDialect;
} // namespace POP

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

/// Options for the KGEN to LLVM pipeline.
struct LowerToLLVMOptions
    : public mlir::PassPipelineOptions<LowerToLLVMOptions> {
  Option<std::string> topLevelKernel{
      *this, "top-level-kernel",
      llvm::cl::desc("The name of the top-level kernel. If specified, the "
                     "signature of the kernel is altered to be C-compatible")};
  Option<bool> emitOpaqueWrappers{
      *this, "emit-opaque-wrappers",
      llvm::cl::desc("Whether to emit opaque function wrappers. If "
                     "specified, all contained functions will receive a "
                     "wrapper with arguments and results tightly packed.")};
};

/// Build the pass pipeline to convert post-elaboration KGEN IR to LLVM IR.
/// The pipeline runs the canonicalizer, the KGEN to LLVM conversion, a series
/// of LLVM lowerings, and the canonicalizer again.
void buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                              const LowerToLLVMOptions &options);

/// Register the lower to LLVM pipeline.
void registerLowerToLLVMPipeline();

/// Create/register the EmitLLVM pass with the given runtime.
std::unique_ptr<mlir::Pass> createEmitLLVMPass(LLCL::Runtime &rt);
void registerEmitLLVMPass(LLCL::Runtime &rt);

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "KGEN/KGENPasses.h.inc"

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

/// Create an instance of the elaborator pass that captures all of the
/// referenced include files.
std::unique_ptr<mlir::Pass>
createElaborateGenerators(SmallVectorImpl<std::string> &includedFiles,
                          const ElaborateGeneratorsOptions &options = {});

} // namespace M::KGEN

#endif // KGEN_KGENPASSES_H
