//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPASSES_H
#define KGEN_KGENPASSES_H

#include "CompilationOptions.h"
#include "KGEN/Elaborator.h"
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

namespace M {
class BuildInfoAttr;
class TargetInfoAttr;

namespace HLCF {
class HLCFDialect;
} // namespace HLCF

namespace LLCL {
class Runtime;
} // namespace LLCL

namespace KGEN {
class KGENCallOpInterface;
class KGENDialect;
class FuncOp;
class GeneratorOp;

namespace POP {
class POPDialect;
} // namespace POP

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

/// Options for the KGEN to LLVM pipeline.
struct LowerToLLVMOptions
    : public mlir::PassPipelineOptions<LowerToLLVMOptions> {
  LowerToLLVMOptions(
      DebugInfo::EmissionKind diLevel = DebugInfo::EmissionKind::None,
      std::optional<CompilationOptions::DebugAtLevel> diAtLevel =
          std::nullopt) {
    debugInfoLevel = diLevel;
    if (diAtLevel)
      debugAtLevel = *diAtLevel;
  }

  Option<DebugInfo::EmissionKind> debugInfoLevel{
      *this, "debug-level",
      llvm::cl::desc("The level of debug info to use during compilation"),
      llvm::cl::values(
          clEnumValN(DebugInfo::EmissionKind::None, "none",
                     "Disable all debug info."),
          clEnumValN(DebugInfo::EmissionKind::LineTablesOnly,
                     "only-line-tables",
                     "Only generate debug info for line number tables."),
          clEnumValN(DebugInfo::EmissionKind::Full, "full",
                     "Generate full debug info.")),
      llvm::cl::init(DebugInfo::EmissionKind::None)};

  Option<CompilationOptions::DebugAtLevel> debugAtLevel{
      *this, "debug-at",
      llvm::cl::desc("The abstraction level to generate debug info at"),
      llvm::cl::values(clEnumValN(KGEN::CompilationOptions::kDebugAtLLVM,
                                  "llvm",
                                  "Generate debug info for the LLVM level."))};

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
createElaborateGenerators(LLCL::Runtime &runtime, TargetInfoAttr target,
                          BuildInfoAttr build,
                          const ElaborateGeneratorsOptions &options = {},
                          EvaluatorExecutorFn evaluatorExecutorFn = {});

//===----------------------------------------------------------------------===//
// Inlining
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
createAlwaysInlineParametric(LLCL::Runtime &runtime,
                             const AlwaysInlineParametricOptions &options = {});
std::unique_ptr<mlir::Pass>
createForceInline(LLCL::Runtime &runtime,
                  const ForceInlineOptions &options = {});

} // namespace KGEN
} // namespace M

#endif // KGEN_KGENPASSES_H
