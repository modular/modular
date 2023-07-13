//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPASSES_H
#define KGEN_KGENPASSES_H

#include "KGEN/CompilationOptions.h"
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
          clEnumValN(DebugInfo::EmissionKind::LineTablesOnly, "line-tables",
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

  Option<bool> isJIT{
      *this, "is-jit",
      llvm::cl::desc("True if the module is being compiled for JIT mode.")};
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

/// This function kind represents a callback to invoke a compiled evaluator
/// function with the compiled candidate functions. This function performs the
/// actual benchmarking of search and must be invoked in isolation. The
/// elaborator ensures that the compiler process is quiet before invoking this
/// function, which is required for stable and accurate results.
using ElaboratorSearchFn = llvm::unique_function<ErrorOr<ssize_t>()>;

/// This function kind represents a callback given the IR for an evaluator
/// function and a list of candidate functions and should perform all necessary
/// JIT compilation on those functions, in preparation for search. The function
/// should return a search execute function, which the elaborator then
/// guarantees executes in isolation.
using EvaluatorExecutorFn = std::function<ErrorOr<ElaboratorSearchFn>(
    FuncOp, const SymbolTable &, TargetInfoAttr, ArrayRef<FuncOp>)>;
using EvaluatorExecutorFnRef = function_ref<ErrorOr<ElaboratorSearchFn>(
    FuncOp, const SymbolTable &, TargetInfoAttr, ArrayRef<FuncOp>)>;

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
