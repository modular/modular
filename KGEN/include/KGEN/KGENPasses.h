//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPASSES_H
#define KGEN_KGENPASSES_H

#include "CompilationOptions.h"
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

namespace M::HLCF {
class HLCFDialect;
} // namespace M::HLCF

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN {
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
createElaborateGenerators(LLCL::Runtime &runtime,
                          SmallVectorImpl<std::string> &includedFiles,
                          const ElaborateGeneratorsOptions &options = {});

//===----------------------------------------------------------------------===//
// ResolveIncludes
//===----------------------------------------------------------------------===//

/// Create an instance of the elaborator pass that captures all of the
/// referenced include files.
std::unique_ptr<mlir::Pass>
createResolveIncludes(SmallVectorImpl<std::string> &includedFiles,
                      const ResolveIncludesOptions &options = {});

//===----------------------------------------------------------------------===//
// Inlining Utils
//===----------------------------------------------------------------------===//

/// Inline a parametric call to a parametric function. All operations in the
/// generator are inlined at the callsite. The results of the call are replaced
/// with the results of the generator. The generator may have multiple return
/// sites, in which case the body of the generator is wrapped in a labelled
/// `hlcf.loop` and all return sites rewritten to `hlcf.break` to that loop.
/// Result parameters are replaced with new parameter declarations. All nested
/// parameters are mangled to avoid collisions in the current scope.
///
/// This function expects the call to be located inside a generator. The
/// original generator is not modified. The operations are cloned and inserted
/// at the callsite. The call is replaced by the results of the callee
void inlineGeneratorCall(KGENCallOpInterface call, GeneratorOp callee);

} // namespace M::KGEN

#endif // KGEN_KGENPASSES_H
