//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_KGENPASSES_H
#define KGEN_TOOLCOMMON_KGENPASSES_H

#include "KGEN/KGENDialect/KGENEnums.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Buffer.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
class ModuleOp;
class OpPassManager;
namespace index {
class IndexDialect;
} // namespace index
namespace LLVM {
class LLVMDialect;
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace M {
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
class PackageArchiveAttr;
class PackageLinkOp;
class SymbolConstantAttr;

namespace POP {
class POPDialect;
} // namespace POP

//===----------------------------------------------------------------------===//
// Shared Enums
//===----------------------------------------------------------------------===//

/// Policy for when to update debug info while inlining.
///
/// When compiling with debug info, an op's location is tagged with a source
/// scope (e.g. function scope). This needs to be updated when the op is inlined
/// into another function (with a different scope).
enum class InlinerDebugInfoUpdateTime {
  // Update debug info for each inlined function immediately after it is
  // inlined. More costly, but allows for more optimization while inlining.
  kImmediate,
  // Update debug info after all inlining is done. This requires tagging
  // each inlined scope with an attribute during inlining, and looking for the
  // tag at the end to update debug scopes.
  kDeferred,
  // Never update debug info. This is only legal when compiling without any
  // debug info.
  kNever
};

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "KGEN/KGENPasses.h.inc"

/// Register all passes with default options and the provided runtime.
void registerDefaultKGENPasses(LLCL::Runtime &runtime);

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

/// This struct represents the result of a cross-device compilation, which is a
/// function or closure reference.
struct CrossDeviceFunction {
  /// The compiled cross-device function contents, which can be assembly, object
  /// code, or something else.
  StringAttr contents;
  /// The number of captures that need to be propagated to the cross-device
  /// closure.
  unsigned numCaptures;
  /// A function for populating the captured values opaquely.
  /// FIXME(#22670): The expected API is tightly bound with the GPU module in
  /// the standard library.
  OwningOpRef<Operation *> populateCapturesFn;
};

/// Function to slice and compile the generator to assembly with the provided
/// input parameters and target. The expected mangled name of the generate is
/// passed to be used as the entry point.
using ElaboratorCompileAsmFn = std::function<ErrorOr<CrossDeviceFunction>(
    GeneratorOp, SymbolConstantAttr, StringAttr, const SymbolTable &,
    TargetInfoAttr, EmissionKind)>;

/// During module elaboration, `kgen.package_link` ops that link to `.mojopkg`
/// packages may appear in the module. These linked packages may only contain
/// post-parser MLIR bytecode for the target being built. In that case, this
/// callback is invoked. The callback is expected to return an attribute
/// containing the MLIR bytecode that the `materialize-packages` pass will load
/// into the module that is importing the package (i.e.: the module that
/// contains the `kgen.package_link` op). The function can return a null
/// attribute to indicate that compilation should proceed with no precompiled
/// reference.
using PackageLinkHandlerFn =
    std::function<ErrorOr<PackageArchiveAttr>(PackageLinkOp, TargetInfoAttr)>;

/// Create an instance of the elaborator pass that captures all of the
/// referenced include files.
std::unique_ptr<mlir::Pass>
createElaborateGenerators(LLCL::Runtime &runtime, TargetInfoAttr target,
                          const ElaborateGeneratorsOptions &options = {},
                          EvaluatorExecutorFn evaluatorExecutorFn = {},
                          ElaboratorCompileAsmFn compileAsmFn = {},
                          PackageLinkHandlerFn packageHandlerFn = {});

//===----------------------------------------------------------------------===//
// Inlining
//===----------------------------------------------------------------------===//

/// Create a ForceInline pass with an LLCL runtime.
std::unique_ptr<mlir::Pass>
createInlineParametric(LLCL::Runtime &runtime,
                       const InlineParametricOptions &options = {});

/// Create a ForceInline pass with an LLCL runtime instance and a
/// function pass pipeline to run.
std::unique_ptr<mlir::Pass> createForceInline(
    LLCL::Runtime &runtime, const ForceInlineOptions &options = {},
    std::function<void(mlir::OpPassManager &)> buildFuncPasses = {});

/// Create an AutomaticInline pass with an LLCL runtime instance and a
/// function pass pipeline to run.
std::unique_ptr<mlir::Pass> createAutomaticInline(
    LLCL::Runtime &runtime, const AutomaticInlineOptions &options = {},
    std::function<void(mlir::OpPassManager &)> buildFuncPasses = {});

/// Create a ResolveCompilerPromises pass with an LLCL runtime.
std::unique_ptr<mlir::Pass>
createResolveCompilerPromises(LLCL::Runtime &runtime);

/// Create a DeadArgumentElimination pass with an LLCL runtime.
std::unique_ptr<mlir::Pass>
createDeadArgumentElimination(LLCL::Runtime &runtime);

//===----------------------------------------------------------------------===//
// LowerToLLVMPipeline
//===----------------------------------------------------------------------===//

/// Options for the KGEN to LLVM pipeline.
struct LowerToLLVMOptions
    : public mlir::PassPipelineOptions<LowerToLLVMOptions> {
  LowerToLLVMOptions(
      DebugInfo::EmissionKind diLevel = DebugInfo::EmissionKind::None,
      std::optional<CompilationOptions::DebugAtLevel> diAtLevel = std::nullopt,
      llvm::dwarf::SourceLanguage diLanguage = llvm::dwarf::DW_LANG_Mojo) {
    debugInfoLevel = diLevel;
    if (diAtLevel)
      debugAtLevel = *diAtLevel;
    debugInfoLanguage = diLanguage;
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

  Option<llvm::dwarf::SourceLanguage> debugInfoLanguage{
      *this, "debug-info-language",
      llvm::cl::desc("The DWARF language to specify in the debug info. "
                     "Either `C` or `Mojo`. Defaults to `Mojo`."),
      llvm::cl::values(
          clEnumValN(llvm::dwarf::DW_LANG_C, "C", "C language."),
          clEnumValN(llvm::dwarf::DW_LANG_Mojo, "Mojo", "Mojo language")),
      llvm::cl::init(llvm::dwarf::DW_LANG_Mojo)};

  Option<std::string> alignedAllocFnName{
      *this, "aligned-alloc-fn-name",
      llvm::cl::desc("The name of the aligned allocator function"),
      llvm::cl::init("kgenAlignedAlloc")};

  Option<std::string> alignedFreeFnName{
      *this, "aligned-free-fn-name",
      llvm::cl::desc("The name of the aligned free function"),
      llvm::cl::init("kgenAlignedFree")};

  Option<std::string> globalCtorFnName{
      *this, "global-ctor-fn-name",
      llvm::cl::desc("The name of the global init function in JIT mode."),
      llvm::cl::init("kgenGlobalCtor")};

  Option<std::string> globalDtorFnName{
      *this, "global-dtor-fn-name",
      llvm::cl::desc("The name of the global deinit function in JIT mode"),
      llvm::cl::init("kgenGlobalDtor")};
};

/// Build the pass pipeline to convert post-elaboration KGEN IR to LLVM IR.
/// The pipeline runs the canonicalizer, the KGEN to LLVM conversion, a series
/// of LLVM lowerings, and the canonicalizer again.
void buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                              const LowerToLLVMOptions &options);

/// Register the lower to LLVM pipeline.
void registerLowerToLLVMPipeline();

//===----------------------------------------------------------------------===//
// MaterializePackages
//===----------------------------------------------------------------------===//

/// When materializing packages, `kgen.package_link` ops that link to `.mojopkg`
/// packages may appear in the module. These linked packages may only contain
/// post-parse MLIR bytecode. In that case, this callback is invoked. The
/// callback is expected to return an attribute containing the MLIR bytecode
/// that the `materialize-packages` pass will load into the module that is
/// importing the package (i.e.: the module that contains the
/// `kgen.package_link` op).
using PackageGenLibraryFn =
    std::function<ErrorOr<DenseResourceElementsAttr>(PackageLinkOp)>;

/// Create a MaterializePackages pass with the specified behavior.
std::unique_ptr<mlir::Pass>
createMaterializePackages(PackageGenLibraryFn packageGenLibraryFn = nullptr);

//===----------------------------------------------------------------------===//
// CHECKLITPipeline
//===----------------------------------------------------------------------===//

/// This populates the post-parser pipeline that checks and lowers source-level
/// LIT constructs.
void buildCheckLITPipeline(mlir::PassManager &pm, LLCL::Runtime &runtime,
                           const CompilationOptions &options);

//===----------------------------------------------------------------------===//
// GenerateLibraryPipeline
//===----------------------------------------------------------------------===//

/// This populates the pre-elaboration phase passes of the KGEN compiler. The
/// distribution format of a KGEN library is essentially what comes just before
/// elaboration because the parameter system allows significant extension.
void buildGenerateLibraryPipeline(mlir::PassManager &pm, LLCL::Runtime &runtime,
                                  const CompilationOptions &options);

//===----------------------------------------------------------------------===//
// ElaborateModulePipeline
//===----------------------------------------------------------------------===//

/// This populates the passes to produce a fully concrete KGEN module. That
/// means it runs the elaborator and any dependent passes.
void buildElaborateModulePipeline(mlir::PassManager &pm, LLCL::Runtime &runtime,
                                  TargetInfoAttr target,
                                  const CompilationOptions &options,
                                  EvaluatorExecutorFn evaluatorExecutorFn,
                                  ElaboratorCompileAsmFn compileAsmFn,
                                  PackageGenLibraryFn packageGenLibraryFn,
                                  PackageLinkHandlerFn packageLinkHandlerFn);

//===----------------------------------------------------------------------===//
// PostElaborationPipeline
//===----------------------------------------------------------------------===//

/// This populates the post-elaboration optimization and simplification passes.
/// These passes are intended to run immediately after the elaborator.
void buildPostElaborationPipeline(mlir::PassManager &pm, LLCL::Runtime &runtime,
                                  const CompilationOptions &options);

} // namespace KGEN
} // namespace M

#endif // KGEN_TOOLCOMMON_KGENPASSES_H
