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
class MDialect;

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
class PackageLinkOp;
class SymbolConstantAttr;

namespace POP {
class POPDialect;
} // namespace POP

namespace Custom {
class CustomDialect;
} // namespace Custom

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

/// Register all passes with default options.
void registerDefaultKGENPasses();

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

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

/// Create an instance of the elaborator pass that captures all of the
/// referenced include files.
std::unique_ptr<mlir::Pass>
createElaborateGenerators(TargetInfoAttr target,
                          const ElaborateGeneratorsOptions &options = {},
                          ElaboratorCompileAsmFn compileAsmFn = {});

//===----------------------------------------------------------------------===//
// Inlining
//===----------------------------------------------------------------------===//

/// Create an AutomaticInline pass with a function pass pipeline to run.
std::unique_ptr<mlir::Pass> createAutomaticInline(
    const AutomaticInlineOptions &options = {},
    std::function<void(mlir::OpPassManager &)> buildFuncPasses = {});

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

/// Register the lower to LLVM pipeline.
void registerLowerToLLVMPipeline();

//===----------------------------------------------------------------------===//
// MaterializePackages
//===----------------------------------------------------------------------===//

/// When materializing packages, `kgen.package_link` ops that link to `.mojopkg`
/// packages may appear in the module. These linked packages may only contain
/// post-parse MLIR bytecode. In that case, this callback is invoked. The
/// callback is expected to return a buffer containing the MLIR bytecode that
/// the `materialize-packages` pass will load into the module that is importing
/// the package (i.e.: the module that contains the `kgen.package_link` op).
using PackageGenLibraryFn = std::function<ErrorOr<BufferRef>(PackageLinkOp)>;

/// Create a MaterializePackages pass with the specified behavior.
std::unique_ptr<mlir::Pass>
createMaterializePackages(PackageGenLibraryFn packageGenLibraryFn);

} // namespace KGEN
} // namespace M

#endif // KGEN_TOOLCOMMON_KGENPASSES_H
