//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides various utilities for configuring and compiling Mojo.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
#define KGEN_TOOLS_MOJO_COMMON_COMPILATION_H

#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"
#include "llvm/Option/ArgList.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class PassManager;
class TimingScope;
} // namespace mlir

namespace M {
namespace AsyncRT {
class Runtime;
} // namespace AsyncRT

namespace KGEN::LIT {
struct ParserConfig;
} // namespace KGEN::LIT

class TargetInfoAttr;

/// Parse the common configuration options for Mojo related to compilation,
/// populating the provided `compilationOptions` argument. An error is returned
/// if any of the provided option values are invalid.
ErrorOrSuccess
parseCompilationOptions(const State &state, const llvm::opt::InputArgList &args,
                        KGEN::CompilationOptions &compilationOptions,
                        llvm::SourceMgr &sourceMgr, MLIRContext &ctx,
                        llvm::opt::OptSpecifier includeDirsId,
                        llvm::opt::OptSpecifier noOptimizationId = {},
                        llvm::opt::OptSpecifier debugLevelId = {},
                        llvm::opt::OptSpecifier sanitizeId = {},
                        llvm::opt::OptSpecifier debugInfoLanguageId = {});

/// Parse the common configuration options for Mojo related to target info,
/// populating the provided `compilationOptions` argument. On success, `target`
/// is populated with the selected compilation target.
ErrorOrSuccess parseTargetOptions(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, TargetInfoAttr &target, llvm::opt::OptSpecifier tripleId,
    llvm::opt::OptSpecifier cpuId, llvm::opt::OptSpecifier featuresId,
    llvm::opt::OptSpecifier marchId, llvm::opt::OptSpecifier mcpuId,
    llvm::opt::OptSpecifier mtuneId);

/// Wrap a parser invocation to Mojo, populating the necessary parsing context,
/// and attaching post parse metadata. On success, returns the parsed module
/// operation.
ErrorOr<OwningOpRef<ModuleOp>> invokeMojoParser(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, MLIRContext *ctx,
    AsyncRT::Runtime &runtime, llvm::opt::OptSpecifier docDiagnoseMissingId,
    llvm::opt::OptSpecifier docErrorOnInvalidDocId,
    llvm::opt::OptSpecifier maxNotesId, llvm::opt::OptSpecifier definesId,
    function_ref<OwningOpRef<ModuleOp>(KGEN::LIT::ParserConfig &,
                                       mlir::TimingScope &)>
        parseFn);
} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
