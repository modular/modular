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

#include "KGEN/Compiler/ExecutionEngine.h"
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
namespace LLCL {
class Runtime;
} // namespace LLCL

namespace KGEN::LIT {
struct ParserConfig;
} // namespace KGEN::LIT

class TargetInfoAttr;

/// Parse the common compilation options for Mojo related to configuration,
/// populating the provided `compilationOptions` argument. On success, `target`
/// is populated with the selected compilation target.
ErrorOrSuccess parseCompilationOptions(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, TargetInfoAttr &target,
    llvm::opt::OptSpecifier includeDirsId, llvm::opt::OptSpecifier tripleId,
    llvm::opt::OptSpecifier cpuId, llvm::opt::OptSpecifier featuresId,
    llvm::opt::OptSpecifier marchId, llvm::opt::OptSpecifier mcpuId,
    llvm::opt::OptSpecifier mtuneId, llvm::opt::OptSpecifier noOptimizationId,
    llvm::opt::OptSpecifier debugLevelId, llvm::opt::OptSpecifier sanitizeId,
    llvm::opt::OptSpecifier debugInfoLanguageId,
    llvm::opt::OptSpecifier noAlnumSymbolsId);

/// Wrap a parser invocation to Mojo, populating the necessary parsing context,
/// and attaching post parse metadata. On success, returns the parsed module
/// operation.
ErrorOr<OwningOpRef<ModuleOp>> invokeMojoParser(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, MLIRContext *ctx,
    LLCL::Runtime &runtime, llvm::opt::OptSpecifier docWarnMissingId,
    llvm::opt::OptSpecifier maxNotesId, llvm::opt::OptSpecifier definesId,
    llvm::opt::OptSpecifier parsingStdlibId,
    function_ref<OwningOpRef<ModuleOp>(KGEN::LIT::ParserConfig &,
                                       mlir::TimingScope &)>
        parseFn);
} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
