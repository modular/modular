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

#include "KGEN/CompilationOptions.h"
#include "KGEN/ExecutionEngine.h"
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

struct MojoParserConfig;
class TargetInfoAttr;

/// Parse the common compilation options for Mojo related to configuration,
/// populating the provided `compilationOptions` argument.
ErrorOrSuccess parseCompilationOptions(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    llvm::opt::OptSpecifier includeDirsId, llvm::opt::OptSpecifier linkDirsId,
    llvm::opt::OptSpecifier tripleId, llvm::opt::OptSpecifier cpuId,
    llvm::opt::OptSpecifier featuresId,
    llvm::opt::OptSpecifier noOptimizationId,
    llvm::opt::OptSpecifier debugLevelId);

/// Wrap a parser invocation to Mojo, populating the necessary parsing context,
/// and attaching post parse metadata. On success, returns the parsed module
/// operation.
ErrorOr<OwningOpRef<ModuleOp>> invokeMojoParser(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, MLIRContext *ctx,
    LLCL::Runtime &runtime, llvm::opt::OptSpecifier docValidateId,
    llvm::opt::OptSpecifier maxNotesId, llvm::opt::OptSpecifier definesId,
    function_ref<OwningOpRef<ModuleOp>(MojoParserConfig &, mlir::TimingScope &)>
        parseFn);

/// Sets up an ExecutionEngine instance for compiling Mojo. It handles
/// initializing the LLVM MC targets, the target machine, the cache backends,
/// and the execution engine itself. On success, the execution engine is
/// returned, and the used target is returned in `target`.
ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>>
initializeExecutionEngine(LLCL::Runtime &runtime, mlir::PassManager &pm,
                          const KGEN::CompilationOptions &compilationOptions,
                          KGEN::ExecutionEngineOptions executionEngineOptions,
                          bool isJIT, TargetInfoAttr &target);
} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
