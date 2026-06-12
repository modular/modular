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
namespace MLRT {
class CPUDevice;
struct CPUDeviceOptions;
} // namespace MLRT

namespace KGEN::LIT {
struct ParserConfig;
} // namespace KGEN::LIT

class TargetInfoAttr;

/// Holds the option IDs that are common between mojo build and mojo run.
/// These are passed to parseCommonMojoArguments to avoid duplicating the
/// option ID mappings.
struct CommonOptionIDs {
  llvm::opt::OptSpecifier help;
  llvm::opt::OptSpecifier helpHidden;
  llvm::opt::OptSpecifier diagnosticFormat;
  llvm::opt::OptSpecifier disableWarnings;
  llvm::opt::OptSpecifier warningsAsErrors;
  llvm::opt::OptSpecifier noWarningsAsErrors;
  llvm::opt::OptSpecifier ignoreIncompatiblePrecompiledFileErrors;
  llvm::opt::OptSpecifier unknown;
  llvm::opt::OptSpecifier input;

  // Compilation options
  llvm::opt::OptSpecifier includeDirs;
  llvm::opt::OptSpecifier optimizationLevel;
  llvm::opt::OptSpecifier debugLevel;
  llvm::opt::OptSpecifier sanitize;
  llvm::opt::OptSpecifier sharedLibasan;
  llvm::opt::OptSpecifier externalLibasan;
  llvm::opt::OptSpecifier bitcodeLibs;
  llvm::opt::OptSpecifier debugInfoLanguage;
  llvm::opt::OptSpecifier numThreads;
  llvm::opt::OptSpecifier mojoSearchPaths;
  llvm::opt::OptSpecifier loopUnrollingWarnThreshold;
  llvm::opt::OptSpecifier elaborationErrorLimit;
  llvm::opt::OptSpecifier elaborationErrorIncludePrelude;
  llvm::opt::OptSpecifier elaborationErrorVerbose;
  llvm::opt::OptSpecifier elaborationMaxDepth;

  // Target options
  llvm::opt::OptSpecifier targetTriple;
  llvm::opt::OptSpecifier targetCpu;
  llvm::opt::OptSpecifier targetFeatures;
  llvm::opt::OptSpecifier march;
  llvm::opt::OptSpecifier mcpu;
  llvm::opt::OptSpecifier mtune;
  llvm::opt::OptSpecifier targetAccelerator;
  llvm::opt::OptSpecifier mcmodel;
  llvm::opt::OptSpecifier largeDataThreshold;
  llvm::opt::OptSpecifier relocationModel;

  // Parser options
  llvm::opt::OptSpecifier diagnoseMissingDocStrings;
  llvm::opt::OptSpecifier maxNotes;
  llvm::opt::OptSpecifier defines;
  llvm::opt::OptSpecifier stripFilePrefix;
  llvm::opt::OptSpecifier disableBuiltins;
  llvm::opt::OptSpecifier fixit;
  llvm::opt::OptSpecifier exportFixit;

  // Stability options
  llvm::opt::OptSpecifier warnOnUnstableAPIs;

  // Linker options
  llvm::opt::OptSpecifier lldPath;
};

/// Configuration flags for common argument parsing behavior.
struct CommonParseConfig {
  /// If true, parse all arguments normally. If false (for `mojo run`), only
  /// parse arguments up to and including the input file, treating remaining
  /// arguments as program arguments to pass to the Mojo executable.
  bool parseAllArguments = true;

  /// If true, require exactly one input file. If false, allow zero or more.
  bool requireSingleInput = true;
};

/// Result of parsing common Mojo arguments.
struct CommonParseResult {
  /// If set, the caller should exit immediately with this code.
  std::optional<int> exitCode;

  /// The parsed argument list. For `mojo run`, this includes only arguments
  /// up to and including the input file.
  llvm::opt::InputArgList args;

  /// The parsed compilation options.
  KGEN::CompilationOptions compilationOptions;

  /// The parsed target information.
  TargetInfoAttr target;
};

/// Parse arguments common to both mojo build and mojo run.
///
/// This function extracts the common argument parsing logic shared between
/// the two commands, including:
/// - Diagnostic format parsing
/// - Unknown argument rejection
/// - Input file validation and opening
/// - Source manager setup
/// - Compilation option parsing
/// - Target option parsing
///
/// Note: Help text handling is intentionally left to the caller, as each
/// command has different help text files that cannot be easily parameterized.
/// Callers should check for help flags before calling this function.
///
/// Returns a CommonParseResult containing either an exit code (if parsing
/// failed) or the parsed arguments and options.
ErrorOr<CommonParseResult> parseCommonMojoArguments(
    State &state, llvm::SourceMgr &sourceManager, MLIRContext &ctx,
    const llvm::opt::PrecomputedOptTable &optTable,
    const CommonOptionIDs &optionIDs, const CommonParseConfig &config);

/// Parse the common configuration options for Mojo related to compilation,
/// populating the provided `compilationOptions` argument. An error is returned
/// if any of the provided option values are invalid.
ErrorOrSuccess parseCompilationOptions(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, llvm::opt::OptSpecifier includeDirsId,
    llvm::opt::OptSpecifier optimizationLevelId = {},
    llvm::opt::OptSpecifier debugLevelId = {},
    llvm::opt::OptSpecifier sanitizeId = {},
    llvm::opt::OptSpecifier sharedLibasan = {},
    llvm::opt::OptSpecifier externalLibasan = {},
    llvm::opt::OptSpecifier bitcodeLibs = {},
    llvm::opt::OptSpecifier debugInfoLanguageId = {},
    llvm::opt::OptSpecifier numThreadsId = {},
    llvm::opt::OptSpecifier stdLibPath = {},
    llvm::opt::OptSpecifier loopUnrollingWarnThresholdId = {},
    llvm::opt::OptSpecifier elaborationErrorLimitId = {},
    llvm::opt::OptSpecifier elaborationErrorIncludePreludeId = {},
    llvm::opt::OptSpecifier elaborationErrorVerbose = {},
    llvm::opt::OptSpecifier elaborationMaxDepth = {},
    llvm::opt::OptSpecifier ignoreIncompatiblePrecompiledFilesId = {});

/// Warn users when doing debug builds with a compiler in debug mode.
void warnBuildingForDebugWithDebugBuiltCompiler(
    const State &state,
    KGEN::CompilationOptions::DebugInfoLevel debugInfoLevel);

/// Parse the common configuration options for Mojo related to target info,
/// populating the provided `compilationOptions` argument. On success, `target`
/// is populated with the selected compilation target.
ErrorOrSuccess parseTargetOptions(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, TargetInfoAttr &target, llvm::opt::OptSpecifier tripleId,
    llvm::opt::OptSpecifier cpuId, llvm::opt::OptSpecifier featuresId,
    llvm::opt::OptSpecifier marchId, llvm::opt::OptSpecifier mcpuId,
    llvm::opt::OptSpecifier mtuneId,
    llvm::opt::OptSpecifier targetAcceleratorId,
    llvm::opt::OptSpecifier mcmodelId,
    llvm::opt::OptSpecifier largeDataThresholdId,
    llvm::opt::OptSpecifier relocationModelId = {});

/// Wrap a parser invocation to Mojo, populating the necessary parsing context,
/// and attaching post parse metadata. On success, returns the parsed module
/// operation. If the `autoFixIt` flag is set and the parser collects any
/// fix-its, they will be applied, and the returned module will be null.
/// If `exportFixit` is set, fix-its will be exported to a YAML file instead.
ErrorOr<OwningOpRef<ModuleOp>> invokeMojoParser(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, MLIRContext *ctx,
    MLRT::CPUDevice &cpuDevice, llvm::opt::OptSpecifier docDiagnoseMissingId,
    llvm::opt::OptSpecifier maxNotesId, llvm::opt::OptSpecifier definesId,
    llvm::opt::OptSpecifier stripFilePrefixId,
    llvm::opt::OptSpecifier disableBuiltins, llvm::opt::OptSpecifier stdlibPath,
    llvm::opt::OptSpecifier autoFixIt, llvm::opt::OptSpecifier exportFixit,
    function_ref<OwningOpRef<ModuleOp>(KGEN::LIT::ParserConfig &,
                                       mlir::TimingScope &)>
        parseFn);

/// Configure cpuDevice options based on compilation options.
/// Currently handles thread pool configuration based on numThreads.
void configureCPUDeviceOptions(MLRT::CPUDeviceOptions &cpuDeviceOptions,
                               const KGEN::CompilationOptions &options);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
