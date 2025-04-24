//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Compilation.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

ErrorOrSuccess M::parseCompilationOptions(
    const State &state, const llvm::opt::InputArgList &args,
    CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, llvm::opt::OptSpecifier includeDirsId,
    llvm::opt::OptSpecifier optimizationLevelId,
    llvm::opt::OptSpecifier debugLevelId, llvm::opt::OptSpecifier sanitizeId,
    llvm::opt::OptSpecifier sharedLibasan,
    llvm::opt::OptSpecifier externalLibasan,
    llvm::opt::OptSpecifier debugInfoLanguageId,
    llvm::opt::OptSpecifier numThreadsId, llvm::opt::OptSpecifier stdLibPath) {
  // Process the sanitizers.
  if (sanitizeId.isValid()) {
    StringRef sanitizer = args.getLastArgValue(sanitizeId);
    if (args.hasMultipleArgs(sanitizeId))
      return Error("too many specified sanitizers, expected exactly one");
    if (!sanitizer.empty()) {
      if (!llvm::is_contained({"address", "thread"}, sanitizer)) {
        return Error("invalid sanitizer '" + sanitizer +
                     "', expected one of: `address` or `thread`");
      }
      if (sanitizer == "address")
        compilationOptions.sanitizers.enable(Sanitizers::kAddress);
      else if (sanitizer == "thread")
        compilationOptions.sanitizers.enable(Sanitizers::kThread);
    }
  }

  if (sharedLibasan.isValid() && args.hasArg(sharedLibasan)) {
    if (!compilationOptions.sanitizers.has(Sanitizers::kAddress))
      return Error(
          "cannot enable --shared-libasan without enabling --sanitize=address");
    compilationOptions.sharedLibasan = true;
  }

  if (externalLibasan.isValid()) {
    StringRef libPath = args.getLastArgValue(externalLibasan);
    if (args.hasMultipleArgs(externalLibasan))
      return Error("too many external libasan paths, expected exactly one");
    if (!libPath.empty()) {
      if (compilationOptions.sharedLibasan)
        return Error("--external-libasan cannot be used with --shared-libasan");
      if (!compilationOptions.sanitizers.has(Sanitizers::kAddress))
        return Error("cannot use --external-libasan without enabling "
                     "--sanitize=address");
      compilationOptions.externalLibasan = libPath;
    }
  }

  // Enable overwritting of the auto-imported paths, which is where the compiler
  // first looks for builtins.
  if (stdLibPath.isValid()) {
    StringRef value = args.getLastArgValue(stdLibPath);
    compilationOptions.searchPaths = value;
  }

  // Process the debug info language.
  if (debugInfoLanguageId.isValid()) {
    StringRef debugInfoLanguage = args.getLastArgValue(debugInfoLanguageId);
    if (args.hasMultipleArgs(debugInfoLanguageId))
      return Error(
          "too many specified debug info languages, expected exactly one");
    if (!debugInfoLanguage.empty()) {
      if (!llvm::is_contained({"C", "Mojo"}, debugInfoLanguage)) {
        return Error("invalid debug info language '" + debugInfoLanguage +
                     "', expected one of: `C` or `Mojo`");
      }
      if (debugInfoLanguage == "C")
        compilationOptions.debugInfoLanguage = CompilationOptions::kLangC;
    }
  }

  // Set up the optimization level.
  if (optimizationLevelId.isValid()) {
    StringRef levelStr = args.getLastArgValue(optimizationLevelId, "3");
    int level = -1;
    if (levelStr.size() == 1)
      level = levelStr[0] - '0';
    if (0 <= level && level <= 3) {
      compilationOptions.optimizationLevel = level;
    } else {
      return Error(llvm::formatv("invalid optimization level '{0}', expected "
                                 "number from 0-3 inclusive",
                                 levelStr));
    }
  }

  // Setup the debug level.
  if (debugLevelId.isValid()) {
    StringLiteral kDebugLevelNone = "none";
    StringLiteral kDebugLevelLineTables = "line-tables";
    StringLiteral kDebugLevelFull = "full";
    StringRef level = args.getLastArgValue(debugLevelId, kDebugLevelNone);
    if (!llvm::is_contained(
            {kDebugLevelNone, kDebugLevelLineTables, kDebugLevelFull}, level)) {
      return Error(llvm::formatv("invalid debug level '{0}', expected one of: "
                                 "`{1}` (the default value), `{2}`, or `{3}`",
                                 level, kDebugLevelNone, kDebugLevelLineTables,
                                 kDebugLevelFull));
    }
    compilationOptions.debugLevel =
        llvm::StringSwitch<CompilationOptions::DebugInfoLevel>(level)
            .Case(kDebugLevelNone, CompilationOptions::kNoDebug)
            .Case(kDebugLevelLineTables, CompilationOptions::kLineTablesOnly)
            .Case(kDebugLevelFull, CompilationOptions::kFullDebugInfo);
  }
  if (numThreadsId.isValid()) {
    if (args.hasMultipleArgs(numThreadsId)) {
      return Error("Number of threads can only be specified once");
    }
    StringRef numThreadsStr = args.getLastArgValue(numThreadsId);
    if (!numThreadsStr.empty()) {
      // Next line has a side effect of setting compilationOptions.numThreads
      if (!llvm::to_integer(numThreadsStr, compilationOptions.numThreads))
        return Error("Invalid number of threads: " + numThreadsStr.str());
    }
  }
  sourceMgr.setIncludeDirs(args.getAllArgValues(includeDirsId));

  // Initialize the MLIR context.
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registerKGENToLLVMTranslation(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadDialect<MDialect>();

  return success();
}

void M::warnBuildingForDebugWithDebugBuiltCompiler(
    const State &state,
    KGEN::CompilationOptions::DebugInfoLevel debugInfoLevel) {
#ifdef MODULAR_DEBUG
  if (debugInfoLevel == M::KGEN::CompilationOptions::kFullDebugInfo)
    state.reportWarning(
        "Performing a debug build with the Mojo compiler built in debug mode:\n"
        "It is not necessary to use a debug build of the Mojo compiler to "
        "produce debuggable Mojo programs.\n"
        "You can safely use a release build of the compiler for this, which "
        "will result in faster build times.\n");
#endif
}

ErrorOrSuccess M::parseTargetOptions(
    const State &state, const llvm::opt::InputArgList &args,
    CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, TargetInfoAttr &target, llvm::opt::OptSpecifier tripleId,
    llvm::opt::OptSpecifier cpuId, llvm::opt::OptSpecifier featuresId,
    llvm::opt::OptSpecifier marchId, llvm::opt::OptSpecifier mcpuId,
    llvm::opt::OptSpecifier mtuneId,
    llvm::opt::OptSpecifier targetAcceleratorId,
    llvm::opt::OptSpecifier mcmodelId,
    llvm::opt::OptSpecifier largeDataThresholdId,
    llvm::opt::OptSpecifier loopUnrollingWarnThresholdId) {
  StringRef targetTriple = args.getLastArgValue(tripleId);
  if (args.hasMultipleArgs(tripleId))
    return Error("too many specified target triples, expected exactly one");

  StringRef targetCpu = args.getLastArgValue(cpuId);
  if (args.hasMultipleArgs(cpuId))
    return Error("too many specified target CPUs, expected exactly one");

  StringRef targetFeatures = args.getLastArgValue(featuresId);
  if (args.hasMultipleArgs(featuresId))
    return Error("too many specified target features, expected exactly one");

  StringRef mArch = args.getLastArgValue(marchId);
  if (args.hasMultipleArgs(marchId))
    return Error(
        "too many specified target architectures, expected exactly one");

  StringRef mCpu = args.getLastArgValue(mcpuId);
  if (args.hasMultipleArgs(mcpuId))
    return Error("too many specified target cpus, expected exactly one");

  StringRef mTune = args.getLastArgValue(mtuneId);
  if (args.hasMultipleArgs(mtuneId))
    return Error("too many specified tune cpus, expected exactly one");

  StringRef targetAccelerator = args.getLastArgValue(targetAcceleratorId);
  if (args.hasMultipleArgs(targetAcceleratorId))
    return Error(
        "too many specified target accelerators, expected exactly one");

  StringRef mcmodel = args.getLastArgValue(mcmodelId);
  if (args.hasMultipleArgs(mcmodelId))
    return Error("too many specified CodeModel, expected exactly one");

  StringRef largeDataThreshold = args.getLastArgValue(largeDataThresholdId);
  if (args.hasMultipleArgs(largeDataThresholdId))
    return Error(
        "too many specified large data threshold, expected exactly one");

  StringRef loopUnrollingWarnThreshold =
      args.getLastArgValue(loopUnrollingWarnThresholdId);
  if (args.hasMultipleArgs(loopUnrollingWarnThresholdId))
    return Error("too many specified loop unroll factor threshold, expected "
                 "exactly one");

  // If the user specified the triple, the target CPU, or the target feature
  // set, use those to override the defaults.
  if (!targetTriple.empty())
    compilationOptions.targetTriple = targetTriple.str();
  if (!targetCpu.empty())
    compilationOptions.targetCpu = targetCpu.str();
  if (!targetFeatures.empty())
    compilationOptions.targetFeatures = targetFeatures.str();
  if (!targetAccelerator.empty()) {
    compilationOptions.targetAccelerator = targetAccelerator.str();
    compilationOptions.isCrossCompilation = true;
  }

  if (!mcmodel.empty()) {
    if (!llvm::is_contained({"small", "medium", "large"}, mcmodel)) {
      return Error("invalid mcmodel'" + mcmodel +
                   "', expected one of: `small`, `medium` or `large`");
    }
    if (mcmodel == "small")
      compilationOptions.mcmodel = llvm::CodeModel::Small;
    else if (mcmodel == "medium")
      compilationOptions.mcmodel = llvm::CodeModel::Medium;
    else if (mcmodel == "large")
      compilationOptions.mcmodel = llvm::CodeModel::Large;
  }

  if (!largeDataThreshold.empty()) {
    uint64_t value;
    if (!llvm::to_integer(largeDataThreshold, value)) {
      return Error("invalid large-data-threshold'" + largeDataThreshold +
                   "', expected a positive integer number");
    }
    compilationOptions.largeDataThreshold = value;
  }

  if (!loopUnrollingWarnThreshold.empty()) {
    uint64_t value;
    if (!llvm::to_integer(loopUnrollingWarnThreshold, value)) {
      return Error("invalid loop-unrolling-warn-threshold'" +
                   loopUnrollingWarnThreshold +
                   "', expected an integer number");
    }
    compilationOptions.loopUnrollingWarnThreshold = value;
  }

  // Initialize targets first - we rely on this for getTargetInfo as well as for
  // the ExecutionEngine.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Construct a target specification using the command line options.
  ErrorOr<TargetInfoAttr> targetOr = nullptr;
  if (!mArch.empty() || !mCpu.empty()) {
    // Use `-march` to determine the feature set.
    targetOr = getMArchFeatures(&ctx, mArch, mCpu, mTune);
  } else {
    // Use the full triple, specific CPU, and manually specified features to
    // get the target info.
    targetOr = getTargetInfoFor(
        &ctx, compilationOptions.targetTriple, compilationOptions.targetCpu,
        compilationOptions.targetFeatures, /*tuneCpu=*/"",
        /*acceleratorArch=*/compilationOptions.targetAccelerator);
  }
  if (targetOr.isError())
    return targetOr.takeError();
  target = targetOr.takeValue();

  return success();
}

ErrorOr<OwningOpRef<ModuleOp>> M::invokeMojoParser(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, MLIRContext *ctx,
    AsyncRT::Runtime &runtime, llvm::opt::OptSpecifier docDiagnoseMissingId,
    llvm::opt::OptSpecifier docErrorOnInvalidDocId,
    llvm::opt::OptSpecifier maxNotesId, llvm::opt::OptSpecifier definesId,
    llvm::opt::OptSpecifier stripFilePrefixId,
    llvm::opt::OptSpecifier disableBuiltins, llvm::opt::OptSpecifier stdLibPath,
    function_ref<OwningOpRef<ModuleOp>(ParserConfig &, mlir::TimingScope &)>
        parseFn) {
  // We don't allow users to configure the time profiler.
  mlir::DefaultTimingManager timingManager;
  mlir::TimingScope timing = timingManager.getRootScope();

  DialectRegistry registry;
  registerAllKGENDialects(registry);
  ctx->appendDialectRegistry(registry);
  if (stdLibPath.isValid()) {
    StringRef value = args.getLastArgValue(stdLibPath);
    compilationOptions.searchPaths = value;
  }
  // Parse the input Mojo file into an MLIR module.
  ParserConfig parseConfig(ctx, compilationOptions);
  parseConfig.diagnoseMissingDocStrings = args.hasArg(docDiagnoseMissingId);
  parseConfig.errorOnInvalidDocStrings = args.hasArg(docErrorOnInvalidDocId);
  int maxNotes = 0;
  if (!args.getLastArgValue(maxNotesId).getAsInteger(10, maxNotes))
    parseConfig.maxNotesPerDiagnostic = maxNotes;
  parseConfig.stripFilePrefix = args.getLastArgValue(stripFilePrefixId);
  parseConfig.useBuiltinModule = !args.hasArg(disableBuiltins);

  mlir::TimingScope mojoScope = timing.nest("Import Mojo");
  OwningOpRef<ModuleOp> module = parseFn(parseConfig, mojoScope);
  if (!module)
    return Error("failed to parse the provided Mojo source module");

  // Tag the module with the environment, which includes any definitions the
  // user may have specified on the command line.
  ctx->loadDialect<KGENDialect>();
  if (definesId.isValid()) {
    ErrorOr<EnvAttr> envOrErr = compilationOptions.parseDefinesWithDefaults(
        ctx, args.getAllArgValues(definesId));
    if (failed(envOrErr)) {
      return Error(
          llvm::formatv("an internal error occurred when initializing the Mojo "
                        "MLIR module: {0}",
                        envOrErr.getError()));
    }
    (*module)->setAttr(EnvAttr::getEnvAttrName(), *envOrErr);
  }
  return module;
}

void M::configureRuntimeOptions(AsyncRT::RuntimeOptions &runtimeOptions,
                                const KGEN::CompilationOptions &options) {
  if (options.numThreads != 0) {
    // AsyncRT has a sophisticated thread pool configuration system.
    // There are two parameters: numThreads and maxThreads.
    // Setting numThreads to non-zero means "Use exactly this many threads"
    // Setting maxThreads to non-zero means numThreads=min(maxThreads,
    // heuristicBasedOnNumCores)
    //
    // Here we take the stance that mojo build -j 5 means "Use at most 5
    // threads"
    runtimeOptions.numThreads = 0;
    runtimeOptions.maxThreads = options.numThreads;
  }
}
