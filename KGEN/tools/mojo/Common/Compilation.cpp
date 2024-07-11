//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Compilation.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
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
    llvm::opt::OptSpecifier noOptimizationId,
    llvm::opt::OptSpecifier debugLevelId, llvm::opt::OptSpecifier sanitizeId,
    llvm::opt::OptSpecifier debugInfoLanguageId) {
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

  // Disabled optimizations.
  if (noOptimizationId.isValid() && args.hasArg(noOptimizationId))
    compilationOptions.optimizationLevel = 0;

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

  sourceMgr.setIncludeDirs(args.getAllArgValues(includeDirsId));

  // Initialize the MLIR context.
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registerKGENToLLVMTranslation(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadDialect<MDialect>();

  return success();
}

ErrorOrSuccess M::parseTargetOptions(
    const State &state, const llvm::opt::InputArgList &args,
    CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, TargetInfoAttr &target, llvm::opt::OptSpecifier tripleId,
    llvm::opt::OptSpecifier cpuId, llvm::opt::OptSpecifier featuresId,
    llvm::opt::OptSpecifier marchId, llvm::opt::OptSpecifier mcpuId,
    llvm::opt::OptSpecifier mtuneId) {
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

  // If the user specified the triple, the target CPU, or the target feature
  // set, use those to override the defaults.
  if (!targetTriple.empty())
    compilationOptions.targetTriple = targetTriple.str();
  if (!targetCpu.empty())
    compilationOptions.targetCpu = targetCpu.str();
  if (!targetFeatures.empty())
    compilationOptions.targetFeatures = targetFeatures.str();

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
    targetOr = getTargetInfoFor(&ctx, compilationOptions.targetTriple,
                                compilationOptions.targetCpu,
                                compilationOptions.targetFeatures);
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
    function_ref<OwningOpRef<ModuleOp>(ParserConfig &, mlir::TimingScope &)>
        parseFn) {
  // We don't allow users to configure the time profiler.
  mlir::DefaultTimingManager timingManager;
  mlir::TimingScope timing = timingManager.getRootScope();

  DialectRegistry registry;
  registerAllKGENDialects(registry);
  ctx->appendDialectRegistry(registry);

  // Parse the input Mojo file into an MLIR module.
  ParserConfig parseConfig(ctx, compilationOptions);
  parseConfig.diagnoseMissingDocStrings = args.hasArg(docDiagnoseMissingId);
  parseConfig.errorOnInvalidDocStrings = args.hasArg(docErrorOnInvalidDocId);
  int maxNotes = 0;
  if (!args.getLastArgValue(maxNotesId).getAsInteger(10, maxNotes))
    parseConfig.maxNotesPerDiagnostic = maxNotes;

  mlir::TimingScope mojoScope = timing.nest("Import Mojo");
  OwningOpRef<ModuleOp> module = parseFn(parseConfig, mojoScope);
  if (!module)
    return Error("failed to parse the provided Mojo source module");

  // Tag the module with the environment, which includes any definitions the
  // user may have specified on the command line.
  ctx->loadDialect<KGENDialect>();
  if (definesId.isValid()) {
    ErrorOr<EnvAttr> envOrErr =
        EnvAttr::parseDefines(ctx, args.getAllArgValues(definesId));
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
