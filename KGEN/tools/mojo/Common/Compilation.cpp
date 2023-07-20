//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Compilation.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/MojoParser.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace M::KGEN;

ErrorOrSuccess M::parseCompilationOptions(
    const State &state, const llvm::opt::InputArgList &args,
    CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, TargetInfoAttr &target,
    llvm::opt::OptSpecifier includeDirsId, llvm::opt::OptSpecifier linkDirsId,
    llvm::opt::OptSpecifier tripleId, llvm::opt::OptSpecifier cpuId,
    llvm::opt::OptSpecifier featuresId, llvm::opt::OptSpecifier marchId,
    llvm::opt::OptSpecifier mcpuId, llvm::opt::OptSpecifier mtuneId,
    llvm::opt::OptSpecifier noOptimizationId,
    llvm::opt::OptSpecifier debugLevelId) {
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

  // Disabled optimizations.
  if (args.hasArg(noOptimizationId))
    compilationOptions.optimizationLevel = 0;

  // Setup the debug level.
  StringRef level = args.getLastArgValue(debugLevelId, "none");
  if (!llvm::is_contained({"none", "line", "full"}, level)) {
    return Error("invalid debug level '" + level +
                 "', expected one of: `none` (the default value), "
                 "`line-tables`, or `full`");
  }
  compilationOptions.debugLevel =
      llvm::StringSwitch<CompilationOptions::DebugInfoLevel>(level)
          .Case("none", CompilationOptions::kNoDebug)
          .Case("line-tables", CompilationOptions::kLineTablesOnly)
          .Case("full", CompilationOptions::kFullDebugInfo);

  sourceMgr.setIncludeDirs(args.getAllArgValues(includeDirsId));
  compilationOptions.linkDirs = args.getAllArgValues(linkDirsId);

  // Initialize the MLIR context.
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadDialect<MDialect>();

  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx.allowUnregisteredDialects();

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
    LLCL::Runtime &runtime, llvm::opt::OptSpecifier docValidateId,
    llvm::opt::OptSpecifier maxNotesId, llvm::opt::OptSpecifier definesId,
    function_ref<OwningOpRef<ModuleOp>(MojoParserConfig &, mlir::TimingScope &)>
        parseFn) {
  // We don't allow users to configure the time profiler.
  mlir::DefaultTimingManager timingManager;
  mlir::TimingScope timing = timingManager.getRootScope();

  // Parse the input Mojo file into an MLIR module.
  MojoParserConfig parseConfig(ctx, runtime, compilationOptions);
  parseConfig.validateDocStrings = args.hasArg(docValidateId);
  int maxNotes = 0;
  if (!args.getLastArgValue(maxNotesId).getAsInteger(10, maxNotes))
    parseConfig.maxNotesPerDiagnostic = maxNotes;

  mlir::TimingScope mojoScope = timing.nest("Import Mojo");
  OwningOpRef<ModuleOp> module = parseFn(parseConfig, mojoScope);
  if (!module)
    return Error("failed to parse the provided Mojo");

  // Tag the module with the environment, which includes any definitions the
  // user may have specified on the command line.
  ctx->loadDialect<KGENDialect>();
  ErrorOr<EnvAttr> envOrErr =
      EnvAttr::parseDefines(ctx, args.getAllArgValues(definesId));
  if (failed(envOrErr)) {
    return Error(
        llvm::formatv("an internal error occurred when initializing the Mojo "
                      "MLIR module: {0}",
                      envOrErr.getError()));
  }
  (*module)->setAttr(EnvAttr::getEnvAttrName(), *envOrErr);
  return module;
}

ErrorOr<std::unique_ptr<ExecutionEngine>>
M::initializeExecutionEngine(LLCL::Runtime &runtime, mlir::PassManager &pm,
                             const CompilationOptions &compilationOptions,
                             ExecutionEngineOptions executionEngineOptions,
                             bool isJIT, TargetInfoAttr target) {
  MLIRContext *ctx = pm.getContext();

  // Now create the execution engine so we can JIT.
  auto tmOr = createTargetMachine(compilationOptions, isJIT);
  if (tmOr.isError())
    return tmOr.takeError();

  auto engineOr = ExecutionEngine::createWithStandardLayers(
      std::move(executionEngineOptions), **tmOr);
  if (failed(engineOr))
    return engineOr.takeError();
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  // Add the object compiler layer.
  auto compiler =
      ObjectCompiler::create(runtime, pm, ".kgen_cache", compilationOptions);
  if (failed(compiler))
    return compiler.takeError();

  auto &objLayer = engine->addLayer<ObjectCompilerLayer>(
      std::move(*compiler), engine->getLinkingLayer());

  // If we aren't jitting, notify the object layer that anything we build is not
  // for immediate execution.
  if (!isJIT)
    objLayer.notForImmediateExecution();

  // Add the KGEN compiler layer.
  // First though, get the backend chains to pass into the compile layer.
  auto transformCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".kgen_cache") / "transform").string(),
      KGEN_VERSION_STRING);
  if (transformCacheBackend.isError())
    return transformCacheBackend.takeError();

  auto regionCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".kgen_cache") / "region").string(),
      KGEN_VERSION_STRING);
  if (regionCacheBackend.isError())
    return regionCacheBackend.takeError();

  // Get the build info from the current build.
  BuildInfoAttr build = BuildInfoAttr::getForCurrentBuild(ctx);

  engine->addLayer<KGENCompilerLayer>(
      pm, runtime, target, build, compilationOptions, objLayer,
      std::move(*transformCacheBackend), std::move(*regionCacheBackend));
  return std::move(engine);
}
