//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Init/Init.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "Config/Version.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "KGEN/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/CommonCLOptions.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/DebugInfoDialect/Transforms/SnapshotDebugInfo.h"
#include "Support/FileSystemExtras.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/Process.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/Timing.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {

class CLOptions : public KGENOptions {
public:
  KGENCLOptions parser;

  CLOptions(int argc, char **argv, bool skipInitLLVM = false)
      : parser(argc, argv, *this, skipInitLLVM) {}

  M::cl::MListOpt<std::string> inputFiles{llvm::cl::Positional,
                                          cl::desc("<input files>")};

  M::cl::MOpt<bool> emitTextualAsm{
      "S", cl::desc("Print MLIR output files in textual form")};

  M::cl::MOpt<bool> ignoreFailures{
      "ignore-failure",
      cl::desc("Ignore execution failures. Any messages are still printed, but "
               "failures don't mean the tool fails to execute.")};

  M::cl::MOpt<bool> disablePrebuiltPackages{
      "disable-prebuilt-packages",
      cl::desc("Disable prebuilt packages when parsing the input Mojo file."),
      llvm::cl::init(false)};

  M::cl::MOpt<std::string> dependencyFilename{
      "d", llvm::cl::desc("Path of the dependency file to generate"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

  /// We default to printing diagnostics through llvm::SourceMgr to enable
  /// source ranges and fixit hints, but allow disabling this for testing.
  M::cl::MOpt<bool> enableMLIRDiagnostics{
      "enable-mlir-diagnostics",
      cl::desc("Print .mojo diagnostics through MLIR."), llvm::cl::init(false)};

  /// Add all the input files provided on the command line to the SourceMgr.
  /// This is how MLIR parses multiple files.
  ErrorOrSuccess addInputFilesToSourceMgr(llvm::SourceMgr &mgr);
  void addInputFilesToSourceMgrOrExit(llvm::SourceMgr &mgr);
};
} // namespace

ErrorOrSuccess CLOptions::addInputFilesToSourceMgr(llvm::SourceMgr &mgr) {
  if (inputFiles.empty())
    mgr.AddNewSourceBuffer(openInputFileOrExit(), llvm::SMLoc());

  for (StringRef in : inputFiles) {
    std::error_code ec;
    std::filesystem::path fullPath = std::filesystem::absolute(in.str(), ec);
    if (ec) {
      return Error(
          llvm::formatv("failed to resolve the absolute path for '{0}': {1}",
                        in.str(), ec.message()));
    }
    std::string errorMsg;
    auto result = mlir::openInputFile(fullPath.string(), &errorMsg);
    if (!result)
      return Error(errorMsg);

    mgr.AddNewSourceBuffer(std::move(result), llvm::SMLoc());
  }

  return M::success();
}

void CLOptions::addInputFilesToSourceMgrOrExit(llvm::SourceMgr &mgr) {
  if (auto err = addInputFilesToSourceMgr(mgr))
    exit(reportError(err.getError()));
}

/// Emit the IR for `theModule` to a file.
static LogicalResult emitModuleIR(ModuleOp theModule, const CLOptions &opts) {
  CompilerTimeTraceScope traceScope("emit-module",
                                    theModule.getSymName().value_or(""));
  if (opts.emitTextualAsm.getValue()) {
    auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/false, ".mlir");
    if (!outFile)
      return failure();

    theModule.print(outFile->os());
    // `print` does not insert a newline, so add one here.
    outFile->os() << "\n";
    outFile->keep();
  } else {
    auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/true, ".mlirbc");
    if (!outFile)
      return failure();

    if (failed(mlir::writeBytecodeToFile(theModule, outFile->os())))
      return failure();
    outFile->keep();
  }

  // Try to save the textual IR as an intermediate file.
  if (auto irFile = opts.getIntermediateFile(opts.outputFilename, ".mlir")) {
    theModule.print(irFile->os());
    irFile->keep();
  }

  return mlir::success();
}

/// Create a dependency file for the `-d` option.
///
/// This functionality is generally only for the benefit of the build system,
/// and informs it of the dependencies of the input files.
static LogicalResult createDependencyFile(const CLOptions &clOptions,
                                          ArrayRef<std::string> includedFiles) {
  // It only makes sense to output a dependency file that can map inputs to
  // outputs. If the output file already exists and is not a regular file --
  // like `"-"` for stdout, or a character file like `/dev/null` -- then fail.
  if (clOptions.outputFilename == "-" ||
      (llvm::sys::fs::exists(clOptions.outputFilename) &&
       !llvm::sys::fs::is_regular_file(clOptions.outputFilename))) {
    return failure(clOptions.reportError(
        "can only create dependency file for outputs written to files"));
  }

  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(clOptions.dependencyFilename, &errorMessage);
  if (!outputFile)
    return failure(clOptions.reportError(errorMessage));

  // Resolve each of the dependencies and add them to the file.
  outputFile->os() << clOptions.outputFilename << ":";
  for (StringRef includeFile : includedFiles)
    outputFile->os() << ' ' << includeFile;

  outputFile->os() << "\n";
  outputFile->keep();
  return mlir::success();
}

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the funcs it operates on.
static LogicalResult runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                                     CLOptions &clOptions) {
  DialectRegistry registry;
  TraceProfiler tracer(clOptions.timeTrace, clOptions.timeTraceGranularity);

  if (clOptions.enableMLIRCrashReproducer) {
    // If the reproducer is enable, turn off all threading.
    ctx->disableMultithreading();
    clOptions.useSingleThreadedWorkqueue();
  }

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registerKGENToLLVMTranslation(registry);

  // Create our context, with a runtime; this should not fail.
  LLCL::RuntimeOptions &runtimeOpts = clOptions.parser.options;
  if (runtimeOpts.workQueueType ==
      LLCL::RuntimeOptions::WorkQueueType::kSingleThread)
    runtimeOpts.singleThreaded = true;
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen", Init::Options().withRuntimeOptions(
                  clOptions.parser.options.withCPUAffinity(false)));
  if (ctxOr.isError())
    return failure();
  if (!clOptions.enableMLIRCrashReproducer)
    registerContext(registry, *ctxOr);
  LLCL::Runtime &runtime = *(*ctxOr)->get<LLCL::Runtime>();

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);

  CompilationOptions options = clOptions.getCompilationOptions();

  OwningOpRef<ModuleOp> theModule;
  auto inputFileName = llvm::StringRef(clOptions.inputFilename);

  // Initialize the timing manager.
  std::unique_ptr<mlir::TimingManager> timingManager;
  if (clOptions.timeTrace) {
    if constexpr (KGEN::kIsTracingEnabled)
      timingManager = std::make_unique<TimeProfilerTimingManager>();
    else
      llvm::errs() << "-time-trace specified but tracing isn't coded on";
  } else {
    auto defaultManager = std::make_unique<mlir::DefaultTimingManager>();
    applyDefaultTimingManagerCLOptions(*defaultManager);
    timingManager = std::move(defaultManager);
  }
  TimingScope timing = timingManager->getRootScope();

  PassManagerConfigOptions pmOptions;
  pmOptions.applyPassManagerCLOptions = true;
  pmOptions.enableTiming = true;
  pmOptions.timingScope = &timing;
  pmOptions.crashReproducerOptions.enable = clOptions.enableMLIRCrashReproducer;
  pmOptions.crashReproducerOptions.inputFileName = clOptions.inputFilename;
  pmOptions.crashReproducerOptions.enableLocalMLIRReproducer =
      clOptions.enableLocalMLIRReproducer;

  KGENCompiler compiler(*ctx, options, std::move(pmOptions));

  // The set of files included during processing, used to generate the
  // dependency file.
  SmallVector<std::string> includedFiles;

  if (inputFileName.ends_with(".mojo") || inputFileName.ends_with(".🔥")) {
    TimingScope litScope = timing.nest("Import Mojo source");
    LIT::ParserConfig config(ctx, options);
    config.useMLIRDiagnostics = clOptions.enableMLIRDiagnostics;
    config.disablePrebuiltPackages = clOptions.disablePrebuiltPackages;
    theModule = importMojoFile(runtime, mgr, config, litScope, &includedFiles);
  } else if (options.getDebugInfoLevelForInput() >
             CompilationOptions::kSynthetic) {
    ctx->loadDialect<DebugInfo::DebugInfoDialect>();
    theModule = DebugInfo::parseSourceFileWithDebugInfo(
        mgr, ctx, options.getDIEmissionKind(),
        static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage));
  } else {
    theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  }
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

  // Tag the module with the environment parsed from the defines.
  ctx->loadDialect<KGENDialect>();

  // Populate the module with the user-provided -D options.
  ErrorOr<EnvAttr> env = EnvAttr::parseDefines(ctx, clOptions.defines);
  if (env.isError())
    return failure(clOptions.reportError(env.takeError().get()));
  theModule.get()->setAttr(EnvAttr::getEnvAttrName(), env.takeValue());

  // Extend the module with the Module env-attrs.
  extendWithModularEnvAttr(theModule.get());

  // If we are generating a dependency file, do so now.
  if (!clOptions.dependencyFilename.empty()) {
    if (failed(createDependencyFile(clOptions, includedFiles)))
      return failure(
          clOptions.reportError("failed to create a dependency file"));
  }

  // Find a target specification or construct one using the commandline options.
  TargetInfoAttr target = getTargetInfo(*theModule);
  if (target) {
    if (!clOptions.march.empty()) {
      mlir::emitWarning(theModule->getLoc(),
                        "overriding module target specification with -march");
    } else if (target.getTripleStr() != clOptions.targetTriple ||
               target.getArch() != clOptions.targetCpu ||
               target.getFeatures() != clOptions.targetFeatures) {
      mlir::emitWarning(theModule->getLoc(),
                        "module target does not match command line "
                        "specification and will be overwritten");
    }
    target = nullptr;
  }
  if (!target) {
    ErrorOr<TargetInfoAttr> targetOr = nullptr;
    if (!clOptions.march.empty() || !clOptions.mcpu.empty()) {
      // Detect if the user accidentally specified any of the `--target-*`.
      if (options.targetTriple != llvm::sys::getDefaultTargetTriple() ||
          options.targetCpu != llvm::sys::getHostCPUName() ||
          options.targetFeatures != getHostCPUFeatures())
        return failure(clOptions.reportError(
            "--target-triple, --target-cpu, or --target-features specified at "
            "the same time as -march or -mcpu"));

      // Use `-march` to determine the feature set.
      targetOr = getMArchFeatures(ctx, clOptions.march, clOptions.mcpu,
                                  clOptions.mtune);
    } else {
      // If the user provided the target triple without specifying a CPU,
      // default to `generic`.
      if (options.targetTriple != llvm::sys::getDefaultTargetTriple()) {
        if (options.targetCpu == llvm::sys::getHostCPUName())
          options.targetCpu = "generic";
        if (options.targetFeatures == getHostCPUFeatures())
          options.targetFeatures = "";
      }

      // Use the full triple, specific CPU, and manually specified features to
      // get the target info.
      targetOr =
          getTargetInfoFor(ctx, clOptions.targetTriple, clOptions.targetCpu,
                           clOptions.targetFeatures);
    }
    if (targetOr.isError())
      return failure(clOptions.reportError(targetOr.getError()));
    target = targetOr.takeValue();
    options.targetTriple = target.getTripleStr();
    options.targetCpu = target.getArch();
    options.targetFeatures = target.getFeatures();
  }

  // Generate a library file or go all the way through elaboration.
  if (clOptions.cmd == Command::kGenLibraryFile) {
    if (failed(compiler.runGenerateLibraryPipeline(*theModule)))
      return failure(clOptions.reportError("compilation failed"));
    return emitModuleIR(*theModule, clOptions);
  }

  auto compilerOr = ObjectCompiler::create(".mojo_cache", options,
                                           clOptions.cmd == Command::kExecute,
                                           *ctx, pmOptions);
  if (failed(compilerOr))
    return failure(clOptions.reportError(compilerOr.getError()));
  ObjectCompiler &objCompiler = **compilerOr;

  // Compiles the module through KGEN compiler pipeline.
  // We don't need to try to look anything up.
  if (ErrorOrSuccess err = compiler.runKGENPipeline(*theModule, target))
    return failure(clOptions.reportError(err.getError()));

  // If all we're doing is generating a library file or elaborating, we're done
  // now.
  if (clOptions.cmd == Command::kElaborate)
    return emitModuleIR(*theModule, clOptions);

  // Construct the symbol table and the export map.
  SymbolTable symtab(*theModule);
  ExportMap exportedSymbols = getExportedSymbols(*theModule);

  // Handle LLVM output.
  if (clOptions.cmd == Command::kEmitLLVM) {
    llvm::LLVMContext llvmCtx;
    ErrorOr<std::unique_ptr<llvm::Module>> llvmModuleOr =
        objCompiler.lowerAllFuncsToLLVM(llvmCtx, *theModule);

    if (llvmModuleOr)
      return failure(clOptions.reportError(
          Twine("could not lower funcs to LLVM, ") + llvmModuleOr.getError()));

    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".ll");
    if (!outFile)
      return failure(clOptions.reportError("could not open .ll output file"));

    std::unique_ptr<llvm::Module> llvmModule = llvmModuleOr.takeValue();
    llvmModule->print(outFile->os(), nullptr);
    outFile->keep();
    return mlir::success();
  }

  // Handle assembly output.
  if (clOptions.cmd == Command::kEmitAssembly) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".s");
    if (!outFile)
      return failure(clOptions.reportError("could not open .s output file"));

    auto standaloneOr = objCompiler.emitAssembly(*theModule, outFile->os());
    if (failed(standaloneOr))
      return failure(
          clOptions.reportError("could not produce standalone asm: " +
                                Twine(standaloneOr.getError())));
    outFile->keep();
    return mlir::success();
  }

  // Handle header emission, we don't need to generate an archive for this.
  if (clOptions.cmd == Command::kEmitHeader) {
    LogicalResult result = failure();
    auto writeFn = [&](raw_ostream &os) {
      result =
          objCompiler.emitCXXHeader(*theModule, clOptions.outputFilename, os);
    };
    if (clOptions.outputFilename == "-") {
      auto writeContents = [&](raw_ostream &os) {
        writeFn(os);
        os.flush();
        return llvm::Error::success();
      };
      if (llvm::Error err =
              llvm::writeToOutput(clOptions.outputFilename, writeContents)) {
        return failure(
            clOptions.reportError(toModularError(std::move(err)).get()));
      }

      // Safely process creating the header, taking into account that we may
      // have different processes trying to produce this header in parallel.
    } else if (ErrorOr<std::filesystem::path> err =
                   writeFileUnderLock(clOptions.outputFilename, writeFn);
               err.isError()) {
      return failure(clOptions.reportError(err.getError()));
    }
    return mlir::success();
  }

  // If there are no exported symbols, there's nothing to codegen. Report this
  // as an error.
  if (exportedSymbols.empty()) {
    return failure(
        clOptions.reportError("module does not `@export` any symbols or define "
                              "a `main` function; nothing to codegen"));
  }

  // If we need to execute, grab the function metadata before the module is
  // consumed.
  struct FunctionExecution {
    StringAttr name;
    Location loc;
    FunctionType type;
    CommandLineFunc clFunc;
  };
  SmallVector<FunctionExecution> funcExecs;
  StringSet<> foundFuncs;
  if (clOptions.cmd == Command::kExecute) {
    for (auto fn : theModule->getOps<FuncOp>()) {
      StringAttr name = fn.getSymNameAttr();
      // See if we were asked to execute this function.
      if (std::optional<CommandLineFunc> clFunc =
              clOptions.shouldExecuteFunc(name)) {
        funcExecs.push_back(FunctionExecution{name, fn.getLoc(),
                                              fn.getFunctionType(), *clFunc});
        foundFuncs.insert(name);
        if (auto err = clFunc->verifyFuncSignature(funcExecs.back().type)) {
          mlir::emitError(fn.getLoc(), err.getError());
          if (!clOptions.ignoreFailures)
            return failure();
        }
      }
    }
    // If we didn't find a function the user asked to execute, emit an error.
    for (const auto &fn : clOptions.funcs) {
      if (!foundFuncs.count(fn.name)) {
        return mlir::emitError(theModule->getLoc(),
                               "could not find func '@" + fn.name + "'");
      }
    }
  }

  // -emit and -execute both require compiled objects.
  ErrorOr<BufferRef> archiveOr = objCompiler.emitArchive(*theModule);
  if (failed(archiveOr)) {
    return failure(clOptions.reportError("failed to emit archive: " +
                                         Twine(archiveOr.getError())));
  }
  BufferRef archive = archiveOr.takeValue();

  // If we're emitting the archive, do it.
  if (clOptions.cmd == Command::kEmit) {
    // Look up the first item in the exported symbols to trigger archive
    // generation.
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".o");
    if (!outFile)
      return failure(clOptions.reportError("could not open .o output file"));

    outFile->os() << archive->getBuffer();
    outFile->keep();
    return mlir::success();
  }

  ExecutionEngineOptions eeOptions;
  if (options.debugLevel != CompilationOptions::kNoDebug)
    eeOptions.registerDebugPlugins = true;
  // Detect cross-compilation by checking whether the target CPU is the same as
  // the host CPU.
  eeOptions.crossCompiling = options.targetCpu != llvm::sys::getHostCPUName();

  auto engineOr = initializeExecutionEngine(*ctx, options, std::move(eeOptions),
                                            /*isJIT=*/true, pmOptions);
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));
  ExecutionEngine &engine = **engineOr;

  // Helper to execute a func.
  auto execFunc = [&](const FunctionExecution &func, StringAttr name,
                      const CommandLineFunc &clFunc) -> LogicalResult {
    CompilerTimeTraceScope traceScope("execute-function", name);
    // Trigger compilation so we can pull out the archive.
    ErrorOr<CompiledFunc> funcOr = engine.lookup(name);
    if (failed(funcOr))
      return failure(clOptions.reportError(funcOr.getError()));

    if (auto err = clFunc.executeAndPrint(*funcOr)) {
      mlir::emitError(func.loc, err.getError());
      return failure(!clOptions.ignoreFailures);
    }
    return mlir::success();
  };

  // Pass the compiled archive to the execution engine.
  if (ErrorOrSuccess err =
          engine.addIfAbsent<StaticArchiveLayer>("exec", std::move(archive)))
    return failure(clOptions.reportError(err.getError()));

  // Loop over the functions, executing as necessary.
  for (const FunctionExecution &func : funcExecs) {
    if (failed(execFunc(func, func.name, func.clFunc))) {
      return failure(
          clOptions.reportError("failed to execute " + func.name.strref()));
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Override the default version printer.
  llvm::cl::SetVersionPrinter([](raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << "KGEN compiler:\n  ";
    os << "Modular version: " << version.major << '.' << version.minor << '.'
       << version.patch << version.label << "\n  ";
    os << "Git SHA: " << version.revision << "\n  ";
    os << "Build config: " << version.buildType << "\n\n";

    // Print the host target config.
    llvm::sys::printDefaultTargetAndDetectedCPU(os);
    // Print all registered targets.
    llvm::TargetRegistry::printRegisteredTargetsForVersion(os);
  });

  // Enable command line options for various MLIR internals.
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  registerDefaultTimingManagerCLOptions();
  KGEN::registerDefaultKGENPasses();
  registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file(s).
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(clOptions.getIncludePaths());
  clOptions.addInputFilesToSourceMgrOrExit(sourceManager);

  return failed(clOptions.configureMLIRContextAndExecute(
      sourceManager, [&](MLIRContext *ctx) -> LogicalResult {
        ctx->printOpOnDiagnostic(true);
        return runToolPipeline(ctx, sourceManager, clOptions);
      }));
}
