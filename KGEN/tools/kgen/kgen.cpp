//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "Config/Version.h"
#include "KGEN/CLOptions.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/EmitFuncHeader.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/MojoParser.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/SnapshotDebugInfo.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {
class CLOptions : public KGENCLOptions {
public:
  using KGENCLOptions::KGENCLOptions;

  cl::list<std::string> inputFiles{llvm::cl::Positional,
                                   cl::desc("<input files>")};

  cl::opt<bool> emitTextualAsm{
      "S", cl::desc("Print MLIR output files in textual form")};

  cl::opt<bool> ignoreFailures{
      "ignore-failure",
      cl::desc("Ignore execution failures. Any messages are still printed, but "
               "failures don't mean the tool fails to execute.")};

  cl::opt<std::string> dependencyFilename{
      "d", llvm::cl::desc("Path of the dependency file to generate"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

  /// We default to printing diagnostics through llvm::SourceMgr to enable
  /// source ranges and fixit hints, but allow disabling this for testing.
  cl::opt<bool> enableMLIRDiagnostics{
      "enable-mlir-diagnostics",
      cl::desc("Print .mojo diagnostics through MLIR."), cl::init(false)};

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
    std::string errorMsg;
    auto result = mlir::openInputFile(in, &errorMsg);
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
  TimeTraceScope<> traceScope("emit-module",
                              theModule.getSymName().value_or(""));
  if (opts.emitTextualAsm) {
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

    mlir::writeBytecodeToFile(theModule, outFile->os());
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
  TraceProfiler tracer(clOptions);

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                  index::IndexDialect, LLVM::LLVMDialect>();

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);
  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx->allowUnregisteredDialects();

  CompilationOptions compilationOptions = clOptions.getCompilationOptions();
  OwningOpRef<ModuleOp> theModule;
  auto inputFileName = llvm::StringRef(clOptions.inputFilename.getValue());

  // Initialize the timing manager.
  std::unique_ptr<mlir::TimingManager> timingManager;
  if (clOptions.timeTrace) {
    timingManager = std::make_unique<TimeProfilerTimingManager>();
  } else {
    auto defaultManager = std::make_unique<mlir::DefaultTimingManager>();
    applyDefaultTimingManagerCLOptions(*defaultManager);
    timingManager = std::move(defaultManager);
  }
  TimingScope timing = timingManager->getRootScope();

  mlir::PassManager pm(ctx);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableTiming(timing);
  if (clOptions.enableMLIRCrashReproducer.getValue()) {
    ctx->disableMultithreading();
    pm.enableCrashReproducerGeneration(clOptions.inputFilename.getValue() +
                                           ".repro.mlir",
                                       /*genLocalReproducer=*/false);
  }

  // Set up the runtime.
  std::unique_ptr<LLCL::Runtime> runtime = clOptions.createRuntime();

  // The set of files included during processing, used to generate the
  // dependency file.
  SmallVector<std::string> includedFiles;

  if (inputFileName.ends_with(".mojo") || inputFileName.ends_with(".🔥")) {
    TimingScope litScope = timing.nest("Import Mojo source");
    MojoParserConfig config(ctx, *runtime, compilationOptions);
    config.useMLIRDiagnostics = clOptions.enableMLIRDiagnostics;
    theModule = importMojoFile(mgr, config, litScope, &includedFiles);
  } else if (compilationOptions.getDebugInfoLevelForInput()) {
    theModule = DebugInfo::parseSourceFileWithDebugInfo(
        mgr, ctx, compilationOptions.getDIEmissionKind());
  } else {
    theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  }
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

  // If we are generating a dependency file, do so now.
  if (!clOptions.dependencyFilename.empty()) {
    if (failed(createDependencyFile(clOptions, includedFiles)))
      return failure(
          clOptions.reportError("failed to create a dependency file"));
  }

  // Find a target specification or construct one using the commandline options.
  TargetInfoAttr target = getTargetInfo(*theModule);
  if (target) {
    if (target.getTripleStr() != clOptions.targetTriple ||
        target.getCpu() != clOptions.targetCpu ||
        target.getFeatures() != clOptions.targetFeatures) {
      mlir::emitWarning(theModule->getLoc(),
                        "module target does not match command line "
                        "specification and will be overwritten");
    }
    target = nullptr;
  }
  if (!target) {
    ErrorOr<TargetInfoAttr> targetOr =
        getTargetInfoFor(ctx, clOptions.targetTriple, clOptions.targetCpu,
                         clOptions.targetFeatures);
    if (targetOr.isError())
      return mlir::emitError(theModule->getLoc(), targetOr.getError());
    target = targetOr.takeValue();
  }

  // Get the build info from the current build.
  BuildInfoAttr build = BuildInfoAttr::getForCurrentBuild(ctx);

  // Now create the execution engine so we can JIT.
  auto tmOr = createTargetMachine(compilationOptions,
                                  /*isJIT=*/clOptions.cmd == Command::kExecute);
  if (tmOr.isError())
    return failure(clOptions.reportError(tmOr.getError()));

  auto engineOr = ExecutionEngine::createWithStandardLayers(
      {/*registerDebugPlugins=*/compilationOptions.debugLevel !=
       CompilationOptions::DebugInfoLevel::kNoDebug},
      **tmOr);
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  // TODO (8082): This should not be necessary.
  std::vector<std::pair<StringLiteral, void *>> compilerRTFunctions;
  registerIntelAMX(compilerRTFunctions);
  registerLLCL(compilerRTFunctions);
  registerPython(compilerRTFunctions);
  registerMemory(compilerRTFunctions);
  registerPrint(compilerRTFunctions);
  registerRandom(compilerRTFunctions);
  registerSystem(compilerRTFunctions);
  registerTracing(compilerRTFunctions);
  for (auto [name, ptr] : compilerRTFunctions)
    if (auto err = engine->add<StaticSymbolLayer>("exec", name, ptr))
      return failure(clOptions.reportError(err.getError()));

  // Add the object compiler layer.
  auto compiler =
      ObjectCompiler::create(*runtime, pm, ".kgen_cache", compilationOptions);
  if (failed(compiler)) {
    return failure(clOptions.reportError(
        Twine("could not create object compiler: ") + compiler.getError()));
  }
  auto &objLayer = engine->addLayer<ObjectCompilerLayer>(
      std::move(*compiler), engine->getLinkingLayer());

  // Add the KGEN compiler layer.
  // First though, get the backend chains to pass into the compile layer.
  auto transformCacheBackend = Cache::getLocalDefaultBackendChain(
      *runtime, (std::filesystem::path(".kgen_cache") / "transform").string(),
      KGEN_VERSION_STRING);
  if (transformCacheBackend.isError())
    return failure(clOptions.reportError(transformCacheBackend.getError()));

  auto regionCacheBackend = Cache::getLocalDefaultBackendChain(
      *runtime, (std::filesystem::path(".kgen_cache") / "region").string(),
      KGEN_VERSION_STRING);
  if (regionCacheBackend.isError())
    return failure(clOptions.reportError(transformCacheBackend.getError()));

  auto &compileLayer = engine->addLayer<KGENCompilerLayer>(
      pm, *runtime, target, build, clOptions.getCompilationOptions(), objLayer,
      std::move(*transformCacheBackend), std::move(*regionCacheBackend));

  // Generate a library file or go all the way through elaboration.
  if (clOptions.cmd == Command::kGenLibraryFile) {
    populateGenerateLibraryFilePasses(pm, *runtime);
    if (failed(pm.run(*theModule)))
      return failure(clOptions.reportError("compilation failed"));
    return emitModuleIR(*theModule, clOptions);
  }

  // This currently compiles the module, so we don't need to try to look
  // anything up.
  // TODO(#10893): We will have to look up the symbols we want to emit at some
  //   point.
  if (auto err = compileLayer.add("exec", *theModule))
    return failure(clOptions.reportError("compilation failed"));

  // If all we're doing is generating a library file or elaborating, we're done
  // now.
  if (clOptions.cmd == Command::kElaborate)
    return emitModuleIR(*theModule, clOptions);

  // Construct the symbol table and the export map.
  SymbolTable symtab(*theModule);
  llvm::MapVector<StringAttr, ExportedSymbol> exportedSymbols =
      getExportedSymbols(*theModule);

  // Handle LLVM output.
  if (clOptions.cmd == Command::kEmitLLVM) {
    llvm::LLVMContext llvmCtx;
    auto llvmModule = objLayer.getRawCompiler().lowerAllFuncsToLLVM(
        symtab, exportedSymbols, llvmCtx);
    if (!llvmModule)
      return failure(clOptions.reportError("could not lower funcs to LLVM"));
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".ll");
    if (!outFile)
      return failure(clOptions.reportError("could not open .ll output file"));

    llvmModule->print(outFile->os(), nullptr);
    outFile->keep();
    return mlir::success();
  }

  // Handle assembly output.
  if (clOptions.cmd == Command::kEmitAssembly) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".s");
    if (!outFile)
      return failure(clOptions.reportError("could not open .s output file"));

    auto standaloneOr = objLayer.getRawCompiler().produceStandaloneAssembly(
        symtab, exportedSymbols, target, outFile->os());
    if (failed(standaloneOr))
      return failure(
          clOptions.reportError("could not produce standalone asm: " +
                                Twine(standaloneOr.getError())));
    outFile->keep();
    return mlir::success();
  }

  // Handle header emission, we don't need to generate an archive for this.
  if (clOptions.cmd == Command::kEmitHeader)
    return emitHeader(symtab, exportedSymbols, *compiler,
                      clOptions.outputFilename);

  // If the module is empty, just return - don't bother trying codegen or
  // emitting anything.
  if (theModule->getOps().empty())
    return mlir::success();

  // If there are no exported symbols, then we won't codegen anything. That
  // means we need to add an exported symbol if there aren't any. Use the first
  // symbol op in the module just so we have something.
  // TODO(#10893): This behavior is sketchy. We should be exporting the roots of
  //   callstacks we want codegen'd. This requires updating tests.
  if (exportedSymbols.empty()) {
    auto firstFunc = *theModule->getOps<mlir::SymbolOpInterface>().begin();
    exportedSymbols.insert(
        {firstFunc.getNameAttr(), {firstFunc.getNameAttr(), false}});
  }

  // If we're emitting the archive, do it.
  if (clOptions.cmd == Command::kEmit) {
    // Notify the object layer that this is not for immediate execution.
    objLayer.notForImmediateExecution();
    // Look up the first item in the exported symbols to trigger archive
    // generation.
    ErrorOr<CompiledFunc> funcOr =
        engine->lookup(exportedSymbols.front().second.alias);
    if (funcOr.isError())
      return failure(clOptions.reportError(funcOr.getError()));
    // And lookup the archive.
    std::optional<Cache::BufferRef> archive =
        engine->getLayer<ObjectCompilerLayer>().lookupArchive(*theModule);
    if (!archive.has_value())
      return failure(clOptions.reportError("compiled archive was missing"));
    return clOptions.emitArchive((*archive)->getBuffer());
  }

  // Helper to execute a func.
  auto execFunc = [&](FuncOp theFunc, StringAttr name,
                      const CommandLineFunc &clFunc) -> LogicalResult {
    TimeTraceScope<> traceScope("execute-function", name);
    auto compiledFuncOr = engine->lookup(name);
    if (failed(compiledFuncOr))
      return failure(clOptions.reportError(compiledFuncOr.getError()));

    if (auto err = clFunc.verifyFuncSignature(theFunc.getFunctionType())) {
      mlir::emitError(theFunc.getLoc(), err.getError());
      return failure(!clOptions.ignoreFailures);
    }

    if (auto err = clFunc.executeAndPrint(*compiledFuncOr)) {
      mlir::emitError(theFunc.getLoc(), err.getError());
      return failure(!clOptions.ignoreFailures);
    }
    return mlir::success();
  };

  // Loop over the functions, executing as necessary.
  llvm::DenseSet<StringRef> foundFuncs;
  for (auto fn : theModule->getOps<FuncOp>()) {
    StringAttr name = fn.getNameAttr();

    // If this function was exported, grab the alias it was exported as.
    auto it = exportedSymbols.find(name);
    if (it != exportedSymbols.end())
      name = it->second.alias;

    // If we were asked to handle this func, do so.
    if (std::optional<CommandLineFunc> clFunc =
            clOptions.shouldExecuteFunc(name)) {
      switch (clOptions.cmd) {
      case Command::kGenLibraryFile:
      case Command::kElaborate:
      case Command::kEmitLLVM:
      case Command::kEmitAssembly:
      case Command::kEmit:
      case Command::kEmitHeader:
        break;
      case Command::kExecute: {
        if (failed(execFunc(fn, name, *clFunc)))
          return failure(
              clOptions.reportError("failed to execute " + name.getValue()));
      }
      }
    }
    foundFuncs.insert(name);
  }

  // Validate that the user didn't pass in any funcs we don't have. This would
  // be super confusing if the user simply gets no response for something that
  // isn't defined, so put up an actual error.
  for (const auto &fn : clOptions.funcs)
    if (!foundFuncs.count(fn.name))
      return mlir::emitError(theModule->getLoc(),
                             "could not find func '@" + fn.name + "'");

  return mlir::success();
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Initialize the compiler runtime.
  KGEN_CompilerRT_Initialize();

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Override the default version printer.
  llvm::cl::SetVersionPrinter([](raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << "KGEN compiler:\n  ";
    os << "Modular version " << version.major << '.' << version.minor << '.'
       << version.patch << "\n  ";
    os << "Git SHA " << version.revision << "\n  ";
    os << "Build config " << version.buildType << "\n\n";

    // Print the host target config.
    llvm::sys::printDefaultTargetAndDetectedCPU(os);
    // Print all registered targets.
    llvm::TargetRegistry::printRegisteredTargetsForVersion(os);
  });

  // Enable command line options for various MLIR internals.
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file(s).
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(clOptions.getIncludePaths());
  clOptions.addInputFilesToSourceMgrOrExit(sourceManager);

  return failed(clOptions.configureMLIRContextAndExecute(
      sourceManager, [&](MLIRContext *ctx) -> LogicalResult {
        return runToolPipeline(ctx, sourceManager, clOptions);
      }));
}
