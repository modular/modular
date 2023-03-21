//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "Config/Config.h"
#include "KGEN/CLOptions.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/EmitFuncHeader.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/ParseLit.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/SnapshotDebugInfo.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/Timing.h"
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
      cl::desc("Print .lit parser diagnostics through MLIR."), cl::init(false)};

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
  auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/true, ".mlirbc");
  if (!outFile)
    return mlir::failure();

  mlir::writeBytecodeToFile(theModule, outFile->os());
  outFile->keep();

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
                                     const CLOptions &clOptions) {
  DialectRegistry registry;
  TraceProfiler tracer(clOptions);

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                  HLCF::HLCFDialect, index::IndexDialect, LLVM::LLVMDialect>();

  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
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
  applyPassManagerCLOptions(pm);
  pm.enableTiming(timing);
  if (clOptions.enableMLIRCrashReproducer.getValue()) {
    ctx->disableMultithreading();
    pm.enableCrashReproducerGeneration(clOptions.inputFilename.getValue() +
                                           ".repro.mlir",
                                       /*genLocalReproducer=*/true);
  }

  // Set up the runtime.
  std::unique_ptr<LLCL::Runtime> runtime = clOptions.createRuntime();

  // The set of files included during processing, used to generate the
  // dependency file.
  SmallVector<std::string> includedFiles;

  if (inputFileName.ends_with(".lit") || inputFileName.ends_with(".mojo")) {
    TimingScope litScope = timing.nest("Import Lit");
    theModule = importLitFile(mgr, ctx, litScope, compilationOptions,
                              clOptions.enableMLIRDiagnostics, *runtime,
                              /*validateDocStrings=*/false, &includedFiles);
  } else if (compilationOptions.getDebugInfoLevelForInput()) {
    theModule = DebugInfo::parseSourceFileWithDebugInfo(
        mgr, ctx, compilationOptions.getDIEmissionKind());
  } else {
    theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  }
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

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

  // Generate a library file or go all the way through elaboration.
  if (clOptions.cmd == Command::kGenLibraryFile)
    populateGenerateLibraryFilePasses(pm);
  else
    populateElaborateModulePasses(pm, *runtime, target,
                                  {clOptions.enableSearch});

  if (failed(pm.run(*theModule)))
    return failure(clOptions.reportError("compilation failed"));

  // If we are generating a dependency file, do so now.
  if (!clOptions.dependencyFilename.empty()) {
    if (failed(createDependencyFile(clOptions, includedFiles)))
      return failure();
  }

  // If all we're doing is generating a library file or elaborating, we're done
  // now.
  if (clOptions.cmd == Command::kGenLibraryFile ||
      clOptions.cmd == Command::kElaborate)
    return emitModuleIR(*theModule, clOptions);

  SymbolTable symtab(*theModule);
  auto compiler =
      ObjectCompiler::create(*runtime, pm, ".kgen_cache", compilationOptions);
  if (failed(compiler)) {
    return failure(clOptions.reportError(
        Twine("could not create object compiler: ") + compiler.getError()));
  }
  llvm::MapVector<StringAttr, ExportedSymbol> exportedSymbols =
      getExportedSymbols(*theModule);

  // Handle LLVM output.
  if (clOptions.cmd == Command::kEmitLLVM) {
    llvm::LLVMContext ctx;
    auto llvmModule =
        compiler->lowerAllFuncsToLLVM(symtab, exportedSymbols, ctx);
    if (!llvmModule)
      return failure();
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".ll");
    if (!outFile)
      return failure();

    llvmModule->print(outFile->os(), nullptr);
    outFile->keep();
    return mlir::success();
  }

  // Handle assembly output.
  if (clOptions.cmd == Command::kEmitAssembly) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".s");
    if (!outFile)
      return failure();

    auto standaloneOr = compiler->produceStandaloneAssembly(
        symtab, exportedSymbols, target, outFile->os());
    if (failed(standaloneOr))
      return failure();
    outFile->keep();
    return mlir::success();
  }

  // Handle header emission, we don't need to generate an archive for this.
  if (clOptions.cmd == Command::kEmitHeader)
    return emitHeader(symtab, exportedSymbols, *compiler,
                      clOptions.outputFilename);

  // This produces a standalone archive for all the objects we requested.
  auto standaloneOr = compiler->produceStandaloneArchive(
      symtab, exportedSymbols,
      /*isJIT=*/clOptions.cmd == Command::kExecute);
  if (failed(standaloneOr) && !clOptions.ignoreFailures)
    return failure();
  Cache::BufferRef standaloneArchive = std::move(*standaloneOr);

  // If we're emitting the archive, do it.
  if (clOptions.cmd == Command::kEmit)
    return clOptions.emitArchive(standaloneArchive->getBuffer());

  // Now we can load it into the JIT - we're definitely executing the thing.

  // Now create the execution engine so we can JIT.
  auto tmOr = createTargetMachine(compilationOptions,
                                  /*isJIT=*/true);
  if (tmOr.isError())
    return failure(clOptions.reportError(tmOr.getError()));

  auto engineOr = ExecutionEngine::create(
      {/*registerDebugPlugins=*/compilationOptions.debugLevel !=
       CompilationOptions::DebugInfoLevel::kNoDebug},
      **tmOr);
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));
  ExecutionEngine engine = std::move(*engineOr);

  // TODO (8082): This should not be necessary.
  std::vector<std::pair<StringLiteral, void *>> compilerRTFunctions;
  KGEN::registerIntelAMX(compilerRTFunctions);
  KGEN::registerLLCL(compilerRTFunctions);
  KGEN::registerMemory(compilerRTFunctions);
  KGEN::registerPrint(compilerRTFunctions);
  KGEN::registerSystem(compilerRTFunctions);
  KGEN::registerTracing(compilerRTFunctions);
  for (auto [name, ptr] : compilerRTFunctions)
    if (auto err = engine.add("exec", name, ptr))
      return failure(clOptions.reportError(err.getError()));

  if (auto err = engine.add("exec", std::move(standaloneArchive)))
    return failure(clOptions.reportError(err.getError()));

  // Helper to execute a func.
  auto execFunc = [&](FuncOp theFunc, StringAttr name,
                      const CommandLineFunc &clFunc) -> LogicalResult {
    TimeTraceScope<> traceScope("execute-function", name);
    auto compiledFuncOr = engine.lookup(name);
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
          return failure();
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
    os << "KGEN compiler:\n  ";
    os << "Modular version " << MODULAR_VERSION_MAJOR << '.'
       << MODULAR_VERSION_MINOR << '.' << MODULAR_VERSION_PATCH << "\n  ";
    os << "Git SHA " << MODULAR_VERSION_REVISION << "\n  ";
    os << "Build config " << MODULAR_BUILD_TYPE << "\n\n";

    // Print the host target config.
    llvm::sys::printDefaultTargetAndDetectedCPU(os);
    // Print all registered targets.
    llvm::TargetRegistry::printRegisteredTargetsForVersion(os);
  });

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file(s).
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(clOptions.searchPaths);
  clOptions.addInputFilesToSourceMgrOrExit(sourceManager);

  return failed(clOptions.configureMLIRContextAndExecute(
      sourceManager, [&](MLIRContext *ctx) -> LogicalResult {
        return runToolPipeline(ctx, sourceManager, clOptions);
      }));
}
