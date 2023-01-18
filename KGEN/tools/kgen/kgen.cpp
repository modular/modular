//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "EmitFuncHeader.h"
#include "KGEN/CLOptions.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/ParseLit.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommonCLOptions.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/SnapshotDebugInfo.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {
class CLOptions : public KGENCLOptions {
public:
  using KGENCLOptions::KGENCLOptions;

  cl::list<std::string> inputFiles{llvm::cl::Positional,
                                   cl::desc("<input files>")};

  cl::opt<bool> timeTrace{
      "time-trace",
      cl::desc("Turn on time profiler. Generates JSON file "
               "called kgen.trace.json in the derived directory.")};

  cl::opt<int> timeTraceGranularity{
      "time-trace-granularity",
      cl::desc("Minimum time granularity (in microseconds) "
               "traced by time profiler."),
      cl::init(0)};

  cl::opt<bool> ignoreFailures{
      "ignore-failure",
      cl::desc("Ignore execution failures. Any messages are still printed, but "
               "failures don't mean the tool fails to execute.")};

  cl::opt<bool> enableSearch{
      "enable-search", cl::init(false),
      cl::desc("Do search when an evaluator is provided.")};

  cl::list<std::string> searchPaths{
      "I", cl::desc("Path to use to search for included files.")};

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

struct TraceProfiler {
  TraceProfiler(const CLOptions &clOptions) {
    if (!clOptions.timeTrace)
      return;
    timeTraceProfilerInitialize(clOptions.timeTraceGranularity, "kgen");

    std::error_code ec;
    std::filesystem::path derived = std::filesystem::absolute(
        llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."), ec);
    if (ec)
      clOptions.reportError("cannot get the modular derived path: " +
                            ec.message());

    outputFilePath = derived / "kgen.trace.json";
    isActive = true;
  }

  ~TraceProfiler() {
    if (!isActive)
      return;

    if (auto err = timeTraceProfilerWrite(outputFilePath.string(), "-"))
      llvm::errs() << "unable to write trace file: " << err.getError();
    timeTraceProfilerCleanup();
  }

private:
  bool isActive = false;
  std::filesystem::path outputFilePath;
};

/// Any dialect that has this interface attached will be legal to inline (by
/// force).
struct ForceInlineDialectInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const override {
    return true;
  }

  bool isLegalToInline(Operation *, Operation *,
                       bool wouldBeCloned) const override {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const override {
    return true;
  }
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
  auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/true);
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
static LogicalResult
createDependencyFile(const CLOptions &clOptions,
                     SmallVectorImpl<std::string> &includedFiles) {
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

  // Setup the search paths.
  SmallVector<std::filesystem::path> paths;
  for (const auto &p : clOptions.searchPaths)
    paths.push_back(p);
  paths.push_back(std::filesystem::path("."));

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
                  HLCF::HLCFDialect, index::IndexDialect, LLVM::LLVMDialect,
                  scf::SCFDialect>();

  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
  // Add a basic inliner interface to the debug info and builtin dialect.
  ctx->getOrLoadDialect<DebugInfo::DebugInfoDialect>()
      ->addInterface<ForceInlineDialectInterface>();
  ctx->getOrLoadDialect<BuiltinDialect>()
      ->addInterface<ForceInlineDialectInterface>();
  ctx->getOrLoadDialect<index::IndexDialect>()
      ->addInterface<ForceInlineDialectInterface>();
  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx->allowUnregisteredDialects();

  CompilationOptions compilationOptions = clOptions.getCompilationOptions();
  OwningOpRef<ModuleOp> theModule;
  auto inputFileName = llvm::StringRef(clOptions.inputFilename.getValue());
  mlir::TimingScope ts;
  mlir::PassManager pm(ctx);
  if (inputFileName.ends_with(".lit")) {
    theModule = importLitFile(mgr, ctx, ts, compilationOptions,
                              clOptions.enableMLIRDiagnostics);
    pm.addPass(createLowerLITTerminators());
  } else if (compilationOptions.getDebugInfoLevelForInput()) {
    theModule = DebugInfo::parseSourceFileWithDebugInfo(
        mgr, ctx, compilationOptions.getDIEmissionKind());
  } else {
    theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  }
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

  // The set of files included during processing, used to generate the
  // dependency file.
  SmallVector<std::string> includedFiles;

  // Set up the runtime.
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());

  // Generate a library file or go all the way through elaboration.
  if (clOptions.cmd == Command::kGenLibraryFile) {
    generateLibraryFile(pm);
  } else {
    elaborateModule(pm, runtime,
                    {clOptions.searchPaths, clOptions.enableSearch},
                    includedFiles);
  }

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
  auto compiler = ObjectCompiler::create(runtime, ".kgen_cache", symtab,
                                         compilationOptions);
  if (failed(compiler)) {
    return failure(clOptions.reportError(
        Twine("could not create object compiler: ") + compiler.getError()));
  }

  TargetInfoAttr attr = TargetInfoAttr::getForHost(ctx);

  // Handle LLVM output.
  if (clOptions.cmd == Command::kEmitLLVM) {
    llvm::LLVMContext ctx;
    auto llvmModule = compiler->lowerAllFuncsToLLVM(ctx);
    if (!llvmModule)
      return failure();
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false);
    if (!outFile)
      return failure();

    llvmModule->print(outFile->os(), nullptr);
    outFile->keep();
    return mlir::success();
  }

  // Handle assembly output.
  if (clOptions.cmd == Command::kEmitAssembly) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false);
    if (!outFile)
      return failure();

    auto standaloneOr =
        compiler->produceStandaloneAssembly(attr, outFile->os());
    if (failed(standaloneOr))
      return failure();
    outFile->keep();
    return mlir::success();
  }

  // This produces a standalone object for all the objects we requested.
  auto standaloneOr = compiler->produceStandaloneObject(
      attr, /*isJIT=*/clOptions.cmd == Command::kExecute);
  if (failed(standaloneOr) && !clOptions.ignoreFailures)
    return failure();
  Cache::BufferRef standaloneObject = std::move(*standaloneOr);

  // If we're emitting the object, do it.
  if (clOptions.cmd == Command::kEmit) {
    if (failed(clOptions.emitObject(standaloneObject->getBuffer())))
      return failure();

    auto headerPath = clOptions.getHeaderOutputPath();
    // If we have no output path, we can't emit headers so return.
    if (!headerPath)
      return mlir::success();

    // Finish off by producing a header file with the decls.
    return emitHeader(*compiler, *headerPath);
  }

  // Now we can load it into the JIT - we're definitely executing the thing.

  // Now create the execution engine so we can JIT.
  auto engineOr = ExecutionEngine::create(compilationOptions);
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));
  ExecutionEngine engine = std::move(*engineOr);

  if (auto err = engine.add("exec", std::move(standaloneObject)))
    return failure(clOptions.reportError(err.getError()));

  // Helper to execute a func.
  auto execFunc = [&](FuncOp theFunc,
                      const CommandLineFunc &clFunc) -> LogicalResult {
    TimeTraceScope<> traceScope("execute-function", theFunc.getSymName());
    auto compiledFuncOr = engine.lookup("exec", theFunc.getNameAttr());
    if (failed(compiledFuncOr))
      return failure(clOptions.reportError(compiledFuncOr.getError()));

    if (auto err = clFunc.verifyFuncSignature(theFunc.getFunctionType())) {
      mlir::emitError(theFunc.getLoc(), err.getError());
      return mlir::failure(!clOptions.ignoreFailures);
    }

    if (auto err = clFunc.executeAndPrint(*compiledFuncOr)) {
      mlir::emitError(theFunc.getLoc(), err.getError());
      return mlir::failure(!clOptions.ignoreFailures);
    }
    return mlir::success();
  };

  llvm::DenseSet<StringRef> foundFuncs;
  // Loop over the funcs and maybe emit the func as an object file or maybe
  // execute it.
  for (auto fn : theModule->getOps<FuncOp>()) {
    TimeTraceScope<> traceScope("emit", fn.getName());
    foundFuncs.insert(fn.getName());

    // If we were asked to handle this func, do so.
    if (std::optional<CommandLineFunc> clFunc =
            clOptions.shouldExecuteFunc(fn.getName())) {
      switch (clOptions.cmd) {
      case Command::kGenLibraryFile:
      case Command::kElaborate:
      case Command::kEmitLLVM:
      case Command::kEmitAssembly:
      case Command::kEmit:
        break;
      case Command::kExecute: {
        if (failed(execFunc(fn, *clFunc)))
          return failure();
      }
      }
    }
  }

  // Validate that the user didn't pass in any funcs we don't have. This would
  // be super confusing if the user simply gets no response for something that
  // isn't defined, so put up an actual error.
  for (const auto &fn : clOptions.funcs) {
    if (!foundFuncs.count(fn.name))
      return mlir::emitError(theModule->getLoc(),
                             "could not find func '@" + fn.name + "'");
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Initialize the compiler runtime.
  KGEN_CompilerRT_Initialize();

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
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
