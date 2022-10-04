//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "EmitFuncHeader.h"
#include "EmitFuncObject.h"
#include "KGEN/CLOptions.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/Elaborator.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/CommonCLOptions.h"
#include "Support/IndexDialect/IndexDialect.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
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

  cl::opt<bool> ignoreFailures{
      "ignore-failure",
      cl::desc("Ignore execution failures. Any messages are still printed, but "
               "failures don't mean the tool fails to execute.")};

  cl::list<std::string> searchPaths{
      "I", cl::desc("Path to use to search for included files.")};

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

/// This function creates the elaborator pass and forwards the correct
/// arguments. If it fails, it fails with a fatal error.
static std::unique_ptr<Pass> createElaboratorPass(const CLOptions &clOptions) {
  auto elaborate = createElaborateGenerators();
  std::string includes;
  llvm::raw_string_ostream includeStr(includes);
  for (StringRef include : clOptions.searchPaths)
    includeStr << "search-path=" << include << " ";

  if (failed(elaborate->initializeOptions(includeStr.str())))
    llvm::report_fatal_error("unable to initialize elaborator options");

  return elaborate;
}

/// Emit the IR for `theModule` to a file.
static LogicalResult emitModuleIR(ModuleOp theModule, const CLOptions &opts) {
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

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the funcs it operates on.
static LogicalResult runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                                     const CLOptions &clOptions) {
  DialectRegistry registry;

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<index::IndexDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect>();

  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx->allowUnregisteredDialects();

  OwningOpRef<ModuleOp> theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

  // Set up the pass pipeline.
  mlir::PassManager pm(ctx);
  pm.addPass(createLowerLIT());

  // FIXME: This has to be disabled to avoid lowering buffer types before
  // elaboration.
  // pm.addPass(createLowerZAPToPOPPass());

  pm.addPass(mlir::createCanonicalizerPass());
  if (clOptions.cmd != Command::kGenLibraryFile)
    pm.addPass(createElaboratorPass(clOptions));

  // Run the pass manager.
  if (failed(pm.run(*theModule)))
    return failure(clOptions.reportError("compilation failed"));

  // If all we're doing is generating a library file or elaborating, we're done
  // now.
  if (clOptions.cmd == Command::kGenLibraryFile ||
      clOptions.cmd == Command::kElaborate)
    return emitModuleIR(*theModule, clOptions);

  // Now create the execution engine so we can JIT.
  auto engineOr = ExecutionEngine::create();
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));

  ExecutionEngine engine = std::move(*engineOr);

  // Add the module to the execution engine. This will perform all the slicing
  // necessary.
  if (auto err = engine.add(*theModule))
    return failure(clOptions.reportError(err.getError()));

  // Helper to execute a func.
  auto execFunc = [&](FuncOp theFunc,
                      const CommandLineFunc &clFunc) -> LogicalResult {
    auto compiledFuncOr = engine.lookup(theFunc);
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
    foundFuncs.insert(fn.getName());

    // If we were asked to handle this func, do so.
    if (Optional<CommandLineFunc> clFunc = clOptions.shouldHandleFunc(fn)) {
      switch (clOptions.cmd) {
      case Command::kGenLibraryFile:
      case Command::kElaborate:
        break;
      case Command::kEmit: {
        // If the filename is not provided, then default to the current working
        // directory.
        std::filesystem::path objPath = clFunc->outputFilename;
        if (!objPath.is_absolute())
          objPath = std::filesystem::current_path() / clFunc->outputFilename;

        if (failed(emitObjectForFunc(engine, fn, objPath)))
          return failure();

        if (failed(emitHeaderForFunc(fn,
                                     objPath.replace_extension(".h").string())))
          return failure();
        break;
      }
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
  clOptions.addInputFilesToSourceMgrOrExit(sourceManager);

  return failed(clOptions.configureMLIRContextAndExecute(
      sourceManager, [&](MLIRContext *ctx) -> LogicalResult {
        return runToolPipeline(ctx, sourceManager, clOptions);
      }));
}
