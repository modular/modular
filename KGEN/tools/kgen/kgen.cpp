//===- kgen.cpp -----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "EmitKernelHeader.h"
#include "EmitKernelObject.h"
#include "KGEN/CLOptions.h"
#include "KGEN/Elaborator.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/CommonCLOptions.h"
#include "Support/IndexDialect/IndexDialect.h"
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
  auto elaborate = createElaborateGeneratorsPass();
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
  // TODO: change this to `true` when we emit the module in its binary format.
  auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/false);
  if (!outFile)
    return mlir::failure();

  theModule->print(outFile->os());
  outFile->keep();
  return mlir::success();
}

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the kernels it operates on.
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
  pm.addPass(createLowerHLKGENPass());
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

  // Helper to execute a kernel.
  auto execKernel = [&](KernelOp theKernel,
                        const CommandLineKernel &clKernel) -> LogicalResult {
    auto compiledKernelOr = engine.lookup(theKernel);
    if (failed(compiledKernelOr))
      return failure(clOptions.reportError(compiledKernelOr.getError()));

    if (auto err =
            clKernel.verifyKernelSignature(theKernel.getFunctionType())) {
      mlir::emitError(theKernel.getLoc(), err.getError());
      return mlir::failure(!clOptions.ignoreFailures);
    }

    if (auto err = clKernel.executeAndPrint(*compiledKernelOr)) {
      mlir::emitError(theKernel.getLoc(), err.getError());
      return mlir::failure(!clOptions.ignoreFailures);
    }
    return mlir::success();
  };

  llvm::DenseSet<StringRef> foundKernels;
  // Loop over the kernels and maybe emit the kernel as an object file or maybe
  // execute it.
  for (auto k : theModule->getOps<KernelOp>()) {
    foundKernels.insert(k.getName());

    // If we were asked to handle this kernel, do so.
    if (Optional<CommandLineKernel> clKernel =
            clOptions.shouldHandleKernel(k)) {
      switch (clOptions.cmd) {
      case Command::kGenLibraryFile:
      case Command::kElaborate:
        break;
      case Command::kEmit: {
        // If the filename is not provided, then default to the current working
        // directory.
        std::filesystem::path objPath = clKernel->outputFilename;
        if (!objPath.is_absolute())
          objPath = std::filesystem::current_path() / clKernel->outputFilename;

        if (failed(emitObjectForKernel(engine, k, objPath)))
          return failure();

        if (failed(emitHeaderForKernel(
                k, objPath.replace_extension(".h").string())))
          return failure();
        break;
      }
      case Command::kExecute: {
        if (failed(execKernel(k, *clKernel)))
          return failure();
      }
      }
    }
  }

  // Validate that the user didn't pass in any kernels we don't have. This would
  // be super confusing if the user simply gets no response for something that
  // isn't defined, so put up an actual error.
  for (const auto &k : clOptions.kernels) {
    if (foundKernels.find(k.name) == foundKernels.end())
      return mlir::emitError(theModule->getLoc(),
                             "could not find kernel '@" + k.name + "'");
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

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
