//===- kgen-elaborate.cpp - The kgen-elaborate driver ---------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Internals.h"
#include "KGEN/InitAllDialects.h"
#include "Support/CommonCLOptions.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
using namespace M;

static DialectRegistry getDialects() {
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registry.insert<mlir::arith::ArithmeticDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect>();
  return registry;
}

class CLOptions : public CommonCLOptions {
public:
  CLOptions(StringRef programName) : CommonCLOptions(programName) {}

  //===--------------------------------------------------------------------===//
  // Input specification

  cl::opt<bool> splitInputFile{
      "split-input-file",
      cl::desc("Split the input file into pieces and process each "
               "chunk independently"),
      cl::init(false)};

  //===--------------------------------------------------------------------===//
  // Library specification.

  cl::opt<std::string> libraryFilename{"library", cl::desc("<input file>")};

  //===--------------------------------------------------------------------===//
  // Emission Options

  cl::opt<std::string> outputFilename{"o", cl::desc("Output filename"),
                                      cl::value_desc("filename"),
                                      cl::init("-")};

  // Determine an output file name and open it.
  std::unique_ptr<llvm::ToolOutputFile> getOutputFile() const {
    std::string errorMessage;
    auto result =
        mlir::openOutputFile(outputFilename.getValue(), &errorMessage);
    if (!result)
      exit(reportError(errorMessage));
    return result;
  }
};

/// Generate all the kernels specified with a single input file.  This requires
/// parsing the file and its library.
static LogicalResult processFile(MLIRContext *ctx, llvm::SourceMgr &sourceMgr,
                                 llvm::raw_ostream &outputStream,
                                 const CLOptions &clOptions) {
  ctx->appendDialectRegistry(getDialects());
  ctx->loadAllAvailableDialects();
  ctx->allowUnregisteredDialects(true);
  ctx->printOpOnDiagnostic(false);

  // Open the input file.
  OwningOpRef<ModuleOp> primaryModule(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, ctx));
  if (!primaryModule)
    return failure(clOptions.reportError("could not parse input file"));

  OwningOpRef<ModuleOp> libraryModule(
      mlir::parseSourceFile<ModuleOp>(clOptions.libraryFilename, ctx));
  if (!libraryModule) {
    return failure(clOptions.reportError(
        Twine("could not parse library file: ") + clOptions.libraryFilename));
  }

  // Elaborate kernels for the primary module.  If any errors are emitted, we
  // let the current diagnostic handler decide what to do with them.
  // -verify-diagnostics doesn't consider errors to be a tool failure if they
  // are matched correctly.
  if (succeeded(elaborateKernels(primaryModule.get(), libraryModule.get()))) {
    // If the generator thought it succeeded, double check that the IR is valid.
    (void)verify(primaryModule.get());
  }

  primaryModule->print(outputStream);
  return success();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Enable command line options for various MLIR internals.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  CLOptions clOptions(argv[0]);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      clOptions.openInputFileOrExit();

  // Get the output file now so that we can use it in the lambdas below.
  std::unique_ptr<llvm::ToolOutputFile> outputFile = clOptions.getOutputFile();

  // Provide a tool function that runs the requested ops, again, so we can
  // re-use it.
  auto toolFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                    raw_ostream &os) {
    return clOptions.configureMLIRContextAndSourceMgrAndExecute(
        std::move(chunkBuffer),
        [&](MLIRContext *ctx, llvm::SourceMgr &sourceMgr) {
          return processFile(ctx, sourceMgr, os, clOptions);
        });
  };

  // Either split the input file (or don't) and process.
  auto result = mlir::splitAndProcessBuffer(
      std::move(inputFile), toolFn, outputFile->os(), clOptions.splitInputFile);
  // Only keep the output file if we succeeded.
  if (succeeded(result))
    outputFile->keep();

  return failed(result);
}
