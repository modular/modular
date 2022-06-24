//===- kgen-generate.cpp - The kgen-generate driver -----------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Internals.h"
#include "KGEN/InitAllDialects.h"
#include "Support/CommonCLOptions.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
using namespace M;

static DialectRegistry getDialects() {
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registry.insert<mlir::arith::ArithmeticDialect>();
  return registry;
}

struct CLOptions : public CommonCLOptions {
  CLOptions(const char *toolName) : toolName(toolName) {}

  // This is argv[0] of the invoking command.
  const char *const toolName;

  // Emit an error prefixed with the argv[0] tool name.
  int reportError(Twine message) const {
    llvm::errs() << toolName << ": " << message << '\n';
    return 1;
  }

  //===--------------------------------------------------------------------===//
  // Library specification.

  cl::opt<std::string> libraryFilename{"library", cl::desc("<input file>")};

  /// Open the filename specified on the command line and return a memory
  /// buffer, or an error message on failure.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> openLibraryFile() const {
    std::string errorMsg;
    auto result = mlir::openInputFile(libraryFilename, &errorMsg);
    if (result)
      return result;
    return Error(errorMsg);
  }

  std::unique_ptr<llvm::MemoryBuffer> openLibraryFileOrExit() const {
    auto result = openLibraryFile();
    if (failed(result))
      exit(reportError(result.getError()));
    return result.takeValue();
  }

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
static void processFile(MLIRContext *ctx, llvm::SourceMgr &sourceMgr,
                        const CLOptions &options) {
  ctx->appendDialectRegistry(getDialects());
  ctx->loadAllAvailableDialects();
  ctx->allowUnregisteredDialects(true);
  ctx->printOpOnDiagnostic(false);

  // Open the input file.
  OwningOpRef<ModuleOp> module(mlir::parseSourceFile<ModuleOp>(sourceMgr, ctx));
  if (!module)
    exit(1);

  // Open the library file.
  std::unique_ptr<llvm::MemoryBuffer> libraryFile =
      options.openLibraryFileOrExit();
  OwningOpRef<ModuleOp> library(
      mlir::parseSourceString<ModuleOp>(libraryFile->getBuffer(), ctx));
  if (!library)
    exit(1);

  if (failed(generateKernels(module.get(), library.get())))
    exit(1);

  std::unique_ptr<llvm::ToolOutputFile> outputFile = options.getOutputFile();
  module->print(outputFile->os());
  outputFile->keep();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Enable command line options for various MLIR internals.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  CLOptions options(argv[0]);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      options.openInputFileOrExit(argv[0]);

  return failed(options.configureMLIRContextAndSourceMgrAndExecute(
      std::move(inputFile), [&](MLIRContext *ctx, llvm::SourceMgr &sourceMgr) {
        processFile(ctx, sourceMgr, options);
      }));
}
