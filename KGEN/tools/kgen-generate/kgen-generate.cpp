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
  OwningOpRef<ModuleOp> primaryModule(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, ctx));
  if (!primaryModule)
    exit(1);

  OwningOpRef<ModuleOp> libraryModule(
      mlir::parseSourceFile<ModuleOp>(options.libraryFilename, ctx));
  if (!libraryModule)
    exit(1);

  // Generate kernels for the primary module.  If any errors are emitted, we let
  // the current diagnostic handler decide what to do with them.
  // -verify-diagnostics doesn't consider errors to be a tool failure if they
  // are matched correctly.
  (void)generateKernels(primaryModule.get(), libraryModule.get());

  std::unique_ptr<llvm::ToolOutputFile> outputFile = options.getOutputFile();
  primaryModule->print(outputFile->os());
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
