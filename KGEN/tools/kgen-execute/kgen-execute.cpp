//===- kgen-execute.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "Support/CommonCLOptions.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/ToolOutputFile.h"
using namespace M;
using namespace mlir;

/// Get all the dialects the executor tool will need, and register the
/// conversion to LLVMIR.
static DialectRegistry getDialects() {
  DialectRegistry registry;
  mlir::registerLLVMDialectTranslation(registry);
  return registry;
}

class CLOptions : public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  cl::list<ExecutableKernel, bool, ExecutableKernelParser> kernelsToExecute{
      "run-kernel",
      cl::desc(
          "Names and signatures of kernels to execute. Each name must match "
          "the kernel's unmangled symbol name exactly, and the signature must "
          "as well.")};

  cl::list<EmittableKernel, bool, EmittableKernelParser> kernelsToEmit{
      "emit-kernel",
      cl::desc("Names and output filenames of kernels to emit. Each name must "
               "match the kernel's unmangled symbol name exactly.")};
};

//===--------------------------------------------------------------------===//
// ProcessBuffer
//===--------------------------------------------------------------------===//

namespace {
/// This struct essentially provides the body of the execution flow that we pass
/// to configureMLIRContextAndSourceMgrAndExecute. It's long enough that we
/// don't want to have it inline, and pulling it out into a functor makes it
/// more readable.
struct ProcessBuffer {
  KGEN::ExecutionEngine &execEngine;
  CLOptions &clOptions;

  LogicalResult operator()(MLIRContext *ctx, llvm::SourceMgr &sourceMgr) const {
    ctx->appendDialectRegistry(getDialects());
    ctx->loadAllAvailableDialects();

    // Open the input file.
    OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, ctx));
    if (!module)
      return failure(clOptions.reportError("could not parse input file"));

    SymbolTable symtab(*module);
    auto lookupKernel =
        [&](StringRef kernelName) -> ErrorOr<mlir::LLVM::LLVMFuncOp> {
      auto kernel = symtab.lookup<LLVM::LLVMFuncOp>(kernelName);
      if (!kernel)
        return Error("could not find kernel '" + kernelName + "'.");
      return kernel;
    };

    for (const auto &k : clOptions.kernelsToEmit) {
      auto kernelOr = lookupKernel(k.name);
      if (kernelOr.isError())
        return failure(clOptions.reportError(kernelOr.getError()));

      // Compile the kernel.
      if (auto err = execEngine.add(*kernelOr))
        return failure(clOptions.reportError(err.getError()));

      // Get the compiled object.
      auto objOr = execEngine.getObject(*kernelOr);
      if (objOr.isError())
        return failure(clOptions.reportError(objOr.getError()));

      // Open the output file and write the compiled object to it.
      std::string errMsg;
      auto outFile = mlir::openOutputFile(k.outputFilename, &errMsg);
      if (!outFile)
        return failure(clOptions.reportError(errMsg));

      outFile->os().write((*objOr)->getBufferStart(),
                          (*objOr)->getBufferSize());
      outFile->keep();
    }

    for (const auto &k : clOptions.kernelsToExecute) {
      auto kernelOr = lookupKernel(k.name);
      if (kernelOr.isError())
        return failure(clOptions.reportError(kernelOr.getError()));

      auto kernel = *kernelOr;
      if (auto err = k.verifyKernelSignature(kernel.getFunctionType()))
        return failure(clOptions.reportError(err.getError()));

      if (auto err = execEngine.add(kernel))
        return failure(clOptions.reportError(err.getError()));

      if (auto err = k.executeAndPrint(execEngine))
        return failure(clOptions.reportError(err.getError()));
    }

    return mlir::success();
  }
};
} // namespace

//===--------------------------------------------------------------------===//
// main
//===--------------------------------------------------------------------===//

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      clOptions.openInputFileOrExit();

  auto engineOr = KGEN::ExecutionEngine::create();
  if (engineOr.isError())
    clOptions.reportError(engineOr.getError());

  auto execEngine = std::move(*engineOr);

  // Provide a tool function that runs the requested ops, again, so we can
  // re-use it.
  auto toolFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                    raw_ostream &os) {
    return clOptions.configureMLIRContextAndSourceMgrAndExecute(
        std::move(chunkBuffer), ProcessBuffer{execEngine, clOptions});
  };

  // Process the file.
  return failed(
      splitAndProcessBuffer(std::move(inputFile), toolFn, llvm::outs(), false));
}
