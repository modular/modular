//===- kgen-execute.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "Support/CommonCLOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/ToolOutputFile.h"
using namespace M;
using namespace mlir;

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
  KGENCLOptions &clOptions;

  LogicalResult operator()(MLIRContext *ctx, llvm::SourceMgr &sourceMgr) const {
    DialectRegistry registry;
    // Don't need HLKGEN here.
    registry.insert<KGEN::KGENDialect, KGEN::MetaDialect, KGEN::POPDialect>();
    mlir::registerLLVMDialectTranslation(registry);

    ctx->appendDialectRegistry(registry);
    ctx->loadAllAvailableDialects();

    // Open the input file.
    OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, ctx);
    if (!module)
      return failure(clOptions.reportError("could not parse input file"));

    SymbolTable symtab(*module);
    auto lookupKernel = [&](StringRef kernelName) -> ErrorOr<KGEN::KernelOp> {
      auto kernel = symtab.lookup<KGEN::KernelOp>(kernelName);
      if (!kernel)
        return Error("could not find kernel '" + kernelName + "'.");
      return kernel;
    };

    // Add the module to the execution engine.
    if (auto err = execEngine.add(*module))
      return failure(clOptions.reportError(err.getError()));

    for (const auto &k : clOptions.kernels) {
      auto kernelOr = lookupKernel(k.name);
      if (kernelOr.isError())
        return failure(clOptions.reportError(kernelOr.getError()));

      KGEN::KernelOp kernel = *kernelOr;

      // And now we diverge.
      switch (clOptions.cmd) {
      case Command::kExecute: {
        if (auto err = k.verifyKernelSignature(kernel.getFunctionType()))
          return failure(clOptions.reportError(err.getError()));

        if (auto err = k.executeAndPrint(execEngine))
          return failure(clOptions.reportError(err.getError()));
        break;
      }
      case Command::kEmit: {
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
        break;
      }
      }
    }

    return mlir::success();
  }
};
} // namespace

//===--------------------------------------------------------------------===//
// main
//===--------------------------------------------------------------------===//

int main(int argc, char **argv) {
  KGENCLOptions clOptions(argc, argv);

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
