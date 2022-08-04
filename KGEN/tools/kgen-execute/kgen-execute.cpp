//===- kgen-execute.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "Support/CommonCLOptions.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/InitLLVM.h"
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

namespace {
/// This struct provides a way to parse a kernel name and signature from the
/// command line. It's a simple abstraction that allows
struct ExecutableKernel {
  std::string name;
  std::string signature;

  ErrorOrSuccess executeAndPrint(KGEN::ExecutionEngine &engine);
  ErrorOrSuccess verifyKernelSignature(mlir::LLVM::LLVMFuncOp kernelOp);
};

/// Parse ExecutableKernel objects from the command line flags provided.
class ExecutableKernelParser : public llvm::cl::parser<ExecutableKernel> {
public:
  using llvm::cl::parser<ExecutableKernel>::parser;

  bool parse(llvm::cl::Option &O, StringRef ArgName, StringRef ArgValue,
             ExecutableKernel &Val);
};

class CLOptions : public CommonCLOptions {
public:
  CLOptions(StringRef programName) : CommonCLOptions(programName) {}

  cl::list<ExecutableKernel, bool, ExecutableKernelParser> kernelsToExecute{
      "run-kernel",
      cl::desc(
          "Names and signatures of kernels to execute. Each name must match "
          "the kernel's symbol name exactly, and the signature must as well.")};
};
} // namespace

//===--------------------------------------------------------------------===//
// ExecutableKernelParser implementation
//===--------------------------------------------------------------------===//

bool ExecutableKernelParser::parse(llvm::cl::Option &O, StringRef ArgName,
                                   StringRef ArgValue, ExecutableKernel &Val) {
  // Split at the colon.
  auto [kernelName, kernelSignature] = ArgValue.split(':');

  Val.name = kernelName;
  Val.signature = kernelSignature;

  return false;
}

//===--------------------------------------------------------------------===//
// ExecutableKernel implementation
//===--------------------------------------------------------------------===//

ErrorOrSuccess
ExecutableKernel::executeAndPrint(KGEN::ExecutionEngine &engine) {
  if (signature == "f32()") {
    auto outOr = engine.invoke<float>(name);
    if (outOr.isError())
      return outOr.takeError();

    printf("--- Kernel '%s' returned %f\n", name.c_str(), *outOr);
  }

  return M::success();
}

ErrorOrSuccess
ExecutableKernel::verifyKernelSignature(mlir::LLVM::LLVMFuncOp kernelOp) {
  if (signature == "f32()") {
    if (kernelOp.getFunctionType().getNumParams() != 0 ||
        kernelOp.getFunctionType().getReturnType() !=
            mlir::Float32Type::get(kernelOp.getContext()))
      return Error("kernel signature does not match the IR signature.");
  }

  return M::success();
}

//===--------------------------------------------------------------------===//
// main
//===--------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  CLOptions clOptions(argv[0]);
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
        std::move(chunkBuffer),
        [&](MLIRContext *ctx, llvm::SourceMgr &sourceMgr) {
          ctx->appendDialectRegistry(getDialects());
          ctx->loadAllAvailableDialects();

          // Open the input file.
          OwningOpRef<ModuleOp> module(
              parseSourceFile<ModuleOp>(sourceMgr, ctx));
          if (!module)
            return failure(clOptions.reportError("could not parse input file"));

          SymbolTable symtab(*module);
          for (auto k : clOptions.kernelsToExecute) {
            auto kernel = symtab.lookup<LLVM::LLVMFuncOp>(k.name);
            if (!kernel)
              return failure(clOptions.reportError("could not find kernel '" +
                                                   k.name + "'."));

            if (auto err = k.verifyKernelSignature(kernel))
              return failure(clOptions.reportError(err.getError()));

            if (auto err = execEngine.add(kernel))
              return failure(clOptions.reportError(err.getError()));

            if (auto err = k.executeAndPrint(execEngine))
              return failure(clOptions.reportError(err.getError()));
          }

          return mlir::success();
        });
  };

  // Process the file.
  return failed(
      splitAndProcessBuffer(std::move(inputFile), toolFn, llvm::outs(), false));
}
