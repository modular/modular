//===- CLOptions.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"

using namespace M;

//===--------------------------------------------------------------------===//
// ExecutableKernelParser implementation
//===--------------------------------------------------------------------===//

bool ExecutableKernelParser::parse(llvm::cl::Option &o, StringRef argName,
                                   StringRef argValue, ExecutableKernel &val) {
  std::tie(val.name, val.signature) = argValue.split(':');

  return false;
}

//===--------------------------------------------------------------------===//
// EmittableKernelParser implementation
//===--------------------------------------------------------------------===//

bool EmittableKernelParser::parse(llvm::cl::Option &o, StringRef argName,
                                  StringRef argValue, EmittableKernel &val) {
  std::tie(val.name, val.outputFilename) = argValue.split(':');

  return false;
}

//===--------------------------------------------------------------------===//
// ExecutableKernel implementation
//===--------------------------------------------------------------------===//

ErrorOrSuccess ExecutableKernel::verifyKernelSignature(
    mlir::LLVM::LLVMFunctionType kernelType) const {
  if (signature == "f32()") {
    if (kernelType.getNumParams() != 0 ||
        kernelType.getReturnType() !=
            mlir::Float32Type::get(kernelType.getContext()))
      return Error(
          "command-line specified signature does not match the IR signature.");
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

ErrorOrSuccess
ExecutableKernel::executeAndPrint(KGEN::ExecutionEngine &engine) const {
  if (signature == "f32()") {
    auto outOr = engine.invoke<float>(name);
    if (outOr.isError())
      return outOr.takeError();

    printf("--- Kernel '%s' returned %f\n", name.c_str(), *outOr);
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}
