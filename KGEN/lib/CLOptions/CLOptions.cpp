//===- CLOptions.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"

using namespace M;

//===--------------------------------------------------------------------===//
// CommandLineKernel implementation
//===--------------------------------------------------------------------===//

ErrorOrSuccess CommandLineKernel::verifyKernelSignature(
    mlir::LLVM::LLVMFunctionType kernelType) const {
  if (signature == "f32()") {
    if (kernelType.getNumParams() != 0 ||
        kernelType.getReturnType() !=
            mlir::Float32Type::get(kernelType.getContext())) {
      std::string ktype;
      llvm::raw_string_ostream os(ktype);
      os << kernelType;
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   ktype + ", but got " + signature);
    }
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

ErrorOrSuccess
CommandLineKernel::executeAndPrint(KGEN::ExecutionEngine &engine) const {
  if (signature == "f32()") {
    auto outOr = engine.invoke<float>(name);
    if (outOr.isError())
      return outOr.takeError();

    printf("--- Kernel '%s' returned %f\n", name.c_str(), *outOr);
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

bool CommandLineKernelParser::parse(llvm::cl::Option &o, StringRef argName,
                                    StringRef argValue,
                                    CommandLineKernel &val) {
  SmallVector<StringRef, 3> parts;
  argValue.split(parts, ':');

  if (parts.size() != 3)
    return o.error("'" + argValue +
                   "' invalid: must provide name:signature:filename");

  val.name = parts[0];
  val.signature = parts[1];
  val.outputFilename = parts[2];
  return false;
}
