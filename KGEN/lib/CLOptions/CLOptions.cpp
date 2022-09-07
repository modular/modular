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

ErrorOrSuccess
CommandLineKernel::verifyKernelSignature(mlir::FunctionType kernelType) const {
  if (signature == "f32()") {
    if (kernelType.getNumInputs() != 0 || kernelType.getNumResults() != 1 ||
        kernelType.getResult(0) !=
            mlir::Float32Type::get(kernelType.getContext())) {
      std::string ktype;
      llvm::raw_string_ostream os(ktype);
      os << kernelType;
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   ktype + ", but got " + signature);
    }
    return M::success();
  } else if (signature == "()") {
    if (kernelType.getNumResults() != 0 || kernelType.getNumInputs() != 0) {
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
CommandLineKernel::executeAndPrint(KGEN::CompiledKernel &compiledKernel) const {
  if (signature == "f32()") {
    printf("--- Kernel '%s' returned %f\n", name.c_str(),
           compiledKernel.invoke<float>());
    return M::success();
  } else if (signature == "()") {
    compiledKernel.invoke<void>();
    printf("--- Kernel '%s' finished\n", name.c_str());
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

bool CommandLineKernelParser::parse(llvm::cl::Option &o, StringRef argName,
                                    StringRef argValue,
                                    CommandLineKernel &val) {
  SmallVector<StringRef, 3> parts;
  argValue.split(parts, ':');

  // If only 2 are provided, parse it into name + output filename.
  if (parts.size() == 2) {
    val.name = parts[0];
    val.outputFilename = parts[1];
    return false;
  }

  if (parts.size() != 3)
    return o.error("'" + argValue +
                   "' invalid: must provide name:signature:filename");

  val.name = parts[0];
  val.signature = parts[1];
  val.outputFilename = parts[2];
  return false;
}
