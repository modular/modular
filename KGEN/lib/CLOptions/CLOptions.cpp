//===- CLOptions.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"

using namespace M;

//===--------------------------------------------------------------------===//
// CommandLineFunc implementation
//===--------------------------------------------------------------------===//

ErrorOrSuccess
CommandLineFunc::verifyFuncSignature(mlir::FunctionType funcType) const {
  if (signature == "f32()") {
    if (funcType.getNumInputs() != 0 || funcType.getNumResults() != 1 ||
        funcType.getResult(0) !=
            mlir::Float32Type::get(funcType.getContext())) {
      std::string ktype;
      llvm::raw_string_ostream os(ktype);
      os << funcType;
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   ktype + ", but got " + signature);
    }
    return M::success();
  } else if (signature == "()") {
    if (funcType.getNumResults() != 0 || funcType.getNumInputs() != 0) {
      std::string ktype;
      llvm::raw_string_ostream os(ktype);
      os << funcType;
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   ktype + ", but got " + signature);
    }
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

ErrorOrSuccess
CommandLineFunc::executeAndPrint(KGEN::CompiledFunc &compiledFunc) const {
  if (signature == "f32()") {
    printf("--- '%s' returned %f\n", name.c_str(),
           compiledFunc.invoke<float>());
    return M::success();
  } else if (signature == "()") {
    compiledFunc.invoke<void>();
    printf("--- '%s' finished\n", name.c_str());
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

bool CommandLineFuncParser::parse(llvm::cl::Option &o, StringRef argName,
                                  StringRef argValue, CommandLineFunc &val) {
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
