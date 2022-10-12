//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "llvm/Support/ToolOutputFile.h"
#include <filesystem>

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
  SmallVector<StringRef, 2> parts;
  argValue.split(parts, ':');

  // If only one is provided, parse it into just a name.
  if (parts.size() == 1) {
    val.name = parts[0];
    return false;
  }

  if (parts.size() != 2)
    return o.error("'" + argValue + "' invalid: must provide name:signature");

  val.name = parts[0];
  val.signature = parts[1];
  return false;
}

std::string KGENCLOptions::getOutputPath() const {
  if (outputFilename.empty() || outputFilename == "-")
    return "";

  // If the filename is not provided, then default to the current working
  // directory.
  std::filesystem::path objPath =
      std::filesystem::absolute(outputFilename.getValue());

  return objPath.string();
}

LogicalResult
KGENCLOptions::emitObject(std::unique_ptr<llvm::MemoryBuffer> object) const {
  std::unique_ptr<llvm::ToolOutputFile> outFile =
      getOutputFile(/*hasBinaryOutput=*/true);
  if (!outFile)
    return failure();

  outFile->os().write(object->getBufferStart(), object->getBufferSize());
  outFile->keep();

  return mlir::success();
}
