//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/MLIRToString.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

LLVM_DUMP_METHOD std::string M::mlirToString(mlir::Operation *op) {
  // The implementation of debugString(op) prints a pointer, unhelpful.
  std::string outStr;
  llvm::raw_string_ostream out(outStr);
  out << *op;
  return out.str();
}

LLVM_DUMP_METHOD std::string M::mlirToString(mlir::Attribute attr) {
  return debugString(attr);
}

LLVM_DUMP_METHOD std::string M::mlirToString(mlir::Type type) {
  return debugString(type);
}
