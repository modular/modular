//===- GenerateKernels.cpp - Kernel generator driver ----------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains logic to lower a file full of kernel generators into
//
//===----------------------------------------------------------------------===//

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
using namespace M;

namespace M {
LogicalResult generateKernels(ModuleOp module, ModuleOp library);
}

LogicalResult M::generateKernels(ModuleOp module, ModuleOp library) {
  // TODO!
  return success();
}
