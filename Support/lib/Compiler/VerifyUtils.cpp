//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/VerifyUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"

mlir::LogicalResult M::checkResultTypes(Operation *op,
                                        TypeRange expectedTypes) {
  if (op->getNumOperands() != expectedTypes.size()) {
    return op->emitOpError("expected ")
           << expectedTypes.size() << " operands for enclosing op";
  }

  for (size_t i = 0, e = op->getNumOperands(); i != e; ++i) {
    auto t = op->getOperand(i).getType();
    if (t != expectedTypes[i]) {
      return op->emitOpError("operand #")
             << i << " has type " << t << " but should be " << expectedTypes[i];
    }
  }
  return mlir::success();
}
