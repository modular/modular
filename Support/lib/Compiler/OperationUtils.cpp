//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringExtras.h"

bool M::operationIsIsolatedFromAbove(Operation *op,
                                     SmallVectorImpl<Value> *captures) {
  bool result = true;
  op->walk<mlir::WalkOrder::PreOrder>([&](Operation *nested) {
    // Skip over isolated operations. There's nothing to check in them.
    if (nested->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      return WalkResult::skip();

    for (Value operand : nested->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        // If the top-level operation does not contain the defining op, this
        // value is captured from above.
        if (!op->isAncestor(defOp)) {
          result = false;
          if (captures)
            captures->push_back(operand);
        }
      } else {
        Block *parent = cast<BlockArgument>(operand).getParentBlock();
        // If the defining block contains the top-level operation, the block
        // argument is captured from above.
        if (parent->findAncestorOpInBlock(*op)) {
          result = false;
          if (captures)
            captures->push_back(operand);
        }
      }
    }
    return WalkResult::advance();
  });
  return result;
}

std::string M::getUniqueSymbolName(std::string baseName, SymbolTable &symtab,
                                   unsigned &counter) {
  std::string uniqueName = baseName;
  while (symtab.lookup(uniqueName))
    uniqueName = (baseName + "_" + Twine(counter++)).str();
  return uniqueName;
}

std::string M::makeCIdentifier(StringRef ident) {
  std::string res(ident.str());
  for (char &c : res)
    // Only allow [0-9a-zA-Z_].
    if (!llvm::isAlnum(c) && c != '_')
      c = '_';
  return res;
}
