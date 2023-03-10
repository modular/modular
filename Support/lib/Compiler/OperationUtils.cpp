//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"

bool M::operationIsIsolatedFromAbove(Operation *op,
                                     llvm::SetVector<Value> *captures,
                                     bool allowIsolated) {
  bool result = true;
  op->walk<mlir::WalkOrder::PreOrder>([&](Operation *nested) {
    // Skip over isolated operations. There's nothing to check in them.
    if (!allowIsolated &&
        nested->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      return WalkResult::skip();

    for (Value operand : nested->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        // If the top-level operation does not contain the defining op, this
        // value is captured from above.
        if (!op->isAncestor(defOp)) {
          result = false;
          if (captures)
            captures->insert(operand);
        }
      } else {
        Block *parent = cast<BlockArgument>(operand).getParentBlock();
        // If the defining block contains the top-level operation, the block
        // argument is captured from above.
        if (parent->findAncestorOpInBlock(*op)) {
          result = false;
          if (captures)
            captures->insert(operand);
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

std::string M::makeCWrapperName(StringRef ident) {
  std::string res(ident.str());
  if (!ident.empty() && !llvm::isAlnum(res[0]))
    res[0] = 'x';
  for (char &c : res)
    // Only allow [0-9a-zA-Z_].
    if (!llvm::isAlnum(c) && c != '_')
      c = '_';
  res.append("_c");
  return res;
}

bool M::isCIdentifier(StringRef ident) {
  if (ident.empty() || !llvm::isAlnum(ident[0]))
    return false;
  for (char c : ident)
    // Only allow [0-9a-zA-Z_].
    if (!llvm::isAlnum(c) && c != '_')
      return false;
  return true;
}
